"""Admit, audit and time a guest-proposed four-source-step load block."""
import argparse
from collections import defaultdict
import json
import math
import statistics
import struct
from time import perf_counter_ns
from unittest.mock import patch

import torch
import wasmtime

import terrain_verify as terrain
from prove_composed_loads import OUT, VERSION, bindings
from replay_frozen_multistep import CALLS, state_equal
from replay_frozen_recurrent import ARCHIVE, PACKED, graph_case, sha, signed, wasmtime_run
from replay_guest_rewrite import EXECUTOR, invoke
from replay_reusable_region_rules import compare_access
from replay_rule_portfolio import (
    OUT as PORTFOLIO, PAIR, PRIOR, check_rules, prepare, verify_histories, with_rows,
)
from visuaml.compiler_lab.frozen_calls_v58 import initialize, load_call_image

PATTERN = ("local.get", "i32.load", "local.get", "i32.load")


def certificate():
    check_rules()
    raw = (OUT / "rules.json").read_bytes()
    proof = json.loads(raw)
    if (sha(raw) != (OUT / "rules.sha256").read_text().strip() or proof["version"] != VERSION
            or proof["bindings"] != bindings() or len(proof["obligations"]) != 5
            or any(row["result"] != "unsat" for row in proof["obligations"])):
        raise ValueError("Unbound composed rule")
    return sha(raw)


def proposals(base, planner, pc, source, plans, challenge=False):
    descriptors = [word for i, op in enumerate(source)
                   for word in (int(op != PATTERN[i]), pc + i)]
    flags = [int(plans[pc][0] != 2), int(plans[pc + 2][0] != 2)]
    operands = [signed(x) for i in (pc, pc + 2) for x in plans[i][1:]]
    words = [*descriptors, *flags, int(challenge), *operands]
    page = bytearray(65536)
    struct.pack_into("<27i", page, 512, *words, *([-1] * 12))
    memory, replay = invoke(base, planner, 512, page)
    candidate = struct.unpack_from("<12i", memory, 572)
    assert replay["status"] == 1 and replay["return_i32"] == candidate[0]
    assert wasmtime_run(planner, 512, page) == (candidate[0], memory)
    assert memory[:572] == page[:572] and memory[620:] == page[620:]
    return {"input_words": words, "emitted_plan": candidate, "optimizer_replay": replay}


def accept(proposal, pc, plans):
    operands = [signed(x) for i in (pc, pc + 2) for x in plans[i][1:]]
    if tuple(proposal["emitted_plan"]) != (1, pc, pc + 1, pc + 2, pc + 3, 2, 4, 0, *operands):
        raise ValueError("Composition disagrees with the certified source-boundary schedule")


def discover(plain, candidates, selected):
    chosen = set(selected)
    # Include route targets conservatively, even if a dynamic guard rarely takes them.
    incoming = defaultdict(set)
    for source, row in enumerate(plain.rows[:-1].tolist()):
        for target in set(row[3:5]):
            incoming[target].add(source)
    sites = {}
    for start, function in zip(plain.starts, plain.functions, strict=True):
        for rel in range(len(function) - 4):
            pc = start + rel
            if tuple(function[rel:rel + 4]) != PATTERN or not {pc, pc + 2} <= chosen:
                continue
            if any(candidates[i][0] != 2 for i in (pc, pc + 2)):
                continue
            rows = plain.rows[pc:pc + 4].tolist()
            if any(row[3:5] != [pc + i + 1] * 2 or any(row[5:8])
                   for i, row in enumerate(rows)):
                continue
            if any(incoming[inner] != {inner - 1} for inner in range(pc + 1, pc + 4)):
                continue
            sites[pc] = tuple(function[rel:rel + 4])
    return sites


def extend(image, plans, chosen_pc):
    extra = torch.full((len(image.rows), 2), -1, dtype=torch.int64)
    if chosen_pc is not None:
        extra[chosen_pc] = torch.tensor(plans[chosen_pc + 2][1:])
    return with_rows(image, torch.cat((image.rows, extra), 1))


def execute(graph, image, arg, page):
    state = initialize(image, arg, page, 100_000)
    calls = blocks = 0
    start = perf_counter_ns()
    while int(state.vm.status) == 0:
        out = graph(image.rows, state, 64)
        state = out.state
        calls += 1
        blocks += getattr(out, "fused_blocks", 0)
    elapsed = perf_counter_ns() - start
    assert int(state.vm.status) == 1
    return state.vm.page.numpy().tobytes(), {
        "ns": elapsed, "calls": calls, "blocks": blocks,
        "steps": 100_000 - int(state.vm.fuel), "return": signed(state.vm.stack[0])}


def benchmark(variants, raw, arg, page, moves, steps):
    labels = tuple(variants)
    samples = {name: [] for name in labels}
    for i, delta in enumerate(moves):
        if delta is not None:
            terrain.write(page, 320, [4 * m for m in delta])
        expected, oracle = wasmtime_run(raw, arg, page)
        offset = i % len(labels)
        for name in labels[offset:] + labels[:offset]:
            graph, image = variants[name]
            changed, row = execute(graph, image, arg, page)
            assert changed == oracle and row["return"] == expected and row["steps"] == steps
            samples[name].append({**row, "page_sha256": sha(changed)})
        if delta is not None:
            page = bytearray(oracle)
    output = {}
    for name, rows in samples.items():
        warm = sorted(r["ns"] for r in (rows[1:] if len(rows) > 8 else rows))
        output[name] = {"median_ns": statistics.median(warm), "warm_samples": len(warm), "rows": rows}
        if len(warm) > 8:
            output[name]["p95_ns"] = warm[math.ceil(.95 * len(warm)) - 1]
    return output


def segmentation(graph, image, arg, page, sites):
    state = initialize(image, arg, page, 100_000)
    for _ in range(8):
        graph(image.rows, state, 0)
    start = perf_counter_ns()
    for _ in range(500):
        graph(image.rows, state, 0)
    empty_ns = (perf_counter_ns() - start) / 500
    groups = defaultdict(lambda: {"calls": 0, "source_steps": 0, "ns": 0})
    previous = None
    edges = defaultdict(int)
    while int(state.vm.status) == 0:
        pc = int(state.vm.pc)
        budget = 2 if pc in sites else 1
        start = perf_counter_ns()
        out = graph(image.rows, state, budget)
        elapsed = perf_counter_ns() - start
        covered = out.fused_pairs + out.fused_loads + out.fused_arithmetic > 0
        label = "covered" if covered else "ordinary_or_fallback"
        if previous is not None:
            edges[previous + " -> " + label] += 1
        name = "covered_after_covered" if covered and previous == "covered" else label
        previous = label
        group = groups[name]
        group["calls"] += 1
        group["ns"] += elapsed
        group["source_steps"] += int(state.vm.fuel) - int(out.state.vm.fuel)
        state = out.state
    return {"groups": dict(groups), "empty_invocation_proxy_ns": empty_ns, "edges": dict(edges)}


def prefixes(one, wide, plain, image, state, pc):
    assert int(state.vm.pc) == pc
    source = state
    checked = []
    for count in range(1, 5):
        out = one(plain.rows, source)
        source = out.state
        got = wide(image.rows, state, count)
        state_equal(source, got.state)
        compare_access(out.access, got.access)
        checked.append({"budget": count, "pc_after": int(source.vm.pc), "fuel_after": int(source.vm.fuel),
                        "stack": source.vm.stack[:int(source.vm.height)].tolist(),
                        "access": out.access.tolist(), "composed_blocks": got.fused_blocks})
    for cut in (1, 2, 3):
        ref = state._replace(vm=state.vm._replace(fuel=torch.tensor(cut)))
        start = ref
        while int(ref.vm.status) == 0:
            ref = one(plain.rows, ref).state
        observed = wide(image.rows, start, 64)
        state_equal(ref, observed.state)
        assert int(ref.vm.status) == 2 and observed.fused_blocks == 0
    return checked


def local_profile(old, wide, one, plain, prior, image, arg, page, pc):
    state = initialize(plain, arg, page, 5000)
    while int(state.vm.pc) != pc:
        state = one(plain.rows, state).state
        assert int(state.vm.status) == 0
    checked = prefixes(one, wide, plain, image, state, pc)
    assert checked[-1]["composed_blocks"] == 1
    variants = (("portfolio", old, prior), ("composed", wide, image))
    raw = {name: [] for name, _, _ in variants}
    operators = {}
    for name, graph, layout in variants:
        for _ in range(8):
            graph(layout.rows, state, 4)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
            graph(layout.rows, state, 4)
        operators[name] = {item.key: item.count for item in prof.key_averages()}
    for i in range(31):
        for name, graph, layout in variants[::(1 if i % 2 == 0 else -1)]:
            start = perf_counter_ns()
            for _ in range(16):
                graph(layout.rows, state, 4)
            raw[name].append((perf_counter_ns() - start) / 16)
    return {"source_prefixes": checked, "pc": pc, "runtime_operators": operators, "raw_ns": raw,
            "median_ns": {name: statistics.median(values) for name, values in raw.items()},
            "samples": 31, "calls_per_sample": 16}


def audit(one, wide, plain, image, arg, page):
    a = initialize(plain, arg, page, 5000)
    b = initialize(image, arg, page, 5000)
    calls = 0
    while int(a.vm.status) == 0:
        pc = int(a.vm.pc)
        count = 4 if int(image.rows[pc, 14]) >= 0 else 1
        for prefix in range(1, count + 1):
            reference = one(plain.rows, a)
            a = reference.state
            current = wide(image.rows, b, prefix)
            # The marked block contains reads only. Ordinary stores use just
            # the one-iteration branch above, preserving private-page ownership.
            state_equal(a, current.state)
            compare_access(reference.access, current.access)
            calls += 1
            if int(a.vm.status):
                break
        b = current.state
    return {"source_steps": 5000 - int(a.vm.fuel), "boundaries_checked": calls,
            "page_sha256": sha(a.vm.page.numpy().tobytes())}


def transfer(base, one, old, wide, planner, planners):
    path = OUT / "after-freeze.wat"
    raw = path.with_suffix(".wasm").read_bytes()
    assert bytes(wasmtime.wat2wasm(path.read_text())) == raw
    plain = load_call_image(raw)
    images, plans, selected, _ = prepare(plain, base, planners, bytes((PORTFOLIO / "chooser.wasm").read_bytes()), {})
    sites = discover(plain, plans, selected)
    assert list(sites) == [0]
    proposed = proposals(base, planner, 0, sites[0], plans)
    accept(proposed, 0, plans)
    rejected = proposals(base, planner, 0, sites[0], plans, challenge=True)
    try:
        accept(rejected, 0, plans)
    except ValueError as error:
        reason = str(error)
    else:
        raise AssertionError("Wrong seam was accepted")
    image = extend(images["portfolio"], plans, 0)
    page = bytearray(65536)
    struct.pack_into("<I", page, 104, 7)
    struct.pack_into("<I", page, 112, 11)
    state = initialize(plain, 100, page, 100)
    witnessed = prefixes(one, wide, plain, image, state, 0)
    assert witnessed[1]["stack"] == [7] and witnessed[3]["stack"] == [7, 11]
    assert witnessed[3]["composed_blocks"] == 1
    scenarios = []
    for arg in (100, 65527, 65531):
        for fuel in (0, 1, 2, 3, 4, 100):
            ref = initialize(plain, arg, page, fuel)
            new = initialize(image, arg, page, fuel)
            while int(ref.vm.status) == 0:
                r = one(plain.rows, ref)
                ref = r.state
            out = wide(image.rows, new, 64)
            state_equal(ref, out.state)
            compare_access(r.access, out.access)
            if fuel == 100:
                status = int(ref.vm.status)
                result, oracle = wasmtime_run(raw, arg, page, trap_expected=arg != 100)
                assert oracle == page and status == (1 if arg == 100 else 3)
                if status == 1:
                    assert result == signed(ref.vm.stack[0]) == 18
            scenarios.append({"argument": arg, "fuel": fuel, "status": int(ref.vm.status),
                              "pc": int(ref.vm.pc), "remaining": int(ref.vm.fuel),
                              "event": out.access.tolist(), "blocks": out.fused_blocks})
    return {"binary_sha256": sha(raw), "proposal": proposed,
            "negative": {"proposal": rejected, "reason": reason, "first_word": 7,
                         "incorrect_first_word": 11, "both_final_stack": [7, 11]},
            "prefixes": witnessed, "cases": scenarios}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--transfer-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    start = perf_counter_ns()
    proof = certificate()
    check_ns = perf_counter_ns() - start
    base, one, old, wide = [torch.jit.load(str(path)) for path in (
        EXECUTOR / "machine.pt", CALLS / "machine.pt", PORTFOLIO / "machine.pt", OUT / "machine.pt")]
    planners = {1: (PAIR / "planner.wasm").read_bytes(), 2: (PRIOR / "load-planner.wasm").read_bytes()}
    chooser = (PORTFOLIO / "chooser.wasm").read_bytes()
    planner = (OUT / "planner.wasm").read_bytes()
    output = {"certificate_sha256": proof, "check_ns": check_ns}
    with patch(PACKED, side_effect=AssertionError("Unexpected packed fallback")):
        if args.transfer_only:
            output["transfer"] = transfer(base, one, old, wide, planner, planners)
        else:
            profiles = json.loads((PORTFOLIO / "training.json").read_text())
            argument, graph_page, *_ = graph_case(0)
            world = terrain.scale_session(8, False, 25)
            page = terrain.initial_page(world["initial"])
            terrain.write(page, 320, [4] * 8)
            bfs = (ARCHIVE / "v58-stateful-scale-7125937/bfs.wasm").read_bytes()
            for name, raw, arg, initial, n in (
                ("terrain", terrain.GUEST.read_bytes(), terrain.BASE, page, 1650),
                ("bfs", bfs, argument, graph_page, 41591),
            ):
                start = perf_counter_ns()
                source = load_call_image(raw)
                pc_counts = {int(pc): value for pc, value in profiles[name]["pc_counts"].items()}
                images, plans, selected, _ = prepare(source, base, planners, chooser, pc_counts)
                prep_ns = perf_counter_ns() - start
                start = perf_counter_ns()
                sites = discover(source, plans, selected)
                pc = max(sites, key=lambda p: (pc_counts.get(p, 0), -p), default=None)
                proposal = None
                if pc is not None:
                    proposal = proposals(base, planner, pc, sites[pc], plans)
                    accept(proposal, pc, plans)
                optimized = extend(images["portfolio"], plans, pc)
                composed_prep_ns = perf_counter_ns() - start
                options = {"portfolio": (old, images["portfolio"]), "composed": (wide, optimized)}
                if name == "bfs":
                    options["arithmetic_only"] = old, images["new"]
                else:
                    options["same_graph_without_block"] = wide, extend(images["portfolio"], plans, None)
                rows = benchmark(options, raw, arg, initial, [x["moves"] for x in world["inputs"]]
                                 if name == "terrain" else [None] * 4, n)
                evidence = {"binary_sha256": sha(raw), "prep_ns": prep_ns, "compose_prep_ns": composed_prep_ns,
                            "source_sites": {p: pc_counts.get(p, 0) for p in sites}, "chosen_pc": pc,
                            "proposal": proposal, "timings": rows,
                            "segmented_profile": segmentation(old, images["portfolio"], arg, initial, set(selected))}
                if name == "terrain":
                    evidence["local"] = local_profile(old, wide, one, source, images["portfolio"],
                                                      optimized, arg, initial, pc)
                    evidence["audit"] = audit(one, wide, source, optimized, arg, initial)
                    evidence["histories"] = verify_histories(wide, optimized, raw)
                output[name] = evidence
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
