"""Train bounded site scores, run the guest chooser, audit and benchmark selection."""
import argparse
from collections import Counter
import json
import math
import statistics
import struct
from time import perf_counter_ns
from unittest.mock import patch

import torch
import wasmtime

import terrain_verify as terrain
from prove_rule_portfolio import OUT, VERSION, bindings
from replay_frozen_calls import run
from replay_frozen_multistep import CALLS, state_equal
from replay_frozen_recurrent import ARCHIVE, PACKED, graph_case, sha, signed, wasmtime_run
from replay_guest_rewrite import EXECUTOR, invoke
from replay_reusable_region_rules import (
    OUT as PRIOR, PAIR, admit, audit, checked_certificate, compare_access, with_rows,
)
from visuaml.compiler_lab.frozen_calls_v58 import initialize, load_call_image
from visuaml.compiler_lab.frozen_recurrent_v58 import OPCODES

ARITHMETIC = {"i32.add", "i32.mul", "i32.lt_s"}
WEIGHTS = {1: 1, 2: 2, 3: 2}
MODES = ("existing", "new", "portfolio")


def check_rules():
    checked_certificate()
    raw = (OUT / "rules.json").read_bytes()
    proof = json.loads(raw)
    if (sha(raw) != (OUT / "rules.sha256").read_text().strip() or proof["version"] != VERSION
            or proof["bindings"] != bindings() or len(proof["obligations"]) != 4
            or any(row["result"] != "unsat" for row in proof["obligations"])):
        raise ValueError("Portfolio proof or semantics binding rejected")
    return sha(raw)


def arithmetic_sites(image):
    result = {}
    for start, function in zip(image.starts, image.functions, strict=True):
        for offset, (first, second) in enumerate(zip(function, function[1:])):
            if first != "i32.const" or second not in ARITHMETIC:
                continue
            pc = start + offset
            rows = image.rows[pc:pc + 2].tolist()
            a, b = rows
            if (a[0] != OPCODES.index(first) or b[0] != OPCODES.index(second)
                    or a[3:5] != [pc + 1] * 2 or b[3:5] != [pc + 2] * 2
                    or any(a[5:8]) or any(b[5:8]) or b[10] or pc + 2 >= start + len(function)
                    or not 0 <= a[1] <= 0xFFFFFFFF):
                raise ValueError("Arithmetic site fails source identity or fall-through")
            result[pc] = (3, b[0], a[1])
    return result


def validate_selection(candidates, selected):
    ordered = sorted(selected)
    if any(pc not in candidates for pc in ordered) or any(b - a < 2 for a, b in zip(ordered, ordered[1:])):
        raise ValueError("Unknown, duplicate or overlapping region entry")


def choose(machine, chooser, candidates, counts):
    picked, decisions = [], []
    for pc in sorted(candidates):
        if not picked or pc - picked[-1] >= 2:
            picked.append(pc)
            continue
        other = picked[-1]
        scores = [counts.get(i, 0) * WEIGHTS[candidates[i][0]] for i in (other, pc)]
        page = bytearray(65536)
        struct.pack_into("<3i", page, 512, *scores, -1)
        result, record = invoke(machine, chooser, 512, page)
        choice, = struct.unpack_from("<i", result, 520)
        if (choice not in (0, 1) or record["status"] != 1 or record["return_i32"] != choice
                or choice != int(scores[1] > scores[0])):
            raise ValueError("Guest chooser emitted an invalid selection")
        picked[-1] = (other, pc)[choice]
        decisions.append({"candidates": [other, pc], "scores": scores, "chosen": picked[-1],
                          "guest": record})
    validate_selection(candidates, picked)
    return picked, decisions


def greedy(candidates):
    selected = []
    for pc in sorted(candidates):
        if not selected or pc - selected[-1] >= 2:
            selected.append(pc)
    return selected


def lower(image, candidates, selected):
    validate_selection(candidates, selected)
    metadata = torch.full((len(image.rows), 3), -1, dtype=torch.int64)
    for pc in selected:
        kind, a, b = candidates[pc]
        metadata[pc] = torch.tensor((a, b, -1) if kind == 1 else (a, -kind, b))
    return with_rows(image, torch.cat((image.rows, metadata), 1))


def profile(one, image, arg, page):
    started = perf_counter_ns()
    final, trace = run(one, image, arg, page, 100_000, trace=True)
    assert int(final.vm.status) == 1
    counts = Counter(row["pc"] + image.starts[row["function"]] for row in trace)
    adjacent = Counter((a["opcode"], b["opcode"]) for a, b in zip(trace, trace[1:])
                       if a["function"] == b["function"] and b["pc"] == a["pc"] + 1)
    return counts, {"audit_collection_ns": perf_counter_ns() - started,
                    "steps": len(trace), "adjacent": [[*key, n] for key, n in adjacent.most_common(28)],
                    "pc_counts": dict(counts)}


def kernel_profile(old, fast, images, candidates, argument, page, sites):
    state = initialize(images["plain"], argument, page, 100_000)
    while int(state.vm.pc) not in sites or candidates[int(state.vm.pc)][0] != 3:
        state = old(images["prior_all"].rows, state, 1).state
        assert int(state.vm.status) == 0
    first = old(images["prior_all"].rows, state, 2)
    fused = fast(images["portfolio"].rows, state, 2)
    assert fused.fused_arithmetic == 1
    state_equal(first.state, fused.state)
    counts, samples = {}, {}
    for name, graph, image in (("existing", old, images["prior_all"]),
                               ("portfolio", fast, images["portfolio"])):
        for _ in range(8):
            graph(image.rows, state, 2)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as recorded:
            graph(image.rows, state, 2)
        counts[name] = {event.key: event.count for event in recorded.key_averages()}
        samples[name] = []
    for n in range(25):
        variants = (("existing", old, images["prior_all"]), ("portfolio", fast, images["portfolio"]))
        for name, graph, image in variants[::(1 if n % 2 == 0 else -1)]:
            started = perf_counter_ns()
            for _ in range(12):
                graph(image.rows, state, 2)
            samples[name].append((perf_counter_ns() - started) / 12)
    return {"site": int(state.vm.pc), "counts": counts, "ns_per_call": samples,
            "median_ns": {k: statistics.median(v) for k, v in samples.items()}}


def execute(graph, image, argument, page):
    state = initialize(image, argument, page, 100_000)
    calls = pair = load = arithmetic = 0
    started = perf_counter_ns()
    while int(state.vm.status) == 0:
        tick = graph(image.rows, state, 64)
        state = tick.state
        calls += 1
        pair += tick.fused_pairs
        load += tick.fused_loads
        arithmetic += getattr(tick, "fused_arithmetic", 0)
    elapsed = perf_counter_ns() - started
    assert int(state.vm.status) == 1
    return state.vm.page.numpy().tobytes(), {"execution_ns": elapsed, "host_calls": calls,
                                            "fused_pairs": pair, "fused_loads": load,
                                            "fused_arithmetic": arithmetic,
                                            "source_steps": 100_000 - int(state.vm.fuel),
                                            "return_i32": signed(state.vm.stack[0])}


def benchmark(graphs, images, raw, argument, page, ticks, source_steps):
    rows = {mode: [] for mode in graphs}
    names = tuple(graphs)
    for i, moves in enumerate(ticks):
        if moves is not None:
            terrain.write(page, 320, [4 * m for m in moves])
        result, oracle = wasmtime_run(raw, argument, page)
        pivot = i % len(names)
        for mode in names[pivot:] + names[:pivot]:
            rebuilt, metrics = execute(graphs[mode], images[mode], argument, page)
            assert rebuilt == oracle and metrics["return_i32"] == result
            assert metrics["source_steps"] == source_steps
            rows[mode].append({**metrics, "page_sha256": sha(rebuilt)})
        if moves is not None:
            page = bytearray(oracle)
    values = {}
    for mode, samples in rows.items():
        warm = sorted(r["execution_ns"] for r in samples[1:]) if len(ticks) > 8 else [
            r["execution_ns"] for r in samples]
        values[mode] = {"warm_samples": len(warm), "median_ns": statistics.median(warm), "samples": samples}
        if len(warm) > 8:
            values[mode]["p95_ns"] = warm[math.ceil(0.95 * len(warm)) - 1]
    return values


def verify_histories(graph, image, raw):
    evidence = {}
    for label in ("terrain", "terrain-omitted"):
        fixture = json.loads((OUT.parents[1] / "fixtures" / f"{label}.proof.json").read_text())
        page = terrain.initial_page(fixture["session"]["initial"])
        evidence[label] = []
        for inputs, saved in zip(fixture["session"]["inputs"], fixture["ticks"], strict=True):
            terrain.write(page, 4, [int(inputs["edit"])])
            terrain.write(page, 320, [4 * m for m in inputs["moves"]])
            after, result = execute(graph, image, terrain.BASE, page)
            assert wasmtime_run(raw, terrain.BASE, page) == (result["return_i32"], after)
            assert sha(after) == saved["page_sha256"] and result["source_steps"] == saved["steps"]
            evidence[label].append({**result, "page_sha256": sha(after)})
            page = bytearray(after)
    return evidence


def composition(one, fast, plain, planned, sites, raw, argument, page):
    a = initialize(plain, argument, page, 100)
    b = initialize(planned, argument, page, 100)
    transitions, edges = [], []
    previous = None
    while int(a.vm.status) == 0:
        pc = int(a.vm.pc)
        start_a, start_b = a, b
        first = one(plain.rows, a)
        prefix = fast(planned.rows, b, 1)
        state_equal(first.state, prefix.state)
        compare_access(first.access, prefix.access)
        if pc in sites and int(first.state.vm.status) == 0:
            second = one(plain.rows, first.state)
            result = fast(planned.rows, b, 2)
            state_equal(second.state, result.state)
            compare_access(second.access, result.access)
            kind = sites[pc][0]
            # A one-fuel cutoff takes the ordinary first source transition.
            cut_a = start_a._replace(vm=start_a.vm._replace(fuel=torch.tensor(1)))
            cut_b = start_b._replace(vm=start_b.vm._replace(fuel=torch.tensor(1)))
            stopped = one(plain.rows, one(plain.rows, cut_a).state)
            actual = fast(planned.rows, cut_b, 64)
            state_equal(stopped.state, actual.state)
            compare_access(stopped.access, actual.access)
            assert int(actual.state.vm.status) == 2 and not (actual.fused_pairs or actual.fused_loads or actual.fused_arithmetic)
            if previous is not None:
                edges.append([previous, kind])
            previous = kind
            transitions.append({"pc": pc, "rule": kind, "prefix_fuel": int(prefix.state.vm.fuel),
                                "end_fuel": int(result.state.vm.fuel), "status": int(result.state.vm.status),
                                "actual_fused": result.fused_pairs + result.fused_loads + result.fused_arithmetic})
            a, b = second.state, result.state
        else:
            a, b = first.state, prefix.state
            previous = None
    return {"regions": transitions, "adjacent_rule_ids": edges, "status": int(a.vm.status),
            "fuel_remaining": int(a.vm.fuel), "page_sha256": sha(a.vm.page.numpy().tobytes())}


def prepare(image, base, planners, chooser, pc_counts):
    started = perf_counter_ns()
    (_, existing_image), old = admit(image, base, planners)
    discovered = arithmetic_sites(image)
    candidates = {**old["sites"], **discovered}
    admit_ns = perf_counter_ns() - started
    clock = perf_counter_ns()
    selected, decisions = choose(base, chooser, candidates, pc_counts)
    choose_ns = perf_counter_ns() - clock
    early = greedy(old["sites"])
    images = {"plain": image, "prior_all": existing_image, "existing": lower(image, old["sites"], early),
              "new": lower(image, discovered, greedy(discovered)), "portfolio": lower(image, candidates, selected)}
    return images, candidates, selected, {
        "admission_ns_including_guest_and_oracle": admit_ns,
        "existing_site_check_ns": old["site_admission_ns"], "chooser_ns": choose_ns,
        "candidates": candidates, "existing_guest_proposals": old["guest_proposals"],
        "existing_earliest": early, "guest_selected": selected, "decisions": decisions,
        "different_overlap_decisions": [d for d in decisions if d["chosen"] != d["candidates"][0]],
        "prepared_image_hashes": {k: v.image_sha256 for k, v in images.items()}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-only", action="store_true")
    parser.add_argument("--transfer-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.train_only:
        certificate, verify_ns = None, None
    else:
        started = perf_counter_ns()
        certificate = check_rules()
        verify_ns = perf_counter_ns() - started
    plain = torch.jit.load(str(EXECUTOR / "machine.pt"))
    old = torch.jit.load(str(PRIOR / "machine.pt"))
    one = torch.jit.load(str(CALLS / "machine.pt"))
    fast = torch.jit.load(str(OUT / "machine.pt"))
    planners = {1: (PAIR / "planner.wasm").read_bytes(), 2: (PRIOR / "load-planner.wasm").read_bytes()}
    chooser = (OUT / "chooser.wasm").read_bytes()
    terrain_raw = terrain.GUEST.read_bytes()
    bfs = (ARCHIVE / "v58-stateful-scale-7125937/bfs.wasm").read_bytes()
    argument, graph_page, *_ = graph_case(0)
    session = terrain.scale_session(8, False, 25)
    world = terrain.initial_page(session["initial"])
    terrain.write(world, 320, [4] * 8)
    output = {}
    with patch(PACKED, side_effect=AssertionError("Unexpected packed fallback")):
        for name, raw, arg, page in (("terrain", terrain_raw, terrain.BASE, world),
                                     ("bfs", bfs, argument, graph_page)):
            if args.transfer_only:
                continue
            image = load_call_image(raw)
            prior_row = json.loads((OUT / "training.json").read_text())[name] if not args.train_only else None
            if prior_row is None:
                (_, warmed), _ = admit(image, plain, planners)
                execute(old, warmed, arg, page)
                times = [execute(old, warmed, arg, page)[1] for _ in range(2)]
                _, data = profile(one, image, arg, page)
                output[name] = {**data, "warm_existing": times, "binary_sha256": sha(raw)}
                continue
            assert prior_row["binary_sha256"] == sha(raw)
            counts = {int(k): n for k, n in prior_row["pc_counts"].items()}
            images, candidates, picked, preparation = prepare(image, plain, planners, chooser, counts)
            start = next((pc for pc in sorted(candidates) if pc + 1 in candidates), None)
            if start is not None:
                try:
                    validate_selection(candidates, [start, start + 1])
                except ValueError as error:
                    preparation["rejected_overlap"] = {"entries": [start, start + 1], "reason": str(error)}
                else:
                    raise AssertionError("Overlapping plan was admitted")
            graphs = {mode: old if mode == "existing" else fast for mode in MODES}
            if name == "terrain":
                graphs["prior_all"] = old
            steps = 1650 if name == "terrain" else 41591
            timings = benchmark(graphs, images, raw, arg, page,
                                [x["moves"] for x in session["inputs"]] if name == "terrain" else [None] * 5,
                                steps)
            result = {"binary_sha256": sha(raw), "preparation": preparation, "timings": timings,
                      "local_profile": kernel_profile(old, fast, images, candidates, arg, page, picked)}
            if name == "terrain":
                result["audit"] = audit(one, fast, image, images["portfolio"],
                                        {pc: candidates[pc] for pc in picked}, arg, page, 5000)
                result["histories"] = verify_histories(fast, images["portfolio"], raw)
            output[name] = result
        if args.transfer_only:
            path = OUT / "after-freeze.wat"
            raw = path.with_suffix(".wasm").read_bytes()
            assert raw == bytes(wasmtime.wat2wasm(path.read_text()))
            image = load_call_image(raw)
            page = bytearray(65536)
            page[9:13] = bytes.fromhex("09000000")
            counts, training = profile(one, image, 8, page)
            images, candidates, picked, prep = prepare(image, plain, planners, chooser, counts)
            sites = {pc: candidates[pc] for pc in picked}
            normal = composition(one, fast, image, images["portfolio"], sites, raw, 8, page)
            trapped = composition(one, fast, image, images["portfolio"], sites, raw, 65532, page)
            assert [2, 3] in normal["adjacent_rule_ids"] and trapped["status"] == 3
            final, metrics = execute(fast, images["portfolio"], 8, page)
            assert wasmtime_run(raw, 8, page) == (metrics["return_i32"], final)
            assert wasmtime_run(raw, 65532, page, trap_expected=True) == (None, bytes(page))
            output["after_freeze"] = {"binary_sha256": sha(raw), "training": training,
                                      "preparation": prep, "normal": normal, "trap": trapped,
                                      "return_i32": metrics["return_i32"]}
    envelope = output if args.train_only else {"certificate_sha256": certificate,
                                               "certificate_check_ns": verify_ns, **output}
    print(json.dumps(envelope, indent=2))


if __name__ == "__main__":
    main()
