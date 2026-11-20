"""Verify and measure prepared effect descriptors on unchanged guest images."""
import argparse
from collections import Counter
import json
import math
import statistics
from time import perf_counter_ns
from unittest.mock import patch

import torch
import wasmtime

import terrain_verify as terrain
from freeze_ordinary_effects import OUT
from replay_composed_loads import OUT as COMPOSED, extend
from replay_frozen_calls import NESTED
from replay_frozen_multistep import CALLS, state_equal
from replay_frozen_recurrent import ARCHIVE, PACKED, graph_case, sha, signed, wasmtime_run
from replay_guest_rewrite import EXECUTOR
from replay_reusable_region_rules import compare_access
from replay_rule_portfolio import OUT as PORTFOLIO, PAIR, PRIOR, check_rules, prepare, with_rows
from visuaml.compiler_lab.frozen_calls_v58 import initialize, load_call_image
from visuaml.compiler_lab.frozen_recurrent_v58 import OPCODES, POPS, PUSHES, snapshot_page


def descriptors(image):
    """Accept only the validated portfolio row layout; attach source effects."""
    assert image.rows.shape[1] == 14
    rows = []
    for row in image.rows.tolist():
        op = row[0]
        assert 0 <= op < len(OPCODES)
        rows.append([POPS[op], PUSHES[op], int(row[9] >= 0 or row[10] != 0)])
    expanded = torch.cat((image.rows, torch.tensor(rows, dtype=torch.int64)), dim=1)
    return with_rows(image, expanded)


def drive(graph, image, arg, page):
    state = initialize(image, arg, page, 100_000)
    calls = fusions = 0
    started = perf_counter_ns()
    while int(state.vm.status) == 0:
        tick = graph(image.rows, state, 64)
        state = tick.state
        calls += 1
        fusions += tick.fused_pairs + tick.fused_loads + tick.fused_arithmetic
    elapsed = perf_counter_ns() - started
    assert int(state.vm.status) == 1
    return snapshot_page(state.vm), {"ns": elapsed, "calls": calls, "fusions": fusions,
                                     "steps": 100_000 - int(state.vm.fuel), "result": signed(state.vm.stack[0])}


def bench(graphs, images, raw, arg, page, inputs, steps):
    modes = tuple(graphs)
    rows = {name: [] for name in modes}
    for ordinal, moves in enumerate(inputs):
        if moves is not None:
            terrain.write(page, 320, [4 * x for x in moves])
        expected, oracle = wasmtime_run(raw, arg, page)
        shift = ordinal % len(modes)
        shared = set()
        for name in modes[shift:] + modes[:shift]:
            final, row = drive(graphs[name], images[name], arg, page)
            assert row["result"] == expected and final == oracle and row["steps"] == steps
            shared.add((row["steps"], row["calls"], row["fusions"]))
            rows[name].append({**row, "page_sha256": sha(final)})
        assert len(shared) == 1
        if moves is not None:
            page = bytearray(oracle)
    results = {}
    for name, runs in rows.items():
        warm = sorted(r["ns"] for r in (runs[1:] if len(runs) > 8 else runs))
        results[name] = {"median_ns": statistics.median(warm), "warm_samples": len(warm), "raw": runs}
        if len(warm) > 8:
            results[name]["p95_ns"] = warm[math.ceil(.95 * len(warm)) - 1]
    return results


def runtime_profile(graph, image, arg, page, sites, sample=256):
    state = initialize(image, arg, page, 100_000)
    generic, fused = Counter(), Counter()
    generic_ns = Counter()
    spans = []
    for _ in range(5):
        graph(image.rows, initialize(image, arg, page, 100_000), 64)
    while int(state.vm.status) == 0:
        pc = int(state.vm.pc)
        budget = 2 if pc in sites else 1
        op = OPCODES[int(image.rows[pc, 0])]
        entry = None
        if len(spans) < sample:
            entry = state._replace(vm=state.vm._replace(page=state.vm.page.clone()))
        started = perf_counter_ns()
        pulse = graph(image.rows, state, budget)
        elapsed = perf_counter_ns() - started
        fusion = pulse.fused_pairs + pulse.fused_loads + pulse.fused_arithmetic
        name = "fused" if fusion else "ordinary"
        if name == "ordinary":
            generic[op] += 1
            generic_ns[op] += elapsed
        else:
            fused[op] += 1
        retired = int(state.vm.fuel) - int(pulse.state.vm.fuel)
        assert retired == (2 if fusion else 1)
        if entry is not None:
            spans.append((entry, budget, name))
        state = pulse.state
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as recorded:
        for entry, budget, name in spans:
            # Replay the independently captured pre-instruction page.
            with torch.profiler.record_function(name):
                graph(image.rows, entry, budget)
    counts, costs = Counter(), Counter()
    for event in recorded.events():
        parent = event.cpu_parent
        while parent is not None and parent.name not in ("ordinary", "fused"):
            parent = parent.cpu_parent
        if parent is not None and parent.name == "ordinary":
            counts[event.name] += 1
            costs[event.name] += event.self_cpu_time_total
    return {"ordinary_opcode_counts": dict(generic), "ordinary_call_ns_including_crossing": dict(generic_ns),
            "covered_starts": dict(fused), "operator_counts_sample": dict(counts),
            "operator_self_cpu_us_sample": dict(costs),
            "ordinary_sample_calls": sum(name == "ordinary" for _, _, name in spans),
            "source_steps": 100_000 - int(state.vm.fuel)}


def audit(one, graph, plain, image, arg, page, fuel, sites):
    ref, got = initialize(plain, arg, page, fuel), initialize(image, arg, page, fuel)
    covered = checked = 0
    while int(ref.vm.status) == 0:
        site = int(ref.vm.pc) in sites
        a = one(plain.rows, ref)
        b = graph(image.rows, got, 1)
        state_equal(a.state, b.state)
        compare_access(a.access, b.access)
        checked += 1
        if site and int(a.state.vm.status) == 0:
            second = one(plain.rows, a.state)
            merged = graph(image.rows, got, 2)
            state_equal(second.state, merged.state)
            compare_access(second.access, merged.access)
            covered += merged.fused_pairs + merged.fused_loads + merged.fused_arithmetic
            checked += 1
            ref, got = second.state, merged.state
        else:
            ref, got = a.state, b.state
    return {"checked": checked, "fusions": covered, "steps": fuel - int(got.vm.fuel),
            "status": int(got.vm.status), "page_sha256": sha(snapshot_page(got.vm)),
            "return_i32": signed(got.vm.stack[0]) if int(got.vm.status) == 1 else None}


def no_regions(image):
    extra = torch.full((len(image.rows), 3), -1, dtype=torch.int64)
    return descriptors(with_rows(image, torch.cat((image.rows, extra), 1)))


def witnesses(one, graph):
    rollback = (ARCHIVE / "v58-array-memory/rollback.wasm").read_bytes()
    before = bytearray(65536)
    before[1:5] = before[7:11] = bytes.fromhex("01000080")
    nested = bytes(wasmtime.wat2wasm(NESTED))
    later_path = OUT / "after-freeze.wat"
    later = later_path.with_suffix(".wasm").read_bytes()
    assert later == bytes(wasmtime.wat2wasm(later_path.read_text()))
    results = {}
    for label, binary, arg, page, fuel in (
        ("nested_return", nested, 0, bytearray(65536), 100),
        ("nested_trap", nested, 65533, bytearray(65536), 100),
        ("rollback", rollback, 0, before, 100),
        ("before_store", rollback, 0, before, 8),
        ("after_store", rollback, 0, before, 9),
        ("after_freeze", later, 7, bytearray(65536), 100),
        ("after_freeze_else", later, 0, bytearray(65536), 100),
    ):
        image = load_call_image(binary)
        evidence = audit(one, graph, image, no_regions(image), arg, page, fuel, set())
        if evidence["status"] in (1, 3):
            val, ref = wasmtime_run(binary, arg, page, trap_expected=evidence["status"] == 3)
            assert val == evidence["return_i32"] and sha(ref) == evidence["page_sha256"]
        results[label] = {**evidence, "binary_sha256": sha(binary), "fuel": fuel}
    assert results["before_store"]["page_sha256"] == sha(before)
    after = before.copy()
    after[7:11] = bytes.fromhex("78563412")
    assert results["after_store"]["page_sha256"] == sha(after)
    return results


def histories(graphs, images, raw):
    saved = {}
    for name in ("terrain", "terrain-omitted"):
        fixture = json.loads((OUT.parents[1] / "fixtures" / f"{name}.proof.json").read_text())
        page = terrain.initial_page(fixture["session"]["initial"])
        checkpoints = []
        for input_, expected in zip(fixture["session"]["inputs"], fixture["ticks"], strict=True):
            terrain.write(page, 4, [int(input_["edit"])])
            terrain.write(page, 320, [4 * x for x in input_["moves"]])
            result, oracle = wasmtime_run(raw, terrain.BASE, page)
            paths = {}
            for mode, graph in graphs.items():
                memory, row = drive(graph, images[mode], terrain.BASE, page)
                assert memory == oracle and row["result"] == result and row["steps"] == expected["steps"]
                paths[mode] = row
            assert len({row["calls"] for row in paths.values()}) == 1
            assert sha(oracle) == expected["page_sha256"]
            checkpoints.append({"page_sha256": sha(oracle), "paths": paths})
            page = bytearray(oracle)
        saved[name] = checkpoints
    return saved


def main():
    parser = argparse.ArgumentParser()
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--witnesses", action="store_true")
    modes.add_argument("--profile-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    check_rules()
    paths = {"portfolio": PORTFOLIO / "machine.pt", "descriptors": OUT / "descriptor.pt", "lazy": OUT / "lazy.pt"}
    graphs = {name: torch.jit.load(str(path)) for name, path in paths.items()}
    one = torch.jit.load(str(CALLS / "machine.pt"))
    output = {"graph_sha256": {k: sha(p.read_bytes()) for k, p in paths.items()}}
    with patch(PACKED, side_effect=AssertionError("Unexpected packed fallback")):
        if args.witnesses:
            output["prefix_witnesses"] = {name: witnesses(one, graph) for name, graph in graphs.items()
                                          if name != "portfolio"}
        else:
            base = torch.jit.load(str(EXECUTOR / "machine.pt"))
            training = json.loads((PORTFOLIO / "training.json").read_text())
            planners = {1: (PAIR / "planner.wasm").read_bytes(), 2: (PRIOR / "load-planner.wasm").read_bytes()}
            chooser = (PORTFOLIO / "chooser.wasm").read_bytes()
            world = terrain.scale_session(8, False, 25)
            page = terrain.initial_page(world["initial"])
            terrain.write(page, 320, [4] * 8)
            arg, graph_page, *_ = graph_case(0)
            bfs = (ARCHIVE / "v58-stateful-scale-7125937/bfs.wasm").read_bytes()
            for label, raw, argument, initial, steps in (
                    ("terrain", terrain.GUEST.read_bytes(), terrain.BASE, page, 1650),
                    ("bfs", bfs, arg, graph_page, 41591)):
                plain = load_call_image(raw)
                count = {int(k): v for k, v in training[label]["pc_counts"].items()}
                prepared, candidate, sites, _ = prepare(plain, base, planners, chooser, count)
                original = prepared["portfolio"]
                start = perf_counter_ns()
                optimized = descriptors(original)
                prep_ns = perf_counter_ns() - start
                images = {"portfolio": original, "descriptors": optimized, "lazy": optimized}
                for mode, graph in graphs.items():
                    drive(graph, images[mode], argument, initial)
                workload = {"wasm_sha256": sha(raw), "image_sha256": optimized.image_sha256,
                            "descriptor_ns": prep_ns, "descriptor_bytes": len(original.rows) * 3 * 8}
                if args.profile_only:
                    output[label] = {"profile": {mode: runtime_profile(
                        graph, images[mode], argument, initial, set(sites)) for mode, graph in graphs.items()}}
                    continue
                workload["timings"] = bench(graphs, images, raw, argument, initial,
                                            [x["moves"] for x in world["inputs"]] if label == "terrain" else [None]*5,
                                            steps)
                if label == "terrain":
                    workload["audits"] = {mode: audit(one, graph, plain, optimized, argument, initial, 5000, set(sites))
                                          for mode, graph in graphs.items() if mode != "portfolio"}
                    workload["histories"] = histories(graphs, images, raw)
                else:
                    # The unchanged BFS binary admits no composed-load site.
                    wide = torch.jit.load(str(COMPOSED / "machine.pt"))
                    empty_composition = extend(original, candidate, None)
                    controls = {"portfolio": graphs["portfolio"], "composed_zero_sites": wide}
                    workload["zero_composition_sites"] = bench(controls, {"portfolio": original,
                        "composed_zero_sites": empty_composition}, raw, argument, initial, [None]*2, steps)
                workload["profile"] = {mode: runtime_profile(graph, images[mode], argument, initial, set(sites))
                                       for mode, graph in graphs.items()}
                output[label] = workload
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
