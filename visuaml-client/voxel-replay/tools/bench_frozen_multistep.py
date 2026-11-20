"""Paired wall timings and a separate operator profile for the frozen runner."""
import argparse
import json
import math
import resource
import statistics
from time import perf_counter_ns
from unittest.mock import patch

import torch

import terrain_verify as terrain
from replay_frozen_calls import OUT as CALLS, run as audit_run
from replay_frozen_recurrent import ARCHIVE, EVIDENCE, PACKED, graph_case, sha, wasmtime_run
from visuaml.compiler_lab.frozen_calls_v58 import initialize, load_call_image
from visuaml.compiler_lab.frozen_recurrent_v58 import snapshot_page

OUT = EVIDENCE.parent / "frozen-multistep"
BUDGETS = (1, 4, 16, 64)


def drive(machine, loop, image, state, budget):
    calls = 0
    while int(state.vm.status) == 0:
        state = (machine(image.rows, state) if budget == 1
                 else loop(image.rows, state, budget)).state
        calls += 1
    return state, calls


def timed_drive(machine, loop, image, page, argument, budget):
    started = perf_counter_ns()
    state = initialize(image, argument, page, 100_000)
    initialized_ns = perf_counter_ns() - started
    started = perf_counter_ns()
    state, calls = drive(machine, loop, image, state, budget)
    elapsed = perf_counter_ns() - started
    return state, {"host_calls": calls, "execution_ns": elapsed,
                   "initialize_ns": initialized_ns, "source_steps": 100_000 - int(state.vm.fuel)}


def paired(machine, loop, image, page, argument, samples, ordinal, expected, oracle):
    rotation = ordinal % len(BUDGETS)
    order = BUDGETS[rotation:] + BUDGETS[:rotation]
    for budget in order:
        state, result = timed_drive(machine, loop, image, page, argument, budget)
        assert int(state.vm.status) == 1 and int(state.vm.stack[0]) == expected
        assert snapshot_page(state.vm) == oracle
        samples[str(budget)].append(result)


def summarize(samples):
    result = {}
    for budget, all_rows in samples.items():
        rows = all_rows[1:] if len(all_rows) > 4 else all_rows
        times = sorted(row["execution_ns"] for row in rows)
        result[budget] = {"warm_samples": len(times), "median_ns": statistics.median(times),
                          "p95_ns": times[math.ceil(.95 * len(times)) - 1],
                          "median_initialize_ns": statistics.median(row["initialize_ns"] for row in rows),
                          "rows": all_rows}
    return result


def diagnose(machine, loop, image, page, argument, steps):
    state = initialize(image, argument, page, 100_000)
    for _ in range(8):
        loop(image.rows, state, 0)
    started = perf_counter_ns()
    for _ in range(1000):
        loop(image.rows, state, 0)
    crossing = (perf_counter_ns() - started) / 1000
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
        for _ in range(steps):
            state = machine(image.rows, state).state
    operators = [{"name": e.key, "self_cpu_us": e.self_cpu_time_total, "count": e.count}
                 for e in profile.key_averages()]
    started = perf_counter_ns()
    for _ in range(8):
        snapshot_page(state.vm)
    snapshot_ns = (perf_counter_ns() - started) / 8
    started = perf_counter_ns()
    _, trace = audit_run(machine, image, argument, page, fuel=steps, trace=True)
    trace_ns = perf_counter_ns() - started
    _, baseline = timed_drive(machine, loop, image, page, argument, 1)
    return {"profiled_transitions": steps, "boundary_proxy_ns": crossing, "snapshot_ns": snapshot_ns,
            "trace_prefix_rows": len(trace), "trace_prefix_ns_with_initialization": trace_ns,
            "baseline_full_execution_ns": baseline["execution_ns"],
            "baseline_full_source_steps": baseline["source_steps"], "operators": operators}


def evidence_costs():
    """Measure Python trace collection at the same fuel and on identical pages."""
    from visuaml.compiler_lab.frozen_multistep_v58 import freeze_multistep

    build_started = perf_counter_ns()
    freeze_multistep()
    freeze_ns = perf_counter_ns() - build_started
    machine = torch.jit.load(str(CALLS / "machine.pt"))
    cases = []
    paths = ((1, terrain.GUEST), (256, CALLS / "terrain-256.wasm"))
    for pages, path in paths:
        image = load_call_image(path.read_bytes(), memory_pages=pages)
        for agents in (2, 8):
            scene = terrain.scale_session(agents, crowded=False, ticks=1)
            page = terrain.initial_page(scene["initial"])
            page.extend(bytes((pages - 1) * terrain.PAGE_BYTES))
            terrain.write(page, 320, [4 * move for move in scene["inputs"][0]["moves"]])
            samples = {"untraced_ns": [], "traced_ns": [], "snapshot_ns": []}
            for sample in range(9):
                for trace in (False, True) if sample % 2 == 0 else (True, False):
                    started = perf_counter_ns()
                    state, _ = audit_run(machine, image, terrain.BASE, page, fuel=64, trace=trace)
                    elapsed = perf_counter_ns() - started
                    if sample:
                        samples["traced_ns" if trace else "untraced_ns"].append(elapsed)
                started = perf_counter_ns()
                snapshot_page(state.vm)
                if sample:
                    samples["snapshot_ns"].append(perf_counter_ns() - started)
            cases.append({"pages": pages, "agents": agents, "source_steps": 64,
                          "raw": samples, "medians": {key: statistics.median(rows)
                                                       for key, rows in samples.items()}})
    return {"freeze_ns": freeze_ns, "cases": cases}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-costs", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.evidence_costs:
        print(json.dumps(evidence_costs(), indent=2))
        return
    started = perf_counter_ns()
    machine = torch.jit.load(str(CALLS / "machine.pt"))
    reference_load_ns = perf_counter_ns() - started
    started = perf_counter_ns()
    loop = torch.jit.load(str(OUT / "machine.pt"))
    loop_load_ns = perf_counter_ns() - started
    cases = []
    with patch(PACKED, side_effect=AssertionError("Unexpected packed fallback")):
        for pages, path in ((1, terrain.GUEST), (256, CALLS / "terrain-256.wasm")):
            raw = path.read_bytes()
            started = perf_counter_ns()
            image = load_call_image(raw, memory_pages=pages)
            prepare_ns = perf_counter_ns() - started
            for agents in (2, 8):
                session = terrain.scale_session(agents, crowded=False, ticks=17)
                page = terrain.initial_page(session["initial"])
                page.extend(bytes((pages - 1) * terrain.PAGE_BYTES))
                samples = {str(budget): [] for budget in BUDGETS}
                for number, inputs in enumerate(session["inputs"]):
                    terrain.write(page, 320, [4 * move for move in inputs["moves"]])
                    expected, oracle = wasmtime_run(raw, terrain.BASE, page)
                    paired(machine, loop, image, page, terrain.BASE, samples, number, expected, oracle)
                    page = bytearray(oracle)
                diagnostic = diagnose(machine, loop, image, page, terrain.BASE, 64)
                cases.append({"pages": pages, "agents": agents, "image_preparation_ns": prepare_ns,
                              "binary_sha256": sha(raw), "timings": summarize(samples),
                              "diagnostic": diagnostic, "process_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})
        raw = (ARCHIVE / "v58-stateful-scale-7125937/bfs.wasm").read_bytes()
        started = perf_counter_ns()
        image = load_call_image(raw)
        prep_ns = perf_counter_ns() - started
        argument, page, *_ = graph_case(0)
        expected, oracle = wasmtime_run(raw, argument, page)
        samples = {str(budget): [] for budget in BUDGETS}
        # Pilot profile samples a prefix. All wall comparisons run full BFS.
        diagnosis = diagnose(machine, loop, image, page, argument, 64)
        for ordinal in (0, 2):
            paired(machine, loop, image, page, argument, samples, ordinal, expected, oracle)
        assert {r["source_steps"] for rows in samples.values() for r in rows} == {41591}
        cases.append({"workload": "full BFS", "page_sha256": sha(oracle), "binary_sha256": sha(raw),
                      "image_preparation_ns": prep_ns, "timings": summarize(samples),
                      "diagnostic": diagnosis, "process_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})
    print(json.dumps({"torch": torch.__version__, "reference_load_ns": reference_load_ns,
                      "loop_load_ns": loop_load_ns, "cases": cases}, indent=2))


if __name__ == "__main__":
    main()
