"""Compare archived budget-64 execution with separately frozen reuse variants."""
import argparse
import json
import resource
import statistics
from time import perf_counter_ns
from unittest.mock import patch

import torch

import terrain_verify as terrain
from bench_frozen_multistep import CALLS, OUT as MULTISTEP, summarize, timed_drive
from replay_frozen_recurrent import ARCHIVE, EVIDENCE, PACKED, graph_case, sha, wasmtime_run
from visuaml.compiler_lab.frozen_calls_v58 import initialize, load_call_image
from visuaml.compiler_lab.frozen_recurrent_v58 import snapshot_page

OUT = EVIDENCE.parent / "frozen-state-reuse"


def measure(machines, cases, raw, image, argument, page, turns):
    samples = {name: [] for name in machines}
    names = tuple(machines)
    for tick, inputs in enumerate(turns):
        if inputs is not None:
            terrain.write(page, 320, [4 * move for move in inputs["moves"]])
        expected, oracle = wasmtime_run(raw, argument, page)
        offset = tick % len(names)
        for name in names[offset:] + names[:offset]:
            graph = machines[name]
            output, sample = timed_drive(graph, graph, image, page, argument, 64)
            assert int(output.vm.status) == 1 and int(output.vm.stack[0]) == expected
            assert snapshot_page(output.vm) == oracle
            samples[name].append(sample)
        if inputs is not None:
            page = bytearray(oracle)
    steps = {row["source_steps"] for rows in samples.values() for row in rows}
    assert len(steps) == 1
    cases.append({"image_sha256": image.image_sha256, "binary_sha256": image.binary_sha256,
                  "oracle_sha256": sha(oracle), "pages": image.memory_pages,
                  "timings": summarize(samples), "source_steps": steps.pop()})
    return page


def profile(machines, image, argument, page):
    profiles = {}
    for name, graph in machines.items():
        for _ in range(3):
            graph(image.rows, initialize(image, argument, page, 100), 64)
        initial = initialize(image, argument, page, 100)
        with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU], profile_memory=True) as records:
            result = graph(image.rows, initial, 64).state
        by_operator = [{"op": item.key, "count": item.count,
                        "self_cpu_us": item.self_cpu_time_total,
                        "net_self_cpu_bytes": item.self_cpu_memory_usage}
                       for item in records.key_averages()]
        positive = [event.self_cpu_memory_usage for event in records.events()
                    if event.self_cpu_memory_usage > 0]
        profiles[name] = {"source_steps": 100 - int(result.vm.fuel), "operators": by_operator,
                          "positive_self_memory_events": len(positive),
                          "positive_self_memory_bytes": sum(positive)}
    return profiles


def main():
    parser = argparse.ArgumentParser()
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--selected", action="store_true")
    modes.add_argument("--memory", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    machines, loads = {}, {}
    paths = [("state", OUT / "state.pt")]
    if args.memory:
        paths = [("selected", OUT / "selected.pt"),
                 ("memory", OUT.parent / "frozen-memory-selection/machine.pt")]
    elif args.selected:
        paths.append(("selected", OUT / "selected.pt"))
    else:
        paths = [("baseline", MULTISTEP / "machine.pt"), ("row", OUT / "row.pt"), *paths]
    for name, path in paths:
        started = perf_counter_ns()
        machines[name] = torch.jit.load(str(path))
        loads[name] = {"load_ns": perf_counter_ns() - started, "sha256": sha(path.read_bytes())}
    cases = []
    with patch(PACKED, side_effect=AssertionError("Unexpected packed fallback")):
        for pages, path in ((1, terrain.GUEST), (256, CALLS / "terrain-256.wasm")):
            raw = path.read_bytes()
            started = perf_counter_ns()
            image = load_call_image(raw, memory_pages=pages)
            prepared = perf_counter_ns() - started
            session = terrain.scale_session(8, crowded=False, ticks=18 if args.selected else 22)
            page = terrain.initial_page(session["initial"])
            page.extend(bytes((pages - 1) * terrain.PAGE_BYTES))
            measure(machines, cases, raw, image, terrain.BASE, page, session["inputs"])
            terrain.write(page, 320, [4] * 8)
            costs = []
            for _ in range(9):
                state = initialize(image, terrain.BASE, page, 100).vm
                started = perf_counter_ns()
                snapshot_page(state)
                costs.append(perf_counter_ns() - started)
            cases[-1].update(workload="8-agent terrain", preparation_ns=prepared,
                             snapshot_ns=costs, snapshot_median_ns=statistics.median(costs),
                             profile=profile(machines, image, terrain.BASE, page))
        raw = (ARCHIVE / "v58-stateful-scale-7125937/bfs.wasm").read_bytes()
        started = perf_counter_ns()
        image = load_call_image(raw)
        prepared = perf_counter_ns() - started
        argument, page, *_ = graph_case(0)
        measure(machines, cases, raw, image, argument, page, (None, None) if args.selected or args.memory else (None, None, None))
        assert cases[-1]["source_steps"] == 41591
        cases[-1].update(workload="full BFS", preparation_ns=prepared,
                         profile=profile(machines, image, argument, page))
    print(json.dumps({"torch": torch.__version__, "graphs": loads, "cases": cases,
                      "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}, indent=2))


if __name__ == "__main__":
    main()
