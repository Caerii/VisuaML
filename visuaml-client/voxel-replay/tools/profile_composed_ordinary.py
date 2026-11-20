"""Attribute a short sampled portfolio prefix to ordinary and fused operators."""
from collections import Counter
import json

import torch

import terrain_verify as terrain
from replay_composed_loads import EXECUTOR, PORTFOLIO, PAIR, PRIOR, prepare
from replay_frozen_recurrent import ARCHIVE, PACKED, graph_case
from unittest.mock import patch
from visuaml.compiler_lab.frozen_calls_v58 import initialize, load_call_image


def group_prefix(graph, image, argument, page, sites):
    state = initialize(image, argument, page, 100_000)
    groups = {"ordinary": Counter(), "covered": Counter()}
    times = {name: Counter() for name in groups}
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
        for _ in range(256):
            if int(state.vm.status):
                break
            fused = int(state.vm.pc) in sites
            name = "covered" if fused else "ordinary"
            with torch.profiler.record_function(name):
                state = graph(image.rows, state, 2 if fused else 1).state
    for event in profile.events():
        parent = event.cpu_parent
        while parent is not None and parent.name not in groups:
            parent = parent.cpu_parent
        if parent is not None:
            groups[parent.name][event.name] += 1
            times[parent.name][event.name] += event.self_cpu_time_total
    return {name: {"counts": dict(counts), "self_cpu_us": dict(times[name])}
            for name, counts in groups.items()}


def main():
    torch.set_num_threads(1)
    base = torch.jit.load(str(EXECUTOR / "machine.pt"))
    graph = torch.jit.load(str(PORTFOLIO / "machine.pt"))
    planners = {1: (PAIR / "planner.wasm").read_bytes(), 2: (PRIOR / "load-planner.wasm").read_bytes()}
    chooser = (PORTFOLIO / "chooser.wasm").read_bytes()
    training = json.loads((PORTFOLIO / "training.json").read_text())
    argument, bfs_page, *_ = graph_case(0)
    page = terrain.initial_page(terrain.scale_session(8, False, 1)["initial"])
    terrain.write(page, 320, [4] * 8)
    out = {}
    with patch(PACKED, side_effect=AssertionError("Unexpected packed fallback")):
        for name, raw, arg, initial in (
                ("terrain", terrain.GUEST.read_bytes(), terrain.BASE, page),
                ("bfs", (ARCHIVE / "v58-stateful-scale-7125937/bfs.wasm").read_bytes(), argument, bfs_page)):
            image = load_call_image(raw)
            counts = {int(k): v for k, v in training[name]["pc_counts"].items()}
            images, _, selected, _ = prepare(image, base, planners, chooser, counts)
            for _ in range(8):
                graph(images["portfolio"].rows, initialize(image, arg, initial, 100_000), 64)
            out[name] = group_prefix(graph, images["portfolio"], arg, initial, set(selected))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
