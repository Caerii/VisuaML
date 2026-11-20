"""Vary capacity and sparse interacting agent count on one frozen graph."""
import json
import statistics
import time
from unittest.mock import patch

import torch
import wasmtime

import terrain_verify as terrain
from replay_frozen_calls import OUT, run
from replay_frozen_recurrent import PACKED, sha
from visuaml.compiler_lab.frozen_calls_v58 import load_call_image
from visuaml.compiler_lab.frozen_recurrent_v58 import snapshot_page


def scale(machine, raw, image, agents, ticks=16):
    scene = terrain.scale_session(agents, crowded=False, ticks=ticks)
    positions, materials = scene["initial"]["positions"], scene["initial"]["materials"]
    page = terrain.initial_page(scene["initial"])
    page.extend(bytes((image.memory_pages - 1) * terrain.PAGE_BYTES))
    store = wasmtime.Store()
    exports = wasmtime.Instance(store, wasmtime.Module(store.engine, raw), []).exports(store)
    memory = exports["memory"]
    memory.write(store, page)
    samples = []
    for inputs in scene["inputs"]:
        positions, materials, *_ = terrain.resolve(positions, materials, inputs)
        terrain.write(page, 320, [4 * move for move in inputs["moves"]])
        start = terrain.BASE + 320
        memory.write(store, page[start:start + 4 * agents], start)
        expected = exports["run"](store, terrain.BASE)
        started = time.perf_counter_ns()
        state, _ = run(machine, image, terrain.BASE, page, fuel=50_000)
        elapsed = time.perf_counter_ns() - started
        assert int(state.vm.status) == 1 and int(state.vm.stack[0]) == expected
        page = bytearray(snapshot_page(state.vm))
        assert page == memory.read(store)
        assert terrain.read(page, 64, agents) == [4 * p for p in positions]
        samples.append({"ns_with_initialization": elapsed, "source_steps": 50_000 - int(state.vm.fuel),
                        "page_sha256": sha(page[:terrain.PAGE_BYTES])})
    warm = [sample["ns_with_initialization"] for sample in samples[1:]]
    return {"memory_pages": image.memory_pages, "agents": agents,
            "binary_sha256": image.binary_sha256, "image_sha256": image.image_sha256,
            "warm_samples": len(warm), "median_ns": statistics.median(warm),
            "agent_ticks_per_second": agents * len(warm) * 1e9 / sum(warm),
            "samples": samples}


def main():
    torch.set_num_threads(1)
    machine_path = OUT / "machine.pt"
    machine = torch.jit.load(str(machine_path))
    cases = []
    with patch(PACKED, side_effect=AssertionError("Unexpected packed fallback")):
        for pages, binary in ((1, terrain.GUEST), (256, OUT / "terrain-256.wasm")):
            raw = binary.read_bytes()
            image = load_call_image(raw, memory_pages=pages)
            for agents in (2, 8):
                cases.append(scale(machine, raw, image, agents))
    for baseline, large in zip(cases[:2], cases[2:], strict=True):
        assert baseline["image_sha256"] == large["image_sha256"]
        assert [(x["source_steps"], x["page_sha256"]) for x in baseline["samples"]] == [
            (x["source_steps"], x["page_sha256"]) for x in large["samples"]]
    print(json.dumps({"machine_sha256": sha(machine_path.read_bytes()), "cases": cases}, indent=2))


if __name__ == "__main__":
    main()
