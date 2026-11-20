"""Matched loop on CPU. Keep cold loading and warm invocation samples separate."""
import json
import platform
from statistics import median
from time import perf_counter_ns
from unittest.mock import patch

import torch
import wasmtime

from internal_tool import OUT, prepare_program
from replay_internal_tool import WAT, oracle
from replay_frozen_recurrent import PACKED
from visuaml_replay import Replay
from visuaml.compiler_lab.frozen_calls_v58 import initialize


def main():
    torch.set_num_threads(1)
    spec = json.loads((OUT / "heldout.json").read_text())
    code = prepare_program(spec["instructions"])
    raw = bytes(wasmtime.wat2wasm(WAT))
    start = perf_counter_ns()
    model = torch.jit.load(str(OUT / "internal-tool.pt"))
    model_load = perf_counter_ns()-start
    start = perf_counter_ns()
    replay = Replay(raw)
    graph_load = perf_counter_ns()-start
    start = perf_counter_ns()
    store = wasmtime.Store()
    exports = wasmtime.Instance(store, wasmtime.Module(store.engine, raw), []).exports(store)
    wasm_load = perf_counter_ns()-start
    tokens, operands, enabled, perturbation = map(torch.tensor, ([4, 1], [2, 4], True, 0))
    page = bytes([2]) + bytes(65535)

    def graph():
        state = initialize(replay.image, 4, page, 1000)
        while int(state.vm.status) == 0:
            state = replay.graph(replay.image.rows, state).state
        return int(state.vm.stack[0])

    def native():
        exports["memory"].write(store, page)
        return exports["run"](store, 4)

    runs = {
        "internal_tool_with_trace": lambda: int(model(tokens, operands, code, enabled, perturbation)[0]),
        "python_logical_trace": lambda: oracle(spec["instructions"], [2, 4])[-1][4],
        "visuaml_source_graph": graph,
        "wasmtime": native,
    }
    result = dict(cpu=platform.machine(), torch=str(torch.__version__), threads=1,
                  cold_load_ns=dict(internal_tool=model_load, visuaml_source_graph=graph_load, wasmtime=wasm_load),
                  warm_ns={key: [] for key in runs})
    with torch.inference_mode(), patch(PACKED, side_effect=AssertionError("Unexpected packed fallback")):
        for function in runs.values():
            for _ in range(5):
                assert function() == 14
        items = tuple(runs.items())
        for i in range(48):
            for key, function in (items if i % 2 == 0 else reversed(items)):
                start = perf_counter_ns()
                answer = function()
                result["warm_ns"][key].append(perf_counter_ns()-start)
                assert answer == 14
    result["median_ns"] = {key: median(values) for key, values in result["warm_ns"].items()}
    (OUT / "benchmark.json").write_text(json.dumps(result, indent=2)+"\n")
    print(result["median_ns"])


if __name__ == "__main__":
    main()
