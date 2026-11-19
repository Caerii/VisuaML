"""Replay an exported session and emit a full-page proof for each tick."""
import argparse
import hashlib
import json
import struct
from pathlib import Path
from time import perf_counter_ns
from unittest.mock import patch

import wasmtime

from visuaml.compiler_lab import neural_microop_executor_v57 as vm
from visuaml.compiler_lab.wasm_control_flow_v51 import decode_control, encode_control

HERE = Path(__file__).resolve().parent
BASE = 256
SIZE = 65_536
GUESTS = {
    "single": HERE.parents[1] / "benchmarks/results/v58-voxel-tick/voxel_tick.wasm",
    "shared": HERE / "guest/shared_tick.wasm",
}
EXPECTED = {
    "single": [([1], [0], False), ([1], [7], False)],
    "shared": [([1, 3], [0, 0], False), ([1, 3], [0, 0], True)],
}


def digest(page):
    return hashlib.sha256(page).hexdigest()


def verify(session, expected=None):
    mode = session["mode"]
    assert session["version"] == 1 and mode in GUESTS
    shared = mode == "shared"
    cells = 5 if shared else 4
    count = 2 if shared else 1
    positions = session["initial"]["positions"]
    materials = session["initial"]["materials"]
    assert len(positions) == count and len(set(positions)) == count and len(materials) == cells
    assert all(type(p) is int and 0 <= p < cells and materials[p] == 0 for p in positions)
    assert all(type(m) is int and 0 <= m <= 0x7fffffff for m in materials)
    raw = GUESTS[mode].read_bytes()
    assert digest(raw) == session["binary_sha256"]
    start = perf_counter_ns()
    program = decode_control(raw, scaled=True)
    assert encode_control(program) == raw
    engine = wasmtime.Engine()
    store = wasmtime.Store(engine)
    exports = wasmtime.Instance(store, wasmtime.Module(engine, raw), []).exports(store)
    preparation_ns = perf_counter_ns() - start
    array_page = bytearray(SIZE)
    struct.pack_into(f"<{count}i", array_page, BASE, *(p * 4 for p in positions))
    struct.pack_into(f"<{cells}i", array_page, BASE + (32 if shared else 16), *materials)
    initial_sha = digest(array_page)
    exports["memory"].write(store, array_page)
    records = []
    start = perf_counter_ns()

    with patch("visuaml.compiler_lab.neural_microstep_executor_v58._NativeMicrostep",
               side_effect=AssertionError("Unexpected packed fallback")):
        for i, inputs in enumerate(session["inputs"]):
            assert len(inputs) == count and all(type(d) is int and d in (-1, 0, 1) for d in inputs)
            targets = [p + d for p, d in zip(positions, inputs, strict=True)]
            assert all(0 <= t < cells for t in targets)
            offset = BASE + (8 if shared else 4)
            packed = struct.pack(f"<{count}i", *(d * 4 for d in inputs))
            array_page[offset:offset + len(packed)] = packed
            exports["memory"].write(store, packed, offset)
            tick_start = perf_counter_ns()
            reference = exports["run"](store, BASE)
            reference_ns = perf_counter_ns() - tick_start

            first_route_ns = None
            route = vm._array_route

            def first_route(*args):
                nonlocal first_route_ns
                first_route_ns = perf_counter_ns()
                vm._array_route = route
                return route(*args)

            with patch.object(vm, "_array_route", first_route):
                tick_start = perf_counter_ns()
                result = vm.execute_microop_routed_control(program, BASE, array_memory=array_page,
                                                          capture_trace=False)
                tick_end = perf_counter_ns()
            assert result.backend == "v58-structured-numpy-memory"
            assert result.result_i32 == reference and array_page == exports["memory"].read(store)
            positions = [p // 4 for p in struct.unpack_from(f"<{count}i", array_page, BASE)]
            observed = list(struct.unpack_from(f"<{count}i", array_page, BASE + (16 if shared else 8)))
            conflict = shared and struct.unpack_from("<i", array_page, BASE + 24)[0] == 1
            assert len(set(positions)) == count
            assert observed == [materials[t] for t in targets]
            if expected is not None:
                assert (positions, observed, conflict) == expected[i]
            records.append({"page_sha256": digest(array_page), "positions": positions,
                            "materials": observed, "conflict": conflict, "steps": result.steps,
                            "numpy_to_first_route_ns": first_route_ns - tick_start,
                            "numpy_first_route_to_return_ns": tick_end - first_route_ns,
                            "wasmtime_call_ns": reference_ns})
    return {"session": session, "initial_page_sha256": initial_sha, "ticks": records,
            "module_preparation_ns": preparation_ns, "replay_ns": perf_counter_ns() - start,
            "engines": ["VisuaML fixed-weight NumPy", "Wasmtime"], "packed_fallback": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session", type=Path)
    parser.add_argument("--fixture", action="store_true", help="Check independent shipped-scene expectations")
    args = parser.parse_args()
    session = json.loads(args.session.read_text())
    expected = EXPECTED[session["mode"]] if args.fixture else None
    if expected is not None:
        assert len(session["inputs"]) == len(expected)
    print(json.dumps(verify(session, expected), indent=2))


if __name__ == "__main__":
    main()
