"""Optional full NumPy correctness check, isolated from voxel timings."""
import hashlib
import json
import struct
from time import perf_counter
from unittest.mock import patch

import wasmtime

from scripts.stateful_scale_workloads_v58 import PUZZLES, sudoku_page, sudoku_reference
from visuaml.compiler_lab.neural_microop_executor_v57 import execute_microop_routed_control
from visuaml.compiler_lab.wasm_control_flow_v51 import decode_control

from scale_terrain import ARCHIVE


def main():
    raw = (ARCHIVE / "sudoku.wasm").read_bytes()
    reference_archive = json.loads((ARCHIVE / "manifest.json").read_text())
    binary_hash = hashlib.sha256(raw).hexdigest()
    assert binary_hash == reference_archive["artifacts"]["sudoku.wasm"]["sha256"]
    givens = tuple(map(int, PUZZLES["escargot"]))
    page, argument = sudoku_page(givens)
    program = decode_control(raw, scaled=True)
    engine = wasmtime.Engine()
    store = wasmtime.Store(engine)
    guest = wasmtime.Instance(store, wasmtime.Module(engine, raw), []).exports(store)
    guest["memory"].write(store, page)
    expected = guest["run"](store, argument)
    start = perf_counter()
    with patch("visuaml.compiler_lab.neural_microstep_executor_v58._NativeMicrostep",
               side_effect=AssertionError("Unexpected packed fallback")):
        result = execute_microop_routed_control(program, argument, array_memory=page,
                                                fuel=2_080_993, capture_trace=False)
    seconds = perf_counter() - start
    assert result.backend == "v58-structured-numpy-memory"
    assert result.result_i32 == expected == 1 and result.steps == 2_080_993
    assert page == guest["memory"].read(store)
    assert sudoku_reference(givens) == (struct.unpack_from("<81i", page),)
    print(json.dumps({"binary_sha256": binary_hash, "steps": result.steps,
                      "page_sha256": hashlib.sha256(page).hexdigest(),
                      "numpy_seconds": seconds}, indent=2))


if __name__ == "__main__":
    main()
