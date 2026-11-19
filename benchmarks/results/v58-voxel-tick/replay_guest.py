"""Replay the exact rustc output across NumPy, packed C++, and Wasmtime."""
import hashlib
import json
import struct
from pathlib import Path
from unittest.mock import patch

import wasmtime

from scripts.verify_packed_control_v58 import prepare
from visuaml.compiler_lab.neural_microop_executor_v57 import execute_microop_routed_control
from visuaml.compiler_lab.wasm_control_flow_v51 import MEMORY_BYTES, decode_control, encode_control

GUEST = Path(__file__).with_name("voxel_tick.wasm")
TRACE_FIELDS = ("pc", "next_pc", "stack_i32", "locals_i32", "control_labels", "memory_access")


def main():
    raw = GUEST.read_bytes()
    program = decode_control(raw, scaled=True)
    assert encode_control(program) == raw
    packed = prepare(program)
    engine = wasmtime.Engine()
    module = wasmtime.Module(engine, raw)
    store = wasmtime.Store(engine)
    exports = wasmtime.Instance(store, module, []).exports(store)
    base = 256
    rows = []
    for name, position, delta, expected_position, expected_material in (
        ("empty", 0, 4, 4, 0),
        ("collision", 4, 4, 4, 7),
        ("reverse", 12, -4, 12, 7),
        ("reverse_clear", 4, -4, 0, 0),
    ):
        initial = bytearray(MEMORY_BYTES)
        struct.pack_into("<8i", initial, base, position, delta, -1, 0, 0, 0, 7, 0)
        expected = initial.copy()
        struct.pack_into("<i", expected, base, expected_position)
        struct.pack_into("<i", expected, base + 8, expected_material)
        array_page = initial.copy()
        with patch("visuaml.compiler_lab.neural_microstep_executor_v58._NativeMicrostep",
                   side_effect=AssertionError("Unexpected packed entry")):
            result = execute_microop_routed_control(program, base, array_memory=array_page)
        assert result.backend == "v58-structured-numpy-memory"
        packed_page = initial.copy()
        native = packed.run_memory(base, packed_page)
        exports["memory"].write(store, initial)
        answer = exports["run"](store, base)
        assert array_page == packed_page == exports["memory"].read(store) == expected
        assert answer == result.result_i32 == native.result_i32 == expected_position
        assert result.steps == native.steps
        for left, right in zip(result.trace, native.trace, strict=True):
            assert all(left[field] == right[field] for field in TRACE_FIELDS)
        rows.append({"case": name, "initial_position_bytes": position, "delta_bytes": delta,
                     "final_position_bytes": answer, "material": expected_material,
                     "steps": result.steps, "page_sha256": hashlib.sha256(array_page).hexdigest(),
                     "writes": [{"pc": row["pc"], **row["memory_access"]} for row in result.trace
                                if row["opcode"] == "i32.store"]})
    print(json.dumps({"guest_sha256": hashlib.sha256(raw).hexdigest(),
                      "backend": result.backend, "base": base,
                      "opcodes": [op.opcode for op in program.operations], "cases": rows}, indent=2))


if __name__ == "__main__":
    main()
