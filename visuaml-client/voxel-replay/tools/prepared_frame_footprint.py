"""Measure actual scaled-limit setup before the first source route."""
import json
import sys
from statistics import median
from time import perf_counter_ns
from unittest.mock import patch

import wasmtime

from visuaml.compiler_lab import neural_microop_executor_v57 as vm
from visuaml.compiler_lab.compiled_frame_event_v58 import _CompiledFrameRetain
from visuaml.compiler_lab.constructed_alu_v58 import compiled_alu
from visuaml.compiler_lab.neural_frame_circuit_v53 import HEIGHTS, KINDS
from visuaml.compiler_lab.wasm_control_flow_v51 import SCALED_MAX_OPERATIONS, decode_control


class ReachedRoute(Exception):
    pass


def main():
    operations = "nop " * (SCALED_MAX_OPERATIONS - 3)
    raw = bytes(wasmtime.wat2wasm(f"""
        (module (func (export "run") (param i32) (result i32)
          i32.const 0 i32.load {operations})
          (memory (export "memory") 1 1))
    """))
    program = decode_control(raw, scaled=True)
    assert len(program.operations) == SCALED_MAX_OPERATIONS
    start = perf_counter_ns()
    kernels = compiled_alu()
    compile_ns = perf_counter_ns() - start
    records = {}
    for pair in range(9):
        order = (False, True) if pair % 2 == 0 else (True, False)
        for prepared in order:
            name = "prepared" if prepared else "copy"
            record = records.setdefault(name, {"setup_ns": []})

            def bind(*args, **kwargs):
                binding = _CompiledFrameRetain(*args, **kwargs)
                record["row_payload_bytes"] = binding._rows.nbytes
                pointers = binding._row_pointers
                record["pointer_table_shallow_bytes"] = sys.getsizeof(pointers)
                record["pointer_integers_bytes"] = sum(map(sys.getsizeof, pointers))
                return binding

            def stop(*_):
                raise ReachedRoute

            with patch.object(vm, "_CompiledFrameRetain", bind), patch.object(vm, "_array_route", stop):
                start = perf_counter_ns()
                try:
                    vm.execute_microop_routed_control(
                        program, 0, array_memory=bytearray(65536), capture_trace=False, fuel=1,
                        use_compiled_frame_event=True, compiled_frame_retain="copy",
                        prepare_frame_inputs=prepared,
                    )
                except ReachedRoute:
                    if pair:
                        record["setup_ns"].append(perf_counter_ns() - start)
                else:
                    raise AssertionError("Expected first-route observation")
            record["lookup_payload_bytes"] = (KINDS**2 + HEIGHTS**2 + KINDS + HEIGHTS) * 8 if prepared else 0
    for record in records.values():
        record["median_setup_ns"] = median(record["setup_ns"])
    print(json.dumps({"source_instructions": len(program.operations), "compile_and_load_ns": compile_ns,
                      "compiler": kernels.compiler, "measurements": records}, indent=2))


if __name__ == "__main__":
    main()
