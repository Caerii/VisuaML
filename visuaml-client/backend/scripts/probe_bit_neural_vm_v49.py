"""v49: immutable fixed bit-neural WASM experiment with full-width i32 witnesses.

This is the independent experiment, not only unit tests. Every source
binary is compiled by Wasmtime, and every instruction's entire stack/local/
program-counter state is checked against the separately implemented v40
typed machine. Gate weights are identical across all programs.

The experiment supports all signed-i32 *input values* but cannot exhaust
all 2**32 values. Neither arbitrary WASM nor standard softmax transformer
execution nor arbitrary decompilation is claimed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

import numpy as np
import wasmtime

from visuaml.compiler_lab.bit_neural_vm_v49 import (
    construct_bit_weights,
    decode_state,
    decode_tokens,
    initial_state,
    bit_step,
    tokenize,
)
from visuaml.compiler_lab.wasm_state_vm_v40 import (
    I32Program, VMOperation, encode_program, execute_program,
    initial_state as reference_initial,
    step as reference_step,
)

SCHEMA = "visuaml.full-width-i32-neural-circuit.v49"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _op(name: str, immediate: int | None = None) -> VMOperation:
    return VMOperation(name, immediate)


def programs() -> dict[str, I32Program]:
    return {
        "identity": I32Program(0, (_op("local.get", 0), _op("end"))),
        "overflow_add": I32Program(0, (
            _op("local.get", 0), _op("i32.const", 1),
            _op("i32.add"), _op("end"),
        )),
        "borrow_sub": I32Program(0, (
            _op("local.get", 0), _op("i32.const", -1),
            _op("i32.sub"), _op("end"),
        )),
        "square": I32Program(0, (
            _op("local.get", 0), _op("local.get", 0),
            _op("i32.mul"), _op("end"),
        )),
        "max_multiplier": I32Program(0, (
            _op("local.get", 0), _op("i32.const", (1 << 31)-1),
            _op("i32.mul"), _op("end"),
        )),
        "minimum_constant": I32Program(0, (
            _op("i32.const", -(1 << 31)), _op("local.get", 0),
            _op("i32.add"), _op("end"),
        )),
        "signed_lt": I32Program(0, (
            _op("local.get", 0), _op("i32.const", 0),
            _op("i32.lt_s"), _op("end"),
        )),
        "signed_gt": I32Program(0, (
            _op("local.get", 0), _op("i32.const", -2),
            _op("i32.gt_s"), _op("end"),
        )),
        "zero": I32Program(0, (
            _op("local.get", 0), _op("i32.eqz"), _op("end"),
        )),
        "local_tee": I32Program(2, (
            _op("local.get", 0), _op("local.set", 1),
            _op("local.get", 1), _op("i32.const", 2),
            _op("i32.mul"), _op("local.tee", 2),
            _op("local.get", 2), _op("i32.add"), _op("end"),
        )),
        "drop": I32Program(0, (
            _op("local.get", 0), _op("i32.const", 19),
            _op("drop"), _op("end"),
        )),
    }


def inputs() -> tuple[int, ...]:
    adversarial = (
        -(1 << 31), -(1 << 31)+1,
        -65536, -129, -128, -1, 0, 1, 2,
        127, 128, 65535, (1 << 31)-2, (1 << 31)-1,
    )
    random = np.random.default_rng(49).integers(
        -(1 << 31), (1 << 31), size=12, dtype=np.int64
    )
    return (*adversarial, *(int(v) for v in random))


def run() -> dict:
    fixed = construct_bit_weights()
    sha = fixed.digest()
    engine = wasmtime.Engine()
    records = []
    witnesses = []
    count = 0
    for name, source in programs().items():
        binary = encode_program(source)
        wasmtime.Module.validate(engine, binary)
        store = wasmtime.Store(engine)
        instance = wasmtime.Instance(
            store, wasmtime.Module(engine, binary), []
        )
        wasm = instance.exports(store)["run"]
        tokens = tokenize(source)
        if decode_tokens(tokens) != source:
            raise AssertionError("Original typed WASM token roundtrip failed")
        for argument in inputs():
            neural = initial_state(source, argument)
            reference = reference_initial(source, argument)
            detail = []
            for operation in source.operations:
                neural = bit_step(neural, tokens, fixed)
                reference, transition = reference_step(source, reference)
                inspection = decode_state(neural, source)
                if inspection != reference:
                    raise AssertionError(
                        f"Bit VM/typed step divergence: {name}, {argument}, {operation}"
                    )
                count += 1
                if argument in (-(1 << 31), 0, (1 << 31)-1):
                    detail.append({
                        "opcode": operation.opcode,
                        "state": asdict(inspection),
                        "typed_transition": asdict(transition),
                    })
            result = reference.stack[0]
            actual_wasm = wasm(store, argument)
            independent_typed = execute_program(source, argument).value
            if result != actual_wasm or result != independent_typed:
                raise AssertionError(f"Independent original WASM diverged on {name}:{argument}")
            witnesses.append({
                "program": name, "input_i32": argument,
                "wasmtime_i32": actual_wasm, "bit_neural_i32": result,
                "typed_reference_i32": independent_typed,
                "original_operation_steps": len(source.operations),
                "full_step_trace": detail,
                "pass": True,
            })
        records.append({
            "program": name,
            "canonical_source_wasm_sha256": _sha(binary),
            "canonical_source_wasm_hex": binary.hex(),
            "typed_program_sha256": source.digest(),
            "program_tokens_sha256": tokens.digest(),
            "fixed_interpreter_weights_sha256": sha,
            "instructions": [asdict(o) for o in source.operations],
        })
    if len({x["fixed_interpreter_weights_sha256"] for x in records}) != 1:
        raise AssertionError("Fixed weights differ across programs")
    if len({x["program_tokens_sha256"] for x in records}) != len(records):
        raise AssertionError("Programs not separately tokenized")
    backend = Path(__file__).resolve().parents[1]
    source_files = (
        "scripts/probe_bit_neural_vm_v49.py",
        "visuaml/compiler_lab/bit_neural_vm_v49.py",
        "visuaml/compiler_lab/fixed_neural_token_vm_v48.py",
        "visuaml/compiler_lab/wasm_state_vm_v40.py",
    )
    return {
        "schema": SCHEMA,
        "source_commit": os.environ.get("VISUAML_SOURCE_COMMIT"),
        "source_file_sha256": {
            p: _sha((backend / p).read_bytes()) for p in source_files
        },
        "fixed_interpreter_weights_sha256": sha,
        "program_count": len(records),
        "inputs_per_program": len(inputs()),
        "total_output_witnesses": len(witnesses),
        "total_stepwise_state_comparisons": count,
        "programs": records, "witnesses": witnesses,
        "input_range_admitted": [-(1 << 31), (1 << 31)-1],
        "input_values_exhaustively_tested": False,
        "modular_add_sub_mul_and_signed_comparisons": True,
        "bit_gates_calculated_not_trained": True,
        "fixed_weights_reused_across_programs": True,
        "independent_original_wasmtime_executed": True,
        "every_witness_step_state_compared_to_v40": True,
        "standard_softmax_transformer_executed": False,
        "general_wasm_supported": False,
        "memory_or_control_flow_supported": False,
        "arbitrary_neural_decompilation_implemented": False,
        "all_i32_neural_circuit_smt_proved": False,
        "wasmtime_version": version("wasmtime"),
        "numpy_version": np.__version__,
        "limitations": (
            "32-bit modular arithmetic and signed comparisons constructed "
            "from fixed ReLU threshold gates/bilinear partial products, "
            "with data-supplied straight-line programs; no imports, "
            "memory, branches, loops, traps, alternate numeric types, "
            "standard transformer softmax layers, or unconstrained decompilation. "
            "Finite signed-i32 witnesses and exhaustive local gate truth tables, "
            "not an all-2**32-input verified theorem."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    receipt = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(
        receipt, indent=2, sort_keys=True, allow_nan=False
    ) + "\n", encoding="utf-8")
    print(json.dumps({
        "programs": receipt["program_count"],
        "witnesses": receipt["total_output_witnesses"],
        "stepwise": receipt["total_stepwise_state_comparisons"],
        "fixed_weights": receipt["fixed_interpreter_weights_sha256"],
        "wasmtime_reference": receipt["independent_original_wasmtime_executed"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
