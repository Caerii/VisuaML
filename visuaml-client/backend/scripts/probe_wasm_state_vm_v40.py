"""Source-bound finite-domain WASM rewrite experiment.

Independently runs raw original and rewritten WASM in pinned Wasmtime and
checks the exact typed Python machine across every signed-8 input plus four
overflow/boundary cases. The binary rewrite itself does NOT claim validation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from importlib.metadata import version
from pathlib import Path

import wasmtime
from visuaml.compiler_lab.wasm_state_vm_v40 import (
    I32Program,
    VMOperation,
    alpha_rename_locals,
    decode_module,
    encode_program,
    execute_program,
)

SCHEMA = "visuaml.wasm-state-differential.v40"


def program() -> I32Program:
    def op(name: str, value: int | None = None) -> VMOperation:
        return VMOperation(name, value)

    return I32Program(3, (
        op("local.get", 0),
        op("local.set", 1),
        op("local.get", 1),
        op("i32.const", 2),
        op("i32.mul"),
        op("local.set", 2),
        op("local.get", 2),
        op("i32.const", 7),
        op("i32.add"),
        op("local.tee", 3),
        op("i32.const", 3),
        op("i32.sub"),
        op("end"),
    ))


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def run() -> dict:
    original = encode_program(program())
    rewritten, receipt = alpha_rename_locals(original, (2, 0, 1))
    baseline = decode_module(original)
    revised = decode_module(rewritten)
    if original == rewritten or not receipt["binary_roundtrip_exact"]:
        raise AssertionError("Source and target binary identity was not changed")
    engine = wasmtime.Engine()
    validators = []
    for raw in (original, rewritten):
        # Full independent WASM validation includes typechecking and structural constraints.
        wasmtime.Module.validate(engine, raw)
        module = wasmtime.Module(engine, raw)
        store = wasmtime.Store(engine)
        instance = wasmtime.Instance(store, module, [])
        validators.append((store, instance.exports(store)["run"]))
    witnesses = []
    inputs = (*range(-128, 128), -(1 << 31), (1 << 31) - 1, 65535, -65536)
    for x in inputs:
        source = execute_program(baseline.program, x)
        target = execute_program(revised.program, x)
        wasm_source = validators[0][1](validators[0][0], x)
        wasm_target = validators[1][1](validators[1][0], x)
        ok = source.value == target.value == wasm_source == wasm_target
        witnesses.append({
            "input_i32": x,
            "typed_source_output": source.value,
            "typed_target_output": target.value,
            "wasmtime_source_output": wasm_source,
            "wasmtime_target_output": wasm_target,
            "source_trace_sha256": source.trace_sha256,
            "target_trace_sha256": target.trace_sha256,
            "step_count": len(source.trace),
            "pass": ok,
        })
        if not ok:
            raise AssertionError(f"WASM and state-transition mismatch on {x}")
    backend = Path(__file__).resolve().parents[1]
    sources = [
        "visuaml/compiler_lab/wasm_state_vm_v40.py",
        "visuaml/compiler_lab/wasm_bitstream_v39.py",
        "scripts/probe_wasm_state_vm_v40.py",
    ]
    return {
        "schema": SCHEMA,
        "complete": True,
        "independent_wasmtime": True,
        "neural_weights_executed": False,
        "all_i32_inputs_proved": False,
        "source_sha256": _sha(original),
        "target_sha256": _sha(rewritten),
        "source_hex": original.hex(),
        "target_hex": rewritten.hex(),
        "source_program_sha256": baseline.program.digest(),
        "target_program_sha256": revised.program.digest(),
        "rewrite_receipt": receipt,
        "checked_inputs": len(witnesses),
        "checked_domain": "signed i8 exhaustive, plus 4 i32 boundary/adversarial points",
        "witnesses": witnesses,
        "source_file_sha256": {
            name: _sha((backend / name).read_bytes()) for name in sources
        },
        "wasmtime_version": version("wasmtime"),
        "github_checkout_sha": os.environ.get("GITHUB_SHA"),
        "github_head_ref": os.environ.get("GITHUB_HEAD_REF"),
        "limitations": (
            "One canonical (i32)->i32 WASM function, straight-line instructions, "
            "all locals i32, no imports or side effects except locals/operand stack. "
            "Finite input-domain comparison plus syntactic alpha-inverse evidence, "
            "not a machine-checked proof for unrestricted WASM or neural execution."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "all_passed": all(w["pass"] for w in result["witnesses"]),
        "checked_inputs": result["checked_inputs"],
        "original_sha256": result["source_sha256"],
        "rewritten_sha256": result["target_sha256"],
        "source_program_sha256": result["source_program_sha256"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
