"""Source-bound real WASM vs analytically compiled ReLU and manifest-free decoder.

The weights implement *a finite 256-input lookup via a piecewise-affine
circuit*, not a general transformer virtual machine or reverse engineering of
the original source program. The independent runtime is Wasmtime 28.0.0.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from importlib.metadata import version
from pathlib import Path

import numpy as np
import wasmtime
from visuaml.compiler_lab.analytic_relu_specialization_v41 import (
    HIGH,
    LOW,
    AnalyticRelu,
    compile_bounded_program,
    decompile_function_from_weights,
    evaluate_bounded,
)
from visuaml.compiler_lab.wasm_state_vm_v40 import (
    I32Program,
    VMOperation,
    encode_program,
    execute_program,
)

SCHEMA = "visuaml.wasm-neural-specialization.v41"


def program() -> I32Program:
    def op(name: str, value: int | None = None) -> VMOperation:
        return VMOperation(name, value)

    return I32Program(0, (
        op("local.get", 0), op("local.get", 0), op("i32.mul"),
        op("local.get", 0), op("i32.const", 3), op("i32.mul"),
        op("i32.sub"), op("i32.const", 7), op("i32.add"), op("end"),
    ))


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def run() -> dict:
    source = program()
    binary = encode_program(source)
    engine = wasmtime.Engine()
    wasmtime.Module.validate(engine, binary)
    store = wasmtime.Store(engine)
    module = wasmtime.Module(engine, binary)
    instance = wasmtime.Instance(store, module, [])
    execute_wasm = instance.exports(store)["run"]
    net, compiler = compile_bounded_program(source)
    # Discard program provenance from the network passed to the decoder.
    without_manifest = AnalyticRelu(
        net.w_in.copy(), net.b_in.copy(), net.w_out.copy(), net.b_out.copy()
    )
    recovered = decompile_function_from_weights(without_manifest)
    witnesses = []
    for value in range(LOW, HIGH + 1):
        wasm = execute_wasm(store, value)
        typed = execute_program(source, value).value
        neural = evaluate_bounded(net, value)
        decoded = recovered.outputs[value - LOW]
        if wasm != typed or wasm != neural or wasm != decoded:
            raise AssertionError(f"Neural/typed/WASM inconsistency for {value}")
        witnesses.append({
            "input_i32": value, "wasmtime": wasm,
            "typed": typed, "relu": neural,
            "extracted_from_weights": decoded, "pass": True,
        })
    # An intentionally out-of-domain falsifier. Its disagreement must be
    # preserved in the record rather than discarded as a "successful VM".
    beyond = HIGH + 1
    outside = int(float(
        (net.w_out @ np.maximum(
            0, net.w_in @ np.array([float(beyond)]) + net.b_in
        ))[0] + net.b_out[0]
    ))
    actual_wasm = execute_wasm(store, beyond)
    if outside == actual_wasm:
        raise AssertionError("Expected concrete out-of-domain counterexample")
    backend = Path(__file__).resolve().parents[1]
    sources = [
        "visuaml/compiler_lab/analytic_relu_specialization_v41.py",
        "visuaml/compiler_lab/wasm_state_vm_v40.py",
        "scripts/probe_analytic_neural_wasm_v41.py",
    ]
    return {
        "schema": SCHEMA,
        "complete": True,
        "real_wasm_executed": True,
        "analytic_neural_weights_executed": True,
        "manifest_free_function_recovered": True,
        "source_program_recovered": False,
        "universal_neural_vm": False,
        "all_i32_inputs_proved": False,
        "domain": [LOW, HIGH],
        "witnesses": witnesses,
        "checked_inputs": len(witnesses),
        "function_sha256": recovered.function_sha256,
        "source_wasm_sha256": _sha(binary),
        "source_wasm_hex": binary.hex(),
        "compiler": compiler,
        "network": {
            "w_in": net.w_in.tolist(),
            "b_in": net.b_in.tolist(),
            "w_out": net.w_out.tolist(),
            "b_out": net.b_out.tolist(),
        },
        "outside_domain_counterexample": {
            "input": beyond,
            "neural_extrapolation": outside,
            "wasmtime_result": actual_wasm,
            "disagree": outside != actual_wasm,
        },
        "wasmtime_version": version("wasmtime"),
        "numpy_version": np.__version__,
        "source_file_sha256": {
            name: _sha((backend / name).read_bytes()) for name in sources
        },
        "github_checkout_sha": os.environ.get("GITHUB_SHA"),
        "limitations": (
            "Finite 256-input i8-domain tabulation compiled into analytic "
            "float64 ReLU hinges; behavioral function extracted under a known "
            "template. Not a programmable attention VM, neural i32 execution "
            "outside the interval, original program recovery, or all-input proof."
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
        "checked_inputs": result["checked_inputs"],
        "function_sha256": result["function_sha256"],
        "model_type": "closed-world analytic ReLU, 255 hidden units",
        "outside_domain_counterexample": result["outside_domain_counterexample"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
