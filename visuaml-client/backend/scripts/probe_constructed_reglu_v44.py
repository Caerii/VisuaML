"""v44: archive-bound neural→symbolic→two-ReGLU architecture synthesis.

This is an analytical change of activation family. It does not claim
a speedup, all-i32 float64 equivalence, or universal VM behavior.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path

import numpy as np
import wasmtime
from scripts.probe_neural_symbolic_v43 import load_original_analytic_weights
from visuaml.compiler_lab.analytic_relu_specialization_v41 import evaluate_bounded
from visuaml.compiler_lab.constructed_reglu_quadratic_v44 import (
    ConstructedGatedQuadratic,
    bundle_fingerprint,
    check_real_algebraic_identity,
    construct_from_symbolic,
    evaluate_constructed,
    permute_gated_units,
    recover_from_gated_weights,
)
from visuaml.compiler_lab.neural_symbolic_recovery_v43 import (
    check_all_i32_equivalence,
    recover_quadratic_from_weights,
)
from visuaml.compiler_lab.wasm_state_vm_v40 import decode_module, execute_program

SCHEMA = "visuaml.constructive-gated-resynthesis.v44"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def run() -> dict:
    old, v41, original_weight_archive_sha = load_original_analytic_weights()
    symbolic = recover_quadratic_from_weights(old)
    new, construction = construct_from_symbolic(symbolic)
    # Detach the symbol/source metadata from the inverse again.
    extracted = recover_from_gated_weights(ConstructedGatedQuadratic(
        new.gate_weights.copy(), new.value_weights.copy(),
        new.gate_bias.copy(), new.output_weights.copy(), new.skip, new.bias
    ))
    if extracted != symbolic:
        raise AssertionError("New neural architecture failed reverse extraction")
    reordered = permute_gated_units(new, (1, 0))
    if recover_from_gated_weights(reordered) != symbolic:
        raise AssertionError("Gated reordering lost symbolic meaning")
    real_theorem = check_real_algebraic_identity()
    if not real_theorem["all_real_algebraic_identity_smt_checked"]:
        raise AssertionError("ReGLU algebraic proof failed")

    # Program source is consulted only AFTER both neural→symbolic transitions.
    original_wasm = bytes.fromhex(v41["source_wasm_hex"])
    original_program = decode_module(original_wasm).program
    source_vs_symbolic = check_all_i32_equivalence(original_program, extracted)
    if source_vs_symbolic["solver_outcome"] != "unsat":
        raise AssertionError("Extracted symbolic program not i32-equivalent to source")

    engine = wasmtime.Engine()
    wasmtime.Module.validate(engine, original_wasm)
    store = wasmtime.Store(engine)
    instance = wasmtime.Instance(
        store, wasmtime.Module(engine, original_wasm), []
    )
    source_wasm_fn = instance.exports(store)["run"]
    extras = (-10000, -1000, -129, 128, 129, 1000, 10000)
    witnesses = []
    for x in (*range(-128, 128), *extras):
        original_result = source_wasm_fn(store, x)
        typed = execute_program(original_program, x).value
        neural = evaluate_constructed(new, x)
        permuted = evaluate_constructed(reordered, x)
        if len({original_result, typed, neural, permuted}) != 1:
            raise AssertionError(f"Constructive gated circuit mismatch at {x}")
        witness = {
            "input_integer": x, "source_wasmtime_i32": original_result,
            "typed_source_i32": typed, "constructed_relu_gate_float64": neural,
            "permuted_relu_gate_float64": permuted, "pass": True,
            "in_original_v41_domain": -128 <= x <= 127,
        }
        if -128 <= x <= 127:
            old_output = evaluate_bounded(old, x)
            if old_output != neural:
                raise AssertionError("Gated resynthesis lost old neural behavior")
            witness["original_relu_float64"] = old_output
        witnesses.append(witness)

    x = 128
    original_relu_extrapolation = int(float(
        (old.w_out @ np.maximum(
            0, old.w_in @ np.array([float(x)]) + old.b_in
        ))[0] + old.b_out[0]
    ))
    new_out = evaluate_constructed(new, x)
    wasm_result = source_wasm_fn(store, x)
    if (original_relu_extrapolation, new_out, wasm_result) != (
        16005, 16007, 16007
    ):
        raise AssertionError("Constructive resynthesis lost required falsifier")
    source_dir = Path(__file__).resolve().parents[1]
    sources = (
        "visuaml/compiler_lab/constructed_reglu_quadratic_v44.py",
        "visuaml/compiler_lab/neural_symbolic_recovery_v43.py",
        "scripts/probe_constructed_reglu_v44.py",
    )
    return {
        "schema": SCHEMA,
        "source_commit": os.environ.get("VISUAML_SOURCE_COMMIT"),
        "source_file_sha256": {
            path: _sha((source_dir / path).read_bytes()) for path in sources
        },
        "original_v41_weights_archive_sha256": original_weight_archive_sha,
        "original_v41_wasm_sha256": _sha(original_wasm),
        "symbolic": asdict(symbolic),
        "polynomial_expression": symbolic.expression(),
        "original_ordinary_relu_units": 255,
        "new_gated_reglu_units": 2,
        "original_scalar_parameter_count": 766,
        "new_scalar_parameter_count": 10,
        "different_activation_families": True,
        "estimated_speedup": None,
        "actual_speedup_benchmarked": False,
        "gated_network_raw_weights": {
            "gate_weights": new.gate_weights.tolist(),
            "value_weights": new.value_weights.tolist(),
            "gate_bias": new.gate_bias.tolist(),
            "output_weights": new.output_weights.tolist(),
            "skip": new.skip,
            "bias": new.bias,
        },
        "gated_network_bundle_fingerprint": bundle_fingerprint(new),
        "permuted_bundle_fingerprint": bundle_fingerprint(reordered),
        "gated_construction": construction,
        "real_algebraic_identity_smt": real_theorem,
        "source_vs_recovered_symbolic_all_i32_smt": source_vs_symbolic,
        "original_v41_domain": [-128, 127],
        "exact_neural_domain_witnesses": 256,
        "out_of_original_domain_numeric_witnesses": len(extras),
        "witnesses": witnesses,
        "permutation_invariant_symbolic_recovery": True,
        "old_neural_extrapolation_counterexample": {
            "input": 128,
            "original_v41_relu_output": original_relu_extrapolation,
            "constructed_v44_reglu_output": new_out,
            "wasmtime_source_i32_output": wasm_result,
            "v41_and_v44_differ": original_relu_extrapolation != new_out,
        },
        "constructed_network_all_i32_proved": False,
        "float64_all_real_bitwise_proved": False,
        "symbolic_source_program_extraction_uses_manifests": False,
        "wasmtime_version": __import__("importlib.metadata", fromlist=["version"]).version(
            "wasmtime"
        ),
        "numpy_version": np.__version__,
        "limitations": (
            "Analytic two-ReGLU-gate construction of one known quadratic, "
            "with fixed input signs and a separate linear skip. Exact-real "
            "algebraic identity solver-checked by sign cases; exact float64 "
            "integers only under an explicit bound. Unlike original ReLU "
            "network, not a full i32 modular neural implementation. Not a "
            "trained architecture discovery, GPU performance claim, or "
            "universal neural interpreter."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "symbolic": receipt["polynomial_expression"],
        "old_relu_units": receipt["original_ordinary_relu_units"],
        "new_reglu_units": receipt["new_gated_reglu_units"],
        "real_sign_proof": receipt["real_algebraic_identity_smt"],
        "finite_witnesses": len(receipt["witnesses"]),
        "old_neural_counterexample": receipt["old_neural_extrapolation_counterexample"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
