"""v45 independent typed-KAN and MultKAN basis compilation and replay.

Seven graph families are admitted, analyzed symbolically when warranted, then
run in scalar Python, NumPy, PyTorch and native C++17 from the SAME typed graph.
This does not evaluate trained real-world KAN accuracy or prove causal model
interpretability, and native timings are intentionally not promoted as speedup.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from visuaml.compiler_lab.analytic_relu_specialization_v41 import AnalyticRelu
from visuaml.compiler_lab.native_kan_codegen_v45 import run_kan_cpp17
from visuaml.compiler_lab.neural_symbolic_recovery_v43 import recover_quadratic_from_weights
from visuaml.compiler_lab.torch_kan_v45 import TorchKAN
from visuaml.compiler_lab.typed_kan_v45 import (
    KANEdge,
    KANNode,
    KANProgram,
    evaluate_numpy,
    evaluate_scalar,
    from_recovered_v43_polynomial,
    recover_polynomial,
    validate_kan,
)

SCHEMA = "visuaml.typed-kan-multibasis-replay.v45"
V41_SHA = "035a85ab65c9ed395e3dcb0824899609a14e20563113f0c6370731efe0a0675e"
NUMERIC_TOL = 5e-11


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _edge(source: int, family: str, values: tuple[float, ...], *,
          lo=-1.0, hi=1.0, knots=(), scale=1.0):
    return KANEdge(source, family, values, (lo, hi), knots, scale)


def _cases() -> dict[str, KANProgram]:
    cases = {
        "power": KANProgram(1, ((KANNode((
            _edge(0, "power", (0.5, -1.0, 2.0)),)),),)),
        "chebyshev": KANProgram(1, ((KANNode((
            _edge(0, "chebyshev", (0.5, 0.0, 0.5)),)),),)),
        "fourier": KANProgram(1, ((KANNode((
            _edge(0, "fourier", (0.1, 0.8, -0.4, 0.2, -0.3), scale=2.0),)),),)),
        "gaussian": KANProgram(1, ((KANNode((
            _edge(0, "gaussian", (0.7, -0.4, 0.2),
                  knots=(-0.6, 0.1, 0.7), scale=0.35),)),),)),
        "linear_spline": KANProgram(1, ((KANNode((
            _edge(0, "linear_spline", (0.5, -0.75, 1.0),
                  knots=(-1.0, 0.0, 1.0)),)),),)),
        "multkan_product": KANProgram(2, (
            (
                KANNode((_edge(0, "power", (0.0, 1.0)),)),
                KANNode((_edge(1, "power", (0.0, 1.0)),)),
            ),
            (
                KANNode((
                    _edge(0, "power", (0.0, 1.0)),
                    _edge(1, "power", (0.0, 1.0)),
                ), aggregation="product"),
            ),
        )),
    }
    return cases


def _archived_neural_polynomial() -> tuple[KANProgram, dict]:
    root = Path(__file__).resolve().parents[3]
    raw = (root / "benchmarks/analytic_neural_wasm_v41.json").read_bytes()
    meta = json.loads((
        root / "benchmarks/analytic_neural_wasm_v41.origin.json"
    ).read_bytes())
    if _sha(raw) != V41_SHA or meta["raw_sha256"] != V41_SHA:
        raise AssertionError("v41 original neural archive identity violated")
    record = json.loads(raw)
    weights = record["network"]
    detached = AnalyticRelu(
        np.array(weights["w_in"], dtype=np.float64),
        np.array(weights["b_in"], dtype=np.float64),
        np.array(weights["w_out"], dtype=np.float64),
        np.array(weights["b_out"], dtype=np.float64),
    )
    rule = recover_quadratic_from_weights(detached)
    if (rule.a, rule.b, rule.c) != (1, -3, 7):
        raise AssertionError("Archived neural symbolic identity changed")
    return from_recovered_v43_polynomial(rule), {
        "archived_weights_sha256": V41_SHA,
        "polynomial_sha256": rule.digest(),
        "symbolic": rule.expression(),
        "source_program_supplied_to_extractor": False,
    }


def run() -> dict:
    examples = _cases()
    imported, provenance = _archived_neural_polynomial()
    examples["v43_archived_neural_symbol"] = imported
    results = []
    distinct_families = set()
    rng = np.random.default_rng(45)
    for label, program in examples.items():
        schema = validate_kan(program)
        distinct_families.update(schema["basis_families"])
        if label == "v43_archived_neural_symbol":
            inputs = np.linspace(-128, 127, 80, dtype=np.float64)[:, None]
        else:
            inputs = rng.uniform(-0.9, 0.9, (85, program.input_dim)).astype(np.float64)
        scalar = np.array([
            evaluate_scalar(program, tuple(float(v) for v in row))
            for row in inputs
        ], dtype=np.float64)
        numpy_result = evaluate_numpy(program, inputs)
        torch_result = TorchKAN(program)(
            torch.from_numpy(inputs.copy())
        ).detach().numpy()
        cpp_result, native = run_kan_cpp17(program, inputs)
        max_error = 0.0
        for independent in (numpy_result, torch_result, cpp_result):
            if independent.shape != scalar.shape or not np.isfinite(independent).all():
                raise AssertionError("KAN independent framework ABI mismatch")
            distance = np.max(np.abs(independent - scalar))
            max_error = max(max_error, float(distance))
            if not np.allclose(independent, scalar, atol=NUMERIC_TOL, rtol=NUMERIC_TOL):
                raise AssertionError(f"{label}: KAN scalar/native/Torch parity failed")
        try:
            variables, exprs = recover_polynomial(program)
            symbolic = {
                "variables": [str(v) for v in variables],
                "expressions": [str(v) for v in exprs],
                "symbolic_polynomial_extracted": True,
                "real_functional_or_causal_transfer_certified": False,
            }
        except Exception as exc:
            # Only nonpolynomial basis families are permitted to lack a
            # polynomial result; never hide a polynomial frontend failure.
            if all(f in ("power", "chebyshev") for f in schema["basis_families"]):
                raise
            symbolic = {
                "expressions": None,
                "symbolic_polynomial_extracted": False,
                "reason": type(exc).__name__,
                "real_functional_or_causal_transfer_certified": False,
            }
        if label == "v43_archived_neural_symbol":
            expected = inputs[:, 0] ** 2 - 3 * inputs[:, 0] + 7
            if not np.array_equal(scalar[:, 0], expected):
                raise AssertionError("KAN failed exact archived quadratic")
        results.append({
            "name": label,
            "typed_program": asdict(program),
            "semantic_sha256": schema["program_sha256"],
            "basis_families": schema["basis_families"],
            "multkan_product": schema["has_multiplication_nodes"],
            "input_sha256": _sha(inputs.tobytes()),
            "scalar_output_sha256": _sha(scalar.tobytes()),
            "numpy_output_sha256": _sha(numpy_result.tobytes()),
            "torch_output_sha256": _sha(torch_result.tobytes()),
            "native_output_sha256": _sha(cpp_result.tobytes()),
            "native_receipt": native,
            "max_abs_numeric_difference": max_error,
            "numerical_parity_passed": True,
            "symbolic": symbolic,
        })
    backend = Path(__file__).resolve().parents[1]
    paths = [
        "scripts/probe_typed_kan_v45.py",
        "visuaml/compiler_lab/typed_kan_v45.py",
        "visuaml/compiler_lab/torch_kan_v45.py",
        "visuaml/compiler_lab/native_kan_codegen_v45.py",
    ]
    return {
        "schema": SCHEMA,
        "source_commit": os.environ.get("VISUAML_SOURCE_COMMIT"),
        "source_file_sha256": {
            name: _sha((backend / name).read_bytes()) for name in paths
        },
        "cases": results,
        "numerical_tolerance": NUMERIC_TOL,
        "basis_families": sorted(distinct_families),
        "native_cpp17_executed": True,
        "all_numerical_oracles_passed": True,
        "multkan_product_demonstrated": True,
        "v43_symbolic_recovered": True,
        "v43_archived_neural_provenance": provenance,
        "kan_bases_trained_on_pretrained_checkpoint": False,
        "pretrained_interpretability_certified": False,
        "real_accuracy_or_speedup_measured": False,
        "v45_bases_are_drop_in_pykan_equivalents": False,
        "limitations": (
            "Research-only typed KAN graph with degree-1 linear splines, power "
            "and Chebyshev polynomials, Fourier and Gaussian edges and product "
            "nodes, not full pykan B-spline/adaptive-grid/MultKAN semantics. "
            "No real KAN-pretrained checkpoints, causal feature extraction, "
            "adaptive train-to-symbolic workflow, real dataset accuracy "
            "or native speedup comparison. Symbolic polynomial recovery "
            "only for power/Chebyshev edges."
        ),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    report = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "basis_families": report["basis_families"],
        "cases": len(report["cases"]),
        "max_abs_error": max(x["max_abs_numeric_difference"] for x in report["cases"]),
        "native_compiled_graphs": sum(x["native_receipt"]["native_execution"]
                                      for x in report["cases"]),
        "v43_recovered_symbol": report["v43_archived_neural_provenance"]["symbolic"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
