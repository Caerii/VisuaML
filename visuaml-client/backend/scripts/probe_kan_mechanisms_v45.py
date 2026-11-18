"""v45: archived neural weights -> symbolic rule -> modular MultKAN graph.

The KAN front end sees only the independently recovered v43 polynomial.
Original WASM is consulted strictly after KAN construction, for a separate
external Wasmtime and Z3 oracle. No trained KAN or pretrained-model causal
attribution is claimed; all KAN weights here are analytically prescribed.
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
from scripts.probe_neural_symbolic_v43 import load_original_analytic_weights
from visuaml.compiler_lab.analytic_relu_specialization_v41 import evaluate_bounded
from visuaml.compiler_lab.kan_mechanism_ir_v45 import (
    EdgeFunction,
    KANNode,
    KANProgram,
    execute_kan_scalar,
    execute_kan_vector,
    polynomial_identity,
)
from visuaml.compiler_lab.kan_symbolic_frontend_v45 import (
    causal_edge_witnesses,
    quadratic_to_kan,
)
from visuaml.compiler_lab.neural_symbolic_recovery_v43 import (
    check_all_i32_equivalence,
    compile_polynomial_wasm,
    recover_quadratic_from_weights,
)
from visuaml.compiler_lab.wasm_state_vm_v40 import decode_module, execute_program

SCHEMA = "visuaml.kan-symbolic-construction-replay.v45"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _basis_probes() -> tuple[KANProgram, ...]:
    """Small independent extension witnesses: four actual univariate bases."""
    edges = (
        EdgeFunction("quadratic", "x", "polynomial", (1., -2., 0.5)),
        EdgeFunction("piecewise", "x", "linear_spline", (0., 1., 0.),
                     (-1., 0., 1.)),
        EdgeFunction("gaussian", "x", "gaussian_rbf", (1., -0.25),
                     (-1., 1., 0.5, 1.)),
        EdgeFunction("harmonics", "x", "fourier", (0.2, 1., -0.3),
                     (1.,)),
    )
    return tuple(KANProgram(
        ("x",), (KANNode("out", "sum", (edge,)),), ("out",), (-2., 2.)
    ) for edge in edges)


def run() -> dict:
    original_weights, archived, archive_hash = load_original_analytic_weights()
    recovered = recover_quadratic_from_weights(original_weights)
    compiled = quadratic_to_kan(recovered)
    kan = compiled.program

    # The symbolic identity is recomputed from the actual graph, not copied
    # from the source polynomial digest.
    exact = polynomial_identity(kan, output="result")
    if exact != (7, -3, 1):
        raise AssertionError("MultKAN did not independently normalize to the quadratic")
    binary_source = bytes.fromhex(archived["source_wasm_hex"])
    original_machine = decode_module(binary_source).program
    proof = check_all_i32_equivalence(original_machine, recovered)
    if proof["solver_outcome"] != "unsat":
        raise AssertionError("Original symbolic program not all-i32 equivalent")

    reconstructed_machine, binary_new = compile_polynomial_wasm(recovered)
    engine = wasmtime.Engine()
    runners = []
    for binary in (binary_source, binary_new):
        wasmtime.Module.validate(engine, binary)
        store = wasmtime.Store(engine)
        instance = wasmtime.Instance(store, wasmtime.Module(engine, binary), [])
        runners.append((store, instance.exports(store)["run"]))
    samples = np.arange(-128, 128, dtype=np.float64)
    vector = execute_kan_vector(kan, {"x": samples})["result"]
    witnesses = []
    for x in range(-128, 128):
        original = runners[0][1](runners[0][0], x)
        regenerated = runners[1][1](runners[1][0], x)
        neural = evaluate_bounded(original_weights, x)
        scalar = execute_kan_scalar(kan, {"x": x})["result"]
        batch = float(vector[x + 128])
        reference = execute_program(original_machine, x).value
        rebuilt = execute_program(reconstructed_machine, x).value
        if len({original, regenerated, neural, scalar, batch, reference, rebuilt}) != 1:
            raise AssertionError(f"KAN vs neural vs typed and real WASM failed at x={x}")
        witnesses.append({
            "input": x, "original_wasmtime": original,
            "symbolic_wasmtime": regenerated, "original_relu": neural,
            "kan_scalar": scalar, "kan_vector": batch,
            "original_typed": reference, "symbolic_typed": rebuilt,
            "pass": True,
        })

    basis_tests = []
    inputs = np.linspace(-2., 2., 41, dtype=np.float64)
    for program in _basis_probes():
        native = execute_kan_vector(program, {"x": inputs})["out"]
        scalar = np.asarray([execute_kan_scalar(program, {"x": float(x)})["out"]
                             for x in inputs])
        delta = np.abs(native - scalar)
        success = bool(np.isfinite(native).all() and np.all(delta <= 1e-12 + 1e-12 * np.abs(scalar)))
        if not success:
            raise AssertionError("KAN basis vector/scalar parity failed")
        basis_tests.append({
            "basis": program.nodes[0].edges[0].basis,
            "program_sha256": program.digest(),
            "max_abs_error": float(np.max(delta)),
            "sample_count": len(inputs),
            "all_passed": True,
            "exact_symbolic_proved": program.nodes[0].edges[0].basis == "polynomial",
        })

    interventions = causal_edge_witnesses(kan, (-5, -1, 0, 1, 5))
    original_outside = archived["outside_domain_counterexample"]
    if (original_outside["input"] != 128 or
            original_outside["neural_extrapolation"] == original_outside["wasmtime_result"]):
        raise AssertionError("v41 out-of-domain falsifier missing")
    root = Path(__file__).resolve().parents[1]
    source_files = (
        "scripts/probe_kan_mechanisms_v45.py",
        "visuaml/compiler_lab/kan_mechanism_ir_v45.py",
        "visuaml/compiler_lab/kan_symbolic_frontend_v45.py",
    )
    return {
        "schema": SCHEMA,
        "source_commit": os.environ.get("VISUAML_SOURCE_COMMIT"),
        "source_file_sha256": {
            p: _sha((root / p).read_bytes()) for p in source_files
        },
        "original_v41_archive_sha256": archive_hash,
        "original_wasm_sha256": _sha(binary_source),
        "regenerated_wasm_sha256": _sha(binary_new),
        "kan_graph_digest": kan.digest(),
        "kan_graph": asdict(kan),
        "independently_normalized_polynomial_coefficients": [
            str(c) for c in exact
        ],
        "recovered_polynomial": asdict(recovered),
        "source_symbol_sha256": compiled.source_symbol_sha256,
        "graph_symbolically_certified": True,
        "all_i32_symbolic_vs_original_wasm": proof,
        "bounded_neural_domain": [-128, 127],
        "full_domain_bounded_witnesses": len(witnesses),
        "witnesses": witnesses,
        "basis_extensions": basis_tests,
        "kan_graph_edge_ablations": interventions,
        "original_outside_domain_counterexample": original_outside,
        "kan_is_pretrained_mechanism": False,
        "pretrained_feature_discovery_completed": False,
        "all_i32_kan_graph_proved": False,
        "kan_speedup_measured": False,
        "z3_solver_version": proof["solver"],
        "wasmtime_version": version("wasmtime"),
        "numpy_version": np.__version__,
        "limitations": (
            "Typed constructed polynomial MultKAN, four bounded basis families, "
            "full 256-point signed-i8 replay, independent Wasmtime, and exact "
            "rational graph identity. Causal ablation applies to constructed KAN "
            "only; there is NO evidence that any extracted KAN edge is a feature "
            "of a pretrained model. Real-valued transcendental basis evaluations "
            "have finite numeric tests, not exact semantic equivalence proofs."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    record = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "symbolic_coefficients": record["independently_normalized_polynomial_coefficients"],
        "kan_nodes": len(record["kan_graph"]["nodes"]),
        "neural_replay_witnesses": record["full_domain_bounded_witnesses"],
        "basis_families_tested": [v["basis"] for v in record["basis_extensions"]],
        "all_i32_original_vs_symbolic_z3": record["all_i32_symbolic_vs_original_wasm"]["solver_outcome"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
