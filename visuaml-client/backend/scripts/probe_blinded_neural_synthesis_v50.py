"""v50 original fixed-weight bit-neural behavior -> blinded typed program recovery.

This experiment is not unique-source decompilation and does not inspect
program tokens or model weights inside the synthesizer. Five real canonical
source WASM programs are opaque neural execution oracles. Independently,
Z3 receives the original typed source only inside a verifier callback.

After accepting a candidate it is compiled to DIFFERENT canonical WASM
bytes, checked with Wasmtime on heldout signed-i32 inputs, and the original
neural VM is checked on the same inputs. All claims are bound to source
hashes and the complete query/SMT counterexample transcript.
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
    run_bit_vm,
)
from visuaml.compiler_lab.blinded_behavioral_synthesis_v50 import (
    SCHEMA,
    BehavioralSynthesisError,
    Candidate,
    emit_canonical_wasm,
    synthesize_from_oracle,
    verify_against_typed_source,
)
from visuaml.compiler_lab.wasm_state_vm_v40 import (
    I32Program,
    VMOperation,
    encode_program,
    execute_program,
)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _op(name: str, immediate: int | None = None) -> VMOperation:
    return VMOperation(name, immediate)


def original_programs() -> dict[str, I32Program]:
    return {
        "quadratic": I32Program(0, (
            _op("local.get", 0), _op("local.get", 0), _op("i32.mul"),
            _op("local.get", 0), _op("i32.const", 3), _op("i32.mul"),
            _op("i32.sub"), _op("i32.const", 7), _op("i32.add"), _op("end"),
        )),
        "identity_plus_zero": I32Program(0, (
            _op("local.get", 0), _op("i32.const", 0),
            _op("i32.add"), _op("end"),
        )),
        "zero_test": I32Program(0, (
            _op("local.get", 0), _op("i32.eqz"), _op("end"),
        )),
        "signed_negative": I32Program(0, (
            _op("local.get", 0), _op("i32.const", 0),
            _op("i32.lt_s"), _op("end"),
        )),
        "signed_threshold_5": I32Program(0, (
            _op("local.get", 0), _op("i32.const", 5),
            _op("i32.lt_s"), _op("end"),
        )),
    }


def _session(binary: bytes, engine):
    wasmtime.Module.validate(engine, binary)
    store = wasmtime.Store(engine)
    function = wasmtime.Instance(
        store, wasmtime.Module(engine, binary), []
    ).exports(store)["run"]
    return store, function


def run() -> dict:
    fixed = construct_bit_weights()
    engine = wasmtime.Engine()
    rng = np.random.default_rng(50)
    heldout = (
        -(1 << 31), (1 << 31)-1, -(1 << 31)+1, (1 << 31)-2,
        -1024, -33, 33, 1024,
        *(int(v) for v in rng.integers(
            -(1 << 31), 1 << 31, size=8, dtype=np.int64
        )),
    )
    reports = []
    for name, original_source in original_programs().items():
        original_wasm = encode_program(original_source)

        # The synthesizer is passed two callable oracles, not the source,
        # its tokenization, its construction manifest, or any neural weights.
        def opaque_neural_oracle(x: int, original=original_source) -> int:
            return run_bit_vm(original, x, fixed=fixed)["result"]

        def opaque_smt_verifier(candidate: Candidate, original=original_source) -> dict:
            return verify_against_typed_source(original, candidate)

        recovered = synthesize_from_oracle(
            opaque_neural_oracle, opaque_smt_verifier,
            max_refinements=16,
        )
        symbolic = Candidate(**recovered["accepted_candidate"])
        regenerated, recovered_wasm = emit_canonical_wasm(symbolic)
        if recovered_wasm.hex() != recovered["accepted_wasm_hex"]:
            raise AssertionError("Synthesized typed WASM bytes unexpectedly changed")
        original_store, original_fn = _session(original_wasm, engine)
        regenerated_store, regenerated_fn = _session(recovered_wasm, engine)
        witnesses = []
        for x in heldout:
            baseline = original_fn(original_store, x)
            recompiled = regenerated_fn(regenerated_store, x)
            neural = opaque_neural_oracle(x)
            typed_source = execute_program(original_source, x).value
            if len({baseline, recompiled, neural, typed_source}) != 1:
                raise AssertionError(f"Recovered program diverged on {name} {x}")
            witnesses.append({
                "input_i32": x, "source_wasmtime": baseline,
                "recompiled_wasmtime": recompiled,
                "fixed_neural_bit_gate_output": neural,
                "original_typed_i32": typed_source,
                "pass": True,
            })
        reports.append({
            "name": name,
            "original_wasm_sha256": _sha(original_wasm),
            "original_wasm_hex": original_wasm.hex(),
            "recovered_wasm_sha256": _sha(recovered_wasm),
            "recovered_wasm_hex": recovered_wasm.hex(),
            "recovered_typed_program_sha256": regenerated.digest(),
            "candidate": asdict(symbolic),
            "candidate_expression": symbolic.expression(),
            "source_program_not_passed_into_synthesizer": True,
            "fixed_weights_not_passed_into_synthesizer": True,
            "synthesis_report": recovered,
            "heldout_witnesses": witnesses,
        })
    if not any(
        r["original_wasm_sha256"] != r["recovered_wasm_sha256"]
        and r["synthesis_report"]["all_i32_source_vs_candidate_equivalent"]
        for r in reports
    ):
        raise AssertionError("Source non-uniqueness was not demonstrated")

    # Explicitly refuse a cubic source outside the admitted synthesis grammar,
    # even though the underlying bit-neural arithmetic can evaluate it.
    cubic = I32Program(0, (
        _op("local.get", 0), _op("local.get", 0), _op("i32.mul"),
        _op("local.get", 0), _op("i32.mul"), _op("end"),
    ))
    rejected = False
    try:
        synthesize_from_oracle(
            lambda x: run_bit_vm(cubic, x, fixed=fixed)["result"],
            lambda candidate: verify_against_typed_source(cubic, candidate),
        )
    except BehavioralSynthesisError as exc:
        if "No surviving candidate" not in str(exc):
            raise
        rejected = True
    if not rejected:
        raise AssertionError("Out-of-grammar cubic falsely claimed as decompiled")

    backend = Path(__file__).resolve().parents[1]
    sources = (
        "scripts/probe_blinded_neural_synthesis_v50.py",
        "visuaml/compiler_lab/blinded_behavioral_synthesis_v50.py",
        "visuaml/compiler_lab/bit_neural_vm_v49.py",
    )
    return {
        "schema": "visuaml.blinded-neural-behavior-evidence.v50",
        "synthesis_schema": SCHEMA,
        "source_commit": os.environ.get("VISUAML_SOURCE_COMMIT"),
        "source_file_sha256": {
            p: _sha((backend / p).read_bytes()) for p in sources
        },
        "fixed_bit_weights_sha256": fixed.digest(),
        "program_count": len(reports),
        "heldout_values_per_program": len(heldout),
        "true_source_wasm_compiled_by_wasmtime": True,
        "all_i32_source_candidate_smt_checked": True,
        "multiple_source_programs_same_behavior_possible": True,
        "cubic_outside_grammar_rejected": rejected,
        "black_box_oracle_was_fixed_neural_bit_vm": True,
        "no_source_passed_into_synthesizer": True,
        "no_weights_passed_into_synthesizer": True,
        "unique_historical_program_recovery_claimed": False,
        "arbitrary_neural_model_decompilation_claimed": False,
        "neural_gate_all_i32_equivalence_proved": False,
        "results": reports,
        "wasmtime_version": version("wasmtime"),
        "numpy_version": np.__version__,
        "limitations": (
            "Black-box behavior and opaque equivalence oracle. Only a "
            "small, predeclared polynomial/predicate grammar. Z3 proves "
            "source typed i32 semantics against the recovered candidate; "
            "finite independent original-Wasm and constructed neural "
            "execution witnesses do not formally prove the neural circuit "
            "for all 2^32 inputs. No unique historical source, learned "
            "transformer decompilation, memory/flow or arbitrary WASM."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "programs": report["program_count"],
        "heldout_each": report["heldout_values_per_program"],
        "synthesized": [x["candidate_expression"] for x in report["results"]],
        "source_ambiguity_demonstrated": report[
            "multiple_source_programs_same_behavior_possible"
        ],
        "arbitrary_model_decompilation_claimed": False,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
