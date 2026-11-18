"""v50: blinded behavioral program recovery, SMT-refined and falsifiable.

The candidate synthesizer receives ONLY a callable to the constructed v49
neural machine, not source instructions, weights, program token arrays or
construction manifests. After the neural candidate is chosen, a separately
implemented Z3 bit-vector verifier proves/disproves all-i32 source semantics.
"""
from __future__ import annotations

from dataclasses import replace

import pytest

from visuaml.compiler_lab.bit_neural_vm_v49 import (
    construct_bit_weights,
    run_bit_vm,
)
from visuaml.compiler_lab.blinded_behavioral_synthesis_v50 import (
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

BOUNDARY = (-(1 << 31), (1 << 31)-1, -65536, -1, 0, 1, 2, 65535)


def op(name: str, immediate: int | None = None) -> VMOperation:
    return VMOperation(name, immediate)


def source_programs() -> tuple[tuple[str, I32Program, Candidate], ...]:
    return (
        ("quadratic", I32Program(0, (
            op("local.get", 0), op("local.get", 0), op("i32.mul"),
            op("local.get", 0), op("i32.const", 3), op("i32.mul"),
            op("i32.sub"), op("i32.const", 7), op("i32.add"), op("end"),
        )), Candidate("poly", a=1, b=-3, c=7)),
        ("identity-via-zero", I32Program(0, (
            op("local.get", 0), op("i32.const", 0),
            op("i32.add"), op("end"),
        )), Candidate("poly", a=0, b=1, c=0)),
        ("zero-test", I32Program(0, (
            op("local.get", 0), op("i32.eqz"), op("end"),
        )), Candidate("eqz")),
        ("signed-negative", I32Program(0, (
            op("local.get", 0), op("i32.const", 0),
            op("i32.lt_s"), op("end"),
        )), Candidate("lt_s", threshold=0)),
        ("signed-positive", I32Program(0, (
            op("local.get", 0), op("i32.const", 0),
            op("i32.gt_s"), op("end"),
        )), Candidate("gt_s", threshold=0)),
    )


def test_blinded_recovery_from_actual_neural_vm_and_external_z3_source_oracle():
    pytest.importorskip("z3")
    fixed = construct_bit_weights()
    for name, source, expected in source_programs():
        # This function contains the source but its internals are completely
        # opaque to the synthesizer: it receives only integer I/O results.
        def neural_oracle(x):
            return run_bit_vm(source, x, fixed=fixed)["result"]
        def verifier(hypothesis):
            return verify_against_typed_source(source, hypothesis)
        result = synthesize_from_oracle(neural_oracle, verifier)
        assert result["accepted_candidate"] == expected.__dict__, name
        assert result["all_i32_source_vs_candidate_equivalent"]
        assert not result["source_program_received_by_synthesizer"]
        assert not result["construction_manifest_received_by_synthesizer"]
        assert not result["neural_weights_received_by_synthesizer"]
        assert not result["all_i32_neural_weight_equivalence_proved"]
        assert not result["historical_source_uniquely_recovered"]
        assert not result["arbitrary_unknown_weight_decompiled"]
        regenerated, actual_wasm = emit_canonical_wasm(expected)
        assert result["accepted_wasm_hex"] == actual_wasm.hex()
        assert result["accepted_typed_program_sha256"] == regenerated.digest()
        for x in BOUNDARY:
            assert execute_program(regenerated, x).value == (
                execute_program(source, x).value
            )


def test_canonical_behavioral_recovery_does_not_claim_unique_original_source():
    pytest.importorskip("z3")
    _, original, identity_hypothesis = source_programs()[1]
    minimal, _ = emit_canonical_wasm(identity_hypothesis)
    assert encode_program(minimal) != encode_program(original)
    assert identity_hypothesis.value(7) == 7
    assert verify_against_typed_source(original, identity_hypothesis)[
        "solver_outcome"
    ] == "unsat"


def test_external_solver_returns_real_signed_i32_counterexample_to_wrong_rule():
    pytest.importorskip("z3")
    _, source, correct = source_programs()[0]
    wrong = replace(correct, b=-2)
    response = verify_against_typed_source(source, wrong)
    assert response["solver_outcome"] == "sat"
    x = response["counterexample_i32"]
    assert correct.value(x) != wrong.value(x)
    assert response["source_output_u32"] != response["hypothesis_output_u32"]
    assert response["all_i32_neural_circuit_proved"] is False


def test_declared_grammar_rejects_arbitrary_behavior_and_larger_coefficients():
    source = I32Program(0, (
        op("local.get", 0), op("local.get", 0), op("i32.mul"),
        op("local.get", 0), op("i32.mul"), op("end"),
    ))
    def cubic_oracle(x):
        return execute_program(source, x).value
    with pytest.raises(BehavioralSynthesisError, match="No surviving"):
        synthesize_from_oracle(
            cubic_oracle, lambda _: {"solver_outcome":"unsat"},
        )
    with pytest.raises(BehavioralSynthesisError, match="bounded hypothesis"):
        emit_canonical_wasm(Candidate("poly", c=1000))
    with pytest.raises(BehavioralSynthesisError, match="bounded hypothesis"):
        emit_canonical_wasm(Candidate("untrusted"))


def test_adversarial_oracle_and_verifier_are_rejected_before_acceptance():
    def malformed(x):
        return 3.14
    with pytest.raises(BehavioralSynthesisError, match="non-i32"):
        synthesize_from_oracle(malformed, lambda _: {"solver_outcome":"unsat"})

    with pytest.raises(BehavioralSynthesisError, match="conclusive"):
        synthesize_from_oracle(
            lambda x:x, lambda _: {"solver_outcome":"unknown"},
        )
    with pytest.raises(BehavioralSynthesisError, match="contradicted"):
        synthesize_from_oracle(
            lambda x:x,
            lambda _: {"solver_outcome":"sat", "counterexample_i32":0},
        )
    with pytest.raises(BehavioralSynthesisError, match="Counterexample"):
        synthesize_from_oracle(
            lambda x:x,
            lambda _: {"solver_outcome":"sat", "counterexample_i32":1000},
        )


def test_generated_wasm_runs_in_independent_wasmtime():
    wasmtime = pytest.importorskip("wasmtime")
    engine = wasmtime.Engine()
    for _, original, hyp in source_programs():
        program, wasm = emit_canonical_wasm(hyp)
        wasmtime.Module.validate(engine, wasm)
        store = wasmtime.Store(engine)
        run = wasmtime.Instance(
            store, wasmtime.Module(engine, wasm), []
        ).exports(store)["run"]
        for x in BOUNDARY:
            assert run(store, x) == hyp.value(x)
            assert run(store, x) == execute_program(original, x).value
            assert run(store, x) == execute_program(program, x).value
