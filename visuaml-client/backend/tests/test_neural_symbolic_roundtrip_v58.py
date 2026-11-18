"""v58 recovered symbolic program round-trips into live neural microsteps.

The source stays outside the synthesizer; Z3 separately proves typed source
equivalence over the full i32 domain, and Wasmtime checks the original and
recovered binaries. A finite execution replay checks v49 and v58 behavior,
not arbitrary learned-weight semantics.
"""
from __future__ import annotations

import hashlib

import pytest

from visuaml.compiler_lab.bit_neural_vm_v49 import (
    construct_bit_weights,
    run_bit_vm,
)
from visuaml.compiler_lab.blinded_behavioral_synthesis_v50 import (
    BehavioralSynthesisError,
    Candidate,
    synthesize_from_oracle,
)
from visuaml.compiler_lab.neural_symbolic_roundtrip_v58 import (
    INPUTS,
    as_control,
    source_workloads,
    verify_roundtrip,
)
from visuaml.compiler_lab.wasm_control_flow_v51 import (
    decode_control,
    encode_control,
)
from visuaml.compiler_lab.wasm_state_vm_v40 import (
    I32Program,
    VMOperation,
    encode_program,
)


def test_roundtrip_corpus_has_historical_and_heldout_source_separation():
    items = source_workloads()
    assert len(items) == 13
    assert sum(x.split == "historical" for x in items) == 5
    assert sum(x.split == "heldout" for x in items) == 8
    ids = [hashlib.sha256(encode_program(x.source)).hexdigest() for x in items]
    assert len(set(ids)) == 13
    assert len(INPUTS) == 12
    for item in items:
        controls = as_control(item.source)
        assert encode_control(controls) == encode_program(item.source)
        assert decode_control(encode_control(controls)) == controls


@pytest.mark.parametrize("index", range(5))
def test_historical_constructed_neural_behavior_recovers_provable_wasm(index):
    pytest.importorskip("z3")
    pytest.importorskip("wasmtime")
    case = source_workloads()[index]
    result = verify_roundtrip(case)
    assert result["split"] == "historical"
    assert result["cegis_observation_count"] >= 7
    assert result["finite_replay_inputs"] == 12
    assert result["all_i32_typed_source_equivalence"]
    assert not result["all_i32_neural_weights_equivalence"]
    assert not result["unique_original_source_recovered"]
    assert not result["arbitrary_learned_neural_model_decompiled"]
    assert result["finite_recovered_instruction_transitions"] > 0
    assert all(x["source_vs_recovered_equal"] for x in result["finite_results"])


@pytest.mark.parametrize("index", (5, 10))
def test_heldout_source_compositions_blindly_recover(index):
    pytest.importorskip("z3")
    pytest.importorskip("wasmtime")
    result = verify_roundtrip(source_workloads()[index])
    assert result["split"] == "heldout"
    assert result["all_i32_typed_source_equivalence"]
    assert result["finite_replay_inputs"] == 12
    assert not result["same_source_binary"]


def test_nonadmitted_cubic_neural_behavior_rejected_not_marked_proven():
    source = I32Program(0, (
        VMOperation("local.get", 0),
        VMOperation("local.get", 0),
        VMOperation("i32.mul"),
        VMOperation("local.get", 0),
        VMOperation("i32.mul"),
        VMOperation("end"),
    ))
    fixed = construct_bit_weights()

    def opaque_oracle(x):
        return run_bit_vm(source, x, fixed=fixed)["result"]

    with pytest.raises(BehavioralSynthesisError, match="No surviving"):
        synthesize_from_oracle(
            opaque_oracle,
            lambda _: {"solver_outcome": "unsat"},
        )


def test_synthesis_verifier_rejects_forged_proof_bindings():
    from visuaml.compiler_lab.blinded_behavioral_synthesis_v50 import (
        synthesize_from_oracle,
    )

    with pytest.raises(BehavioralSynthesisError, match="Unbound"):
        synthesize_from_oracle(
            lambda x: x,
            lambda _: {
                "solver_outcome": "unsat",
                "candidate_sha256": Candidate("poly", c=5).digest(),
            },
        )
