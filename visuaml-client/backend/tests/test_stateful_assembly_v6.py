"""Deterministic stateful semantics; does not assert full upstream engine parity."""

from dataclasses import replace
from itertools import pairwise

import numpy as np
import pytest
from visuaml.compiler_lab.assembly_state import (
    AssemblyState,
    AssemblyStateError,
    StepConfig,
    run_assembly_schedule,
    step_assembly,
)
from visuaml.compiler_lab.stateful_ir import (
    StatefulIRError,
    assert_effects_preserved,
    make_assembly_step_ir,
    validate_stateful_ir,
)


def initial():
    weights = np.array(
        [
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [2, 0, 0, 0],
            [0, 0, 0, 1],
        ],
        dtype=np.float32,
    )
    return AssemblyState(weights, np.zeros(4, np.float32), np.array([0, 2]))


def test_manual_recurrent_step_preupdate_charge_and_hebbian_snapshot():
    state = initial()
    end, receipt = step_assembly(
        state, StepConfig(2, beta=0.5, refraction_strength=0.1)
    )
    assert receipt.source == (0, 2)
    assert receipt.winners == (0, 2)
    np.testing.assert_array_equal(np.array(receipt.raw_drive), [3, 0, 1, 0])
    np.testing.assert_array_equal(np.array(receipt.net_drive), [3, 0, 1, 0])
    assert end.weights[0, 0] == np.float32(1.5)
    assert end.weights[2, 0] == np.float32(3.0)
    assert end.weights[0, 2] == np.float32(1.5)
    assert end.weights[2, 2] == 0
    np.testing.assert_allclose(end.bias, [0.3, 0, 0.1, 0], atol=1e-7)
    np.testing.assert_array_equal(state.weights, initial().weights)
    assert end.step_index == 1


def test_state_changes_across_multiple_recurrent_steps():
    state = initial()
    cfg = StepConfig(2, beta=0.2, refraction_strength=0.3)
    end, receipts = run_assembly_schedule(state, cfg, 5)
    assert len(receipts) == 5
    assert end.step_index == 5
    assert len({r.result_sha256 for r in receipts}) == 5
    for a, b in pairwise(receipts):
        assert a.result_sha256 == b.original_sha256
    assert np.all(end.weights >= 0)


def test_external_source_is_distinct_from_current_winners():
    state = initial()
    end, receipt = step_assembly(
        state,
        StepConfig(1, beta=0.4),
        source_mode="external",
        external_source=np.array([1]),
    )
    assert receipt.source == (1,)
    assert receipt.winners == (1,)
    assert end.weights[1, 1] == pytest.approx(1.4)
    np.testing.assert_array_equal(state.winners, [0, 2])


def test_nonlearning_read_advances_activity_but_not_synapses():
    state = initial()
    end, receipt = step_assembly(
        state, StepConfig(1, beta=1, refraction_strength=1, learning_enabled=False)
    )
    np.testing.assert_array_equal(end.weights, state.weights)
    np.testing.assert_array_equal(end.bias, state.bias)
    assert receipt.learning_applied is False
    assert end.step_index == 1


def test_probe_does_not_mutate_any_state():
    state = initial()
    end, receipt = step_assembly(state, StepConfig(1, beta=0.9), commit=False)
    assert end is state
    assert receipt.original_sha256 == receipt.result_sha256
    assert not receipt.learning_applied and not receipt.committed


def test_exact_tie_break_lowest_neuron_ids():
    state = AssemblyState(
        np.zeros((4, 4), np.float32), np.zeros(4, np.float32), np.array([0, 1])
    )
    end, receipt = step_assembly(state, StepConfig(2))
    assert receipt.winners == (0, 1)
    np.testing.assert_array_equal(end.winners, [0, 1])


def test_weight_ceiling_and_no_negative_synapses():
    state = initial()
    end, _ = step_assembly(state, StepConfig(2, beta=20, weight_ceiling=2))
    assert float(end.weights.max()) == 2
    assert np.all(end.weights >= 0)


def test_reproducibility_same_initial_state_and_inputs():
    a, ar = run_assembly_schedule(
        initial(), StepConfig(2, beta=0.1, refraction_strength=0.2), 7
    )
    b, br = run_assembly_schedule(
        initial(), StepConfig(2, beta=0.1, refraction_strength=0.2), 7
    )
    assert a.digest() == b.digest()
    assert [r.as_dict() for r in ar] == [r.as_dict() for r in br]


@pytest.mark.parametrize(
    "bad", [np.array([-1]), np.array([4]), np.array([1, 1]), np.array([1.0])]
)
def test_invalid_sources_fail_closed_without_mutation(bad):
    state = initial()
    before = state.digest()
    with pytest.raises(AssemblyStateError):
        step_assembly(state, StepConfig(2), source_mode="external", external_source=bad)
    assert state.digest() == before


def test_wrong_mode_shape_and_invalid_params_rejected():
    state = AssemblyState(
        np.ones((3, 4), np.float32), np.zeros(4, np.float32), np.array([0, 1])
    )
    with pytest.raises(AssemblyStateError, match="square"):
        step_assembly(state, StepConfig(2))
    with pytest.raises(AssemblyStateError):
        StepConfig(k=0)
    with pytest.raises(AssemblyStateError):
        StepConfig(k=2, beta=-1)
    with pytest.raises(AssemblyStateError):
        StepConfig(k=2, refraction_strength=float("nan"))


def test_state_defensive_copy_no_alias_with_caller():
    orig = np.ones((3, 3), np.float32)
    state = AssemblyState(orig, np.zeros(3, np.float32), np.array([0]))
    orig[0, 0] = 500
    assert state.weights[0, 0] == 1
    with pytest.raises(ValueError):
        state.weights[0, 0] = 9


def test_can_build_and_validate_effect_tokens_for_both_modes():
    for mode in ("external", "recurrent"):
        program = make_assembly_step_ir(mode)
        validate_stateful_ir(program)
        assert len(program.fingerprint()) == 64
        assert_effects_preserved(program, program)


def test_effect_reorder_refused():
    ir = make_assembly_step_ir()
    inst = list(ir.instructions)
    inst[-1], inst[-2] = inst[-2], inst[-1]
    with pytest.raises(StatefulIRError):
        validate_stateful_ir(replace(ir, instructions=tuple(inst)))


def test_elided_hebbian_write_refused():
    ir = make_assembly_step_ir()
    inst = tuple(x for x in ir.instructions if x.op != "hebbian_write")
    with pytest.raises(StatefulIRError):
        validate_stateful_ir(replace(ir, instructions=inst))


def test_stale_weight_read_refused():
    ir = make_assembly_step_ir()
    inst = list(ir.instructions)
    i = 4
    inst[i] = replace(inst[i], reads=(("weights", 99),))
    with pytest.raises(StatefulIRError, match="Stale"):
        validate_stateful_ir(replace(ir, instructions=tuple(inst)))


def test_forged_pure_mutation_refused():
    ir = make_assembly_step_ir()
    inst = list(ir.instructions)
    inst[3] = replace(inst[3], writes=(("weights", 1),))
    with pytest.raises(StatefulIRError):
        validate_stateful_ir(replace(ir, instructions=tuple(inst)))


def test_effectful_rewrite_cannot_change_wiring():
    ir = make_assembly_step_ir()
    inst = list(ir.instructions)
    inst[4] = replace(inst[4], inputs=("selected", "source_indices"))
    with pytest.raises(StatefulIRError):
        assert_effects_preserved(ir, replace(ir, instructions=tuple(inst)))


def test_ir_source_mode_disagreement_refused():
    state = initial()
    with pytest.raises(AssemblyStateError, match="disagree"):
        step_assembly(state, StepConfig(2), program=make_assembly_step_ir("external"))


def test_zero_steps_is_identity_and_invalid_bounds_rejected():
    state = initial()
    out, receipts = run_assembly_schedule(state, StepConfig(2), 0)
    assert out is state and receipts == ()
    with pytest.raises(AssemblyStateError):
        run_assembly_schedule(state, StepConfig(2), -1)


def test_optional_neural_assemblies_explicit_engine_recurrence_parity():
    """Real library comparison, on the shared simple explicit-weight subset.

    This remains an optional gate until the separate package is installed;
    absence MUST NOT be presented as parity with that engine.
    """
    pytest.importorskip("neural_assemblies")
    from neural_assemblies.core.numpy_engine._explicit import NumpyExplicitEngine

    state = initial()
    config = StepConfig(2, beta=0.2, refraction_strength=0, weight_ceiling=20)
    engine = NumpyExplicitEngine(p=0.2, seed=7, w_max=20)
    engine.add_area("A", 4, 2, beta=0.2)
    connection = engine._area_conns["A"]["A"]
    connection.weights[...] = state.weights
    engine.set_winners("A", np.asarray(state.winners, dtype=np.uint32))
    for _ in range(5):
        state, receipt = step_assembly(state, config)
        reference = engine.project_into("A", [], ["A"], plasticity_enabled=True)
        np.testing.assert_array_equal(np.sort(reference.winners), state.winners)
        np.testing.assert_allclose(connection.weights, state.weights, atol=1e-6, rtol=0)
        assert receipt.committed
