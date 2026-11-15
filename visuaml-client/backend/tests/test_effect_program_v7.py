"""V7: reusable effects, bounded loops, and stateful executable parity."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from visuaml.compiler_lab.assembly_state import (
    AssemblyState,
    StepConfig,
    run_assembly_schedule,
)
from visuaml.compiler_lab.effect_ir import (
    EffectIRError,
    EffectProgram,
    Instruction,
    Repeat,
    StateSlot,
    TensorType,
    assert_effect_trace_preserved,
    recurrent_assembly_effect_program,
    validate_effect_program,
)
from visuaml.compiler_lab.effect_runtime import execute_effect_program


def _fixture(n=7, k=3, steps=5):
    rng = np.random.default_rng(31)
    weights = rng.integers(0, 4, size=(n, n)).astype(np.float32)
    bias = np.zeros(n, dtype=np.float32)
    winners = np.array([0, 2, 4][:k], dtype=np.int64)
    return recurrent_assembly_effect_program(
        n, k, steps=steps, beta=0.15, refraction_strength=0.07
    ), {"weights": weights, "bias": bias, "winners": winners}


def test_v6_recurrent_oracle_parity_with_refraction_and_hebbian():
    program, original = _fixture()
    got = execute_effect_program(program, original)
    state = AssemblyState(original["weights"], original["bias"], original["winners"])
    expected, history = run_assembly_schedule(
        state, StepConfig(k=3, beta=0.15, refraction_strength=0.07), 5
    )
    np.testing.assert_array_equal(got.final_state["weights"], expected.weights)
    np.testing.assert_array_equal(got.final_state["bias"], expected.bias)
    np.testing.assert_array_equal(got.final_state["winners"], expected.winners)
    assert len(got.receipts) == len(history) == 5
    assert got.receipts[0].before_sha256 == got.initial_sha256
    assert got.receipts[-1].after_sha256 == got.final_sha256
    assert not got.final_state["weights"].flags.writeable
    assert len(got.report()["program_sha256"]) == 64


def test_state_is_copied_and_unchanged_after_success():
    program, initial = _fixture()
    snapshots = {name: a.copy() for name, a in initial.items()}
    _ = execute_effect_program(program, initial)
    for name, snapshot in snapshots.items():
        np.testing.assert_array_equal(initial[name], snapshot)


def test_ir_fingerprint_includes_semantic_parameters():
    program, _ = _fixture()
    ops = list(program.loops[0].instructions)
    first = ops[6]
    ops[6] = replace(first, params=(("beta", 0.35), ("ceiling", 20.0)))
    other = replace(
        program, loops=(replace(program.loops[0], instructions=tuple(ops)),)
    )
    assert (
        validate_effect_program(other).program_sha256
        != validate_effect_program(program).program_sha256
    )


def test_unknown_operator_fails_closed():
    program, _ = _fixture()
    ops = list(program.loops[0].instructions)
    ops[4] = replace(ops[4], op="eval_python")
    with pytest.raises(EffectIRError, match="Unsupported"):
        validate_effect_program(
            replace(
                program, loops=(replace(program.loops[0], instructions=tuple(ops)),)
            )
        )


def test_effect_token_reuse_rejected():
    program, _ = _fixture()
    ops = list(program.loops[0].instructions)
    ops[-1] = replace(ops[-1], token_in="e1")
    with pytest.raises(EffectIRError, match="effect token"):
        validate_effect_program(
            replace(
                program, loops=(replace(program.loops[0], instructions=tuple(ops)),)
            )
        )


def test_stale_version_read_rejected():
    program, _ = _fixture()
    ops = list(program.loops[0].instructions)
    ops[3] = replace(ops[3], revision=1)
    with pytest.raises(EffectIRError, match="revision mismatch"):
        validate_effect_program(
            replace(
                program, loops=(replace(program.loops[0], instructions=tuple(ops)),)
            )
        )


def test_omitted_write_rejected_by_guard():
    before, _ = _fixture()
    ops = tuple(i for i in before.loops[0].instructions if i.name != "publish_winners")
    after = replace(before, loops=(replace(before.loops[0], instructions=ops),))
    validate_effect_program(
        after
    )  # a different dialect may legitimately write fewer cells
    with pytest.raises(EffectIRError, match="rewired"):
        assert_effect_trace_preserved(before, after)


def test_wrong_shape_and_dtype_prevent_initial_execution():
    prog, initial = _fixture()
    for field, replacement in (
        ("bias", np.zeros(8, dtype=np.float32)),
        ("winners", np.array([0, 2, 4], dtype=np.int32)),
    ):
        invalid = dict(initial)
        invalid[field] = replacement
        with pytest.raises(EffectIRError, match="violates"):
            execute_effect_program(prog, invalid)


def test_nan_and_repeated_indices_are_rejected_without_mutation():
    prog, state = _fixture()
    before = state["weights"].copy()
    broken = dict(state)
    broken["winners"] = np.array([1, 1, 1], dtype=np.int64)
    with pytest.raises(EffectIRError, match="repeated"):
        execute_effect_program(prog, broken)
    np.testing.assert_array_equal(state["weights"], before)
    broken = dict(state)
    broken["weights"] = state["weights"].copy()
    broken["weights"][0, 0] = np.nan
    with pytest.raises(EffectIRError, match="finiteness"):
        execute_effect_program(prog, broken)


def test_budget_rejects_unbounded_loops_and_tensor_dimensions():
    prog, _ = _fixture()
    with pytest.raises(EffectIRError, match="Unbounded"):
        validate_effect_program(
            replace(prog, loops=(replace(prog.loops[0], count=10000),))
        )
    slots = list(prog.state)
    slots[0] = StateSlot("weights", TensorType((1_000_001, 7), "float32"))
    with pytest.raises(EffectIRError, match="dimensions"):
        validate_effect_program(replace(prog, state=tuple(slots)))


def test_read_after_write_uses_latest_cell_version():
    f = TensorType((2,), "float32")
    ops = (
        Instruction(
            "r0",
            "read",
            output="a",
            cell="value",
            token_in="e0",
            token_out="e1",
            revision=0,
        ),
        Instruction(
            "r1",
            "read",
            output="b",
            cell="value",
            token_in="e1",
            token_out="e2",
            revision=0,
        ),
        Instruction("sum", "add", args=("a", "b"), output="c"),
        Instruction(
            "w",
            "write",
            args=("c",),
            cell="value",
            token_in="e2",
            token_out="e3",
            revision=0,
        ),
        Instruction(
            "r2",
            "read",
            output="d",
            cell="value",
            token_in="e3",
            token_out="e4",
            revision=1,
        ),
        Instruction("sum2", "add", args=("d", "a"), output="e"),
        Instruction(
            "w2",
            "write",
            args=("e",),
            cell="value",
            token_in="e4",
            token_out="e5",
            revision=1,
        ),
    )
    p = EffectProgram((StateSlot("value", f),), (), (Repeat("update", 3, ops),))
    result = execute_effect_program(p, {"value": np.array([1, 2], dtype=np.float32)})
    # Every iteration transforms x -> 2x + x == 3x
    np.testing.assert_array_equal(
        result.final_state["value"], np.array([27, 54], dtype=np.float32)
    )
    assert [r.effect_count for r in result.receipts] == [5, 5, 5]


def test_multiple_areas_and_loop_scoped_names():
    f = TensorType((2,), "float32")
    ops = (
        Instruction(
            "read_a",
            "read",
            output="a",
            cell="area_a",
            token_in="e0",
            token_out="e1",
            revision=0,
        ),
        Instruction(
            "read_b",
            "read",
            output="b",
            cell="area_b",
            token_in="e1",
            token_out="e2",
            revision=0,
        ),
        Instruction("sum", "add", args=("a", "b"), output="s"),
        Instruction(
            "set_a",
            "write",
            args=("s",),
            cell="area_a",
            token_in="e2",
            token_out="e3",
            revision=0,
        ),
        Instruction(
            "set_b",
            "write",
            args=("a",),
            cell="area_b",
            token_in="e3",
            token_out="e4",
            revision=0,
        ),
    )
    p = EffectProgram(
        (StateSlot("area_a", f), StateSlot("area_b", f)),
        (),
        (Repeat("coupled", 4, ops),),
    )
    got = execute_effect_program(
        p,
        {
            "area_a": np.array([1, 2], np.float32),
            "area_b": np.array([0, 1], np.float32),
        },
    )
    np.testing.assert_array_equal(
        got.final_state["area_a"], np.array([5, 13], dtype=np.float32)
    )
    np.testing.assert_array_equal(
        got.final_state["area_b"], np.array([3, 8], dtype=np.float32)
    )


def test_effect_rewrite_can_change_pure_ops_but_requires_witness():
    prog, _ = _fixture()
    # This pass is structurally allowed; it is NOT a proof of numeric semantics.
    # A later witness runner must reject a wrong pure computation.
    ops = list(prog.loops[0].instructions)
    ops[4] = replace(ops[4], op="add")
    changed = replace(prog, loops=(replace(prog.loops[0], instructions=tuple(ops)),))
    assert_effect_trace_preserved(prog, changed)
    assert prog.digest() != changed.digest()


def test_integer_boolean_parameter_abuse_rejected():
    prog, _ = _fixture()
    ops = list(prog.loops[0].instructions)
    ops[5] = replace(ops[5], params=(("k", True),))
    with pytest.raises(EffectIRError, match="Parameters"):
        validate_effect_program(
            replace(prog, loops=(replace(prog.loops[0], instructions=tuple(ops)),))
        )


def test_duplicate_state_slots_rejected():
    prog, _ = _fixture()
    with pytest.raises(EffectIRError, match="duplicated"):
        validate_effect_program(replace(prog, state=(*prog.state, prog.state[0])))


def test_effect_rewrite_verified_with_inserted_identity():
    from visuaml.compiler_lab.effect_verification import verify_effect_rewrite

    program, state = _fixture()
    ops = list(program.loops[0].instructions)
    ops.insert(4, Instruction("copy_raw", "identity", args=("raw",), output="raw_copy"))
    # Old operator was op at position 4 (subtract), make it reference copy.
    ops[5] = replace(ops[5], args=("raw_copy", "b0"))
    changed = replace(
        program, loops=(replace(program.loops[0], instructions=tuple(ops)),)
    )
    receipt = verify_effect_rewrite(
        program, changed, [(state, {}), (dict(state), {})], atol=0, rtol=0
    )
    assert receipt.passed and receipt.witnesses_checked == 2
    assert receipt.max_abs_error == 0.0
    assert receipt.before_sha256 != receipt.after_sha256


def test_effect_rewrite_rejects_behavioral_regression_even_if_effects_match():
    from visuaml.compiler_lab.effect_verification import verify_effect_rewrite

    program, state = _fixture()
    ops = list(program.loops[0].instructions)
    ops[4] = replace(ops[4], op="add")
    changed = replace(
        program, loops=(replace(program.loops[0], instructions=tuple(ops)),)
    )
    state = {**state, "bias": np.arange(7, dtype=np.float32) * 0.5}
    with pytest.raises(EffectIRError, match="Rewrite changed cell"):
        verify_effect_rewrite(program, changed, [(state, {})])


def test_two_area_neural_demo_compiles_and_executes():
    from examples.effect_program_v7_demo import build_two_area_program, demo

    program = build_two_area_program()
    report = demo()
    assert report["schema"] == "visuaml.effect-demo.v7"
    assert report["admitted_operations"] == 3 * len(program.loops[0].instructions)
    assert len(report["receipts"]) == 3
    assert len(report["a_winners"]) == len(report["b_winners"]) == 2
    assert report["initial_state_sha256"] != report["final_state_sha256"]


def test_rejects_computation_that_is_small_in_nodes_but_huge_in_flops():
    t = TensorType((1000, 1000), "float32")
    p = EffectProgram(
        (StateSlot("w", t),),
        (),
        (
            Repeat(
                "danger",
                1,
                (
                    Instruction(
                        "read0",
                        "read",
                        output="x",
                        cell="w",
                        token_in="e0",
                        token_out="e1",
                        revision=0,
                    ),
                    Instruction("multiply", "matmul", args=("x", "x"), output="y"),
                    Instruction(
                        "write",
                        "write",
                        args=("y",),
                        cell="w",
                        token_in="e1",
                        token_out="e2",
                        revision=0,
                    ),
                ),
            ),
        ),
    )
    with pytest.raises(EffectIRError, match="compute budget"):
        validate_effect_program(p)
