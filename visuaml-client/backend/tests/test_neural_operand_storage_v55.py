"""v55 bounded exact-i32 operand storage and fixed-weight transition tests."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from visuaml.compiler_lab.bit_neural_vm_v49 import _bits, _int
from visuaml.compiler_lab.neural_operand_storage_v55 import (
    ACTIONS,
    HEIGHTS,
    MAX_STACK,
    OperandCircuitError,
    empty_operands,
    fixed_operand_weights,
    onehot,
    step_operands,
    top_word,
    validate_operands,
)


def apply(state, action, *, value=0, truncate=None, fixed=None):
    return step_operands(
        state,
        onehot(ACTIONS.index(action), len(ACTIONS)),
        value=_bits(value),
        truncate_height=(
            None if truncate is None else onehot(truncate, HEIGHTS)
        ),
        fixed=fixed,
    )


def depth(state):
    return int(np.argmax(state.height))


def test_push_pop_and_truncate_exact_i32_words():
    state = empty_operands()
    for number in (0, 1, -1, -(1 << 31), (1 << 31)-1):
        state, _ = apply(state, "push", value=number)
        assert _int(top_word(state)) == number
    assert depth(state) == 5
    state, popped = apply(state, "pop")
    assert _int(popped) == (1 << 31)-1
    assert depth(state) == 4
    assert _int(top_word(state)) == -(1 << 31)
    state, _ = apply(state, "truncate", truncate=2)
    assert depth(state) == 2
    assert _int(top_word(state)) == 1
    assert not np.any(state.cells[2:])
    state, _ = apply(state, "retain")
    validate_operands(state)


def test_all_capacity_positions_round_trip_without_alias():
    state = empty_operands()
    reference = []
    for index in range(MAX_STACK):
        value = (index * 19088743) % (1 << 31)
        state, _ = apply(state, "push", value=value)
        reference.append(value)
        assert depth(state) == len(reference)
    with pytest.raises(OperandCircuitError):
        apply(state, "push", value=0)
    while reference:
        state, got = apply(state, "pop")
        assert _int(got) == reference.pop()
        assert depth(state) == len(reference)
    with pytest.raises(OperandCircuitError):
        apply(state, "pop")
    assert not np.any(state.cells)


def test_truncation_never_writes_or_resurrects_dead_slots():
    state = empty_operands()
    for index in range(8):
        state, _ = apply(state, "push", value=index + 7)
    state, _ = apply(state, "truncate", truncate=3)
    for index in range(3):
        assert _int(state.cells[index]) == index + 7
    state, _ = apply(state, "push", value=-77)
    assert _int(top_word(state)) == -77
    assert not np.any(state.cells[4:])
    with pytest.raises(OperandCircuitError):
        apply(state, "truncate", truncate=5)


def test_noncanonical_action_and_corrupted_weights_fail_closed():
    state = empty_operands()
    invalid = onehot(1, len(ACTIONS))
    invalid[0] = 1.
    with pytest.raises(OperandCircuitError):
        step_operands(state, invalid, value=_bits(3))
    ablated = fixed_operand_weights()
    fixed = replace(
        ablated, action_gain=np.array([1., 0., 1., 1.])
    )
    with pytest.raises(OperandCircuitError):
        apply(state, "push", value=9, fixed=fixed)


def test_inactive_cells_corruption_rejected():
    state = empty_operands()
    state.cells[17, 0] = 1.
    with pytest.raises(OperandCircuitError):
        validate_operands(state)


def test_retain_produces_exact_same_tensor_and_no_pop_result():
    state, _ = apply(empty_operands(), "push", value=-123)
    after, popped = apply(state, "retain")
    assert np.array_equal(after.height, state.height)
    assert np.array_equal(after.cells, state.cells)
    assert not popped.any()
