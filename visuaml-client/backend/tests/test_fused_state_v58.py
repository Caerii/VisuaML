from dataclasses import replace

import numpy as np
import pytest

from visuaml.compiler_lab.fused_state_v58 import local_access, pop_pair, push_one
from visuaml.compiler_lab.neural_local_bank_v57 import construct_local_bank_weights, read_local, write_local
from visuaml.compiler_lab.neural_microstep_executor_v58 import execute_microstep_control
from visuaml.compiler_lab.neural_operand_storage_v55 import (
    HEIGHTS, MAX_STACK, WIDTH, OperandCircuitError, OperandState, fixed_operand_weights, onehot,
    step_operands, top_word,
)
from visuaml.compiler_lab.typed_microop_ir_v57 import LOCALS_WIDTH, MicroopAdmissionError
from visuaml.compiler_lab.wasm_research_corpus_v57 import generated


def _state(depth):
    cells = np.zeros((MAX_STACK, WIDTH))
    cells[:depth] = np.random.default_rng(depth).integers(0, 2, size=(depth, WIDTH))
    return OperandState(onehot(depth, HEIGHTS), cells)


def test_two_pop_one_push_contractions_match_every_stack_depth():
    fixed = fixed_operand_weights()
    value = np.ones(WIDTH)
    for depth in range(MAX_STACK + 1):
        state = _state(depth)
        for first, second in ((0, 0), (0, 1), (1, 0), (1, 1)):
            if depth < first + second:
                with pytest.raises(OperandCircuitError, match="underflow"):
                    pop_pair(state, first, second, fixed)
                continue
            intermediate, right = step_operands(state, onehot(2 * first, 4), fixed=fixed)
            expected, left = step_operands(intermediate, onehot(2 * second, 4), fixed=fixed)
            actual, top, popped_right, popped_left = pop_pair(state, first, second, fixed)
            np.testing.assert_array_equal(top, top_word(state))
            np.testing.assert_array_equal(popped_right, right)
            np.testing.assert_array_equal(popped_left, left)
            for push in (0, 1):
                if push and depth == MAX_STACK and first + second == 0:
                    with pytest.raises(OperandCircuitError, match="overflow"):
                        push_one(actual, push, value, fixed)
                    continue
                a = push_one(actual, push, value, fixed)
                b, _ = step_operands(expected, onehot(push, 4), value=value, fixed=fixed)
                np.testing.assert_array_equal(a.height, b.height)
                np.testing.assert_array_equal(a.cells, b.cells)


def test_local_contraction_matches_read_and_write_and_rejects_gain_ablations():
    bank = np.random.default_rng(58).integers(0, 2, (LOCALS_WIDTH, WIDTH)).astype(np.float64)
    value = np.ones(WIDTH)
    fixed = construct_local_bank_weights()
    for slot in range(LOCALS_WIDTH):
        selector = onehot(slot, LOCALS_WIDTH)
        for write in (0, 1):
            actual_read, actual_bank = local_access(bank, selector, value, write, fixed)
            np.testing.assert_array_equal(actual_read, read_local(bank, selector, fixed=fixed))
            expected = write_local(bank, selector, value, fixed=fixed) if write else bank
            np.testing.assert_array_equal(actual_bank, expected)
        gain = 1. - selector
        for field in ("read_gain", "write_gain"):
            with pytest.raises(MicroopAdmissionError, match="ablated"):
                local_access(bank, selector, value, 0, replace(fixed, **{field: gain}))


@pytest.mark.parametrize("capture_trace", (True, False))
def test_fused_executor_keeps_live_operand_and_local_authority(capture_trace):
    program = generated(2)
    expected = execute_microstep_control(program, -17, capture_trace=capture_trace)
    assert execute_microstep_control(
        program, -17, capture_trace=capture_trace, use_fused_state=True,
    ) == expected
    for index in range(3):
        gain = np.ones(4)
        gain[index] = 0.
        with pytest.raises(OperandCircuitError, match="one-hot"):
            execute_microstep_control(
                program, -17, capture_trace=capture_trace, use_fused_state=True,
                operand_fixed=replace(fixed_operand_weights(), action_gain=gain),
            )
