"""Focused v53 finite frame-circuit prototype tests.

These do not claim integration into the v52 WASM executor.
"""
import numpy as np
import pytest
from visuaml.compiler_lab.neural_frame_circuit_v53 import (
    ACTIONS,
    HEIGHTS,
    KINDS,
    MAX_NESTING,
    FrameCircuitError,
    FrameCircuitWeights,
    FrameTokens,
    empty_frames,
    fixed_frame_weights,
    onehot,
    step_frames,
    validate_state,
)


def tokens(action, *, kind=None, base=None, depth=None, height=0):
    def maybe(value, size):
        return (np.zeros(size, dtype=np.float64)
                if value is None else onehot(value, size))
    return FrameTokens(
        onehot(action, ACTIONS), maybe(kind, KINDS),
        maybe(base, HEIGHTS), maybe(depth, MAX_NESTING),
        onehot(height, HEIGHTS),
    )


def depth(state):
    return int(np.argmax(state.depth))


def test_enter_close_and_return():
    state = empty_frames()
    state, height = step_frames(
        state, tokens(1, kind=1, base=3, height=3)
    )
    assert depth(state) == 1
    assert state.kinds[0, 1] == 1
    assert state.bases[0, 3] == 1
    assert height[3] == 1
    state, _ = step_frames(state, tokens(1, kind=0, base=5, height=5))
    assert depth(state) == 2
    state, _ = step_frames(state, tokens(2, height=5))
    assert depth(state) == 1
    assert not state.kinds[1].any()
    state, _ = step_frames(state, tokens(4, height=1))
    assert depth(state) == 0
    assert not state.bases.any()
    validate_state(state)


def test_loop_branch_keeps_target_but_exits_nested_block():
    state = empty_frames()
    for kind, base in [(0, 2), (1, 4), (2, 6)]:
        state, _ = step_frames(
            state, tokens(1, kind=kind, base=base, height=base)
        )
    # Branch depth=1 selects loop, retains loop and outer block.
    after, restored = step_frames(
        state, tokens(3, depth=1, height=9)
    )
    assert depth(after) == 2
    assert restored[4] == 1
    assert after.kinds[1, 1] == 1
    assert not after.kinds[2].any()
    # Branch depth=1 selects outer block; exit removes it as well.
    after2, restored2 = step_frames(
        after, tokens(3, depth=1, height=9)
    )
    assert depth(after2) == 0
    assert restored2[2] == 1


def test_retain_is_identical_and_input_height_is_preserved():
    state, _ = step_frames(
        empty_frames(), tokens(1, kind=2, base=1)
    )
    retain = tokens(0, height=7)
    after, height = step_frames(state, retain)
    assert np.array_equal(after.depth, state.depth)
    assert np.array_equal(after.kinds, state.kinds)
    assert np.array_equal(after.bases, state.bases)
    assert height[7] == 1
    for result, source in ((after.depth, state.depth), (after.kinds, state.kinds),
                           (after.bases, state.bases), (height, retain.operand_height)):
        assert not np.shares_memory(result, source)


@pytest.mark.parametrize("kind", [0, 1, 2])
def test_branch_kind_controls_keep_or_exit(kind):
    state, _ = step_frames(
        empty_frames(), tokens(1, kind=kind, base=2)
    )
    after, height = step_frames(
        state, tokens(3, depth=0, height=7)
    )
    assert depth(after) == (1 if kind == 1 else 0)
    assert height[2] == 1


def test_depth_capacity_and_invalid_branch_fail_closed():
    state = empty_frames()
    with pytest.raises(FrameCircuitError):
        step_frames(state, tokens(2))
    with pytest.raises(FrameCircuitError):
        step_frames(state, tokens(3, depth=0))
    state, _ = step_frames(state, tokens(1, kind=1, base=0))
    with pytest.raises(FrameCircuitError):
        step_frames(state, tokens(3, depth=1))
    for _ in range(MAX_NESTING - 1):
        state, _ = step_frames(
            state, tokens(1, kind=1, base=0)
        )
    with pytest.raises(FrameCircuitError):
        step_frames(state, tokens(1, kind=1, base=0))
    after, _ = step_frames(state, tokens(3, depth=MAX_NESTING - 1))
    assert depth(after) == 1


def test_weight_ablation_is_not_silently_repaired():
    state = empty_frames()
    weights = fixed_frame_weights()
    ablated = FrameCircuitWeights(weights.action_weights.copy())
    ablated.action_weights[1] = 0.
    with pytest.raises(FrameCircuitError):
        step_frames(
            state, tokens(1, kind=0, base=0), ablated
        )

    ablated.action_weights[0] = 0.
    with pytest.raises(FrameCircuitError):
        step_frames(state, tokens(0, height=0), ablated)


def test_malformed_frame_metadata_and_noncanonical_action_rejected():
    state = empty_frames()
    state.kinds[0, 0] = 1.
    with pytest.raises(FrameCircuitError):
        validate_state(state)
    state = empty_frames()
    malformed = tokens(0)
    malformed.action[1] = 1.
    with pytest.raises(FrameCircuitError):
        step_frames(state, malformed)
