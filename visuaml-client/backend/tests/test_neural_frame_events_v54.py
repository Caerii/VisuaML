"""Exhaustive bounded truth table and ablations for v54 fixed event gates."""
from __future__ import annotations

from dataclasses import replace
from itertools import product

import numpy as np
import pytest
from visuaml.compiler_lab.neural_frame_events_v54 import (
    ACTIONS,
    OPCODES,
    EventCircuitError,
    decode_opcode_token,
    fixed_event_weights,
    select_frame_action,
)


def bit(value):
    return np.array([float(value)], dtype=np.float64)


@pytest.mark.parametrize("opcode", OPCODES)
def test_exhaustive_opcode_and_four_guard_truth_table(opcode):
    weights = fixed_event_weights()
    for sequential, has_else, active, taken in product((0, 1), repeat=4):
        event = select_frame_action(
            decode_opcode_token(opcode),
            conditional_sequential=bit(sequential),
            has_else=bit(has_else),
            frame_active=bit(active),
            branch_taken=bit(taken),
            weights=weights,
        )
        expected = {
            "block": "push",
            "loop": "push",
            "if": "push" if sequential or has_else else "retain",
            "else": "pop",
            "end": "pop" if active else "retain",
            "br": "branch",
            "br_if": "branch" if taken else "retain",
            "return": "return",
            "other": "retain",
        }[opcode]
        assert np.argmax(event) == ACTIONS.index(expected)
        assert event.sum() == 1.


def test_non_control_opcode_maps_to_retain():
    assert np.argmax(select_frame_action(
        decode_opcode_token("i32.add"),
        conditional_sequential=bit(0),
        has_else=bit(0),
        frame_active=bit(1),
        branch_taken=bit(1),
    )) == 0


def test_corrupted_weights_are_rejected_without_host_fallback():
    fixed = fixed_event_weights()
    broken = fixed.default.copy()
    broken[0, :] = 0.
    with pytest.raises(EventCircuitError):
        select_frame_action(
            decode_opcode_token("block"),
            conditional_sequential=bit(1),
            has_else=bit(0),
            frame_active=bit(0),
            branch_taken=bit(0),
            weights=replace(fixed, default=broken),
        )
    broken_gains = fixed.conditional_gains.copy()
    broken_gains[0] = 0.
    changed = select_frame_action(
        decode_opcode_token("if"),
        conditional_sequential=bit(1),
        has_else=bit(0),
        frame_active=bit(0),
        branch_taken=bit(0),
        weights=replace(fixed, conditional_gains=broken_gains),
    )
    assert np.argmax(changed) == ACTIONS.index("retain")
    assert np.argmax(select_frame_action(
        decode_opcode_token("if"), conditional_sequential=bit(1),
        has_else=bit(0), frame_active=bit(0), branch_taken=bit(0),
    )) == ACTIONS.index("push")


def test_malformed_nonbinary_tokens_fail_closed():
    invalid = decode_opcode_token("br_if")
    invalid[0] = 1.
    with pytest.raises(EventCircuitError):
        select_frame_action(
            invalid, conditional_sequential=bit(0),
            has_else=bit(0), frame_active=bit(0), branch_taken=bit(1),
        )
    with pytest.raises(EventCircuitError):
        select_frame_action(
            decode_opcode_token("br_if"),
            conditional_sequential=bit(0),
            has_else=np.array([0.5]),
            frame_active=bit(0), branch_taken=bit(1),
        )
