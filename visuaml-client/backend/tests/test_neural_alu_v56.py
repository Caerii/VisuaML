"""v56 analytically constructed shared neural ALU regression and falsification."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from visuaml.compiler_lab.bit_neural_vm_v49 import _bits, _int
from visuaml.compiler_lab.neural_alu_v56 import (
    ALU_OPS,
    NeuralALUError,
    construct_alu_weights,
    execute_alu,
    onehot_alu,
)

BOUNDARY = (-(1 << 31), -(1 << 31) + 1, -17, -1, 0, 1, 5, 17,
            (1 << 31) - 2, (1 << 31) - 1)


def signed(value):
    return ((value + (1 << 31)) % (1 << 32)) - (1 << 31)


def reference(opcode, a, b):
    if opcode == "i32.add":
        return signed(a + b)
    if opcode == "i32.sub":
        return signed(a - b)
    if opcode == "i32.mul":
        return signed(a * b)
    if opcode == "i32.lt_s":
        return int(a < b)
    if opcode == "i32.gt_s":
        return int(a > b)
    return int(a == 0)


@pytest.mark.parametrize("opcode", ALU_OPS)
def test_full_width_signed_edges_and_adversarial_values(opcode):
    weights = construct_alu_weights()
    for a in BOUNDARY:
        for b in BOUNDARY:
            actual = execute_alu(onehot_alu(opcode), _bits(a), _bits(b))
            assert _int(actual) == reference(opcode, a, b), (opcode, a, b)
            assert actual.dtype == np.float64
    assert weights.digest() == construct_alu_weights().digest()


def test_random_full_width_witnesses():
    rng = np.random.default_rng(56006)
    for _ in range(36):
        a, b = (int(x) for x in rng.integers(
            -(1 << 31), 1 << 31, size=2, dtype=np.int64
        ))
        for op in ALU_OPS:
            assert _int(execute_alu(
                onehot_alu(op), _bits(a), _bits(b)
            )) == reference(op, a, b)


def test_fixed_gate_ablation_changes_results_or_fails_closed():
    weights = construct_alu_weights()
    changed = weights.output_gains.copy()
    changed[0] = 0.
    with pytest.raises(NeuralALUError):
        execute_alu(
            onehot_alu("i32.add"), _bits(17), _bits(5),
            fixed=replace(weights, output_gains=changed),
        )
    assert weights.output_gains.tolist() == [1.] * len(ALU_OPS)


def test_malformed_and_mixed_opcode_tokens_rejected():
    malformed = onehot_alu("i32.mul")
    malformed[0] = 1.
    with pytest.raises(NeuralALUError):
        execute_alu(malformed, _bits(1), _bits(2))
    malformed = onehot_alu("i32.lt_s")
    malformed[2] = np.nan
    with pytest.raises(NeuralALUError):
        execute_alu(malformed, _bits(1), _bits(2))
    with pytest.raises(NeuralALUError):
        onehot_alu("i64.add")
