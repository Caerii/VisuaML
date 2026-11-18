import shutil
from dataclasses import replace

import numpy as np
import pytest

from visuaml.compiler_lab.bit_neural_vm_v49 import _bits, construct_bit_weights
from visuaml.compiler_lab.compact_routing_v58 import CompactRouter
from visuaml.compiler_lab.exact_binary import is_exact_binary
from visuaml.compiler_lab.neural_control_routing_v52 import (
    TERMINAL, NeuralControlError, _onehot, compile_route_tokens, construct_router_weights,
    neural_pc_step,
)
from visuaml.compiler_lab.neural_microstep_executor_v58 import execute_microstep_control
from visuaml.compiler_lab.wasm_research_corpus_v57 import corpus, countdown, generated

requires_compiler = pytest.mark.skipif(shutil.which("c++") is None, reason="requires C++17")


def test_binary_count_keeps_exact_admission():
    for value in (
        np.ones(128),
        np.ones(129),
        np.concatenate((np.ones(128), [0.5])),
        np.array([-0., 1., 0.]),
        np.array([0., -0., 1., 1.])[::-2],
        np.array([np.nan, 0.]),
        np.array([np.inf, 1.]),
        np.array([np.nextafter(1., 2.), 1.]),
        np.empty(0),
    ):
        assert is_exact_binary(value) == bool(((value == 0.) | (value == 1.)).all())


@requires_compiler
def test_compact_pc_matches_dense_gates_on_every_admitted_row():
    fixed = construct_router_weights()
    bit = construct_bit_weights()
    for program, _ in corpus().values():
        tokens = compile_route_tokens(program)
        candidate = CompactRouter(program, tokens, fixed, bit)
        for index in (*range(len(program.operations)), TERMINAL):
            pc = _onehot(index)
            for argument in (0, 1, -(1 << 31)):
                condition = _bits(argument)
                expected, details = neural_pc_step(pc, condition, tokens, fixed, bit)
                actual, trace = candidate.step(pc, condition)
                np.testing.assert_array_equal(actual, expected)
                assert details == trace


@requires_compiler
def test_compact_pc_uses_live_gains_and_rejects_corruption():
    program = countdown()
    tokens = compile_route_tokens(program)
    fixed = construct_router_weights()
    bit = construct_bit_weights()
    branch_pc = int(np.flatnonzero(tokens.guard_kind[:, 1])[0])
    pc = _onehot(branch_pc)
    condition = _bits(1)
    original, _ = CompactRouter(program, tokens, fixed, bit).step(pc, condition)
    guard = fixed.guard_weights.copy()
    guard[1] = 0.
    changed = replace(fixed, guard_weights=guard)
    actual, trace = CompactRouter(program, tokens, changed, bit).step(pc, condition)
    expected, details = neural_pc_step(pc, condition, tokens, changed, bit)
    np.testing.assert_array_equal(actual, expected)
    assert details == trace
    assert not np.array_equal(original, actual)

    zero_result = replace(fixed, path_weights=np.array([1., 0.]))
    with pytest.raises(NeuralControlError, match="lost unique"):
        CompactRouter(program, tokens, zero_result, bit).step(pc, condition)
    invalid_target = tokens.alternate.copy()
    invalid_target[branch_pc] = np.roll(invalid_target[branch_pc], 1)
    with pytest.raises(NeuralControlError, match="Altered routing targets"):
        CompactRouter(program, replace(tokens, alternate=invalid_target), fixed, bit)
    router = CompactRouter(program, tokens, fixed, bit)
    with pytest.raises(NeuralControlError, match="Invalid binary condition"):
        router.step(pc, np.full(32, 0.5))
    with pytest.raises(NeuralControlError, match="noncanonical active PC"):
        router.step(pc + _onehot(TERMINAL), condition)

    # The zero-test must also use the supplied v49 weights.
    mutated_bit = replace(bit, and_kernel=np.zeros(3))
    normal, _ = router.step(pc, _bits(0))
    actual, trace = CompactRouter(program, tokens, fixed, mutated_bit).step(pc, _bits(0))
    expected, details = neural_pc_step(pc, _bits(0), tokens, fixed, mutated_bit)
    np.testing.assert_array_equal(actual, expected)
    assert trace == details
    assert not np.array_equal(actual, normal)


@requires_compiler
@pytest.mark.parametrize("capture_trace", (True, False))
def test_compact_v58_retains_trace_and_fail_closed_behavior(capture_trace):
    program = generated(2)
    args = {"capture_trace": capture_trace, "use_fused_state": True, "use_compiled_alu": True}
    expected = execute_microstep_control(program, -17, **args)
    assert execute_microstep_control(program, -17, use_compact_routing=True, **args) == expected

    with pytest.raises(NeuralControlError, match="lost unique"):
        execute_microstep_control(
            program, -17, use_compact_routing=True,
            pc_fixed=replace(construct_router_weights(), path_weights=np.array([0., 1.])),
            **args,
        )
