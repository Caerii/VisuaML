"""v52: gate-selected program counters tested against independent v51 + Wasmtime.

The v52 PC itself determines the next executable instruction; v51 is an
independent host-PC oracle consulted ONLY for differential verification.
Source lexical target tables are data; constructed routing coefficients are
one shared configuration across every admitted source program.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from scripts.probe_wasm_gcd_v51_1 import gcd_program
from visuaml.compiler_lab.neural_control_routing_v52 import (
    NeuralControlError,
    compile_route_tokens,
    construct_router_weights,
    execute_routed_control,
    neural_pc_step,
    validate_route_tokens,
)
from visuaml.compiler_lab.wasm_control_flow_v51 import (
    ControlFuelExhausted,
    ControlOperation,
    ControlProgram,
    encode_control,
    execute_control,
)


def op(name, immediate=None):
    return ControlOperation(name, immediate)


def countdown():
    return ControlProgram(1, (
        op("block"), op("loop"), op("local.get", 0),
        op("i32.eqz"), op("br_if", 1),
        op("local.get", 1), op("local.get", 0), op("i32.add"),
        op("local.set", 1), op("local.get", 0),
        op("i32.const", 1), op("i32.sub"),
        op("local.set", 0), op("br", 0),
        op("end"), op("end"), op("local.get", 1), op("end"),
    ))


def ifelse():
    return ControlProgram(1, (
        op("local.get", 0), op("i32.const", 0),
        op("i32.lt_s"), op("if"),
        op("i32.const", 11), op("local.set", 1),
        op("else"), op("i32.const", 22),
        op("local.set", 1), op("end"),
        op("local.get", 1), op("end"),
    ))


def conditional_no_else():
    return ControlProgram(1, (
        op("local.get", 0), op("i32.eqz"),
        op("if"), op("i32.const", 7),
        op("local.set", 1), op("end"),
        op("local.get", 1), op("end"),
    ))


def branch_block():
    return ControlProgram(1, (
        op("block"), op("local.get", 0),
        op("i32.const", 0), op("i32.gt_s"),
        op("br_if", 0),
        op("i32.const", 7), op("local.set", 1),
        op("end"), op("local.get", 1), op("end"),
    ))


def return_early():
    return ControlProgram(0, (
        op("local.get", 0), op("i32.const", 0),
        op("i32.gt_s"), op("if"),
        op("i32.const", 9), op("return"), op("end"),
        op("i32.const", -3), op("end"),
    ))


def infinite():
    return ControlProgram(0, (
        op("loop"), op("br", 0), op("end"),
        op("i32.const", 1), op("end"),
    ))


def programs():
    return {
        "countdown": (countdown(), tuple(range(10))),
        "ifelse": (ifelse(), (-7, -1, 0, 1, 7)),
        "no_else": (conditional_no_else(), (-3, -1, 0, 1, 3)),
        "br_block": (branch_block(), (-3, -1, 0, 1, 3)),
        "early_return": (return_early(), (-7, -1, 0, 1, 7)),
        "gcd_12": (gcd_program(), (-1, 0, 1, 2, 3, 5, 7, 9, 12, 17, 24)),
    }


def test_program_tables_are_different_but_fixed_routing_coefficients_match():
    fixed = construct_router_weights()
    assert fixed.digest() == construct_router_weights().digest()
    tokens = {name: compile_route_tokens(p) for name, (p, _) in programs().items()}
    assert len({t.digest() for t in tokens.values()}) == len(tokens)
    for name, (program, _) in programs().items():
        token = tokens[name]
        validate_route_tokens(token, program)
        assert token.source_program_sha256 == program.digest()
        assert token.source_wasm_sha256 != fixed.digest()
        assert int(np.sum(token.guard_kind[:, 0])) >= 0
    assert fixed.guard_weights.tolist() == [1., 1., 1.]
    assert fixed.path_weights.tolist() == [1., 1.]


@pytest.mark.parametrize("name", list(programs()))
def test_all_state_transitions_match_original_v51_host_control(name):
    p, samples = programs()[name]
    tokens = compile_route_tokens(p)
    fixed = construct_router_weights()
    for x in samples:
        original = execute_control(p, x, fuel=1500)
        actual = execute_routed_control(p, x, fuel=1500, tokens=tokens, fixed=fixed)
        assert actual.result_i32 == original.value, (name, x)
        assert actual.steps == original.transitions
        assert actual.fixed_router_weights_sha256 == fixed.digest()
        assert actual.program_route_tokens_sha256 == tokens.digest()
        for new, old in zip(actual.trace, original.trace, strict=True):
            for key in (
                "step", "pc", "opcode", "next_pc",
                "stack_i32", "locals_i32", "control_labels", "halted",
            ):
                assert new[key] == old[key], (name, x, key, new, old)
            assert new["neural_route"]["next_pc"] == (
                96 if new["halted"] else old["next_pc"]
            ) or (
                new["neural_route"]["next_pc"] == 96
                and old["next_pc"] == len(p.operations)
            )


def test_mutating_gate_weights_changes_route_not_only_its_fingerprint():
    program = branch_block()
    tokens = compile_route_tokens(program)
    fixed = construct_router_weights()
    before = execute_routed_control(program, 4, fuel=100, fixed=fixed)
    assert before.result_i32 == 0
    corrupted = replace(fixed, path_weights=np.array([1., 0.]))
    assert corrupted.digest() != fixed.digest()
    with pytest.raises(NeuralControlError, match="unique next PC"):
        execute_routed_control(program, 4, fuel=100, fixed=corrupted)
    # A corrupted signed-branch guard must disagree with host-PC oracle.
    changed = replace(fixed, guard_weights=np.array([1., 0., 1.]))
    assert changed.digest() != fixed.digest()
    wrong = execute_routed_control(program, 4, fuel=100, fixed=changed)
    assert wrong.result_i32 != before.result_i32


def test_modified_program_route_tokens_fail_closed():
    program = ifelse()
    raw = compile_route_tokens(program)
    alternative = raw.alternate.copy()
    alternative[3, :] = 0.
    alternative[3, 0] = 1.
    with pytest.raises(NeuralControlError, match="Altered routing"):
        validate_route_tokens(replace(raw, alternate=alternative), program)

    guard = raw.guard_kind.copy()
    guard[3] = (1., 1., 0.)
    with pytest.raises(NeuralControlError, match="Ambiguous"):
        validate_route_tokens(replace(raw, guard_kind=guard), program)

    broken = raw.sequential.copy()
    broken[0, :] = 0.
    with pytest.raises(NeuralControlError, match="Incomplete"):
        validate_route_tokens(replace(raw, sequential=broken), program)

    with pytest.raises(NeuralControlError, match="identity"):
        validate_route_tokens(
            replace(raw, source_program_sha256="0"*64), program
        )


def test_pc_gate_uses_actual_binary_condition_not_host_boolean_dispatch():
    program = branch_block()
    tokens = compile_route_tokens(program)
    weights = construct_router_weights()
    from visuaml.compiler_lab.bit_neural_vm_v49 import _bits, construct_bit_weights

    def pc(index):
        v = np.zeros(97, dtype=np.float64)
        v[index] = 1.
        return v

    # br_if is at pc4: if condition=0, fall through; otherwise skip block.
    zero, detail0 = neural_pc_step(
        pc(4), _bits(0), tokens, weights, construct_bit_weights()
    )
    full, detail1 = neural_pc_step(
        pc(4), _bits(-(1 << 31)), tokens, weights, construct_bit_weights()
    )
    assert int(np.argmax(zero)) == 5
    assert int(np.argmax(full)) == 8
    assert detail0["branch_guard_gate"] == 0
    assert detail1["branch_guard_gate"] == 1
    assert detail1["nonzero_condition_gate"] == 1


def test_infinite_loop_and_terminating_large_gcd_are_fuel_distinguished():
    for p, x in ((infinite(), 0), (gcd_program(), (1 << 31)-1)):
        with pytest.raises(ControlFuelExhausted, match="fuel exhausted"):
            execute_routed_control(p, x, fuel=70)
    with pytest.raises(ControlAdmissionError, match="fuel"):
        execute_routed_control(infinite(), 0, fuel=10_000)


def test_independent_wasmtime_executes_same_source_as_gate_routed_pc():
    wasmtime = pytest.importorskip("wasmtime")
    engine = wasmtime.Engine()
    for name, (program, samples) in programs().items():
        wasm = encode_control(program)
        wasmtime.Module.validate(engine, wasm)
        store = wasmtime.Store(engine)
        run = wasmtime.Instance(
            store, wasmtime.Module(engine, wasm), []
        ).exports(store)["run"]
        for x in samples:
            actual = execute_routed_control(program, x, fuel=1500)
            assert run(store, x) == actual.result_i32, (name, x)
