"""v49 exact modular i32 neural gate and original Wasmtime differential tests.

All instructions and original raw WASM use existing v40 typed source semantics.
Per-cell combinational gates get exhaustive Boolean truth tables, while full
32-bit programs get adversarial boundary/random witnesses, not 2**32 exhaust.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from visuaml.compiler_lab.bit_neural_vm_v49 import (
    BitVMAdmissionError,
    _add,
    _and,
    _bits,
    _eqz,
    _int,
    _mul,
    _or,
    _signed_lt,
    _sub,
    _xor,
    bit_step,
    construct_bit_weights,
    decode_state,
    decode_tokens,
    initial_state,
    run_bit_vm,
    tokenize,
)
from visuaml.compiler_lab.wasm_state_vm_v40 import (
    I32Program,
    VMOperation,
    encode_program,
    execute_program,
)
from visuaml.compiler_lab.wasm_state_vm_v40 import initial_state as original_state
from visuaml.compiler_lab.wasm_state_vm_v40 import step as original_step

EDGE_VALUES = (-(1 << 31), -(1 << 31)+1, -65536, -129, -128, -1, 0, 1,
               2, 127, 128, 65535, (1 << 31)-2, (1 << 31)-1)


def _op(name, value=None):
    return VMOperation(name, value)


def programs():
    return (
        ("identity", I32Program(0, (_op("local.get", 0), _op("end")))),
        ("overflow-add", I32Program(0, (
            _op("local.get", 0), _op("i32.const", 1),
            _op("i32.add"), _op("end"),
        ))),
        ("borrow-sub", I32Program(0, (
            _op("local.get", 0), _op("i32.const", -1),
            _op("i32.sub"), _op("end"),
        ))),
        ("multiply", I32Program(0, (
            _op("local.get", 0), _op("local.get", 0),
            _op("i32.mul"), _op("end"),
        ))),
        ("mul-max", I32Program(0, (
            _op("local.get", 0), _op("i32.const", (1 << 31)-1),
            _op("i32.mul"), _op("end"),
        ))),
        ("large-constant", I32Program(0, (
            _op("i32.const", -(1 << 31)), _op("local.get", 0),
            _op("i32.add"), _op("end"),
        ))),
        ("signed-lt", I32Program(0, (
            _op("local.get", 0), _op("i32.const", 0),
            _op("i32.lt_s"), _op("end"),
        ))),
        ("signed-gt", I32Program(0, (
            _op("local.get", 0), _op("i32.const", -2),
            _op("i32.gt_s"), _op("end"),
        ))),
        ("zero", I32Program(0, (
            _op("local.get", 0), _op("i32.eqz"), _op("end"),
        ))),
        ("local-write", I32Program(2, (
            _op("local.get", 0), _op("local.set", 1),
            _op("local.get", 1), _op("i32.const", 2),
            _op("i32.mul"), _op("local.tee", 2),
            _op("local.get", 2), _op("i32.add"), _op("end"),
        ))),
        ("drop", I32Program(0, (
            _op("local.get", 0), _op("i32.const", 19),
            _op("drop"), _op("end"),
        ))),
    )


def test_bit_gates_truth_table_exact_not_floating_approximations():
    for x in (0., 1.):
        for y in (0., 1.):
            assert _and(x, y) == (x and y)
            assert _or(x, y) == (x or y)
            assert _xor(x, y) == int(x != y)
    assert _eqz(_bits(0)) == 1.
    assert _eqz(_bits(-1)) == 0.
    for x in EDGE_VALUES:
        assert _int(_bits(x)) == x


def test_fixed_relu_arithmetic_covers_i32_wrap_and_signed_comparison():
    fixed = construct_bit_weights()
    pairs = (
        (-(1 << 31), -(1 << 31)),
        ((1 << 31)-1, (1 << 31)-1),
        ((1 << 31)-1, 1),
        (-(1 << 31), -1),
        (-1, -1), (-1, 2), (1, -1),
        (123456789, -987654321),
        (0, (1 << 31)-1),
    )
    for a, b in pairs:
        aa, bb = _bits(a), _bits(b)
        def wrap(v):
            u = v & 0xFFFFFFFF
            return u - (1 << 32) if u >= (1 << 31) else u
        assert _int(_add(aa, bb)) == wrap(a+b)
        assert _int(_sub(aa, bb)) == wrap(a-b)
        assert _int(_mul(aa, bb, fixed.carry_thresholds)) == wrap(a*b)
        assert _signed_lt(aa, bb) == int(a < b)
        assert _signed_lt(bb, aa) == int(b < a)
    assert len(fixed.digest()) == 64
    assert fixed.digest() == construct_bit_weights().digest()


def test_one_shared_weight_digest_across_distinct_real_wasm_i32_programs():
    fixed = construct_bit_weights()
    tokens = [tokenize(p) for _, p in programs()]
    assert len({t.digest() for t in tokens}) == len(tokens)
    for token, (_, original) in zip(tokens, programs(), strict=True):
        assert decode_tokens(token) == original
        assert token.original_wasm_sha256 != fixed.digest()


@pytest.mark.parametrize("x", EDGE_VALUES)
def test_per_instruction_state_matches_original_typed_machine(x):
    fixed = construct_bit_weights()
    for name, program in programs():
        token = tokenize(program)
        state = initial_state(program, x)
        reference = original_state(program, x)
        for operation in program.operations:
            state = bit_step(state, token, fixed)
            reference, transition = original_step(program, reference)
            assert decode_state(state, program) == reference, (
                name, x, operation.opcode, transition.pc
            )
        assert reference.stack == (execute_program(program, x).value,)


def test_external_independent_wasmtime_all_programs_and_adversarial_i32():
    wasmtime = pytest.importorskip("wasmtime")
    fixed = construct_bit_weights()
    engine = wasmtime.Engine()
    for name, program in programs():
        wasm = encode_program(program)
        wasmtime.Module.validate(engine, wasm)
        store = wasmtime.Store(engine)
        function = wasmtime.Instance(
            store, wasmtime.Module(engine, wasm), []
        ).exports(store)["run"]
        for x in EDGE_VALUES:
            observed = run_bit_vm(program, x, fixed=fixed)
            expected = function(store, x)
            assert observed["result"] == expected, (name, x)
            assert observed["result"] == execute_program(program, x).value
            assert observed["fixed_bit_weights_sha256"] == fixed.digest()
            assert observed["all_signed_i32_inputs_admitted"]
            assert not observed["full_domain_exhaustively_executed"]
            assert not observed["full_wasm_supported"]
            assert not observed["standard_softmax_transformer_executed"]


def test_fixed_bit_tokens_reject_tampering_and_wrong_program_hash():
    program = programs()[1][1]
    t = tokenize(program)
    first = t.opcodes.copy()
    first[0, 0] = 1.
    with pytest.raises(BitVMAdmissionError, match="one-hot"):
        decode_tokens(replace(t, opcodes=first))
    bad_immediate = t.immediates.copy()
    bad_immediate[0, 0] = 1.
    with pytest.raises(BitVMAdmissionError, match="Noncanonical local selector"):
        decode_tokens(replace(t, immediates=bad_immediate))
    bad_local = t.local_selectors.copy()
    # local.get 0 already has bit 0 set: introduce a *second* selector bit.
    bad_local[0, 1] = 1.
    with pytest.raises(BitVMAdmissionError, match="selector"):
        decode_tokens(replace(t, local_selectors=bad_local))
    with pytest.raises(BitVMAdmissionError, match="digest"):
        decode_tokens(replace(t, original_wasm_sha256="0"*64))
    with pytest.raises(BitVMAdmissionError, match="signed i32"):
        initial_state(program, 1 << 31)
    with pytest.raises(BitVMAdmissionError, match="binary"):
        bit_step(replace(
            initial_state(program, 5),
            stack_bits=np.full((16, 32), 0.5),
        ), tokenize(program), construct_bit_weights())


def test_source_nonuniqueness_survives_full_i32_bit_gating():
    identity = programs()[0][1]
    longer = I32Program(0, (
        _op("local.get", 0), _op("i32.const", 0),
        _op("i32.add"), _op("end"),
    ))
    assert encode_program(identity) != encode_program(longer)
    for x in EDGE_VALUES:
        assert run_bit_vm(identity, x)["result"] == run_bit_vm(longer, x)["result"]



def test_declared_analytic_gate_weights_are_real_execution_parameters():
    fixed = construct_bit_weights()
    assert _and(1., 1., fixed) == 1.
    assert _xor(1., 0., fixed) == 1.
    assert _int(_add(_bits(1), _bits(1), fixed=fixed)) == 2

    # This catches "decorative weights" included in fingerprints but never
    # consumed by a purported neural forward pass. Each mutation must affect
    # the actual corresponding gate computation, not only its SHA digest.
    changed_and = replace(fixed, and_kernel=np.array([0., 0., 0.]))
    assert changed_and.digest() != fixed.digest()
    assert _and(1., 1., changed_and) != _and(1., 1., fixed)

    changed_xor = replace(fixed, xor_kernel=np.zeros((2, 2)))
    assert changed_xor.digest() != fixed.digest()
    assert _xor(1., 0., changed_xor) != _xor(1., 0., fixed)

    changed_adder = replace(
        fixed, adder_thresholds=np.array([2., 3.], dtype=np.float64)
    )
    assert changed_adder.digest() != fixed.digest()
    assert not np.array_equal(
        _add(_bits(1), _bits(1), fixed=fixed),
        _add(_bits(1), _bits(1), fixed=changed_adder),
    )
