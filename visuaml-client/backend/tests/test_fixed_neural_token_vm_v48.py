"""v48: exact per-instruction differential evidence for one fixed neural interpreter.

The reference is the separately implemented typed v40 WASM interpreter and,
at selected inputs, the independent Wasmtime engine executing canonical
source bytes. Program decoding is a structured-token operation, NOT recovery
of unique code from arbitrary learned transformer weights.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from visuaml.compiler_lab.fixed_neural_token_vm_v48 import (
    NeuralTokenVMError,
    construct_fixed_weights,
    compare_typed_programs_smt,
    decode_state,
    detokenize_program,
    initial_neural_state,
    neural_step,
    run_neural,
    tokenize_program,
)
from visuaml.compiler_lab.wasm_state_vm_v40 import (
    I32Program,
    VMOperation,
    encode_program,
    execute_program,
    initial_state,
    step,
)


def op(name: str, immediate: int | None = None) -> VMOperation:
    return VMOperation(name, immediate)


def programs() -> tuple[tuple[str, I32Program], ...]:
    return (
        ("identity", I32Program(0, (
            op("local.get", 0), op("end"),
        ))),
        ("affine", I32Program(0, (
            op("local.get", 0), op("i32.const", 7),
            op("i32.add"), op("end"),
        ))),
        ("quadratic", I32Program(0, (
            op("local.get", 0), op("local.get", 0), op("i32.mul"),
            op("i32.const", 3), op("i32.sub"), op("end"),
        ))),
        ("locals-tee", I32Program(2, (
            op("local.get", 0), op("local.set", 1),
            op("local.get", 1), op("i32.const", 2),
            op("i32.mul"), op("local.tee", 2),
            op("local.get", 2), op("i32.add"), op("end"),
        ))),
        ("eqz", I32Program(0, (
            op("local.get", 0), op("i32.eqz"), op("end"),
        ))),
        ("less-than", I32Program(0, (
            op("local.get", 0), op("i32.const", 3),
            op("i32.lt_s"), op("end"),
        ))),
        ("greater-than-drop", I32Program(0, (
            op("local.get", 0), op("i32.const", 9), op("drop"),
            op("i32.const", 1), op("i32.gt_s"), op("end"),
        ))),
    )


def test_fixed_weights_are_identical_across_programs_and_token_roundtrip():
    fixed = construct_fixed_weights()
    assert fixed.digest() == construct_fixed_weights().digest()
    assert len(fixed.digest()) == 64
    tokens = [tokenize_program(program) for _, program in programs()]
    for (_, original), token in zip(programs(), tokens, strict=True):
        assert detokenize_program(token) == original
        assert token.digest() != fixed.digest()
        assert token.source_wasm_sha256 != fixed.digest()
    assert len({t.digest() for t in tokens}) == len(tokens)


@pytest.mark.parametrize("argument", [-128, -63, -3, -1, 0, 1, 2, 13, 127])
def test_real_numerical_neural_state_matches_independent_v40_each_step(argument):
    fixed = construct_fixed_weights()
    for name, program in programs():
        tokens = tokenize_program(program)
        actual = initial_neural_state(program, argument)
        ref = initial_state(program, argument)
        for operation in program.operations:
            actual = neural_step(actual, tokens, fixed)
            ref, transition = step(program, ref)
            assert transition.opcode == operation.opcode
            assert decode_state(actual, program) == ref, (name, argument, operation)
        assert actual.halted
        assert ref.stack == (execute_program(program, argument).value,)


def test_7_programs_reuse_the_same_weights_on_all_256_signed_eight_bit_inputs():
    fixed = construct_fixed_weights()
    hashes = set()
    total = 0
    for name, program in programs():
        for x in range(-128, 128):
            result = run_neural(program, x, fixed=fixed)
            reference = execute_program(program, x)
            assert result["result_i32"] == reference.value, (name, x)
            assert result["fixed_weights_sha256"] == fixed.digest()
            assert result["program_is_separate_data"]
            assert result["no_weights_trained"]
            assert not result["standard_softmax_transformer_executed"]
            assert not result["general_wasm_support"]
            assert not result["all_i32_modular_arithmetic_verified"]
            assert len(result["step_trace"]) == len(program.operations)
            total += 1
        hashes.add(tokenize_program(program).digest())
    assert total == 7*256
    assert len(hashes) == 7


def test_independent_wasmtime_executes_original_binary_for_each_program():
    wasmtime = pytest.importorskip("wasmtime")
    engine = wasmtime.Engine()
    for _, program in programs():
        raw = encode_program(program)
        wasmtime.Module.validate(engine, raw)
        store = wasmtime.Store(engine)
        instance = wasmtime.Instance(
            store, wasmtime.Module(engine, raw), []
        )
        execute = instance.exports(store)["run"]
        for x in (-128, -64, -1, 0, 1, 7, 127):
            assert execute(store, x) == run_neural(program, x)["result_i32"]
            assert execute(store, x) == execute_program(program, x).value


def test_malformed_token_bytecode_cannot_bypass_strict_isa_decoder():
    token = tokenize_program(programs()[1][1])
    tampered = token.opcode_tokens.copy()
    tampered[0, 0] = 1.
    with pytest.raises(NeuralTokenVMError, match="one-hot"):
        detokenize_program(replace(token, opcode_tokens=tampered))
    tampered = token.immediate_tokens.copy()
    tampered[0] = 9.
    with pytest.raises(NeuralTokenVMError, match="Invalid one-hot local token"):
        detokenize_program(replace(token, immediate_tokens=tampered))
    tampered = token.local_tokens.copy()
    tampered[0, 0] = 1.
    tampered[0, 1] = 1.
    with pytest.raises(NeuralTokenVMError, match="local token"):
        detokenize_program(replace(token, local_tokens=tampered))
    tampered = token.opcode_tokens.copy()
    tampered[-1, 0] = 1.
    with pytest.raises(NeuralTokenVMError, match="padding"):
        detokenize_program(replace(token, opcode_tokens=tampered))
    with pytest.raises(NeuralTokenVMError, match="digest"):
        detokenize_program(replace(token, source_wasm_sha256="0"*64))


def test_finite_neural_execution_refuses_modular_i32_overflow_claim():
    poly4 = I32Program(0, (
        op("local.get", 0), op("local.get", 0), op("i32.mul"),
        op("local.get", 0), op("i32.mul"),
        op("local.get", 0), op("i32.mul"), op("end"),
    ))
    with pytest.raises(NeuralTokenVMError, match="arithmetic exceeded"):
        run_neural(poly4, 127)
    with pytest.raises(NeuralTokenVMError, match="input domain"):
        run_neural(programs()[0][1], 1000000000)
    with pytest.raises(NeuralTokenVMError, match="input domain"):
        run_neural(programs()[0][1], True)


def test_token_decoding_is_not_arbitrary_program_decompilation():
    # Two different source byte sequences implement exactly the same identity
    # function. Behavioral observations cannot reveal which source was used.
    identity = programs()[0][1]
    identity_with_zero = I32Program(0, (
        op("local.get", 0), op("i32.const", 0),
        op("i32.add"), op("end"),
    ))
    assert encode_program(identity) != encode_program(identity_with_zero)
    assert tokenize_program(identity).digest() != tokenize_program(identity_with_zero).digest()
    for x in range(-128, 128):
        assert run_neural(identity, x)["result_i32"] == (
            run_neural(identity_with_zero, x)["result_i32"]
        )
    # No claim that function outputs uniquely recover an original program.


def test_wrong_interpreter_state_and_unadmitted_immediates_fail_closed():
    program = programs()[1][1]
    tokens = tokenize_program(program)
    fixed = construct_fixed_weights()
    state = initial_neural_state(program, 7)
    with pytest.raises(NeuralTokenVMError, match="one-hot"):
        neural_step(replace(
            state, pc=np.full_like(state.pc, 1.),
        ), tokens, fixed)
    with pytest.raises(NeuralTokenVMError, match="Nonexact"):
        neural_step(replace(
            state, locals=np.full_like(state.locals, .5),
        ), tokens, fixed)
    oversize = I32Program(0, (
        op("i32.const", 1000000), op("end"),
    ))
    with pytest.raises(NeuralTokenVMError, match="immediate"):
        tokenize_program(oversize)



def test_all_i32_source_equivalence_exposes_program_non_identifiability():
    pytest.importorskip("z3")
    identity = programs()[0][1]
    identity_plus_zero = I32Program(0, (
        op("local.get", 0), op("i32.const", 0),
        op("i32.add"), op("end"),
    ))
    evidence = compare_typed_programs_smt(identity, identity_plus_zero)
    assert evidence["solver_result"] == "unsat"
    assert evidence["all_i32_typed_programs_equivalent"]
    assert evidence["left_wasm_sha256"] != evidence["right_wasm_sha256"]
    assert not evidence["unique_source_recovered_from_behavior"]
    assert not evidence["neural_all_i32_equivalence_proved"]
    different = I32Program(0, (
        op("i32.const", 0), op("end"),
    ))
    counterexample = compare_typed_programs_smt(identity, different)
    assert counterexample["solver_result"] == "sat"
    assert not counterexample["all_i32_typed_programs_equivalent"]
    assert counterexample["left_result_u32"] != counterexample["right_result_u32"]
