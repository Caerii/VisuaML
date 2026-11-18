"""Adversarial tests for bounded typed i32 VM and real executable WASM rewriting.

The independent Wasmtime engine checks both source and transformed modules.
An exhaustive signed-8 input sweep is finite-domain evidence, not a proof for
all 2^32 i32 inputs or arbitrary WASM programs.
"""
from __future__ import annotations

import hashlib

import pytest
from visuaml.compiler_lab.wasm_bitstream_v39 import MAGIC, _u32
from visuaml.compiler_lab.wasm_state_vm_v40 import (
    I32Program,
    VMAdmissionError,
    VMOperation,
    alpha_inverse_order,
    alpha_rename_locals,
    decode_module,
    encode_program,
    execute_program,
    initial_state,
    step,
    validate_program,
)


def instruction(name: str, value: int | None = None) -> VMOperation:
    return VMOperation(name, value)


def arithmetic_program() -> I32Program:
    # return (x*2 + 7) - 3, with WASM i32 modular arithmetic.
    return I32Program(
        3,
        (
            instruction("local.get", 0),
            instruction("local.set", 1),
            instruction("local.get", 1),
            instruction("i32.const", 2),
            instruction("i32.mul"),
            instruction("local.set", 2),
            instruction("local.get", 2),
            instruction("i32.const", 7),
            instruction("i32.add"),
            instruction("local.tee", 3),
            instruction("i32.const", 3),
            instruction("i32.sub"),
            instruction("end"),
        ),
    )


def test_canonical_module_decode_encode_and_typed_transition():
    p = arithmetic_program()
    raw = encode_program(p)
    assert raw.startswith(MAGIC)
    binding = decode_module(raw)
    assert binding.program == p
    assert binding.source_sha256 == hashlib.sha256(raw).hexdigest()
    assert p.digest() == decode_module(encode_program(p)).program.digest()
    state = initial_state(p, 12)
    first, trace = step(p, state)
    assert trace.opcode == "local.get"
    assert trace.stack_before == ()
    assert first.stack == (12,)
    assert first.locals == (12, 0, 0, 0)
    result = execute_program(p, 12)
    assert result.value == 28
    assert result.final_state.halted
    assert len(result.trace) == len(p.operations)
    assert result.trace[1].writes_local == 1
    assert result.trace[5].writes_local == 2
    assert result.trace[9].writes_local == 3
    assert result.trace_sha256 == execute_program(p, 12).trace_sha256


def test_i32_wrap_signed_comparison_and_eqz():
    p = I32Program(
        0, (instruction("local.get", 0), instruction("i32.const", 1),
            instruction("i32.add"), instruction("i32.eqz"), instruction("end"))
    )
    assert execute_program(p, -1).value == 1
    assert execute_program(p, 0).value == 0
    q = I32Program(
        0, (instruction("local.get", 0), instruction("i32.const", -1),
            instruction("i32.lt_s"), instruction("end"))
    )
    assert execute_program(q, -(1 << 31)).value == 1
    assert execute_program(q, (1 << 31) - 1).value == 0
    for bound in (-(1 << 31), (1 << 31) - 1):
        assert decode_module(encode_program(I32Program(
            0, (instruction("i32.const", bound), instruction("end"))
        ))).program.operations[0].immediate == bound


def test_reversible_local_renumbering_changes_real_wasm_instruction_bytes():
    source = encode_program(arithmetic_program())
    transformed, proof = alpha_rename_locals(source, (2, 0, 1))
    assert source != transformed
    assert proof["source_sha256"] == hashlib.sha256(source).hexdigest()
    assert proof["target_sha256"] == hashlib.sha256(transformed).hexdigest()
    assert proof["changed_byte_count"] > 0
    assert proof["binary_roundtrip_exact"]
    assert proof["typed_alpha_inverse_validated"]
    assert proof["module_typechecked_by_independent_engine"] is False
    assert proof["observational_parity_tested"] is False
    assert decode_module(source).program.operations != decode_module(transformed).program.operations
    restored, _ = alpha_rename_locals(transformed, alpha_inverse_order((2, 0, 1)))
    assert restored == source
    for value in (-2147483648, -32768, -1, 0, 1, 2, 7, 32768, 2147483647):
        original = execute_program(decode_module(source).program, value)
        rewritten = execute_program(decode_module(transformed).program, value)
        assert original.value == rewritten.value
        assert original.final_state.stack == rewritten.final_state.stack
        assert original.trace_sha256 != rewritten.trace_sha256


@pytest.mark.parametrize("bad", [
    (0, 0, 1), (0, 1), (True, 1, 2), (3, 0, 1), [0, 1, 2],
])
def test_local_permutation_rejects_invalid_bijections(bad):
    with pytest.raises(VMAdmissionError, match="bijective"):
        alpha_rename_locals(encode_program(arithmetic_program()), bad)


@pytest.mark.parametrize("code", [
    (instruction("i32.add"), instruction("end")),
    (instruction("local.get", 20), instruction("end")),
    (instruction("local.get", True), instruction("end")),
    (instruction("i32.const", None), instruction("end")),
    (instruction("local.get", 0), instruction("end"), instruction("end")),
    (instruction("i32.const", 1), instruction("drop"), instruction("end")),
    (instruction("i32.const", 1), instruction("i32.const", 2), instruction("end")),
    (instruction("i32.const", 1), instruction("unreachable"), instruction("end")),
])
def test_invalid_typed_programs_fail_before_binary_emission(code):
    with pytest.raises(VMAdmissionError):
        validate_program(I32Program(3, code))


def test_binary_parser_rejects_unsupported_modules_and_noncanonical_encoding():
    raw = encode_program(arithmetic_program())
    for bad in (
        b"", b"wasm", raw[:-1], raw + b"\x00",
        raw.replace(b"\x01\x60\x01\x7f\x01\x7f", b"\x01\x60\x01\x7f\x01\x7e", 1),
        raw.replace(b"\x03run", b"\x03bad", 1),
        raw.replace(b"\x03run", b"\x03ruu", 1),
        raw.replace(b"\x20\x00", b"\x20\x80\x00", 1),
        raw.replace(b"\x41\x02", b"\x41\x82\x00", 1),
    ):
        with pytest.raises(VMAdmissionError):
            decode_module(bad)
    # Valid section envelope can still hold completely unadmitted executable sections.
    with pytest.raises(VMAdmissionError, match="sections"):
        decode_module(MAGIC + b"\x00" + _u32(2) + b"\x01x" + raw[8:])


def test_fuel_is_an_explicit_bound():
    program = arithmetic_program()
    with pytest.raises(VMAdmissionError, match="fuel"):
        execute_program(program, 1, fuel=2)
    with pytest.raises(VMAdmissionError, match="fuel"):
        execute_program(program, 1, fuel=100000)
    with pytest.raises(VMAdmissionError, match="signed i32"):
        initial_state(program, (1 << 31))


def test_independent_wasmtime_executes_original_and_permuted_function():
    wasmtime = pytest.importorskip("wasmtime")
    source = encode_program(arithmetic_program())
    target, _ = alpha_rename_locals(source, (2, 0, 1))
    engine = wasmtime.Engine()
    run_functions = []
    for raw in (source, target):
        wasm = wasmtime.Module(engine, raw)
        store = wasmtime.Store(engine)
        instance = wasmtime.Instance(store, wasm, [])
        run_functions.append((store, instance.exports(store)["run"]))
    # Exhaustive 256-element signed-8 domain, plus adversarial i32 overflow edges.
    for value in (*range(-128, 128), -(1 << 31), (1 << 31) - 1, 65535, -65536):
        source_value = run_functions[0][1](run_functions[0][0], value)
        target_value = run_functions[1][1](run_functions[1][0], value)
        interpreter_value = execute_program(arithmetic_program(), value).value
        assert source_value == target_value == interpreter_value
