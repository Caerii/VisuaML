"""v51 original structured WASM validated independently by Wasmtime.

The execution engine delegates arithmetic to v49's constructed bit gates;
CPU-host branches and manages control frames. These tests check trace shape,
branch targets and fuel separately from Wasmtime final results.
"""
from __future__ import annotations

import hashlib

import pytest
from visuaml.compiler_lab.wasm_control_flow_v51 import (
    ControlAdmissionError,
    ControlExecutionTrap,
    ControlFuelExhausted,
    ControlOperation,
    ControlProgram,
    decode_control,
    encode_control,
    execute_control,
)


def op(name, value=None):
    return ControlOperation(name, value)


def countdown():
    # 0+1+...+n, with explicit loop branch and outer-block escape.
    return ControlProgram(1, (
        op("block"), op("loop"), op("local.get", 0),
        op("i32.eqz"), op("br_if", 1), op("local.get", 1),
        op("local.get", 0), op("i32.add"), op("local.set", 1),
        op("local.get", 0), op("i32.const", 1),
        op("i32.sub"), op("local.set", 0), op("br", 0),
        op("end"), op("end"), op("local.get", 1), op("end"),
    ))


def conditional():
    return ControlProgram(1, (
        op("local.get", 0), op("i32.const", 0),
        op("i32.lt_s"), op("if"),
        op("i32.const", 11), op("local.set", 1), op("else"),
        op("i32.const", 22), op("local.set", 1),
        op("end"), op("local.get", 1), op("end"),
    ))


def break_block():
    return ControlProgram(1, (
        op("block"), op("local.get", 0), op("i32.const", 0),
        op("i32.gt_s"), op("br_if", 0),
        op("i32.const", 7), op("local.set", 1),
        op("end"), op("local.get", 1), op("end"),
    ))


def early_return():
    return ControlProgram(0, (
        op("local.get", 0), op("i32.const", 0),
        op("i32.gt_s"), op("if"), op("i32.const", 9),
        op("return"), op("end"),
        op("i32.const", -3), op("end"),
    ))


def gcd_with_12():
    """Euclid repeated-subtraction GCD(n,12) for nonnegative n.

    Negative values return sentinel 0 (not mathematical GCD).
    The input-zero case returns 12. Branching uses actual WASM control
    constructs, not Python while loops or custom pseudo-opcodes.
    """
    return ControlProgram(1, (
        op("local.get", 0), op("i32.eqz"),
        op("if"), op("i32.const", 12), op("return"), op("end"),
        op("local.get", 0), op("i32.const", 0), op("i32.lt_s"),
        op("if"), op("i32.const", 0), op("return"), op("end"),
        op("i32.const", 12), op("local.set", 1),
        op("block"), op("loop"),
        op("local.get", 0), op("local.get", 1),
        op("i32.sub"), op("i32.eqz"), op("br_if", 1),
        op("local.get", 0), op("local.get", 1), op("i32.gt_s"),
        op("if"),
        op("local.get", 0), op("local.get", 1),
        op("i32.sub"), op("local.set", 0),
        op("else"),
        op("local.get", 1), op("local.get", 0),
        op("i32.sub"), op("local.set", 1),
        op("end"),
        op("br", 0),
        op("end"), op("end"),
        op("local.get", 0), op("end"),
    ))


def infinite():
    return ControlProgram(0, (
        op("loop"), op("br", 0), op("end"),
        op("i32.const", 1), op("end"),
    ))


@pytest.mark.parametrize("program", [
    countdown(), conditional(), break_block(), early_return(), infinite()
])
def test_real_wasm_canonical_binary_roundtrip(program):
    raw = encode_control(program)
    assert raw.startswith(b"\0asm\x01\0\0\0")
    assert decode_control(raw) == program
    assert len(hashlib.sha256(raw).hexdigest()) == 64


@pytest.mark.parametrize("program,x,expected", [
    (countdown(),0,0), (countdown(),1,1),
    (countdown(),2,3), (countdown(),5,15), (countdown(),8,36),
    (conditional(),-5,11), (conditional(),-1,11),
    (conditional(),0,22), (conditional(),5,22),
    (break_block(),-1,7), (break_block(),0,7),
    (break_block(),1,0), (break_block(),9,0),
    (early_return(),-1,-3), (early_return(),0,-3),
    (early_return(),5,9),
])
def test_input_dependent_program_results_and_control_traces(program,x,expected):
    result = execute_control(program, x, fuel=400)
    assert result.value == expected
    assert result.transitions == len(result.trace)
    assert result.trace[0]["pc"] == 0
    assert result.trace[-1]["halted"]
    assert result.host_control_routing
    assert result.trace[-1]["stack_i32"] == [expected]
    assert result.source_sha256 == hashlib.sha256(
        encode_control(program)
    ).hexdigest()


def test_loop_back_edges_preserve_labels_and_mutate_locals():
    result = execute_control(countdown(),5,fuel=400)
    assert result.value == 15
    assert result.trace[0]["control_labels"] == ["block"]
    assert result.trace[1]["control_labels"] == ["block","loop"]
    back = [x for x in result.trace if
            x["opcode"] == "br" and x["next_pc"] <= x["pc"]]
    assert len(back) == 5
    assert all(x["control_labels"] == ["block","loop"] for x in back)
    assert [x["locals_i32"][0] for x in back] == [4,3,2,1,0]
    assert [x["locals_i32"][1] for x in back] == [5,9,12,14,15]


def test_wasmtime_independently_executes_canonical_control_flow():
    wasmtime = pytest.importorskip("wasmtime")
    engine=wasmtime.Engine()
    samples=(
        (countdown(),range(9)),
        (conditional(),(-10,-1,0,1,10)),
        (break_block(),(-10,-1,0,1,10)),
        (early_return(),(-10,-1,0,1,10)),
    )
    for program, values in samples:
        wasm=encode_control(program)
        wasmtime.Module.validate(engine,wasm)
        store=wasmtime.Store(engine)
        func=wasmtime.Instance(store,wasmtime.Module(engine,wasm),[]).exports(store)["run"]
        for x in values:
            assert func(store,x)==execute_control(program,x,fuel=500).value


def test_fuel_exhaustion_is_distinct_from_wasm_language_trap():
    for program,x in ((infinite(),0),(countdown(),-1)):
        with pytest.raises(ControlFuelExhausted,match="fuel exhausted"):
            execute_control(program,x,fuel=40)
    with pytest.raises(ControlAdmissionError,match="fuel"):
        execute_control(infinite(),0,fuel=9999999)


def test_malformed_branch_nesting_and_nonempty_blocks_rejected():
    with pytest.raises(ControlAdmissionError,match="label"):
        encode_control(ControlProgram(0,(
            op("block"),op("br",1),op("end"),op("i32.const",1),op("end"),
        )))
    with pytest.raises(ControlAdmissionError,match="else"):
        encode_control(ControlProgram(0,(op("else"),op("i32.const",0),op("end"))))
    with pytest.raises(ControlAdmissionError,match="Unclosed"):
        encode_control(ControlProgram(0,(op("block"),op("i32.const",0),op("end"))))
    raw=encode_control(conditional())
    with pytest.raises(ControlAdmissionError):
        decode_control(raw.replace(b"\x04\x40",b"\x04\x7f",1))
    with pytest.raises(ControlAdmissionError):
        decode_control(raw+b"\0")


@pytest.mark.parametrize("x, expected", [
    (-1, 0), (0, 12), (1, 1), (2, 2), (3, 3),
    (4, 4), (5, 1), (6, 6), (7, 1),
    (8, 4), (9, 3), (10, 2), (11, 1),
    (12, 12), (15, 3), (16, 4), (18, 6),
    (20, 4), (21, 3), (24, 12),
])
def test_real_wasm_gcd_repeated_subtraction_with_host_bit_neural_arithmetic(x, expected):
    wasmtime = pytest.importorskip("wasmtime")
    p = gcd_with_12()
    wasm = encode_control(p)
    engine = wasmtime.Engine()
    wasmtime.Module.validate(engine, wasm)
    store = wasmtime.Store(engine)
    func = wasmtime.Instance(
        store, wasmtime.Module(engine, wasm), []
    ).exports(store)["run"]
    computed = execute_control(p, x, fuel=1500)
    assert computed.value == expected == func(store, x)
    assert computed.transitions == len(computed.trace)
    if x > 0 and x != 12:
        assert any(w["opcode"] == "br" for w in computed.trace)
        assert any(w["opcode"] == "br_if" for w in computed.trace)
    assert computed.host_control_routing


def test_host_control_arithmetic_uses_modular_i32_gates():
    p=ControlProgram(0,(
        op("local.get",0),op("i32.const",1),op("i32.add"),op("end"),
    ))
    assert execute_control(p,(1<<31)-1).value==-(1<<31)



def test_structural_decoder_is_not_misrepresented_as_full_wasm_type_validator():
    wasmtime = pytest.importorskip("wasmtime")
    # Structurally well-nested, but the i32.add has no operands. The
    # standalone v51 parser intentionally checks the closed-world binary
    # envelope and lexical labels; independent Wasmtime enforces stack types.
    bad = ControlProgram(0, (
        op("i32.add"), op("i32.const", 0), op("end"),
    ))
    raw = encode_control(bad)
    assert decode_control(raw) == bad
    with pytest.raises(wasmtime.WasmtimeError):
        wasmtime.Module.validate(wasmtime.Engine(), raw)
    with pytest.raises(ControlExecutionTrap, match="underflow"):
        execute_control(bad, 0, fuel=20)
