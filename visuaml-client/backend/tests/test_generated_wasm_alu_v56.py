"""v56 reproducible *generated* canonical WASM program differential corpus.

Unlike the six historical handwritten fixtures, these independently
assemble seeded straight-line expression chains from the admitted ISA.
Every generated binary is validated/executed in real Wasmtime.
"""
from __future__ import annotations

import hashlib
import random

import pytest
from visuaml.compiler_lab.neural_alu_executor_v56 import execute_alu_routed_control
from visuaml.compiler_lab.neural_operand_executor_v55 import (
    execute_operand_routed_control,
)
from visuaml.compiler_lab.wasm_control_flow_v51 import (
    ControlOperation,
    ControlProgram,
    encode_control,
    execute_control,
)

OPS = ("i32.add", "i32.sub", "i32.mul", "i32.lt_s", "i32.gt_s")
SAMPLES = (-(1 << 31), -17, -1, 0, 1, 19, (1 << 31) - 1)


def generated(seed: int) -> ControlProgram:
    rng = random.Random(560000 + seed)
    operations = [ControlOperation("local.get", 0)]
    for step in range(4):
        # Interleave i32 arithmetic, signed predicates, unary eqz and
        # mutable i32 locals without ever constructing an invalid stack.
        immediate = rng.choice((
            -(1 << 31), -65537, -3, -1, 0, 1, 5, 4097,
            (1 << 31) - 1,
        ))
        operations.extend((
            ControlOperation("i32.const", immediate),
            ControlOperation(OPS[(seed + 2 * step) % len(OPS)]),
        ))
        if (seed + step) % 3 == 0:
            operations.append(ControlOperation("i32.eqz"))
        if step == 1:
            operations.append(ControlOperation("local.tee", 1))
        if step == 2:
            operations.extend((
                ControlOperation("local.get", 1),
                ControlOperation(OPS[(seed + 1) % len(OPS)]),
            ))
    operations.append(ControlOperation("end"))
    return ControlProgram(1, tuple(operations))


@pytest.mark.parametrize("seed", range(16))
def test_generated_real_wasm_programs_match_all_three_engines(seed):
    wasmtime = pytest.importorskip("wasmtime")
    program = generated(seed)
    raw = encode_control(program)
    assert hashlib.sha256(raw).hexdigest()
    engine = wasmtime.Engine()
    wasmtime.Module.validate(engine, raw)
    store = wasmtime.Store(engine)
    run = wasmtime.Instance(
        store, wasmtime.Module(engine, raw), []
    ).exports(store)["run"]
    for value in SAMPLES:
        newest = execute_alu_routed_control(program, value, fuel=200)
        old = execute_operand_routed_control(program, value, fuel=200)
        host = execute_control(program, value, fuel=200)
        assert newest.result_i32 == old.result_i32 == host.value
        assert newest.result_i32 == run(store, value)
        assert newest.steps == old.steps == host.transitions
        for current, previous, reference in zip(
            newest.trace, old.trace, host.trace, strict=True
        ):
            for key in (
                "pc", "next_pc", "stack_i32", "locals_i32",
                "control_labels", "halted",
            ):
                assert current[key] == previous[key] == reference[key]


def test_generator_produces_distinct_original_source_programs():
    digests = {
        hashlib.sha256(encode_control(generated(seed))).hexdigest()
        for seed in range(16)
    }
    assert len(digests) == 16
