"""Canonical WASM scaling workloads for v58 offline performance experiments.

Each program remains within the admitted v51 structural resource bounds.
These programs test different *growth dimensions*; measured runtime curves
are not automatically claims about asymptotic algorithmic complexity.
"""
from __future__ import annotations

from .wasm_control_flow_v51 import ControlOperation, ControlProgram


def op(name: str, imm: int | None = None) -> ControlOperation:
    return ControlOperation(name, imm)


def arithmetic_chain(terms: int) -> ControlProgram:
    if type(terms) is not int or not 1 <= terms <= 40:
        raise ValueError("Arithmetic chain length must be 1..40")
    ops = [op("local.get", 0)]
    for index in range(terms):
        ops.extend((op("i32.const", index + 1), op("i32.add")))
    ops.append(op("end"))
    return ControlProgram(0, tuple(ops))


def stack_depth(depth: int) -> ControlProgram:
    if type(depth) is not int or not 1 <= depth <= 24:
        raise ValueError("Operand depth outside reviewed capacity")
    ops = [op("i32.const", index + 1) for index in range(depth)]
    ops.extend(op("i32.add") for _ in range(depth-1))
    ops.append(op("end"))
    return ControlProgram(0, tuple(ops))


def nested_frames(depth: int, *, kind: str = "block") -> ControlProgram:
    if (type(depth) is not int or not 1 <= depth <= 12
            or kind not in ("block", "loop")):
        raise ValueError("Frame depth/kind outside reviewed capacity")
    ops = [op(kind) for _ in range(depth)]
    ops.append(op("nop"))
    ops.extend(op("end") for _ in range(depth))
    ops.extend((op("i32.const", 1), op("end")))
    return ControlProgram(0, tuple(ops))


def unwind_frames(depth: int) -> ControlProgram:
    if type(depth) is not int or not 1 <= depth <= 12:
        raise ValueError("Branch unwind depth outside reviewed capacity")
    ops = [op("block") for _ in range(depth)]
    ops.append(op("br", depth - 1))
    ops.extend(op("end") for _ in range(depth))
    ops.extend((op("i32.const", 1), op("end")))
    return ControlProgram(0, tuple(ops))


def scaling_corpus() -> dict[str, tuple[ControlProgram, tuple[int, ...], str]]:
    result = {}
    for n in (2, 4, 8, 16, 32, 40):
        result[f"alu_length_{n:02d}"] = (
            arithmetic_chain(n), (-17, 0, 1, 17), "alu_length",
        )
    for depth in (2, 4, 8, 16, 24):
        result[f"operand_depth_{depth:02d}"] = (
            stack_depth(depth), (0,), "operand_depth",
        )
    for depth in (1, 2, 4, 8, 12):
        result[f"frame_depth_{depth:02d}"] = (
            nested_frames(depth), (0,), "frame_depth",
        )
        result[f"loop_frame_depth_{depth:02d}"] = (
            nested_frames(depth, kind="loop"), (0,), "loop_frame_depth",
        )
        result[f"branch_unwind_{depth:02d}"] = (
            unwind_frames(depth), (0,), "branch_unwind",
        )
    return result
