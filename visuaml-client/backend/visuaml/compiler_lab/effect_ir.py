"""Typed, bounded, effect-linear state programs (VisuaML research dialect v1).

The dialect is intentionally closed-world: no Python callback or code string,
no unbounded loop, dynamic branch, implicit casts, or mutable aliasing.
SSA values are local to each loop iteration. Read/write operations consume
and produce a linear effect token and carry the current relative cell revision.
Successful validation certifies STRUCTURE, not universal numerical equivalence.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from math import prod
from typing import Any, Literal

SCHEMA = "visuaml.effect-program.v1"
_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_.-]{0,63}\Z")


class EffectIRError(ValueError):
    """An invalid instruction, type, effect sequence, or resource admission."""


@dataclass(frozen=True)
class TensorType:
    shape: tuple[int, ...]
    dtype: Literal["float32", "int64"]


@dataclass(frozen=True)
class StateSlot:
    name: str
    tensor: TensorType


@dataclass(frozen=True)
class Instruction:
    name: str
    op: str
    args: tuple[str, ...] = ()
    output: str | None = None
    cell: str | None = None
    params: tuple[tuple[str, float | int], ...] = ()
    token_in: str | None = None
    token_out: str | None = None
    revision: int | None = None


@dataclass(frozen=True)
class Repeat:
    name: str
    count: int
    instructions: tuple[Instruction, ...]


@dataclass(frozen=True)
class EffectProgram:
    state: tuple[StateSlot, ...]
    inputs: tuple[StateSlot, ...]
    loops: tuple[Repeat, ...]
    schema: str = SCHEMA

    def digest(self) -> str:
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True,
            separators=(",", ":"), allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class ValidatedEffectProgram:
    schema: str
    state_cells: tuple[str, ...]
    loop_iterations: int
    expanded_operations: int
    estimated_work_units: int
    ordered_effects: tuple[tuple[str, str, str, int], ...]
    program_sha256: str


def _named(value: str) -> bool:
    return isinstance(value, str) and bool(_NAME.fullmatch(value))


def _tensor(t: TensorType) -> None:
    if not isinstance(t, TensorType) or t.dtype not in ("float32", "int64"):
        raise EffectIRError("Only declared float32/int64 tensor types are supported")
    if not isinstance(t.shape, tuple) or len(t.shape) > 6 or not all(
        type(n) is int and 0 < n <= 100_000 for n in t.shape
    ):
        raise EffectIRError("Tensor shapes must be concrete positive dimensions of rank <=6")
    if prod(t.shape) > 1_000_000:
        raise EffectIRError("Tensor element budget exceeded")


def _params(ins: Instruction, required: tuple[str, ...]) -> dict[str, float | int]:
    params = dict(ins.params)
    if len(params) != len(ins.params) or set(params) != set(required):
        raise EffectIRError(f"{ins.op} requires precisely the parameters {required}")
    import math
    for v in params.values():
        if type(v) not in (int, float) or not math.isfinite(v):
            raise EffectIRError("Parameters must be finite numeric scalars, never booleans")
    return params


def _infer(ins: Instruction, types: dict[str, TensorType]) -> TensorType:
    ts = [types[a] for a in ins.args]
    if ins.op == "identity":
        _params(ins, ())
        if len(ts) != 1:
            raise EffectIRError("identity requires one typed input")
        return ts[0]
    if ins.op in ("add", "subtract"):
        _params(ins, ())
        if len(ts) != 2 or ts[0] != ts[1] or ts[0].dtype != "float32":
            raise EffectIRError("Addition/subtraction requires equal float32 tensors")
        return ts[0]
    if ins.op == "matmul":
        _params(ins, ())
        if len(ts) != 2 or any(t.dtype != "float32" or len(t.shape) != 2 for t in ts) or ts[0].shape[1] != ts[1].shape[0]:
            raise EffectIRError("matmul requires compatible float32 matrices")
        return TensorType((ts[0].shape[0], ts[1].shape[1]), "float32")
    if ins.op in ("relu", "tanh"):
        _params(ins, ())
        if len(ts) != 1 or ts[0].dtype != "float32":
            raise EffectIRError(f"{ins.op} expects one float32 tensor")
        return ts[0]
    if ins.op == "gather_sum":
        _params(ins, ())
        if len(ts) != 2 or len(ts[0].shape) != 2 or ts[0].dtype != "float32" or len(ts[1].shape) != 1 or ts[1].dtype != "int64":
            raise EffectIRError("gather_sum requires a matrix and integer indices")
        return TensorType((ts[0].shape[1],), "float32")
    if ins.op == "topk_indices":
        p = _params(ins, ("k",))
        if len(ts) != 1 or len(ts[0].shape) != 1 or ts[0].dtype != "float32" or type(p["k"]) is not int or not 1 <= p["k"] <= ts[0].shape[0]:
            raise EffectIRError("topk_indices requires finite positive k <= vector length")
        return TensorType((p["k"],), "int64")
    if ins.op == "hebbian_update":
        p = _params(ins, ("beta", "ceiling"))
        if (len(ts) != 3 or len(ts[0].shape) != 2 or ts[0].dtype != "float32"
            or any(len(x.shape) != 1 or x.dtype != "int64" for x in ts[1:])
            or p["beta"] < 0 or p["ceiling"] <= 0):
            raise EffectIRError("hebbian_update requires weights, indices, beta>=0 and positive ceiling")
        return ts[0]
    if ins.op == "refraction_update":
        p = _params(ins, ("strength",))
        if (len(ts) != 3 or ts[0] != ts[1] or len(ts[0].shape) != 1
            or ts[0].dtype != "float32" or ts[2].dtype != "int64"
            or len(ts[2].shape) != 1 or p["strength"] < 0):
            raise EffectIRError("refraction_update requires bias, raw drive, indices and strength>=0")
        return ts[0]
    raise EffectIRError(f"Unsupported pure operation: {ins.op}")


def validate_effect_program(program: EffectProgram) -> ValidatedEffectProgram:
    """Validate types, local SSA, relative state revisions and effect order."""
    if not isinstance(program, EffectProgram) or program.schema != SCHEMA:
        raise EffectIRError("Unsupported effect program schema")
    if not (1 <= len(program.state) <= 64 and len(program.inputs) <= 32 and 1 <= len(program.loops) <= 16):
        raise EffectIRError("State/input/loop budget exceeded")
    cells: dict[str, TensorType] = {}
    initial: dict[str, TensorType] = {}
    for slot in (*program.state, *program.inputs):
        if not isinstance(slot, StateSlot) or not _named(slot.name) or slot.name in cells or slot.name in initial:
            raise EffectIRError("Invalid or duplicated state/input name")
        _tensor(slot.tensor)
        (cells if slot in program.state else initial)[slot.name] = slot.tensor
    allocated_state_bytes = sum(prod(t.shape) * (8 if t.dtype == "int64" else 4) for t in (*cells.values(), *initial.values()))
    if allocated_state_bytes > 64 * 1024 * 1024:
        raise EffectIRError("Combined state/input memory budget exceeded")
    iterations = 0
    expanded = 0
    estimated_work = 0
    effects: list[tuple[str, str, str, int]] = []
    loop_names: set[str] = set()
    for loop in program.loops:
        if not isinstance(loop, Repeat) or not _named(loop.name) or loop.name in loop_names:
            raise EffectIRError("Invalid or duplicate loop")
        loop_names.add(loop.name)
        if type(loop.count) is not int or not 1 <= loop.count <= 256 or not 1 <= len(loop.instructions) <= 256:
            raise EffectIRError("Unbounded or empty repeat block")
        iterations += loop.count
        expanded += loop.count * len(loop.instructions)
        if iterations > 512 or expanded > 8192:
            raise EffectIRError("Expanded execution budget exceeded")
        versions = dict.fromkeys(cells, 0)
        types = dict(initial)
        named_ins: set[str] = set()
        token = "e0"
        used_tokens = {token}
        for ins in loop.instructions:
            if not isinstance(ins, Instruction) or not _named(ins.name) or ins.name in named_ins:
                raise EffectIRError("Invalid or duplicate instruction name")
            named_ins.add(ins.name)
            if not isinstance(ins.args, tuple) or not all(_named(a) and a in types for a in ins.args):
                raise EffectIRError(f"Unknown/forward SSA value used in {ins.name}")
            if ins.op in ("read", "write"):
                if (ins.cell not in cells or ins.token_in != token or not _named(ins.token_out)
                    or ins.token_out in used_tokens or type(ins.revision) is not int
                    or versions[ins.cell] != ins.revision or ins.params):
                    raise EffectIRError("State effect token, cell or revision mismatch")
                used_tokens.add(ins.token_out)
                token = ins.token_out
                if ins.op == "read":
                    if ins.args or not _named(ins.output) or ins.output in types:
                        raise EffectIRError("State read must define a fresh SSA value")
                    types[ins.output] = cells[ins.cell]
                else:
                    if len(ins.args) != 1 or ins.output is not None or types[ins.args[0]] != cells[ins.cell]:
                        raise EffectIRError("State write has incorrect type or SSA signature")
                    versions[ins.cell] += 1
                effects.append((loop.name, ins.op, ins.cell, ins.revision))
            else:
                if ins.cell is not None or ins.token_in is not None or ins.token_out is not None or ins.revision is not None:
                    raise EffectIRError("Pure operator must not have state effects")
                if not _named(ins.output) or ins.output in types:
                    raise EffectIRError("Pure operation must define one fresh SSA value")
                result_type = _infer(ins, types)
                types[ins.output] = result_type
                # Conservative execution-budget estimate; this is NOT wall time.
                size = prod(result_type.shape)
                if ins.op == "matmul":
                    size *= types[ins.args[0]].shape[-1]
                elif ins.op in ("gather_sum", "hebbian_update"):
                    size *= types[ins.args[1]].shape[0]
                elif ins.op == "topk_indices":
                    size = types[ins.args[0]].shape[0] * max(1, types[ins.args[0]].shape[0].bit_length())
                estimated_work += loop.count * size
                if estimated_work > 80_000_000:
                    raise EffectIRError("Aggregate compute budget exceeded")
    return ValidatedEffectProgram(SCHEMA, tuple(cells), iterations, expanded,
        estimated_work, tuple(effects), program.digest())


def assert_effect_trace_preserved(before: EffectProgram, after: EffectProgram) -> None:
    """Conservative rewrite guard: preserve all state accesses and their wiring."""
    validate_effect_program(before)
    validate_effect_program(after)
    if before.state != after.state or before.inputs != after.inputs:
        raise EffectIRError("Rewrite changed the typed state/interface contract")
    if tuple((b.name, b.count) for b in before.loops) != tuple((b.name, b.count) for b in after.loops):
        raise EffectIRError("Rewrite changed the schedule")
    def effect_wiring(p: EffectProgram) -> tuple[tuple, ...]:
        return tuple((loop.name, op) for loop in p.loops for op in loop.instructions if op.op in ("read", "write"))
    if effect_wiring(before) != effect_wiring(after):
        raise EffectIRError("Rewrite reordered, removed or rewired state effects")
    # It is the caller's responsibility to compare executable witnesses for
    # any different PURE computation. This guard is deliberately not a proof.


def recurrent_assembly_effect_program(n: int, k: int, *, steps: int, beta: float,
                                       refraction_strength: float, ceiling: float = 20.0) -> EffectProgram:
    """Construct a canonical recurrent template with externally bound state arrays."""
    if type(n) is not int or type(k) is not int or not 1 <= k <= n:
        raise EffectIRError("Invalid recurrent population dimensions")
    f = TensorType
    instructions = (
        Instruction("src", "read", output="source", cell="winners", token_in="e0", token_out="e1", revision=0),
        Instruction("weights0", "read", output="w0", cell="weights", token_in="e1", token_out="e2", revision=0),
        Instruction("drive", "gather_sum", args=("w0", "source"), output="raw"),
        Instruction("bias0", "read", output="b0", cell="bias", token_in="e2", token_out="e3", revision=0),
        Instruction("net", "subtract", args=("raw", "b0"), output="net_drive"),
        Instruction("select", "topk_indices", args=("net_drive",), output="selected", params=(("k", k),)),
        Instruction("learn", "hebbian_update", args=("w0", "source", "selected"), output="w_next", params=(("beta", beta), ("ceiling", ceiling))),
        Instruction("charge", "refraction_update", args=("b0", "raw", "selected"), output="b_next", params=(("strength", refraction_strength),)),
        Instruction("publish_weights", "write", args=("w_next",), cell="weights", token_in="e3", token_out="e4", revision=0),
        Instruction("publish_bias", "write", args=("b_next",), cell="bias", token_in="e4", token_out="e5", revision=0),
        Instruction("publish_winners", "write", args=("selected",), cell="winners", token_in="e5", token_out="e6", revision=0),
    )
    program = EffectProgram((StateSlot("weights", f((n,n), "float32")),
                           StateSlot("bias", f((n,), "float32")),
                           StateSlot("winners", f((k,), "int64"))), (), (Repeat("recurrent", steps, instructions),))
    validate_effect_program(program)
    return program
