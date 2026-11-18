"""Shared inspectable semantic envelope for captured tensor and effect programs.

Both frontends normalize into one *description* of SSA values, effect ordering,
regions and tensor contracts. Execution continues through separately validated
runtime dialects. This intentionally does not claim full lowering between every
combination of dialects, symbolic shapes, or general Python control flow.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from math import prod
from typing import Literal

from visuaml.compiler_spike import ProgramIR

from .effect_ir import EffectProgram, validate_effect_program


class SemanticIRError(ValueError):
    """Invalid semantic envelope or unsupported canonicalization."""


_NAME = re.compile(r"[A-Za-z_][\w.\-]{0,127}\Z")
_DTYPES = {"float32": 4, "float64": 8, "int64": 8, "int32": 4, "bool": 1, "float16": 2}


@dataclass(frozen=True)
class ValueType:
    shape: tuple[int | str, ...]
    dtype: str
    bytes: int | None


@dataclass(frozen=True)
class SemanticOperation:
    name: str
    dialect: Literal["tensor", "state"]
    op: str
    operands: tuple[str, ...]
    result: str | None
    result_type: ValueType | None
    state_cell: str | None = None
    effect: Literal["pure", "read", "write"] = "pure"
    revision: int | None = None
    token_in: str | None = None
    token_out: str | None = None
    parameters: tuple[tuple[str, float | int], ...] = ()


@dataclass(frozen=True)
class SemanticRegion:
    name: str
    repetitions: int
    operations: tuple[SemanticOperation, ...]


@dataclass(frozen=True)
class SemanticProgram:
    origin: Literal["torch-fx", "effect-program", "onnx"]
    input_types: tuple[tuple[str, ValueType], ...]
    state_types: tuple[tuple[str, ValueType], ...]
    regions: tuple[SemanticRegion, ...]
    outputs: tuple[str, ...]
    source_sha256: str
    schema: str = "visuaml.semantic-envelope.v1"

    def as_dict(self) -> dict:
        return asdict(self)

    def digest(self) -> str:
        payload = json.dumps(
            self.as_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _name(value: str) -> bool:
    return isinstance(value, str) and bool(_NAME.fullmatch(value))


def _type_from_tensor(
    shape: tuple[str, ...], dtype: str, nbytes: int | None
) -> ValueType:
    canonical_dtype = dtype.removeprefix("torch.")
    if canonical_dtype not in _DTYPES or len(shape) > 16:
        raise SemanticIRError("Unsupported tensor dtype or rank")
    dims = tuple(int(d) if d.isdecimal() else d for d in shape)
    if any(isinstance(d, int) and d < 0 for d in dims):
        raise SemanticIRError("Negative tensor dimension")
    if nbytes is not None and (type(nbytes) is not int or nbytes < 0):
        raise SemanticIRError("Invalid declared storage bytes")
    if all(isinstance(d, int) for d in dims):
        expected_bytes = prod(dims) * _DTYPES[canonical_dtype]
        if nbytes is not None and expected_bytes != nbytes:
            raise SemanticIRError("Tensor shape/dtype/byte mismatch")
    return ValueType(dims, canonical_dtype, nbytes)


def canonicalize_tensor_ir(source: ProgramIR) -> SemanticProgram:
    """Import a traced tensor graph, retaining its unknowns rather than guessing."""
    if not isinstance(source, ProgramIR):
        raise SemanticIRError("Expected a traced ProgramIR")
    ops: list[SemanticOperation] = []
    inputs: list[tuple[str, ValueType]] = []
    for node in source.nodes:
        if node.op == "output":
            continue
        if len(node.tensors) > 1:
            raise SemanticIRError(
                "Multi-result FX node requires an explicit tuple dialect"
            )
        ty = (
            _type_from_tensor(
                node.tensors[0].shape, node.tensors[0].dtype, node.tensors[0].bytes
            )
            if node.tensors
            else None
        )
        if node.op == "input":
            if ty is None:
                raise SemanticIRError("Input missing shape/dtype metadata")
            inputs.append((node.id, ty))
        else:
            # The source IR also records get_attr constants. These stay as
            # explicit producers, not free values the verifier can invent.
            ops.append(
                SemanticOperation(node.id, "tensor", node.op, node.inputs, node.id, ty)
            )
    graph_blob = json.dumps(
        source.as_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    program = SemanticProgram(
        "torch-fx",
        tuple(inputs),
        (),
        (SemanticRegion("forward", 1, tuple(ops)),),
        source.output_names,
        hashlib.sha256(graph_blob.encode()).hexdigest(),
    )
    validate_semantic_program(program)
    return program


def canonicalize_effect_ir(source: EffectProgram) -> SemanticProgram:
    """Normalize stateful SSA regions, preserving cell revisions and token order."""
    validate_effect_program(source)
    convert = lambda ty: ValueType(
        ty.shape, ty.dtype, prod(ty.shape) * _DTYPES[ty.dtype]
    )
    regions: list[SemanticRegion] = []
    for loop in source.loops:
        slots = {x.name: convert(x.tensor) for x in source.state}
        ops: list[SemanticOperation] = []
        for ins in loop.instructions:
            # The specialized dialect has already inferred all intermediate
            # types; retain unknown result type explicitly until its verifier
            # offers a public per-operation inference API.
            typ = slots[ins.cell] if ins.op == "read" else None
            ops.append(
                SemanticOperation(
                    ins.name,
                    "state",
                    ins.op,
                    ins.args,
                    ins.output,
                    typ,
                    ins.cell,
                    ins.op if ins.op in ("read", "write") else "pure",
                    ins.revision,
                    ins.token_in,
                    ins.token_out,
                    ins.params,
                )
            )
        regions.append(SemanticRegion(loop.name, loop.count, tuple(ops)))
    program = SemanticProgram(
        "effect-program",
        tuple((x.name, convert(x.tensor)) for x in source.inputs),
        tuple((x.name, convert(x.tensor)) for x in source.state),
        tuple(regions),
        tuple(x.name for x in source.state),
        source.digest(),
    )
    validate_semantic_program(program)
    return program


def validate_semantic_program(program: SemanticProgram) -> None:
    """Structural admission for the shared inspection schema, not a proof."""
    if (
        not isinstance(program, SemanticProgram)
        or program.schema != "visuaml.semantic-envelope.v1"
    ):
        raise SemanticIRError("Unknown semantic schema")
    if not (1 <= len(program.regions) <= 64):
        raise SemanticIRError("Region count outside admission budget")
    if not isinstance(program.source_sha256, str) or not re.fullmatch(
        r"[0-9a-f]{64}", program.source_sha256
    ):
        raise SemanticIRError("Missing source provenance digest")
    known: set[str] = set()
    for name, spec in (*program.input_types, *program.state_types):
        if not _name(name) or name in known or not isinstance(spec, ValueType):
            raise SemanticIRError("Invalid or repeated input/state value")
        if (
            spec.dtype not in _DTYPES
            or len(spec.shape) > 16
            or any(not isinstance(d, (int, str)) for d in spec.shape)
        ):
            raise SemanticIRError("Invalid tensor contract")
        known.add(name)
    regions_seen: set[str] = set()
    operations = 0
    for region in program.regions:
        if (
            not _name(region.name)
            or region.name in regions_seen
            or type(region.repetitions) is not int
            or not 1 <= region.repetitions <= 256
        ):
            raise SemanticIRError("Duplicate, malformed or unbounded region")
        regions_seen.add(region.name)
        local = set(known)
        effect_token = "e0"
        for op in region.operations:
            if not _name(op.name) or not isinstance(op.operands, tuple):
                raise SemanticIRError("Malformed operation")
            if any(x not in local for x in op.operands):
                raise SemanticIRError(f"Use before definition in {op.name}")
            if op.effect in ("read", "write"):
                if (
                    op.state_cell not in dict(program.state_types)
                    or op.token_in != effect_token
                    or not _name(op.token_out)
                ):
                    raise SemanticIRError("Invalid state effect chain")
                effect_token = op.token_out
            elif (
                op.state_cell is not None
                or op.token_in is not None
                or op.token_out is not None
            ):
                raise SemanticIRError("Pure operator cannot carry state effect")
            if op.result is not None:
                if not _name(op.result) or op.result in local:
                    raise SemanticIRError("Repeated SSA value")
                local.add(op.result)
            operations += region.repetitions
            if operations > 16384:
                raise SemanticIRError("Expanded operator admission budget exceeded")
    # Effect programs may expose computed SSA results (e.g. decoder logits)
    # in addition to mutable state cells. Admission still requires every
    # returned result to have a producer in the declared regions.
    allowed_outputs = {x for x, _ in (*program.input_types, *program.state_types)}
    allowed_outputs |= {
        op.result for region in program.regions for op in region.operations
        if op.result is not None
    }
    if not program.outputs or not set(program.outputs).issubset(allowed_outputs):
        raise SemanticIRError("Unknown or missing output")
