"""Symbolic liveness analysis of the bounded v18 semantic decoder graph.

This is a conservative logical tensor-retention model, not GPU/CPU physical
peak RSS, NumPy temporary allocations, or a completed arena allocator. Values
are released after last use by the v18 interpreter; the plan explains exactly
which graph outputs may be retained and their worst admitted shape.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import prod

from .semantic_ir import SemanticProgram, ValueType, validate_semantic_program


class MemoryPlanningError(ValueError):
    """Non-admitted symbolic shape or unsupported graph memory contract."""


@dataclass(frozen=True)
class Lifetime:
    value: str
    defined_at: int
    last_used_at: int
    maximum_bytes: int


@dataclass(frozen=True)
class MemoryPlan:
    semantic_sha256: str
    maximum_query: int
    maximum_context: int
    peak_logical_bytes: int
    peak_logical_values: int
    lifetimes: tuple[Lifetime, ...]

    def as_dict(self) -> dict:
        return {
            "schema": "visuaml.semantic-memory-analysis.v19",
            "semantic_sha256": self.semantic_sha256,
            "maximum_query": self.maximum_query,
            "maximum_context": self.maximum_context,
            "peak_logical_bytes": self.peak_logical_bytes,
            "peak_logical_values": self.peak_logical_values,
            "lifetimes": [vars(item) for item in self.lifetimes],
            "limitations": [
                "Conservative logical SSA values, not physical peak allocation.",
                "Views and temporary NumPy allocations are not deduplicated.",
                "Weights, initial inputs and resident KV storage are excluded.",
            ],
        }


def _byte_size(spec: ValueType, *, query: int, context: int) -> int:
    if not isinstance(spec, ValueType) or spec.dtype not in ("float32", "int64"):
        raise MemoryPlanningError("Unsupported retained tensor type")
    dimensions = []
    for dim in spec.shape:
        if type(dim) is int and dim >= 0:
            dimensions.append(dim)
        elif dim == "Q":
            dimensions.append(query)
        elif dim == "T":
            dimensions.append(context)
        else:
            raise MemoryPlanningError("Unbounded dimension in liveness plan")
    return prod(dimensions) * {"float32": 4, "int64": 8}[spec.dtype]


def plan_decoder_memory(
    program: SemanticProgram, *, maximum_query: int = 32, maximum_context: int = 64
) -> MemoryPlan:
    validate_semantic_program(program)
    if (
        type(maximum_query) is not int or type(maximum_context) is not int
        or not 1 <= maximum_query <= 32
        or not maximum_query <= maximum_context <= 64
        or len(program.regions) != 1 or program.regions[0].repetitions != 1
    ):
        raise MemoryPlanningError("Only bounded single-region decoder plans are admitted")
    operations = program.regions[0].operations
    last_use = {
        name: i
        for i, op in enumerate(operations)
        for name in op.operands
    }
    lengths: list[Lifetime] = []
    for index, op in enumerate(operations):
        if op.result is None:
            continue
        if op.result_type is None:
            raise MemoryPlanningError("SSA result missing a tensor contract")
        lengths.append(Lifetime(
            op.result, index, last_use.get(op.result, index),
            _byte_size(op.result_type, query=maximum_query, context=maximum_context),
        ))
    if not lengths:
        raise MemoryPlanningError("Decoder plan contains no result tensors")
    peak_bytes = peak_count = 0
    for index in range(len(operations)):
        active = [
            item for item in lengths
            if item.defined_at <= index <= item.last_used_at
        ]
        peak_count = max(peak_count, len(active))
        peak_bytes = max(peak_bytes, sum(item.maximum_bytes for item in active))
    return MemoryPlan(
        program.digest(), maximum_query, maximum_context,
        peak_bytes, peak_count, tuple(lengths),
    )
