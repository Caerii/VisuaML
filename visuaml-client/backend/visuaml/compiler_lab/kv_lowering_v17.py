"""Verified semantic-IR lowering for a bounded, dynamic-length resident KV primitive.

This is a real, NumPy-executed stateful operator schedule lowered from VisuaML's
shared SemanticProgram. It is not machine-code generation, a complete decoder
compiler, a pretrained checkpoint importer, or a GPU implementation.

The closed-world v17 dialect admits paired K/V append operations across layers
with one atomic logical commit. It deliberately rejects arbitrary effectful
programs and unbounded symbolic dimensions. All state is session-owned.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from .semantic_ir import (
    SemanticOperation,
    SemanticProgram,
    SemanticRegion,
    ValueType,
    validate_semantic_program,
)

SCHEMA = "visuaml.compiled-resident-kv.v17"
_MAX_RESERVED_BYTES = 64 * 1024 * 1024


class KVLoweringError(ValueError):
    """Unsupported semantic lowering, input, or resident-cache transition."""


def _digest(document: object) -> str:
    return hashlib.sha256(
        json.dumps(document, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _slots(layers: int) -> tuple[str, ...]:
    return tuple(f"{kind}{layer}" for layer in range(layers) for kind in ("k", "v"))


def _admit(batch: int, layers: int, kv_heads: int, head_dim: int, capacity: int) -> None:
    if any(type(x) is not int for x in (batch, layers, kv_heads, head_dim, capacity)):
        raise KVLoweringError("KV dimensions must be integers, not bool or symbols")
    if not (1 <= batch <= 8 and 1 <= layers <= 16 and 1 <= kv_heads <= 16):
        raise KVLoweringError("Batch/layer/KV-head count outside supported bounds")
    if not (2 <= head_dim <= 128 and head_dim % 2 == 0 and 1 <= capacity <= 4096):
        raise KVLoweringError("Head dimension or context capacity outside bounds")
    if batch * layers * kv_heads * head_dim * capacity * 8 > _MAX_RESERVED_BYTES:
        raise KVLoweringError("Resident K/V byte budget exceeded")


def make_kv_semantic_program(
    *, batch: int, layers: int, kv_heads: int, head_dim: int, capacity: int
) -> SemanticProgram:
    """Express per-layer K/V append and ordered publication in the common IR.

    The symbolic 'step' dimension may vary at each invocation; all other
    dimensions and the fixed physical capacity are statically admitted.
    """
    _admit(batch, layers, kv_heads, head_dim, capacity)
    slots = _slots(layers)
    state_shape = (batch, capacity, kv_heads, head_dim)
    chunk_shape = (batch, "step", kv_heads, head_dim)
    state = tuple(
        (name, ValueType(state_shape, "float32", batch * capacity * kv_heads * head_dim * 4))
        for name in slots
    )
    inputs = tuple((f"chunk_{name}", ValueType(chunk_shape, "float32", None)) for name in slots)
    operations: list[SemanticOperation] = []
    for effect_index, name in enumerate(slots):
        result = f"next_{name}"
        operations.append(
            SemanticOperation(
                name=f"append_{name}",
                dialect="state",
                op="kv_append",
                operands=(name, f"chunk_{name}"),
                result=result,
                result_type=ValueType((batch, "live", kv_heads, head_dim), "float32", None),
            )
        )
        operations.append(
            SemanticOperation(
                name=f"publish_{name}",
                dialect="state",
                op="write",
                operands=(result,),
                result=None,
                result_type=None,
                state_cell=name,
                effect="write",
                revision=1,
                token_in=f"e{effect_index}",
                token_out=f"e{effect_index + 1}",
            )
        )
    origin_spec = (SCHEMA, batch, layers, kv_heads, head_dim, capacity)
    program = SemanticProgram(
        origin="effect-program",
        input_types=inputs,
        state_types=state,
        regions=(SemanticRegion("append_transaction", 1, tuple(operations)),),
        outputs=slots,
        source_sha256=_digest(origin_spec),
    )
    validate_semantic_program(program)
    return program


@dataclass(frozen=True)
class CompiledKVPlan:
    """An admitted, source-bound schedule of typed resident K/V writes."""

    semantic: SemanticProgram
    batch: int
    layers: int
    kv_heads: int
    head_dim: int
    capacity: int
    slots: tuple[str, ...]

    @property
    def semantic_sha256(self) -> str:
        return self.semantic.digest()

    def new_session(self) -> KVSession:
        return KVSession(self)

    def manifest(self) -> dict[str, Any]:
        unsigned = {
            "schema": SCHEMA,
            "batch": self.batch,
            "layers": self.layers,
            "kv_heads": self.kv_heads,
            "head_dim": self.head_dim,
            "capacity": self.capacity,
            "semantic_sha256": self.semantic_sha256,
        }
        return {**unsigned, "manifest_sha256": _digest(unsigned)}


def lower_kv_semantic_program(program: SemanticProgram) -> CompiledKVPlan:
    """Lower a canonical effect-typed KV program to an executable NumPy plan.

    Equality against the generated legal program is intentional: the current
    backend has one understood effect schedule, not unrestricted stateful FX.
    """
    if not isinstance(program, SemanticProgram):
        raise KVLoweringError("Expected a typed SemanticProgram")
    try:
        validate_semantic_program(program)
    except (ValueError, TypeError, AttributeError) as exc:
        raise KVLoweringError("Malformed or unverified semantic IR") from exc
    states = program.state_types
    if not states or len(states) % 2 or len(states) > 32:
        raise KVLoweringError("Paired K/V state declarations required")
    shape = states[0][1].shape
    if len(shape) != 4 or any(type(v) is not int for v in shape):
        raise KVLoweringError("Resident KV buffers require concrete allocation dimensions")
    batch, capacity, kv_heads, head_dim = shape
    layers = len(states) // 2
    _admit(batch, layers, kv_heads, head_dim, capacity)
    canonical = make_kv_semantic_program(
        batch=batch, layers=layers, kv_heads=kv_heads,
        head_dim=head_dim, capacity=capacity,
    )
    if program != canonical:
        raise KVLoweringError("Noncanonical KV effects, shapes, source or operation order")
    return CompiledKVPlan(program, batch, layers, kv_heads, head_dim, capacity, _slots(layers))


def load_kv_manifest(document: Mapping[str, Any]) -> CompiledKVPlan:
    """Reconstruct only known programs; never unpickle or execute supplied code."""
    fields = {
        "schema", "batch", "layers", "kv_heads", "head_dim", "capacity",
        "semantic_sha256", "manifest_sha256",
    }
    if not isinstance(document, Mapping) or set(document) != fields:
        raise KVLoweringError("Unknown or incomplete v17 manifest")
    unsigned = {k: document[k] for k in fields if k != "manifest_sha256"}
    try:
        if (
            document["schema"] != SCHEMA
            or document["manifest_sha256"] != _digest(unsigned)
        ):
            raise KVLoweringError("KV manifest integrity or schema mismatch")
        program = make_kv_semantic_program(
            batch=document["batch"], layers=document["layers"],
            kv_heads=document["kv_heads"], head_dim=document["head_dim"],
            capacity=document["capacity"],
        )
        if program.digest() != document["semantic_sha256"]:
            raise KVLoweringError("KV semantic program fingerprint mismatch")
        return lower_kv_semantic_program(program)
    except (TypeError, OverflowError) as exc:
        raise KVLoweringError("Invalid v17 manifest fields") from exc


@dataclass(frozen=True)
class KVSnapshot:
    """Detached snapshot; the hash binds the typed program, not checkpoint weights."""

    semantic_sha256: str
    cursor: int
    buffers: tuple[tuple[str, np.ndarray], ...]


class KVSession:
    """Single-owner NumPy executor with one logical commit across all KV layers.

    Caller inputs are copied before any writes. Failed validation or incomplete
    staging cannot publish a new cursor. This class is NOT thread-safe.
    """

    def __init__(self, plan: CompiledKVPlan):
        self.plan = lower_kv_semantic_program(plan.semantic)
        self._arrays = {
            name: np.zeros(
                (plan.batch, plan.capacity, plan.kv_heads, plan.head_dim),
                dtype=np.float32,
            )
            for name in plan.slots
        }
        self.cursor = 0

    def append(self, chunks: Mapping[str, np.ndarray]) -> int:
        if not isinstance(chunks, Mapping) or set(chunks) != set(self.plan.slots):
            raise KVLoweringError("Must provide exactly one key and value chunk per layer")
        prepared: dict[str, np.ndarray] = {}
        step: int | None = None
        for name in self.plan.slots:
            value = chunks[name]
            if (
                not isinstance(value, np.ndarray)
                or value.dtype != np.float32
                or value.ndim != 4
                or value.shape[0] != self.plan.batch
                or value.shape[2:] != (self.plan.kv_heads, self.plan.head_dim)
            ):
                raise KVLoweringError(f"Invalid float32 KV chunk shape/dtype: {name}")
            if step is None:
                step = value.shape[1]
            if value.shape[1] != step or not np.isfinite(value).all():
                raise KVLoweringError("KV chunks differ in length or contain nonfinite values")
            prepared[name] = value.copy(order="C")
        if step is None or not 1 <= step <= 256 or self.cursor + step > self.plan.capacity:
            raise KVLoweringError("KV append length or capacity exceeded")
        start, end = self.cursor, self.cursor + step
        try:
            # Order comes from the checked semantic-IR lowering, not map iteration.
            for name in self.plan.slots:
                np.copyto(self._arrays[name][:, start:end], prepared[name])
        except Exception:
            for name in self.plan.slots:
                self._arrays[name][:, start:end].fill(0)
            raise
        self.cursor = end
        return end

    def snapshot(self) -> KVSnapshot:
        copies = []
        for name in self.plan.slots:
            result = self._arrays[name][:, :self.cursor].copy()
            result.setflags(write=False)
            copies.append((name, result))
        return KVSnapshot(self.plan.semantic_sha256, self.cursor, tuple(copies))

    def restore(self, snapshot: KVSnapshot) -> None:
        if (
            not isinstance(snapshot, KVSnapshot)
            or snapshot.semantic_sha256 != self.plan.semantic_sha256
            or type(snapshot.cursor) is not int
            or not 0 <= snapshot.cursor <= self.plan.capacity
            or len(snapshot.buffers) != len(self.plan.slots)
        ):
            raise KVLoweringError("Incompatible KV snapshot")
        expected = (
            self.plan.batch, snapshot.cursor, self.plan.kv_heads, self.plan.head_dim
        )
        staged: dict[str, np.ndarray] = {}
        for (name, value), slot in zip(snapshot.buffers, self.plan.slots):
            if (
                name != slot or not isinstance(value, np.ndarray)
                or value.dtype != np.float32 or value.shape != expected
                or not np.isfinite(value).all()
            ):
                raise KVLoweringError("Malformed KV snapshot; state was not modified")
            staged[name] = value.copy(order="C")
        # Build replacement storage first to preserve old state even if OOM.
        replacement = {name: np.zeros_like(arr) for name, arr in self._arrays.items()}
        for name in self.plan.slots:
            replacement[name][:, :snapshot.cursor] = staged[name]
        self._arrays = replacement
        self.cursor = snapshot.cursor

    def reset(self) -> None:
        for arr in self._arrays.values():
            arr.fill(0)
        self.cursor = 0

    def report(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "semantic_sha256": self.plan.semantic_sha256,
            "cursor": self.cursor,
            "capacity": self.plan.capacity,
            "reserved_bytes": sum(x.nbytes for x in self._arrays.values()),
            "live_bytes": (
                self.cursor * self.plan.batch * self.plan.kv_heads
                * self.plan.head_dim * len(self.plan.slots) * 4
            ),
            "backend": "NumPy CPU interpreted KV schedule (not machine-code AOT)",
        }
