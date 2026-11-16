"""Closed-world semantic-IR execution of the bounded v14-v17 stacked decoder.

Unlike v16's handwritten layer loop, this CPU interpreter executes the actual
normalization, projection, RoPE, attention, residual, SwiGLU, and logits
operations enumerated by a typed VisuaML SemanticProgram. Stateful K/V effects
lower to the transactional v17 resident-cache backend.

Admitted model: the existing 1..4-layer synthetic float32 GQA/RoPE/SwiGLU
checkpoint ABI. No arbitrary Python, pretrained importer, native-code AOT,
CUDA, optical hardware, or all-input equivalence proof is implied.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from math import sqrt
from typing import Any

import numpy as np

from .family_blocks_v12 import KVState
from .kv_lowering_v17 import (
    KVSnapshot,
    lower_kv_semantic_program,
    make_kv_semantic_program,
)
from .semantic_ir import (
    SemanticOperation,
    SemanticProgram,
    SemanticRegion,
    ValueType,
    validate_semantic_program,
)
from .stacked_decoder_v14 import (
    StackState,
    StackedWeights,
    checkpoint_digest,
)
from .streaming_decoder_v16 import _snapshot_weights

SCHEMA = "visuaml.semantic-decoder.v18"


class DecoderLoweringError(ValueError):
    """A decoder program or runtime input does not satisfy the closed-world ABI."""


def _hash_document(document: object) -> str:
    wire = json.dumps(document, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(wire.encode()).hexdigest()


def make_decoder_semantic_program(weights: StackedWeights, *, batch: int = 1) -> SemanticProgram:
    """Build the entire forward computation in typed SSA and explicit state effects."""
    if type(batch) is not int or not 1 <= batch <= 8:
        raise DecoderLoweringError("Decoder batch must be an integer in [1,8]")
    vocab, hidden, depth, kv_heads = weights.validate()
    heads = weights.heads
    capacity = weights.max_context
    arrays = weights.arrays()
    inputs = [("tokens", ValueType((batch, "Q"), "int64", None))]
    inputs.extend(
        (name, ValueType(tuple(array.shape), "float32", array.nbytes))
        for name, array in sorted(arrays.items())
    )
    states = tuple(
        (
            f"{kind}{layer}",
            ValueType(
                (batch, capacity, kv_heads, depth), "float32",
                batch * capacity * kv_heads * depth * 4,
            ),
        )
        for layer in range(len(weights.blocks))
        for kind in ("k", "v")
    )
    operations: list[SemanticOperation] = []
    effect_index = 0

    def emit(
        name: str,
        opcode: str,
        operands: tuple[str, ...],
        shape: tuple[int | str, ...] | None,
        *,
        dtype: str = "float32",
        state_cell: str | None = None,
        effect: str = "pure",
        parameters: tuple[tuple[str, int], ...] = (),
    ) -> str:
        nonlocal effect_index
        result = name if shape is not None else None
        result_type = ValueType(shape, dtype, None) if shape is not None else None
        token_in = token_out = None
        revision = None
        if effect != "pure":
            token_in, token_out = f"e{effect_index}", f"e{effect_index + 1}"
            effect_index += 1
            revision = 1
        operations.append(
            SemanticOperation(
                name, "state" if effect != "pure" else "tensor",
                opcode, operands, result, result_type,
                state_cell, effect, revision, token_in, token_out, parameters,
            )
        )
        return name

    hshape = (batch, "Q", hidden)
    qshape = (batch, "Q", heads, depth)
    kvshape = (batch, "Q", kv_heads, depth)
    h = emit("embedding_out", "embedding", ("tokens", "embedding"), hshape)
    for layer, block in enumerate(weights.blocks):
        prefix = f"l{layer}"
        weight = f"layer{layer}"
        n = emit(f"{prefix}_rms_attn", "rmsnorm", (h, f"{weight}.attention.norm"), hshape)
        q = emit(f"{prefix}_q", "matmul", (n, f"{weight}.attention.q"), hshape)
        q = emit(f"{prefix}_q_heads", "split_heads", (q,), qshape, parameters=(("heads", heads),))
        k = emit(
            f"{prefix}_k", "matmul", (n, f"{weight}.attention.k"),
            (batch, "Q", kv_heads * depth),
        )
        k = emit(
            f"{prefix}_k_heads", "split_heads", (k,), kvshape,
            parameters=(("heads", kv_heads),),
        )
        v = emit(
            f"{prefix}_v", "matmul", (n, f"{weight}.attention.v"),
            (batch, "Q", kv_heads * depth),
        )
        v = emit(
            f"{prefix}_v_heads", "split_heads", (v,), kvshape,
            parameters=(("heads", kv_heads),),
        )
        q = emit(f"{prefix}_q_rotary", "rope", (q,), qshape)
        k = emit(f"{prefix}_k_rotary", "rope", (k,), kvshape)
        emit(
            f"{prefix}_stage_key", "stage_key", (k,), None,
            state_cell=f"k{layer}", effect="write",
            parameters=(("layer", layer),),
        )
        emit(
            f"{prefix}_stage_value", "stage_value", (v,), None,
            state_cell=f"v{layer}", effect="write",
            parameters=(("layer", layer),),
        )
        cached_k = emit(
            f"{prefix}_read_key", "read_cache", (f"k{layer}",),
            (batch, "T", kv_heads, depth),
            state_cell=f"k{layer}", effect="read",
        )
        cached_v = emit(
            f"{prefix}_read_value", "read_cache", (f"v{layer}",),
            (batch, "T", kv_heads, depth),
            state_cell=f"v{layer}", effect="read",
        )
        a = emit(
            f"{prefix}_attention", "gqa_attention", (q, cached_k, cached_v), hshape,
        )
        projected = emit(
            f"{prefix}_attn_output", "matmul", (a, f"{weight}.attention.out"), hshape,
        )
        residual = emit(f"{prefix}_residual", "add", (h, projected), hshape)
        final_norm = emit(
            f"{prefix}_rms_final", "rmsnorm",
            (residual, f"{weight}.final_norm"), hshape,
        )
        intermediate = block.gate.shape[1]
        fshape = (batch, "Q", intermediate)
        gate = emit(
            f"{prefix}_gate", "matmul", (final_norm, f"{weight}.gate"), fshape,
        )
        silu = emit(f"{prefix}_silu", "silu", (gate,), fshape)
        up = emit(f"{prefix}_up", "matmul", (final_norm, f"{weight}.up"), fshape)
        product = emit(f"{prefix}_product", "multiply", (silu, up), fshape)
        down = emit(
            f"{prefix}_down", "matmul", (product, f"{weight}.down"), hshape,
        )
        h = emit(f"{prefix}_hidden", "add", (residual, down), hshape)
    logits = emit("logits", "matmul", (h, "lm_head"), (batch, "Q", vocab))
    program = SemanticProgram(
        origin="effect-program",
        input_types=tuple(inputs),
        state_types=states,
        regions=(SemanticRegion("decode_chunk", 1, tuple(operations)),),
        outputs=(logits,),
        source_sha256=_hash_document(
            (SCHEMA, checkpoint_digest(weights), batch, heads, kv_heads, depth, capacity)
        ),
    )
    validate_semantic_program(program)
    return program


@dataclass(frozen=True)
class CompiledDecoderPlan:
    """Checkpoint-bound, validated decoder schedule; compilation is not machine code."""

    semantic: SemanticProgram
    weights: StackedWeights
    batch: int

    @property
    def digest(self) -> str:
        return self.semantic.digest()

    @property
    def checkpoint_sha256(self) -> str:
        return checkpoint_digest(self.weights)

    def new_session(self) -> SemanticDecoderSession:
        # Canonicalize again to reject dataclass.replace with forged plan fields.
        compiled = lower_decoder_semantic_program(self.semantic, self.weights, batch=self.batch)
        return SemanticDecoderSession(compiled)

    def manifest(self) -> dict[str, Any]:
        data = {
            "schema": SCHEMA,
            "semantic_sha256": self.digest,
            "checkpoint_sha256": self.checkpoint_sha256,
            "batch": self.batch,
            "operations": sum(len(x.operations) for x in self.semantic.regions),
            "backend": "interpreted-numpy-cpu",
        }
        return {**data, "manifest_sha256": _hash_document(data)}

    def explain(self) -> dict[str, Any]:
        ops = self.semantic.regions[0].operations
        return {
            "schema": SCHEMA,
            "checkpoint_sha256": self.checkpoint_sha256,
            "semantic_sha256": self.digest,
            "operation_count": len(ops),
            "operator_histogram": dict(sorted(Counter(op.op for op in ops).items())),
            "state_effect_order": [
                {"operation": op.name, "cell": op.state_cell, "effect": op.effect}
                for op in ops if op.effect != "pure"
            ],
            "compiler_limitations": [
                "The CPU executor interprets operations; no machine-code generation.",
                "The input checkpoint ABI is synthetic and not Hugging Face-compatible.",
                "Numerical parity witnesses do not constitute an all-input proof.",
            ],
        }


def lower_decoder_semantic_program(
    program: SemanticProgram, weights: StackedWeights, *, batch: int = 1
) -> CompiledDecoderPlan:
    """Admit only exact, typed SSA programs with their declared checkpoint effects."""
    if not isinstance(program, SemanticProgram):
        raise DecoderLoweringError("Expected a SemanticProgram")
    if type(batch) is not int or not 1 <= batch <= 8:
        raise DecoderLoweringError("Invalid decoder batch")
    try:
        validate_semantic_program(program)
        frozen = _snapshot_weights(weights)
        expected = make_decoder_semantic_program(frozen, batch=batch)
    except (ValueError, TypeError, AttributeError, OverflowError) as exc:
        raise DecoderLoweringError("Malformed decoder program or checkpoint") from exc
    if program != expected:
        raise DecoderLoweringError("Noncanonical decoder graph, types, effects, or checkpoint")
    return CompiledDecoderPlan(program, frozen, batch)


class SemanticDecoderSession:
    """Single-owner CPU execution of the typed decoder graph and v17 cache effects."""

    def __init__(self, plan: CompiledDecoderPlan):
        # The plan is immutable; revalidate the semantic program at session entry.
        if plan.semantic != make_decoder_semantic_program(plan.weights, batch=plan.batch):
            raise DecoderLoweringError("Session received a forged semantic decoder plan")
        self.plan = plan
        weights = plan.weights
        vocab, hidden, depth, kv_heads = weights.validate()
        self.vocab, self.hidden, self.depth = vocab, hidden, depth
        self.heads, self.kv_heads = weights.heads, kv_heads
        self.batch, self.capacity = plan.batch, weights.max_context
        self.cache = lower_kv_semantic_program(
            make_kv_semantic_program(
                batch=self.batch, layers=len(weights.blocks),
                kv_heads=kv_heads, head_dim=depth, capacity=self.capacity,
            )
        ).new_session()
        inv = 1.0 / (10000.0 ** (np.arange(0, depth, 2, dtype=np.float64) / depth))
        theta = np.outer(np.arange(self.capacity, dtype=np.float64), inv)
        self._cos = np.cos(theta).astype(np.float32)
        self._sin = np.sin(theta).astype(np.float32)
        self._weights = weights.arrays()
        self._operations = plan.semantic.regions[0].operations
        self._uses = Counter(arg for op in self._operations for arg in op.operands)

    @property
    def cursor(self) -> int:
        return self.cache.cursor

    def _execute(
        self,
        op: SemanticOperation,
        env: dict[str, np.ndarray],
        pending: dict[int, np.ndarray],
        tx: Any,
        start: int,
        total: int,
    ) -> np.ndarray | None:
        code = op.op
        if code == "read_cache":
            return self.cache._arrays[op.state_cell][:, :total]
        args = tuple(env[k] for k in op.operands)
        if code == "embedding":
            return args[1][args[0]]
        if code == "rmsnorm":
            h, weights = args
            return (h / np.sqrt(np.mean(h * h, axis=-1, keepdims=True) + 1e-5)) * weights
        if code == "matmul":
            return args[0] @ args[1]
        if code == "split_heads":
            tensor = args[0]
            heads = dict(op.parameters)["heads"]
            return tensor.reshape(self.batch, tensor.shape[1], heads, self.depth)
        if code == "rope":
            tensor = args[0]
            n = tensor.shape[1]
            c = self._cos[start:start + n][None, :, None, :]
            s = self._sin[start:start + n][None, :, None, :]
            result = np.empty_like(tensor)
            evens, odds = tensor[..., ::2], tensor[..., 1::2]
            result[..., ::2] = evens * c - odds * s
            result[..., 1::2] = evens * s + odds * c
            return result
        if code == "stage_key":
            layer = dict(op.parameters)["layer"]
            pending[layer] = args[0]
            return None
        if code == "stage_value":
            layer = dict(op.parameters)["layer"]
            tx.stage(layer, pending.pop(layer), args[0])
            return None
        if code == "gqa_attention":
            q, keys, values = args
            query = q.shape[1]
            kv_heads = self.kv_heads
            replicas = self.heads // kv_heads
            grouped_q = q.reshape(
                self.batch, query, kv_heads, replicas, self.depth
            ).transpose(0, 2, 3, 1, 4)
            k = keys.transpose(0, 2, 1, 3)[:, :, None, :, :]
            scores = np.matmul(grouped_q, np.swapaxes(k, -1, -2)) * np.float32(
                1.0 / sqrt(self.depth)
            )
            positions = np.arange(start, total, dtype=np.int64)
            causal = np.arange(total)[None, :] <= positions[:, None]
            scores = np.where(causal[None, None, None, :, :], scores, np.float32(-1e9))
            np.subtract(scores, scores.max(axis=-1, keepdims=True), out=scores)
            np.exp(scores, out=scores)
            scores /= scores.sum(axis=-1, keepdims=True)
            v = values.transpose(0, 2, 1, 3)[:, :, None, :, :]
            return np.matmul(scores, v).transpose(0, 3, 1, 2, 4).reshape(
                self.batch, query, self.hidden
            )
        if code == "add":
            return args[0] + args[1]
        if code == "silu":
            return args[0] / (1.0 + np.exp(-args[0]))
        if code == "multiply":
            return args[0] * args[1]
        raise DecoderLoweringError(f"Unknown executable decoder opcode: {code}")

    def step(self, tokens: np.ndarray) -> np.ndarray:
        """Interpret one symbolic-Q region; publish all-layer state atomically."""
        if (
            not isinstance(tokens, np.ndarray)
            or tokens.dtype != np.int64
            or tokens.ndim != 2
            or tokens.shape[0] != self.batch
            or not 1 <= tokens.shape[1] <= 32
            or self.cursor + tokens.shape[1] > self.capacity
            or np.any(tokens < 0) or np.any(tokens >= self.vocab)
        ):
            raise DecoderLoweringError("Invalid tokens, vocabulary, batch, or context budget")
        start = self.cursor
        query = tokens.shape[1]
        total = start + query
        env = {"tokens": tokens, **self._weights}
        uses = self._uses.copy()
        pending: dict[int, np.ndarray] = {}
        with self.cache.begin_append(query) as tx:
            for op in self._operations:
                value = self._execute(op, env, pending, tx, start, total)
                if op.result is not None:
                    if not isinstance(value, np.ndarray) or op.result_type is None:
                        raise DecoderLoweringError(f"Missing computed output at {op.name}")
                    shape = tuple(
                        query if d == "Q" else total if d == "T" else d
                        for d in op.result_type.shape
                    )
                    if (
                        value.shape != shape
                        or value.dtype != np.dtype(op.result_type.dtype)
                        or not np.isfinite(value).all()
                    ):
                        raise DecoderLoweringError(f"Invalid computed tensor at {op.name}")
                    env[op.result] = value
                elif value is not None:
                    raise DecoderLoweringError(f"Unexpected output from effect {op.name}")
                for arg in op.operands:
                    uses[arg] -= 1
                    if uses[arg] == 0 and arg != "logits":
                        env.pop(arg, None)
            if pending or "logits" not in env:
                raise DecoderLoweringError("Incomplete graph output or uncommitted key")
            result = env["logits"].copy()
            tx.commit()
            return result

    def snapshot(self) -> StackState:
        snap = self.cache.snapshot()
        cells = dict(snap.buffers)
        return StackState(tuple(
            KVState(cells[f"k{layer}"], cells[f"v{layer}"], self.capacity)
            for layer in range(len(self.plan.weights.blocks))
        ))

    def restore(self, state: StackState) -> None:
        if not isinstance(state, StackState) or len(state.layers) != len(self.plan.weights.blocks):
            raise DecoderLoweringError("Wrong decoder cache layers")
        if not state.layers or not isinstance(state.layers[0], KVState):
            raise DecoderLoweringError("Malformed cache snapshot")
        length = state.layers[0].keys.shape[1] if isinstance(
            state.layers[0].keys, np.ndarray
        ) and state.layers[0].keys.ndim == 4 else -1
        if length < 0:
            raise DecoderLoweringError("Invalid cache length")
        cells: list[tuple[str, np.ndarray]] = []
        for layer, state_layer in enumerate(state.layers):
            if not isinstance(state_layer, KVState) or state_layer.max_context != self.capacity:
                raise DecoderLoweringError("Incompatible decoder KV state")
            cells.extend(((f"k{layer}", state_layer.keys), (f"v{layer}", state_layer.values)))
        self.cache.restore(KVSnapshot(self.cache.plan.semantic_sha256, length, tuple(cells)))

    def reset(self) -> None:
        self.cache.reset()

    def generate(self, prompt: np.ndarray, *, steps: int) -> tuple[np.ndarray, StackState]:
        if type(steps) is not int or not 1 <= steps <= 24:
            raise DecoderLoweringError("Generated token count must be in [1,24]")
        if not isinstance(prompt, np.ndarray) or prompt.ndim != 2 or prompt.dtype != np.int64:
            raise DecoderLoweringError("Prompt must be int64 [batch,tokens]")
        if prompt.shape[1] + steps > self.capacity:
            raise DecoderLoweringError("Generation exceeds context capacity")
        self.reset()
        logits = self.step(prompt)
        produced = []
        for _ in range(steps):
            ids = np.argmax(logits[:, -1], axis=-1).astype(np.int64)[:, None]
            produced.append(ids)
            logits = self.step(ids)
        return np.concatenate(produced, axis=1), self.snapshot()

    def report(self) -> dict[str, Any]:
        return {
            **self.cache.report(),
            "schema": SCHEMA,
            "semantic_sha256": self.plan.digest,
            "checkpoint_sha256": self.plan.checkpoint_sha256,
            "ops_per_chunk": len(self._operations),
            "executable_math": True,
            "backend": "NumPy CPU interpreter; no native AOT, GPU, or physical optical device",
        }
