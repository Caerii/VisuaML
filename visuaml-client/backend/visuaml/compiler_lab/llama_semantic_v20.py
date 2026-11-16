"""v20: real Llama weights -> typed VisuaML IR -> streaming CPU execution.

The same checked SemanticProgram drives NumPy and a TorchScript-hybrid CPU
runtime. All mathematical operations and KV effects are explicit. TorchScript
compiles selected arithmetic primitives, while attention, RoPE and cache writes
remain NumPy. No claim of full native compiled Llama, arbitrary checkpoints,
GPU, tokenizer correctness, or physical-device performance is made.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from dataclasses import asdict, dataclass
from math import sqrt
from typing import Literal

import numpy as np

from .family_blocks_v12 import KVState
from .hf_llama_checkpoint_v19 import (
    HFLlamaContractError,
    LlamaLocalCheckpoint,
    LlamaProfile,
    _expected_shapes,
)
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
from .stacked_decoder_v14 import StackState
from .torchscript_primitives_v19 import TorchScriptPrimitives

SCHEMA = "visuaml.real-llama-semantic.v20"
Backend = Literal["numpy", "torchscript_hybrid"]


class LlamaSemanticError(ValueError):
    """Llama semantics, checkpoint identity, or execution failed admission."""


def _sha_document(value: object) -> str:
    wire = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(wire.encode("utf-8")).hexdigest()


def _freeze_checkpoint(checkpoint: LlamaLocalCheckpoint) -> LlamaLocalCheckpoint:
    if not isinstance(checkpoint, LlamaLocalCheckpoint):
        raise LlamaSemanticError("Expected the validated local Llama checkpoint ABI")
    p = checkpoint.profile
    if not isinstance(p, LlamaProfile) or p.head_dim % 2 or p.head_dim > 128:
        raise LlamaSemanticError("Unsupported Llama head profile")
    if not (1 <= p.layers <= 8 and 1 <= p.kv_heads <= 16 and 1 <= p.heads <= 16):
        raise LlamaSemanticError("Unsupported Llama layer/head count")
    if p.hidden != p.heads * p.head_dim or p.heads % p.kv_heads:
        raise LlamaSemanticError("Inconsistent Llama attention geometry")
    if not (0 < p.eps <= 1e-3 and 1 <= p.rope_theta <= 1e7 and p.context >= 1):
        raise LlamaSemanticError("Invalid Llama normalization/position profile")
    expected = _expected_shapes(p)
    if set(checkpoint.arrays) != set(expected):
        raise LlamaSemanticError("Checkpoint tensor key set does not match Llama ABI")
    if not (
        isinstance(checkpoint.checkpoint_sha256, str)
        and re.fullmatch(r"[0-9a-f]{64}", checkpoint.checkpoint_sha256)
        and isinstance(checkpoint.config_sha256, str)
        and re.fullmatch(r"[0-9a-f]{64}", checkpoint.config_sha256)
        and isinstance(checkpoint.declared_revision, str)
        and re.fullmatch(r"[0-9a-f]{40}", checkpoint.declared_revision)
    ):
        raise LlamaSemanticError("Missing checkpoint, config, or revision identity")
    frozen = {}
    total = 0
    for name, shape in expected.items():
        array = checkpoint.arrays[name]
        if (
            not isinstance(array, np.ndarray)
            or array.dtype != np.float32 or array.shape != shape
            or not np.isfinite(array).all()
        ):
            raise LlamaSemanticError("Incompatible finite FP32 checkpoint tensor: " + name)
        total += array.nbytes
        if total > 128 * 1024 * 1024:
            raise LlamaSemanticError("Admitted Llama tensors exceed memory budget")
        contiguous = np.ascontiguousarray(array)
        # A bytes-owned view cannot be made writable and does not alias inputs.
        frozen[name] = np.frombuffer(contiguous.tobytes(), dtype=np.float32).reshape(shape)
    return LlamaLocalCheckpoint(
        p, frozen, checkpoint.checkpoint_sha256,
        checkpoint.config_sha256, checkpoint.declared_revision,
        checkpoint.tokenizer_sha256,
    )


def _identity(checkpoint: LlamaLocalCheckpoint) -> str:
    return _sha_document({
        "schema": SCHEMA,
        "profile": asdict(checkpoint.profile),
        "source_checkpoint": checkpoint.checkpoint_sha256,
        "source_config": checkpoint.config_sha256,
        "source_revision": checkpoint.declared_revision,
        "tensors": {
            key: hashlib.sha256(value.tobytes()).hexdigest()
            for key, value in sorted(checkpoint.arrays.items())
        },
    })


def make_llama_semantic_program(
    checkpoint: LlamaLocalCheckpoint, *, batch: int = 1
) -> SemanticProgram:
    """Canonical typed SSA graph containing the real Llama forward and KV effects."""
    if type(batch) is not int or not 1 <= batch <= 2:
        raise LlamaSemanticError("Only batch sizes one and two are admitted")
    frozen = _freeze_checkpoint(checkpoint)
    p = frozen.profile
    capacity = min(p.context, 128)
    reserved = batch * p.layers * capacity * p.kv_heads * p.head_dim * 8
    if reserved > 64 * 1024 * 1024:
        raise LlamaSemanticError("KV reservation exceeds resource cap")
    hshape = (batch, "Q", p.hidden)
    qshape = (batch, "Q", p.heads, p.head_dim)
    kvshape = (batch, "Q", p.kv_heads, p.head_dim)
    inputs = [("tokens", ValueType((batch, "Q"), "int64", None))]
    inputs.extend(
        (name, ValueType(tuple(arr.shape), "float32", arr.nbytes))
        for name, arr in sorted(frozen.arrays.items())
    )
    states = tuple(
        (
            f"{kind}{layer}",
            ValueType(
                (batch, capacity, p.kv_heads, p.head_dim),
                "float32", batch * capacity * p.kv_heads * p.head_dim * 4,
            ),
        )
        for layer in range(p.layers) for kind in ("k", "v")
    )
    ops: list[SemanticOperation] = []
    next_effect = 0

    def emit(
        name: str, code: str, operands: tuple[str, ...],
        shape: tuple[int | str, ...] | None, *,
        cell: str | None = None, effect: str = "pure",
        parameters: tuple[tuple[str, float | int], ...] = (),
    ) -> str:
        nonlocal next_effect
        typed = ValueType(shape, "float32", None) if shape is not None else None
        old_token = new_token = None
        revision = None
        if effect != "pure":
            old_token = f"e{next_effect}"
            next_effect += 1
            new_token = f"e{next_effect}"
            revision = 1
        ops.append(SemanticOperation(
            name, "state" if effect != "pure" else "tensor",
            code, operands, name if shape is not None else None,
            typed, cell, effect, revision, old_token, new_token,
            parameters,
        ))
        return name

    hidden = emit(
        "embedding_out", "embedding",
        ("tokens", "model.embed_tokens.weight"), hshape,
    )
    for layer in range(p.layers):
        name = f"l{layer}"
        prefix = f"model.layers.{layer}."
        norm = emit(
            f"{name}_attn_norm", "rmsnorm_eps",
            (hidden, prefix + "input_layernorm.weight"), hshape,
            parameters=(("eps", p.eps),),
        )
        q = emit(
            f"{name}_q", "linear", (norm, prefix + "self_attn.q_proj.weight"), hshape,
        )
        q = emit(
            f"{name}_q_heads", "split_heads", (q,), qshape,
            parameters=(("heads", p.heads),),
        )
        k = emit(
            f"{name}_k", "linear", (norm, prefix + "self_attn.k_proj.weight"),
            (batch, "Q", p.kv_heads * p.head_dim),
        )
        k = emit(
            f"{name}_k_heads", "split_heads", (k,), kvshape,
            parameters=(("heads", p.kv_heads),),
        )
        v = emit(
            f"{name}_v", "linear", (norm, prefix + "self_attn.v_proj.weight"),
            (batch, "Q", p.kv_heads * p.head_dim),
        )
        v = emit(
            f"{name}_v_heads", "split_heads", (v,), kvshape,
            parameters=(("heads", p.kv_heads),),
        )
        q = emit(
            f"{name}_q_rotary", "rope_half", (q,), qshape,
            parameters=(("theta", p.rope_theta),),
        )
        k = emit(
            f"{name}_k_rotary", "rope_half", (k,), kvshape,
            parameters=(("theta", p.rope_theta),),
        )
        emit(
            f"{name}_write_k", "stage_key", (k,), None,
            cell=f"k{layer}", effect="write", parameters=(("layer", layer),),
        )
        emit(
            f"{name}_write_v", "stage_value", (v,), None,
            cell=f"v{layer}", effect="write", parameters=(("layer", layer),),
        )
        cache_k = emit(
            f"{name}_read_k", "read_cache", (f"k{layer}",),
            (batch, "T", p.kv_heads, p.head_dim),
            cell=f"k{layer}", effect="read",
        )
        cache_v = emit(
            f"{name}_read_v", "read_cache", (f"v{layer}",),
            (batch, "T", p.kv_heads, p.head_dim),
            cell=f"v{layer}", effect="read",
        )
        attn = emit(
            f"{name}_attention", "gqa_attention", (q, cache_k, cache_v), hshape,
        )
        proj = emit(
            f"{name}_out", "linear",
            (attn, prefix + "self_attn.o_proj.weight"), hshape,
        )
        residual = emit(f"{name}_residual", "add", (hidden, proj), hshape)
        ff_norm = emit(
            f"{name}_ff_norm", "rmsnorm_eps",
            (residual, prefix + "post_attention_layernorm.weight"), hshape,
            parameters=(("eps", p.eps),),
        )
        ff_shape = (batch, "Q", p.intermediate)
        gate = emit(
            f"{name}_gate", "linear",
            (ff_norm, prefix + "mlp.gate_proj.weight"), ff_shape,
        )
        silu = emit(f"{name}_silu", "silu", (gate,), ff_shape)
        up = emit(
            f"{name}_up", "linear",
            (ff_norm, prefix + "mlp.up_proj.weight"), ff_shape,
        )
        fused = emit(f"{name}_fused", "multiply", (silu, up), ff_shape)
        down = emit(
            f"{name}_down", "linear",
            (fused, prefix + "mlp.down_proj.weight"), hshape,
        )
        hidden = emit(f"{name}_hidden", "add", (residual, down), hshape)
    final = emit(
        "final_norm", "rmsnorm_eps", (hidden, "model.norm.weight"), hshape,
        parameters=(("eps", p.eps),),
    )
    head = (
        "model.embed_tokens.weight" if p.tied_head else "lm_head.weight"
    )
    emit("logits", "linear", (final, head), (batch, "Q", p.vocabulary))
    program = SemanticProgram(
        origin="effect-program",
        input_types=tuple(inputs),
        state_types=states,
        regions=(SemanticRegion("llama_decode", 1, tuple(ops)),),
        outputs=("logits",),
        source_sha256=_sha_document({
            "identity": _identity(frozen), "batch": batch,
            "capacity": capacity, "dialect": SCHEMA,
        }),
    )
    validate_semantic_program(program)
    return program


@dataclass(frozen=True)
class LlamaSemanticPlan:
    program: SemanticProgram
    weights: LlamaLocalCheckpoint
    batch: int
    capacity: int

    @property
    def semantic_sha256(self) -> str:
        return self.program.digest()

    def new_session(self, *, backend: Backend = "numpy") -> LlamaSemanticSession:
        admitted = lower_llama_semantic_program(
            self.program, self.weights, batch=self.batch
        )
        return LlamaSemanticSession(admitted, backend=backend)

    def explain(self) -> dict:
        ops = self.program.regions[0].operations
        return {
            "schema": SCHEMA, "semantic_sha256": self.semantic_sha256,
            "checkpoint_sha256": self.weights.checkpoint_sha256,
            "source_revision": self.weights.declared_revision,
            "backend_options": ("numpy", "torchscript_hybrid"),
            "operation_count": len(ops),
            "operator_histogram": dict(sorted(Counter(op.op for op in ops).items())),
            "cache_capacity": self.capacity,
            "limitations": (
                "TorchScript compiles only selected arithmetic primitives.",
                "No generic model family support or native compiled attention.",
            ),
        }


def lower_llama_semantic_program(
    program: SemanticProgram,
    weights: LlamaLocalCheckpoint, *, batch: int = 1
) -> LlamaSemanticPlan:
    """Rebuild the only admissible graph to reject forged effect/shape/operand lists."""
    if not isinstance(program, SemanticProgram):
        raise LlamaSemanticError("Expected a typed semantic program")
    try:
        validate_semantic_program(program)
        frozen = _freeze_checkpoint(weights)
        canonical = make_llama_semantic_program(frozen, batch=batch)
    except (ValueError, TypeError, AttributeError, OverflowError) as exc:
        raise LlamaSemanticError("Malformed decoder graph, weights, or dimensions") from exc
    if program != canonical:
        raise LlamaSemanticError("Noncanonical Llama IR, state, shape, or checkpoint")
    return LlamaSemanticPlan(program, frozen, batch, min(frozen.profile.context, 128))


class LlamaSemanticSession:
    """Dynamic chunk execution with transactional KV state and explicit SSA liveness."""

    def __init__(self, plan: LlamaSemanticPlan, *, backend: Backend = "numpy") -> None:
        if backend not in ("numpy", "torchscript_hybrid"):
            raise LlamaSemanticError("Unknown Llama execution backend")
        if plan.program != make_llama_semantic_program(plan.weights, batch=plan.batch):
            raise LlamaSemanticError("Forged Llama semantic execution plan")
        self.plan = plan
        self.backend = backend
        self.profile = plan.weights.profile
        self.batch = plan.batch
        self.vocabulary = self.profile.vocabulary
        self.cache = lower_kv_semantic_program(
            make_kv_semantic_program(
                batch=plan.batch, layers=self.profile.layers,
                kv_heads=self.profile.kv_heads, head_dim=self.profile.head_dim,
                capacity=plan.capacity,
            )
        ).new_session()
        p = self.profile
        inv = 1.0 / (
            p.rope_theta ** (
                np.arange(0, p.head_dim, 2, dtype=np.float64) / p.head_dim
            )
        )
        angles = np.outer(np.arange(plan.capacity, dtype=np.float64), inv)
        self._cos = np.concatenate((np.cos(angles), np.cos(angles)), axis=-1).astype(np.float32)
        self._sin = np.concatenate((np.sin(angles), np.sin(angles)), axis=-1).astype(np.float32)
        self._operations = plan.program.regions[0].operations
        self._uses = Counter(
            operand for op in self._operations for operand in op.operands
        )
        self._compiled = TorchScriptPrimitives() if backend == "torchscript_hybrid" else None
        self.compiled_calls = 0

    @property
    def cursor(self) -> int:
        return self.cache.cursor

    def _execute(
        self, op: SemanticOperation, env: dict[str, np.ndarray],
        pending: dict[int, np.ndarray], tx: object,
        start: int, total: int,
    ) -> np.ndarray | None:
        if op.op == "read_cache":
            return self.cache._arrays[op.state_cell][:, :total]
        args = tuple(env[name] for name in op.operands)
        code = op.op
        if code == "stage_key":
            pending[dict(op.parameters)["layer"]] = args[0]
            return None
        if code == "stage_value":
            layer = dict(op.parameters)["layer"]
            tx.stage(layer, pending.pop(layer), args[0])
            return None
        if code == "embedding":
            return args[1][args[0]]
        if code == "split_heads":
            heads = dict(op.parameters)["heads"]
            return args[0].reshape(
                self.batch, args[0].shape[1], heads, self.profile.head_dim
            )
        if code == "rope_half":
            tensor = args[0]
            length = tensor.shape[1]
            cosine = self._cos[start:start + length][None, :, None, :]
            sine = self._sin[start:start + length][None, :, None, :]
            half = self.profile.head_dim // 2
            rotated = np.concatenate(
                (-tensor[..., half:], tensor[..., :half]), axis=-1
            )
            return tensor * cosine + rotated * sine
        if code == "gqa_attention":
            q, k, v = args
            query = q.shape[1]
            p = self.profile
            replicas = p.heads // p.kv_heads
            grouped = q.reshape(
                self.batch, query, p.kv_heads, replicas, p.head_dim
            ).transpose(0, 2, 3, 1, 4)
            keys = k.transpose(0, 2, 1, 3)[:, :, None, :, :]
            score = np.matmul(grouped, np.swapaxes(keys, -1, -2)) * np.float32(
                1.0 / sqrt(p.head_dim)
            )
            positions = np.arange(start, total, dtype=np.int64)
            causal = np.arange(total)[None, :] <= positions[:, None]
            score = np.where(
                causal[None, None, None, :, :], score, np.float32(-1e9)
            )
            score = score - score.max(axis=-1, keepdims=True)
            np.exp(score, out=score)
            score = score / score.sum(axis=-1, keepdims=True)
            values = v.transpose(0, 2, 1, 3)[:, :, None, :, :]
            return np.matmul(score, values).transpose(
                0, 3, 1, 2, 4
            ).reshape(self.batch, query, p.hidden)
        if self._compiled is not None and code in ("linear", "add", "multiply", "silu"):
            kind = "matmul" if code == "linear" else code
            operands = (args[0], args[1].T) if code == "linear" else args
            value, _ = self._compiled.run(kind, *operands)
            self.compiled_calls += 1
            return value
        if code == "linear":
            return args[0] @ args[1].T
        if code == "rmsnorm_eps":
            inp, weight = args
            eps = dict(op.parameters)["eps"]
            return (
                inp / np.sqrt(np.mean(inp * inp, axis=-1, keepdims=True) + eps)
            ) * weight
        if code == "silu":
            return args[0] / (1.0 + np.exp(-args[0]))
        if code == "add":
            return args[0] + args[1]
        if code == "multiply":
            return args[0] * args[1]
        raise LlamaSemanticError("Unsupported Llama IR operation: " + code)

    def step(self, tokens: np.ndarray) -> np.ndarray:
        if (
            not isinstance(tokens, np.ndarray) or tokens.dtype != np.int64
            or tokens.ndim != 2 or tokens.shape[0] != self.batch
            or not 1 <= tokens.shape[1] <= 32
            or self.cursor + tokens.shape[1] > self.plan.capacity
            or np.any(tokens < 0) or np.any(tokens >= self.vocabulary)
        ):
            raise LlamaSemanticError("Token, batch, vocabulary or context out of bounds")
        start, query = self.cursor, tokens.shape[1]
        total = start + query
        env = {"tokens": tokens, **self.plan.weights.arrays}
        uses = self._uses.copy()
        pending: dict[int, np.ndarray] = {}
        with self.cache.begin_append(query) as tx:
            for op in self._operations:
                result = self._execute(op, env, pending, tx, start, total)
                if op.result is not None:
                    if result is None or op.result_type is None:
                        raise LlamaSemanticError("Missing output at " + op.name)
                    expected = tuple(
                        query if d == "Q" else total if d == "T" else d
                        for d in op.result_type.shape
                    )
                    if (
                        not isinstance(result, np.ndarray)
                        or result.shape != expected
                        or result.dtype != np.float32
                        or not np.isfinite(result).all()
                    ):
                        raise LlamaSemanticError("Bad result shape/dtype at " + op.name)
                    env[op.result] = result
                elif result is not None:
                    raise LlamaSemanticError("Unexpected effect output at " + op.name)
                for operand in op.operands:
                    uses[operand] -= 1
                    if uses[operand] == 0 and operand != "logits":
                        env.pop(operand, None)
            if pending or "logits" not in env:
                raise LlamaSemanticError("Incomplete Llama semantic program")
            logits = env["logits"].copy()
            tx.commit()
            return logits

    def snapshot(self) -> StackState:
        snap = self.cache.snapshot()
        values = dict(snap.buffers)
        return StackState(tuple(
            KVState(values[f"k{layer}"], values[f"v{layer}"], self.plan.capacity)
            for layer in range(self.profile.layers)
        ))

    def restore(self, state: StackState) -> None:
        if not isinstance(state, StackState) or len(state.layers) != self.profile.layers:
            raise LlamaSemanticError("Wrong number of Llama KV layers")
        first = state.layers[0]
        if not isinstance(first, KVState) or not isinstance(first.keys, np.ndarray):
            raise LlamaSemanticError("Malformed Llama KV snapshot")
        if first.keys.ndim != 4:
            raise LlamaSemanticError("Wrong Llama KV dimensions")
        length = first.keys.shape[1]
        pairs = []
        for layer, cell in enumerate(state.layers):
            if not isinstance(cell, KVState) or cell.max_context != self.plan.capacity:
                raise LlamaSemanticError("Incompatible Llama cache format")
            pairs.extend(((f"k{layer}", cell.keys), (f"v{layer}", cell.values)))
        self.cache.restore(
            KVSnapshot(self.cache.plan.semantic_sha256, length, tuple(pairs))
        )

    def reset(self) -> None:
        self.cache.reset()

    def generate(self, prompt: np.ndarray, *, steps: int) -> tuple[np.ndarray, StackState]:
        if type(steps) is not int or not 1 <= steps <= 24:
            raise LlamaSemanticError("Generation step count out of range")
        if not isinstance(prompt, np.ndarray) or prompt.ndim != 2:
            raise LlamaSemanticError("Generation prompt must be [batch,tokens]")
        if prompt.shape[1] + steps > self.plan.capacity:
            raise LlamaSemanticError("Generation exceeds admitted context")
        self.reset()
        logits = self.step(prompt)
        produced = []
        for _ in range(steps):
            next_id = np.argmax(logits[:, -1], axis=-1).astype(np.int64)[:, None]
            produced.append(next_id)
            logits = self.step(next_id)
        return np.concatenate(produced, axis=1), self.snapshot()

    def report(self) -> dict:
        return {
            **self.cache.report(),
            "schema": SCHEMA,
            "backend": self.backend,
            "semantic_sha256": self.plan.semantic_sha256,
            "checkpoint_sha256": self.plan.weights.checkpoint_sha256,
            "compiled_primitive_calls": self.compiled_calls,
            "compiled_primitive_specializations": (
                self._compiled.specializations if self._compiled else 0
            ),
            "limitations": (
                "TorchScript hybrid compiles selected arithmetic only, not KV/attention.",
                "No GPU, arbitrary model, or physical-device measurement.",
            ),
        }
