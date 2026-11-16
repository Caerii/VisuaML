"""Bounded, resident-KV streaming CPU adapter for the VisuaML v14 decoder.

This is an actual executable NumPy software runtime, NOT a compiled NumPy-AOT
program or an accelerator backend. Unlike v14's shape-specialized functional
AOT interface, this implementation preallocates immutable-identity KV storage
and executes different admissible token lengths without recompilation.

Correctness is established by independent v14 and NumPy-AOT differential tests,
not algebraic proof. It is NOT thread-safe and does not accept untrusted Python
source or model files. The checkpoint is snapshotted at construction.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

import numpy as np

from .family_blocks_v12 import DecoderWeights, KVState
from .stacked_decoder_v14 import (
    StackContractError,
    StackedWeights,
    StackState,
    TinyDecoderWeights,
    checkpoint_digest,
)


@dataclass(frozen=True)
class ResidentCacheReport:
    schema: str
    batch: int
    layers: int
    capacity_tokens: int
    live_tokens: int
    reserved_bytes: int
    live_bytes: int
    checkpoint_sha256: str
    runtime: str = "NumPy CPU resident KV adapter; no accelerator, no AOT dispatch"

    def as_dict(self) -> dict:
        from dataclasses import asdict
        return asdict(self)


def _readonly_copy(arr: np.ndarray) -> np.ndarray:
    if not isinstance(arr, np.ndarray) or arr.dtype != np.float32 or not np.isfinite(arr).all():
        raise StackContractError("Streaming decoder requires finite float32 checkpoint tensors")
    source = np.ascontiguousarray(arr)
    # A bytes-backed view stays unwriteable even if a caller attempts to
    # re-enable WRITEABLE on the ndarray. It does not alias caller memory.
    return np.frombuffer(source.tobytes(), dtype=source.dtype).reshape(source.shape)


def _snapshot_weights(weights: StackedWeights) -> StackedWeights:
    weights.validate()
    immutable = {name: _readonly_copy(value) for name, value in weights.arrays().items()}
    blocks: list[TinyDecoderWeights] = []
    for idx in range(len(weights.blocks)):
        prefix = f"layer{idx}."
        attention = DecoderWeights(
            *(immutable[prefix + "attention." + name] for name in ("q", "k", "v", "out", "norm"))
        )
        blocks.append(
            TinyDecoderWeights(
                immutable["embedding"], attention,
                immutable[prefix + "final_norm"],
                immutable[prefix + "gate"], immutable[prefix + "up"],
                immutable[prefix + "down"], immutable["lm_head"],
            )
        )
    frozen = StackedWeights(
        immutable["embedding"], tuple(blocks), immutable["lm_head"],
        weights.heads, weights.max_context,
    )
    frozen.validate()
    return frozen


class StreamingStackedDecoder:
    """A single-session, bounded streaming CPU adapter with explicit KV state.

    `step` writes *only* at indices >= the previously committed cache cursor.
    On an error the cursor and all previous cache values remain unchanged.
    Caller-provided weights are immutable-snapshotted; `snapshot()` makes a
    detached read-only state copy for parity and checkpoint handoff.
    """

    def __init__(self, weights: StackedWeights, *, batch: int = 1) -> None:
        if type(batch) is not int or not 1 <= batch <= 8:
            raise StackContractError("Streaming batch must be in [1,8]")
        self._weights = _snapshot_weights(weights)
        self.checkpoint_sha256 = checkpoint_digest(self._weights)
        vocab, hidden, depth, kv_heads = self._weights.validate()
        self.batch = batch
        self.vocab = vocab
        self.hidden = hidden
        self.depth = depth
        self.kv_heads = kv_heads
        self.heads = self._weights.heads
        self.capacity = self._weights.max_context
        self.cursor = 0
        self._keys = tuple(
            np.empty((batch, self.capacity, kv_heads, depth), dtype=np.float32)
            for _ in self._weights.blocks
        )
        self._values = tuple(np.empty_like(key) for key in self._keys)
        # float64 angle formation then float32 trigonometry, as in the v12
        # independent NumPy reference. This table is shared by all blocks.
        inv = 1.0 / (
            10000.0 ** (np.arange(0, depth, 2, dtype=np.float64) / depth)
        )
        theta = np.outer(np.arange(self.capacity, dtype=np.float64), inv)
        self._cos = np.cos(theta).astype(np.float32)
        self._sin = np.sin(theta).astype(np.float32)

    def report(self) -> ResidentCacheReport:
        byte_per_token = sum(k.strides[1] * 2 * self.batch for k in self._keys)
        return ResidentCacheReport(
            "visuaml.resident-kv.v16", self.batch, len(self._keys),
            self.capacity, self.cursor,
            sum(k.nbytes + v.nbytes for k, v in zip(self._keys, self._values)),
            self.cursor * byte_per_token,
            self.checkpoint_sha256,
        )

    def reset(self) -> None:
        """Start a new sequence; clear old values so stale data is inaccessible."""
        for key, value in zip(self._keys, self._values):
            key.fill(0)
            value.fill(0)
        self.cursor = 0

    def snapshot(self) -> StackState:
        def detached(value: np.ndarray) -> np.ndarray:
            result = value[:, :self.cursor].copy()
            result.setflags(write=False)
            return result
        return StackState(tuple(
            KVState(detached(k), detached(v), self.capacity)
            for k, v in zip(self._keys, self._values)
        ))

    def restore(self, state: StackState) -> None:
        """Atomically restore a validated, externally supplied fixed-shape state."""
        if not isinstance(state, StackState) or len(state.layers) != len(self._keys):
            raise StackContractError("Streaming restore requires one KV pair per layer")
        length: int | None = None
        for cache in state.layers:
            if not isinstance(cache, KVState) or cache.max_context != self.capacity:
                raise StackContractError("Incompatible streaming KV checkpoint")
            if length is None:
                if not isinstance(cache.keys, np.ndarray) or cache.keys.ndim != 4:
                    raise StackContractError("Malformed streaming key tensor")
                length = cache.keys.shape[1]
            expected = (self.batch, length, self.kv_heads, self.depth)
            if length > self.capacity or any(
                not isinstance(a, np.ndarray) or a.shape != expected
                or a.dtype != np.float32 or not np.isfinite(a).all()
                for a in (cache.keys, cache.values)
            ):
                raise StackContractError("Streaming restore KV shape/dtype/finiteness mismatch")
        # Only publish after ALL layers are validated. Snapshot inputs first
        # because callers may supply views into resident storage.
        copies = tuple((cache.keys.copy(), cache.values.copy()) for cache in state.layers)
        self.reset()
        length = length or 0
        for (new_keys, new_values), old_keys, old_values in zip(copies, self._keys, self._values):
            old_keys[:, :length] = new_keys
            old_values[:, :length] = new_values
        self.cursor = length

    def _rope(self, tensor: np.ndarray, start: int) -> np.ndarray:
        n = tensor.shape[1]
        c = self._cos[start:start+n][None, :, None, :]
        s = self._sin[start:start+n][None, :, None, :]
        transformed = np.empty_like(tensor)
        even, odd = tensor[..., ::2], tensor[..., 1::2]
        transformed[..., ::2] = even * c - odd * s
        transformed[..., 1::2] = even * s + odd * c
        return transformed

    def step(self, tokens: np.ndarray) -> np.ndarray:
        """Execute one chunk; commit the cache cursor only after all blocks succeed."""
        if (
            not isinstance(tokens, np.ndarray) or tokens.dtype != np.int64
            or tokens.ndim != 2 or tokens.shape[0] != self.batch
            or not 1 <= tokens.shape[1] <= 32
            or self.cursor + tokens.shape[1] > self.capacity
            or np.any(tokens < 0) or np.any(tokens >= self.vocab)
        ):
            raise StackContractError("Streaming token IDs or context outside admitted domain")
        start, query = self.cursor, tokens.shape[1]
        total = start + query
        h = self._weights.embedding[tokens]
        positions = np.arange(start, total, dtype=np.int64)
        causal = np.arange(total)[None, :] <= positions[:, None]
        scale = np.float32(1.0 / sqrt(self.depth))
        replicas = self.heads // self.kv_heads
        try:
            for layer, block in enumerate(self._weights.blocks):
                a = block.attention
                normed = (h / np.sqrt(np.mean(h * h, axis=-1, keepdims=True) + 1e-5)) * a.norm
                q = self._rope((normed @ a.q).reshape(self.batch, query, self.heads, self.depth), start)
                k = self._rope((normed @ a.k).reshape(self.batch, query, self.kv_heads, self.depth), start)
                v = (normed @ a.v).reshape(self.batch, query, self.kv_heads, self.depth)
                self._keys[layer][:, start:total] = k
                self._values[layer][:, start:total] = v
                # Grouped-query attention via broadcasting: never materialize
                # repeated KV heads. Q: [B,KV,R,Q,D], K/V: [B,KV,1,T,D].
                grouped_q = q.reshape(
                    self.batch, query, self.kv_heads, replicas, self.depth
                ).transpose(0, 2, 3, 1, 4)
                keys = self._keys[layer][:, :total].transpose(0, 2, 1, 3)[:, :, None, :, :]
                scores = np.matmul(grouped_q, np.swapaxes(keys, -1, -2)) * scale
                scores = np.where(causal[None, None, None, :, :], scores, np.float32(-1e9))
                np.subtract(scores, scores.max(axis=-1, keepdims=True), out=scores)
                np.exp(scores, out=scores)
                scores /= scores.sum(axis=-1, keepdims=True)
                values = self._values[layer][:, :total].transpose(0, 2, 1, 3)[:, :, None, :, :]
                context = np.matmul(scores, values).transpose(0, 3, 1, 2, 4).reshape(
                    self.batch, query, self.hidden
                )
                residual = h + context @ a.out
                normal = (
                    residual / np.sqrt(np.mean(residual * residual, axis=-1, keepdims=True) + 1e-5)
                ) * block.final_norm
                gate = normal @ block.gate
                silu = gate / (1.0 + np.exp(-gate))
                h = (residual + (silu * (normal @ block.up)) @ block.down).astype(np.float32)
            logits = (h @ self._weights.lm_head).astype(np.float32)
            if not np.isfinite(logits).all():
                raise StackContractError("Nonfinite streaming logits")
        except Exception:
            # Old cache prefix remains untouched; new entries have not been
            # published via the cursor. Clear scratch suffix for hygiene.
            for key, value in zip(self._keys, self._values):
                key[:, start:total] = 0
                value[:, start:total] = 0
            raise
        self.cursor = total
        return logits.copy()

    def generate(self, prompt: np.ndarray, *, steps: int) -> tuple[np.ndarray, StackState]:
        if type(steps) is not int or not 1 <= steps <= 24:
            raise StackContractError("Generation step count must be in [1,24]")
        if not isinstance(prompt, np.ndarray) or prompt.ndim != 2 or prompt.dtype != np.int64:
            raise StackContractError("Generation prompt must be int64 [batch, tokens]")
        if prompt.shape[1] + steps > self.capacity:
            raise StackContractError("Generation context budget exceeded")
        self.reset()
        logits = self.step(prompt)
        produced = []
        for _ in range(steps):
            ids = np.argmax(logits[:, -1], axis=-1).astype(np.int64)[:, None]
            produced.append(ids)
            logits = self.step(ids)
        return np.concatenate(produced, axis=1), self.snapshot()
