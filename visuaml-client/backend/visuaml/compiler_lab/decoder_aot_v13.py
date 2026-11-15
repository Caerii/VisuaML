"""Shape-specialized portable NumPy-AOT for the v12 tiny GQA decoder.

A compiled chunk consumes token IDs and immutable KV tensors and produces
(logits, next_keys, next_values). No Python callbacks run inside the artifact.
Each (batch, query length, cache length) is a different *concrete* program;
cache capacity and token bounds remain explicit at the driver boundary.
This is a CPU-only synthetic-fixture compiler, not a Llama checkpoint loader.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from torch import nn

from visuaml.compiler_spike import capture

from .aot_compile import AOTArtifact, compile_numpy_aot
from .family_blocks_v12 import KVState, empty_cache
from .tiny_decoder_v12 import TinyDecoderWeights, decoder_tokens_numpy, decoder_tokens_torch


class DecoderCompileError(ValueError):
    """A request is outside the tiny decoder's certified fixed-shape dialect."""


class _DecoderChunk(nn.Module):
    def __init__(self, w: TinyDecoderWeights, *, batch: int, query: int, past: int, heads: int):
        super().__init__()
        hidden = int(w.embedding.shape[1])
        if (type(batch) is not int or type(query) is not int or type(past) is not int
            or type(heads) is not int or not 1 <= batch <= 8 or not 1 <= query <= 32
            or not 0 <= past <= 64 or past + query > 64 or hidden % heads):
            raise DecoderCompileError("Invalid concrete decoder specialization")
        depth = hidden // heads
        kv_heads = int(w.attention.k.shape[1]) // depth
        if depth % 2 or not (1 <= kv_heads <= heads and heads % kv_heads == 0):
            raise DecoderCompileError("Unsupported rotary/grouped-head topology")
        self.batch, self.query, self.past, self.heads = batch, query, past, heads
        self.hidden, self.depth, self.kv_heads = hidden, depth, kv_heads
        # All weights are finite float32 buffers, NOT executable Python code.
        for name, a in {
            'embedding': w.embedding, 'norm_attn': w.attention.norm,
            'q': w.attention.q, 'k': w.attention.k, 'v': w.attention.v,
            'out': w.attention.out, 'norm_final': w.final_norm,
            'gate': w.gate, 'up': w.up, 'down': w.down, 'lm_head': w.lm_head,
        }.items():
            if not isinstance(a, np.ndarray) or a.dtype != np.float32 or not np.isfinite(a).all():
                raise DecoderCompileError(f"Invalid finite float32 weight: {name}")
            self.register_buffer(name, torch.tensor(a.copy(), dtype=torch.float32))
        positions = np.arange(past, past + query, dtype=np.float64)
        inv = 1.0 / (10000.0 ** (np.arange(0, depth, 2, dtype=np.float64) / depth))
        theta = positions[:, None] * inv[None, :]
        self.register_buffer('rope_cos', torch.tensor(np.cos(theta)[None, :, None, :].astype(np.float32)))
        self.register_buffer('rope_sin', torch.tensor(np.sin(theta)[None, :, None, :].astype(np.float32)))
        allowed = np.arange(past + query)[None, :] <= positions[:, None]
        bias = np.where(allowed, 0.0, -1e9).astype(np.float32)[None, None, :, :]
        self.register_buffer('causal_bias', torch.tensor(bias))
        self.inv_depth = float(depth ** -0.5)

    def _rope(self, t: torch.Tensor, *, kv: bool) -> torch.Tensor:
        count = self.kv_heads if kv else self.heads
        even, odd = t[:, :, :, ::2], t[:, :, :, 1::2]
        re = even * self.rope_cos - odd * self.rope_sin
        im = even * self.rope_sin + odd * self.rope_cos
        return torch.stack((re, im), dim=-1).reshape(self.batch, self.query, count, self.depth)

    def forward(self, token_ids: torch.Tensor, prev_k: torch.Tensor, prev_v: torch.Tensor):
        x = self.embedding[token_ids]
        normed = x * torch.rsqrt(torch.sum(x * x, dim=-1, keepdim=True) / self.hidden + 1e-5) * self.norm_attn
        q = self._rope(torch.matmul(normed, self.q).reshape(self.batch, self.query, self.heads, self.depth), kv=False)
        k = self._rope(torch.matmul(normed, self.k).reshape(self.batch, self.query, self.kv_heads, self.depth), kv=True)
        v = torch.matmul(normed, self.v).reshape(self.batch, self.query, self.kv_heads, self.depth)
        next_k = torch.cat((prev_k, k), dim=1)
        next_v = torch.cat((prev_v, v), dim=1)
        ks = torch.repeat_interleave(next_k, self.heads // self.kv_heads, dim=2).permute(0, 2, 1, 3)
        vs = torch.repeat_interleave(next_v, self.heads // self.kv_heads, dim=2).permute(0, 2, 1, 3)
        scores = torch.matmul(q.permute(0, 2, 1, 3), ks.transpose(-1, -2)) * self.inv_depth + self.causal_bias
        probs = torch.softmax(scores, dim=-1)
        attn = torch.matmul(probs, vs).permute(0, 2, 1, 3).reshape(self.batch, self.query, self.hidden)
        residual = x + torch.matmul(attn, self.out)
        normalized = residual * torch.rsqrt(torch.sum(residual * residual, dim=-1, keepdim=True) / self.hidden + 1e-5) * self.norm_final
        gate = torch.matmul(normalized, self.gate)
        silu = gate / (1.0 + torch.exp(gate * -1.0))
        mlp = torch.matmul(silu * torch.matmul(normalized, self.up), self.down)
        logits = torch.matmul(residual + mlp, self.lm_head)
        return logits, next_k, next_v


def compile_decoder_chunk(
    weights: TinyDecoderWeights, *, batch: int, query: int, past: int, heads: int = 4
) -> AOTArtifact:
    model = _DecoderChunk(weights, batch=batch, query=query, past=past, heads=heads).eval()
    tokens = torch.zeros((batch, query), dtype=torch.int64)
    cache = torch.zeros((batch, past, model.kv_heads, model.depth), dtype=torch.float32)
    captured = capture(model, tokens, cache, cache.clone())
    artifact = compile_numpy_aot(captured)
    artifact.manifest['source'] = 'v13 shape-specialized tiny-decoder prefill/decode'
    artifact.manifest['input_constraints'] = {
        artifact.manifest['inputs'][0]['name']: {
            'min_inclusive': 0,
            'max_exclusive': int(weights.embedding.shape[0]),
        }
    }
    artifact.manifest['numeric_contract'] = (
        'Concrete [batch,query,past]; immutable float32 KV inputs and outputs; '
        'token range checked by portable manifest and host driver; witness-limited'
    )
    import json, hashlib
    unsigned = {k: v for k, v in artifact.manifest.items() if k != 'manifest_sha256'}
    payload = json.dumps(unsigned, sort_keys=True, separators=(',', ':'), allow_nan=False)
    artifact.manifest['manifest_sha256'] = hashlib.sha256(payload.encode()).hexdigest()
    return artifact


@dataclass
class CompiledTinyDecoder:
    """Bounded shape-specialized AOT driver; caches compiled shapes, not KV state."""
    weights: TinyDecoderWeights
    heads: int = 4
    max_context: int = 64
    _compiled: dict[tuple[int, int, int], AOTArtifact] = field(default_factory=dict)

    def step(self, tokens: np.ndarray, state: KVState) -> tuple[np.ndarray, KVState]:
        if not isinstance(tokens, np.ndarray) or tokens.ndim != 2 or tokens.dtype != np.int64:
            raise DecoderCompileError('Token IDs must be int64 [batch, query]')
        batch, query = tokens.shape
        if (not 1 <= batch <= 8 or not 1 <= query <= 32
            or np.any(tokens < 0) or np.any(tokens >= self.weights.embedding.shape[0])):
            raise DecoderCompileError('Token IDs outside vocabulary or request budget')
        depth = self.weights.embedding.shape[1] // self.heads
        kv = self.weights.attention.k.shape[1] // depth
        if (not isinstance(state, KVState) or state.max_context != self.max_context
            or state.keys.dtype != np.float32 or state.values.dtype != np.float32
            or state.keys.shape != state.values.shape
            or state.keys.ndim != 4 or state.keys.shape[0] != batch
            or state.keys.shape[2:] != (kv, depth)
            or not np.isfinite(state.keys).all() or not np.isfinite(state.values).all()):
            raise DecoderCompileError('Invalid typed KV-cache state')
        past = state.keys.shape[1]
        if past + query > self.max_context:
            raise DecoderCompileError('KV cache exceeds context capacity')
        key = (batch, query, past)
        if key not in self._compiled:
            if len(self._compiled) >= 64:
                raise DecoderCompileError('Too many shape-specialized artifacts')
            self._compiled[key] = compile_decoder_chunk(self.weights, batch=batch, query=query, past=past, heads=self.heads)
        logits, nk, nv = self._compiled[key].run(tokens, state.keys, state.values)
        if (logits.shape != (batch, query, self.weights.embedding.shape[0])
            or nk.shape != (batch, past+query, kv, depth) or nv.shape != nk.shape
            or not np.isfinite(logits).all() or not np.isfinite(nk).all()
            or not np.isfinite(nv).all()):
            raise DecoderCompileError('Compiled artifact violated result contract')
        return logits, KVState(nk.copy(), nv.copy(), self.max_context)

    def generate(self, tokens: np.ndarray, *, steps: int) -> tuple[np.ndarray, KVState]:
        if type(steps) is not int or not 1 <= steps <= 32:
            raise DecoderCompileError('Generation steps must be in [1,32]')
        if not isinstance(tokens, np.ndarray) or tokens.ndim != 2 or tokens.shape[1] + steps > self.max_context:
            raise DecoderCompileError('Generation exceeds declared cache capacity')
        depth = self.weights.embedding.shape[1] // self.heads
        kv = self.weights.attention.k.shape[1] // depth
        state = empty_cache(tokens.shape[0], kv, depth, self.max_context)
        logits, state = self.step(tokens, state)
        produced=[]
        for _ in range(steps):
            next_id=np.argmax(logits[:, -1, :], axis=-1).astype(np.int64)[:, None]
            produced.append(next_id)
            logits, state=self.step(next_id, state)
        return np.concatenate(produced, axis=1), state


def verify_decoder_artifact(weights: TinyDecoderWeights, tokens: np.ndarray, chunks: tuple[int, ...]) -> dict[str, Any]:
    """Independent NumPy/PyTorch witness including every incremental KV tensor."""
    if sum(chunks) != tokens.shape[1] or any(type(n) is not int or n <= 0 for n in chunks):
        raise DecoderCompileError('Chunks must partition token input')
    batch=tokens.shape[0]
    depth=weights.embedding.shape[1]//4
    kv=weights.attention.k.shape[1]//depth
    a=CompiledTinyDecoder(weights)
    ca=empty_cache(batch,kv,depth,a.max_context)
    nr=empty_cache(batch,kv,depth,a.max_context)
    tr=empty_cache(batch,kv,depth,a.max_context)
    count=0; worst=0.0; offset=0
    for step in chunks:
        window=tokens[:, offset:offset+step]
        observed, ca=a.step(window, ca)
        n, nr=decoder_tokens_numpy(window, weights, nr)
        t, tr=decoder_tokens_torch(window, weights, tr)
        for left,right in ((observed,n),(observed,t),(ca.keys,nr.keys),(ca.keys,tr.keys),(ca.values,nr.values),(ca.values,tr.values)):
            np.testing.assert_allclose(left, right, atol=4e-5, rtol=4e-5)
            worst=max(worst,float(np.max(np.abs(left.astype(np.float64)-right.astype(np.float64)))))
            count+=1
        offset+=step
    return {'schema':'visuaml.tiny-decoder-aot-witness.v13','passed':True,
            'chunks':len(chunks),'arrays_checked':count,'max_abs_error':worst,
            'specializations':len(a._compiled),'backend':'numpy-aot CPU; no physical accelerator'}
