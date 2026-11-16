"""Bounded multi-layer, shape-specialized decoder compiler with checkpoint ABI.

This is a numerical research fixture, not a pretrained model importer.  The
optional on-disk checkpoint is a canonical NumPy weight set, not a promise to
accept arbitrary Hugging Face / GPT-NeoX / Llama formats.  AOT artifacts run
through the closed-world NumPy interpreter, independently of PyTorch.
"""

from __future__ import annotations

import hashlib
import json
import operator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from zipfile import BadZipFile, ZipFile

import numpy as np
import torch
from torch import nn
from torch.fx import Graph, GraphModule, symbolic_trace
from torch.fx.passes.shape_prop import ShapeProp

from visuaml.compiler_spike import CapturedProgram, inspect_fx

from .aot_compile import AOTArtifact, compile_numpy_aot
from .decoder_aot_v13 import _DecoderChunk
from .family_blocks_v12 import KVState, decoder_step_numpy, empty_cache
from .numpy_runtime import PreparedNumpyExecutor, array_digest, prepare_numpy_aot
from .tiny_decoder_v12 import TinyDecoderWeights


class StackContractError(ValueError):
    """Checkpoint, state, or numerical execution was outside the admitted subset."""


SCHEMA = "visuaml.stacked-decoder-checkpoint.v14"


@dataclass(frozen=True)
class StackedWeights:
    """Shared embedding/head with 1..4 GQA/RoPE/SwiGLU residual blocks."""

    embedding: np.ndarray
    blocks: tuple[TinyDecoderWeights, ...]
    lm_head: np.ndarray
    heads: int = 4
    max_context: int = 64

    @classmethod
    def seeded(cls, seed: int = 19, *, layers: int = 2) -> StackedWeights:
        if type(seed) is not int or not 0 <= seed <= 100_000:
            raise StackContractError("Seed must be a bounded integer")
        if type(layers) is not int or not 1 <= layers <= 4:
            raise StackContractError("Layer count must be in [1,4]")
        blocks = tuple(TinyDecoderWeights.seeded(seed=seed + i * 100_001) for i in range(layers))
        first = blocks[0]
        w = cls(first.embedding.copy(), blocks, first.lm_head.copy())
        w.validate()
        return w

    def arrays(self) -> dict[str, np.ndarray]:
        out = {"embedding": self.embedding, "lm_head": self.lm_head}
        for i, block in enumerate(self.blocks):
            for name in ("q", "k", "v", "out", "norm"):
                out[f"layer{i}.attention.{name}"] = getattr(block.attention, name)
            for name in ("final_norm", "gate", "up", "down"):
                out[f"layer{i}.{name}"] = getattr(block, name)
        return out

    def validate(self) -> tuple[int, int, int, int]:
        if not 1 <= len(self.blocks) <= 4 or type(self.heads) is not int or not 1 <= self.heads <= 16:
            raise StackContractError("1..4 residual blocks and bounded heads required")
        if type(self.max_context) is not int or not 1 <= self.max_context <= 64:
            raise StackContractError("Context limit must be in [1,64]")
        if not isinstance(self.embedding, np.ndarray) or self.embedding.ndim != 2:
            raise StackContractError("Embedding must be a matrix")
        vocab, hidden = self.embedding.shape
        if not (2 <= vocab <= 128 and 2 <= hidden <= 128 and hidden % self.heads == 0):
            raise StackContractError("Unsupported vocab/hidden/head dimensions")
        dim = hidden // self.heads
        if dim % 2 or self.lm_head.shape != (hidden, vocab):
            raise StackContractError("Rotary dimensions or vocabulary head invalid")
        kv_heads = -1
        for block in self.blocks:
            if not isinstance(block, TinyDecoderWeights):
                raise StackContractError("Every block must have the declared decoder layout")
            attention = block.attention
            if not all(isinstance(getattr(attention, attr), np.ndarray)
                       for attr in ("q", "k", "v", "out", "norm")):
                raise StackContractError("Attention projections must be arrays")
            this_kv = attention.k.shape[1] // dim if attention.k.ndim == 2 else -1
            if (
                attention.q.shape != (hidden, hidden)
                or attention.k.shape != (hidden, this_kv * dim)
                or attention.v.shape != attention.k.shape
                or attention.out.shape != (hidden, hidden)
                or attention.norm.shape != (hidden,)
                or block.final_norm.shape != (hidden,)
                or block.gate.ndim != 2
                or block.gate.shape[0] != hidden
                or block.up.shape != block.gate.shape
                or block.down.shape != (block.gate.shape[1], hidden)
                or not 1 <= this_kv <= self.heads
                or self.heads % this_kv
            ):
                raise StackContractError("Incompatible block matrix/head dimensions")
            if kv_heads >= 0 and this_kv != kv_heads:
                raise StackContractError("Stacked ABI requires a shared KV head count")
            kv_heads = this_kv
        arrays = self.arrays()
        if (
            any(not isinstance(a, np.ndarray) or a.dtype != np.float32 or not np.isfinite(a).all()
                for a in arrays.values())
            or sum(a.nbytes for a in arrays.values()) > 8 * 1024 * 1024
        ):
            raise StackContractError("Weights must be finite float32 under the allocation limit")
        return vocab, hidden, dim, kv_heads


def checkpoint_digest(weights: StackedWeights) -> str:
    weights.validate()
    payload = {
        "schema": SCHEMA,
        "heads": weights.heads,
        "max_context": weights.max_context,
        "layers": len(weights.blocks),
        "arrays": {k: {"shape": list(a.shape), "dtype": str(a.dtype), "sha256": array_digest(a)}
                   for k, a in sorted(weights.arrays().items())},
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def write_canonical_checkpoint(weights: StackedWeights, directory: str | Path, *, provenance: str) -> Path:
    """Store a strict, non-pickled canonical checkpoint plus content manifest."""
    weights.validate()
    if not isinstance(provenance, str) or not provenance.strip() or len(provenance) > 512:
        raise StackContractError("Checkpoint provenance must be declared")
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(root / "weights.npz", **weights.arrays())
    document = {
        "schema": SCHEMA,
        "source_kind": "user-declared-canonical-weights",
        "provenance": provenance,
        "heads": weights.heads,
        "max_context": weights.max_context,
        "layers": len(weights.blocks),
        "weight_sha256": checkpoint_digest(weights),
    }
    (root / "checkpoint.json").write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    return root


def load_canonical_checkpoint(directory: str | Path) -> StackedWeights:
    """Load only the canonical two-layer schema; never evaluate checkpoint code."""
    root = Path(directory)
    metadata = root / "checkpoint.json"
    archive = root / "weights.npz"
    if metadata.stat().st_size > 4096 or archive.stat().st_size > 16 * 1024 * 1024:
        raise StackContractError("Checkpoint metadata or archive exceeds byte budget")
    doc = json.loads(metadata.read_text(encoding="utf-8"))
    if not isinstance(doc, dict):
        raise StackContractError("Checkpoint metadata must be an object")
    if set(doc) != {"schema", "source_kind", "provenance", "heads", "max_context", "layers", "weight_sha256"} or doc["schema"] != SCHEMA:
        raise StackContractError("Unknown or malformed checkpoint schema")
    if doc["source_kind"] != "user-declared-canonical-weights" or not isinstance(doc["provenance"], str) or not doc["provenance"].strip():
        raise StackContractError("Checkpoint provenance missing or incompatible")
    try:
        with ZipFile(archive) as zf:
            members = zf.infolist()
            if (not 11 <= len(members) <= 38 or sum(x.file_size for x in members) > 8 * 1024 * 1024
                or len({x.filename for x in members}) != len(members)
                or any(x.flag_bits & 1 or not x.filename.endswith(".npy") for x in members)):
                raise StackContractError("Malformed or oversized weight archive")
    except BadZipFile as exc:
        raise StackContractError("Invalid NPZ archive") from exc
    with np.load(archive, allow_pickle=False) as file:
        arrays = {name: file[name] for name in file.files}
    # The keys below are a closed schema, even though the arrays were stored in a ZIP.
    from .family_blocks_v12 import DecoderWeights

    if type(doc["layers"]) is not int or not 1 <= doc["layers"] <= 4:
        raise StackContractError("Invalid declared layer count")
    expected_names = {"embedding", "lm_head"}
    for i in range(doc["layers"]):
        expected_names.update(f"layer{i}.attention.{name}" for name in ("q", "k", "v", "out", "norm"))
        expected_names.update(f"layer{i}.{name}" for name in ("final_norm", "gate", "up", "down"))
    if set(arrays) != expected_names:
        raise StackContractError("Unexpected canonical weight keys")
    blocks = []
    for i in range(doc["layers"]):
        p = f"layer{i}."
        a = DecoderWeights(*(arrays[p + "attention." + name] for name in ("q", "k", "v", "out", "norm")))
        # Redundant embedding/head in TinyDecoderWeights are inactive at block level.
        blocks.append(TinyDecoderWeights(arrays["embedding"], a, arrays[p + "final_norm"],
                                         arrays[p + "gate"], arrays[p + "up"], arrays[p + "down"], arrays["lm_head"]))
    weights = StackedWeights(arrays["embedding"], tuple(blocks), arrays["lm_head"],
                             doc["heads"], doc["max_context"])
    weights.validate()
    if checkpoint_digest(weights) != doc["weight_sha256"]:
        raise StackContractError("Checkpoint content digest mismatch")
    return weights


@dataclass(frozen=True)
class StackState:
    layers: tuple[KVState, ...]


def empty_stack(weights: StackedWeights, batch: int) -> StackState:
    _vocab, _hidden, depth, kv = weights.validate()
    if type(batch) is not int or not 1 <= batch <= 8:
        raise StackContractError("Batch must be in [1,8]")
    return StackState(tuple(empty_cache(batch, kv, depth, weights.max_context) for _ in weights.blocks))


def _check_request(weights: StackedWeights, tokens: np.ndarray, state: StackState) -> tuple[int, int, int]:
    vocab, _hidden, depth, kv = weights.validate()
    if (not isinstance(tokens, np.ndarray) or tokens.dtype != np.int64 or tokens.ndim != 2):
        raise StackContractError("Token IDs must be int64 [batch, tokens]")
    batch, query = tokens.shape
    if (not 1 <= batch <= 8 or not 1 <= query <= 32
        or np.any(tokens < 0) or np.any(tokens >= vocab)):
        raise StackContractError("Token IDs outside vocabulary or query budget")
    if not isinstance(state, StackState) or len(state.layers) != len(weights.blocks):
        raise StackContractError("KV-cache layer count differs from checkpoint")
    if not isinstance(state.layers[0], KVState) or not isinstance(state.layers[0].keys, np.ndarray) or state.layers[0].keys.ndim != 4:
        raise StackContractError("Invalid first-layer KV state")
    past = state.layers[0].keys.shape[1]
    if past + query > weights.max_context:
        raise StackContractError("Context budget exceeded")
    for layer in state.layers:
        if (not isinstance(layer, KVState) or layer.max_context != weights.max_context
            or any(not isinstance(a, np.ndarray) or a.dtype != np.float32
                   or a.shape != (batch, past, kv, depth) or not np.isfinite(a).all()
                   for a in (layer.keys, layer.values))):
            raise StackContractError("KV state violates per-layer dtype/shape/finiteness")
    return batch, query, past


def reference_stack_step(weights: StackedWeights, tokens: np.ndarray, state: StackState) -> tuple[np.ndarray, StackState]:
    """Independent NumPy reference using the previously validated attention block."""
    _check_request(weights, tokens, state)
    h = weights.embedding[tokens]
    caches: list[KVState] = []
    for block, prior in zip(weights.blocks, state.layers):
        attention, next_cache = decoder_step_numpy(h, block.attention, prior, heads=weights.heads)
        residual = h + attention
        normed = residual / np.sqrt(np.mean(residual * residual, axis=-1, keepdims=True) + 1e-5) * block.final_norm
        gate = normed @ block.gate
        silu = gate / (1 + np.exp(-gate))
        h = (residual + (silu * (normed @ block.up)) @ block.down).astype(np.float32)
        caches.append(next_cache)
    return (h @ weights.lm_head).astype(np.float32), StackState(tuple(caches))


class _HiddenBlock(_DecoderChunk):
    """Trace a v13 block as hidden -> hidden with its own immutable KV values.

    The v13 decoder itself is deliberately left unchanged so its certification
    and published source fingerprint remain valid.
    """

    def forward(self, x: torch.Tensor, prev_k: torch.Tensor, prev_v: torch.Tensor):
        normed = x * torch.rsqrt(
            torch.sum(x * x, dim=-1, keepdim=True) / self.hidden + 1e-5
        ) * self.norm_attn
        q = self._rope(
            torch.matmul(normed, self.q).reshape(
                self.batch, self.query, self.heads, self.depth
            ), kv=False
        )
        k = self._rope(
            torch.matmul(normed, self.k).reshape(
                self.batch, self.query, self.kv_heads, self.depth
            ), kv=True
        )
        v = torch.matmul(normed, self.v).reshape(
            self.batch, self.query, self.kv_heads, self.depth
        )
        next_k = torch.cat((prev_k, k), dim=1)
        next_v = torch.cat((prev_v, v), dim=1)
        ks = torch.repeat_interleave(next_k, self.heads // self.kv_heads, dim=2).permute(0, 2, 1, 3)
        vs = torch.repeat_interleave(next_v, self.heads // self.kv_heads, dim=2).permute(0, 2, 1, 3)
        scores = (torch.matmul(q.permute(0, 2, 1, 3), ks.transpose(-1, -2))
                  * self.inv_depth + self.causal_bias)
        probs = torch.softmax(scores, dim=-1)
        attn = torch.matmul(probs, vs).permute(0, 2, 1, 3).reshape(
            self.batch, self.query, self.hidden
        )
        residual = x + torch.matmul(attn, self.out)
        normal = residual * torch.rsqrt(
            torch.sum(residual * residual, dim=-1, keepdim=True) / self.hidden + 1e-5
        ) * self.norm_final
        gate = torch.matmul(normal, self.gate)
        silu = gate / (1.0 + torch.exp(gate * -1.0))
        mlp = torch.matmul(silu * torch.matmul(normal, self.up), self.down)
        return residual + mlp, next_k, next_v


class _StackedChunk(nn.Module):
    def __init__(self, weights: StackedWeights, *, batch: int, query: int, past: int):
        super().__init__()
        weights.validate()
        self.register_buffer("embedding", torch.from_numpy(weights.embedding.copy()))
        self.register_buffer("lm_head", torch.from_numpy(weights.lm_head.copy()))
        self.blocks = nn.ModuleList([_HiddenBlock(block, batch=batch, query=query, past=past, heads=weights.heads)
                                     for block in weights.blocks])

    def forward(self, tokens: torch.Tensor, packed_caches: tuple[torch.Tensor, ...]):
        x = self.embedding[tokens]
        updated: list[torch.Tensor] = []
        for i, block in enumerate(self.blocks):
            x, k, v = block(x, packed_caches[2 * i], packed_caches[2 * i + 1])
            updated.extend((k, v))
        return (torch.matmul(x, self.lm_head), *updated)


def _trace_flattened_caches(model: _StackedChunk, t: torch.Tensor,
                            flat_caches: tuple[torch.Tensor, ...]) -> CapturedProgram:
    """Convert an FX tuple cache boundary into independent typed tensor inputs.

    The operation graph is copied, never rewritten via eval/exec or source
    generation. A cache-tuple use other than indexed extraction is rejected.
    """
    traced = symbolic_trace(model)
    original_nodes = list(traced.graph.nodes)
    tuple_input = next((n for n in original_nodes if n.op == "placeholder" and n.target == "packed_caches"), None)
    if tuple_input is None:
        raise StackContractError("FX failed to preserve the cache boundary")
    new_graph = Graph()
    mapped: dict[Any, Any] = {}
    flat_inputs: list[Any] = []
    for node in original_nodes:
        if node is tuple_input:
            flat_inputs = [new_graph.placeholder(f"cache_{i}") for i in range(len(flat_caches))]
            continue
        if (node.op == "call_function" and node.target is operator.getitem
            and node.args[0] is tuple_input and type(node.args[1]) is int):
            index = node.args[1]
            if not 0 <= index < len(flat_inputs):
                raise StackContractError("Cache reference index out of bounds")
            mapped[node] = flat_inputs[index]
            continue
        if tuple_input in node.all_input_nodes:
            raise StackContractError("Cache tuple used outside indexed extraction")
        mapped[node] = new_graph.node_copy(node, lambda dep: mapped[dep])
    lowered = GraphModule(traced, new_graph)
    lowered.graph.lint()
    with torch.no_grad():
        ShapeProp(lowered).propagate(t, *flat_caches)
    return CapturedProgram(lowered, inspect_fx(lowered))


def compile_stacked_chunk(weights: StackedWeights, *, batch: int, query: int, past: int) -> AOTArtifact:
    _vocab, _hidden, depth, kv = weights.validate()
    if (type(batch) is not int or type(query) is not int or type(past) is not int
        or not 1 <= batch <= 8 or not 1 <= query <= 32 or not 0 <= past
        or past + query > weights.max_context):
        raise StackContractError("Invalid fixed-shape specialization")
    model = _StackedChunk(weights, batch=batch, query=query, past=past).eval()
    z = torch.zeros((batch, past, kv, depth), dtype=torch.float32)
    t = torch.zeros((batch, query), dtype=torch.int64)
    captured = _trace_flattened_caches(model, t, tuple(z.clone() for _ in range(2 * len(weights.blocks))))
    artifact = compile_numpy_aot(captured)
    artifact.manifest["source"] = "v14 1..4 layer GQA/RoPE immutable KV checkpoint-specialized decoder"
    artifact.manifest["input_constraints"] = {
        artifact.manifest["inputs"][0]["name"]: {"min_inclusive": 0, "max_exclusive": int(weights.embedding.shape[0])}
    }
    artifact.manifest["numeric_contract"] = (
        "Concrete token/cache shapes; ordered GQA residual blocks; immutable KV; "
        "independent per-layer cache updates; NumPy CPU witness-limited"
    )
    unsigned = {k: v for k, v in artifact.manifest.items() if k != "manifest_sha256"}
    artifact.manifest["manifest_sha256"] = hashlib.sha256(
        json.dumps(unsigned, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    return artifact


@dataclass
class CompiledStackedDecoder:
    weights: StackedWeights
    _compiled: dict[tuple[int, int, int], AOTArtifact] = field(default_factory=dict)
    _prepared: dict[tuple[int, int, int], PreparedNumpyExecutor] = field(default_factory=dict)
    _weight_sha256: str | None = None

    def step(self, tokens: np.ndarray, state: StackState) -> tuple[np.ndarray, StackState]:
        batch, query, past = _check_request(self.weights, tokens, state)
        digest = checkpoint_digest(self.weights)
        if self._weight_sha256 is None:
            self._weight_sha256 = digest
        elif digest != self._weight_sha256:
            raise StackContractError("Checkpoint weights mutated after compilation; create a new driver")
        key = batch, query, past
        if key not in self._compiled:
            if len(self._compiled) >= 48:
                raise StackContractError("Exceeded shape specialization cache budget")
            artifact = compile_stacked_chunk(self.weights, batch=batch, query=query, past=past)
            executor = prepare_numpy_aot(artifact.manifest, artifact.constants)
            self._compiled[key] = artifact
            self._prepared[key] = executor
        cache_inputs = tuple(value for layer in state.layers for value in (layer.keys, layer.values))
        result = self._prepared[key].run(tokens, *cache_inputs)
        if not isinstance(result, tuple) or len(result) != 1 + 2 * len(self.weights.blocks):
            raise StackContractError("Compiled output arity violated")
        logits, *cache_outputs = result
        _vocab, _hidden, depth, kv = self.weights.validate()
        expected_shape = (batch, past + query, kv, depth)
        if (logits.shape != (batch, query, self.weights.embedding.shape[0])
            or logits.dtype != np.float32 or not np.isfinite(logits).all()
            or any(a.shape != expected_shape or a.dtype != np.float32 or not np.isfinite(a).all()
                   for a in cache_outputs)):
            raise StackContractError("Compiled result violated shape/finiteness contract")
        return logits.copy(), StackState(tuple(
            KVState(cache_outputs[2 * i].copy(), cache_outputs[2 * i + 1].copy(), self.weights.max_context)
            for i in range(len(self.weights.blocks))
        ))

    def generate(self, tokens: np.ndarray, *, steps: int) -> tuple[np.ndarray, StackState]:
        if (type(steps) is not int or not 1 <= steps <= 24
            or not isinstance(tokens, np.ndarray) or tokens.ndim != 2
            or tokens.shape[1] + steps > self.weights.max_context):
            raise StackContractError("Generation outside budget")
        state = empty_stack(self.weights, tokens.shape[0])
        logits, state = self.step(tokens, state)
        outputs: list[np.ndarray] = []
        for _ in range(steps):
            ids = np.argmax(logits[:, -1], axis=-1).astype(np.int64)[:, None]
            outputs.append(ids)
            logits, state = self.step(ids, state)
        return np.concatenate(outputs, axis=1), state


def verify_stacked_witness(weights: StackedWeights, tokens: np.ndarray, chunks: tuple[int, ...]) -> dict[str, Any]:
    """Compare entire output AND all four cache tensors after each chunk."""
    if sum(chunks) != tokens.shape[1] or not chunks or any(type(x) is not int or x <= 0 for x in chunks):
        raise StackContractError("Invalid witness segmentation")
    driver = CompiledStackedDecoder(weights)
    observed_state = empty_stack(weights, tokens.shape[0])
    ref_state = empty_stack(weights, tokens.shape[0])
    worst = 0.0
    checked = 0
    position = 0
    for length in chunks:
        x = tokens[:, position:position+length]
        observed, observed_state = driver.step(x, observed_state)
        expected, ref_state = reference_stack_step(weights, x, ref_state)
        parts = [(observed, expected)]
        for got, want in zip(observed_state.layers, ref_state.layers):
            parts.extend(((got.keys, want.keys), (got.values, want.values)))
        for actual, expected_value in parts:
            np.testing.assert_allclose(actual, expected_value, rtol=5e-5, atol=5e-5)
            checked += 1
            if actual.size:
                worst = max(worst, float(np.max(np.abs(actual.astype(np.float64) - expected_value.astype(np.float64)))))
        position += length
    return {"schema": "visuaml.stacked-decoder-witness.v14", "source_kind": "synthetic-cpu-fixture",
            "passed": True, "layers": len(weights.blocks), "chunks": len(chunks), "arrays_checked": checked,
            "max_abs_error": worst, "specializations": len(driver._compiled),
            "checkpoint_sha256": checkpoint_digest(weights), "backend": "NumPy AOT CPU (no device acceleration)"}
