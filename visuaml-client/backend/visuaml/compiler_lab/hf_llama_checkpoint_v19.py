"""Guarded local Hugging Face Llama safetensors importer and reference executor.

This is an experimental compatibility bridge for *real on-disk weights*, not
an assertion that a named pretrained checkpoint has already passed parity.
The admitted family is unscaled, bias-free Llama-style RMSNorm / SwiGLU /
half-rotary causal GQA with a final model-wide RMSNorm. A dedicated NumPy CPU
reference path is intentionally separate from v18's synthetic checkpoint ABI.

No downloads, pickle loading, Python model imports, plugins or remote execution.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from math import sqrt
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors import safe_open


class HFLlamaContractError(ValueError):
    """Unsupported model configuration, tensor set or untrusted local checkpoint."""


@dataclass(frozen=True)
class LlamaProfile:
    hidden: int
    layers: int
    heads: int
    kv_heads: int
    head_dim: int
    intermediate: int
    vocabulary: int
    context: int
    eps: float
    rope_theta: float
    tied_head: bool


@dataclass(frozen=True)
class LlamaLocalCheckpoint:
    profile: LlamaProfile
    arrays: dict[str, np.ndarray]
    checkpoint_sha256: str
    config_sha256: str
    declared_revision: str
    tokenizer_sha256: str | None

    def forward(self, tokens: np.ndarray) -> np.ndarray:
        """Deterministic full-prefill logits, no incremental KV cache or tokenizer."""
        p = self.profile
        if (
            not isinstance(tokens, np.ndarray) or tokens.dtype != np.int64
            or tokens.ndim != 2 or not 1 <= tokens.shape[0] <= 2
            or not 1 <= tokens.shape[1] <= min(64, p.context)
            or np.any(tokens < 0) or np.any(tokens >= p.vocabulary)
        ):
            raise HFLlamaContractError("Invalid batch, token IDs, or prefill length")
        b, length = tokens.shape
        arrays = self.arrays
        x = arrays["model.embed_tokens.weight"][tokens]
        inv_freq = 1.0 / (
            p.rope_theta ** (np.arange(0, p.head_dim, 2, dtype=np.float64) / p.head_dim)
        )
        angles = np.outer(np.arange(length, dtype=np.float64), inv_freq)
        cos = np.concatenate((np.cos(angles), np.cos(angles)), axis=-1).astype(np.float32)
        sin = np.concatenate((np.sin(angles), np.sin(angles)), axis=-1).astype(np.float32)
        cos = cos[None, :, None, :]
        sin = sin[None, :, None, :]

        def rms(inp: np.ndarray, weight: np.ndarray) -> np.ndarray:
            return inp / np.sqrt(np.mean(inp * inp, axis=-1, keepdims=True) + p.eps) * weight

        def rope(inp: np.ndarray) -> np.ndarray:
            # Hugging Face Llama uses rotate_half rather than v18's
            # interleaved even/odd rotations.
            half = p.head_dim // 2
            rotated = np.concatenate((-inp[..., half:], inp[..., :half]), axis=-1)
            return inp * cos + rotated * sin

        causal = np.tril(np.ones((length, length), dtype=np.bool_))
        for layer in range(p.layers):
            prefix = "model.layers." + str(layer) + "."
            residual = x
            norm = rms(x, arrays[prefix + "input_layernorm.weight"])
            q = (norm @ arrays[prefix + "self_attn.q_proj.weight"].T).reshape(
                b, length, p.heads, p.head_dim
            )
            k = (norm @ arrays[prefix + "self_attn.k_proj.weight"].T).reshape(
                b, length, p.kv_heads, p.head_dim
            )
            v = (norm @ arrays[prefix + "self_attn.v_proj.weight"].T).reshape(
                b, length, p.kv_heads, p.head_dim
            )
            q, k = rope(q), rope(k)
            replicas = p.heads // p.kv_heads
            group_q = q.reshape(
                b, length, p.kv_heads, replicas, p.head_dim
            ).transpose(0, 2, 3, 1, 4)
            keys = k.transpose(0, 2, 1, 3)[:, :, None, :, :]
            scores = np.matmul(group_q, np.swapaxes(keys, -1, -2)) * np.float32(
                1.0 / sqrt(p.head_dim)
            )
            scores = np.where(causal[None, None, None, :, :], scores, np.float32(-1e9))
            scores = scores - scores.max(axis=-1, keepdims=True)
            np.exp(scores, out=scores)
            scores = scores / scores.sum(axis=-1, keepdims=True)
            values = v.transpose(0, 2, 1, 3)[:, :, None, :, :]
            attn = np.matmul(scores, values).transpose(0, 3, 1, 2, 4).reshape(
                b, length, p.hidden
            )
            x = residual + attn @ arrays[prefix + "self_attn.o_proj.weight"].T
            normal = rms(x, arrays[prefix + "post_attention_layernorm.weight"])
            gate = normal @ arrays[prefix + "mlp.gate_proj.weight"].T
            up = normal @ arrays[prefix + "mlp.up_proj.weight"].T
            activated = gate / (1.0 + np.exp(-gate))
            x = x + (activated * up) @ arrays[prefix + "mlp.down_proj.weight"].T
        final = rms(x, arrays["model.norm.weight"])
        head = (
            arrays["model.embed_tokens.weight"] if p.tied_head
            else arrays["lm_head.weight"]
        )
        logits = (final @ head.T).astype(np.float32)
        if logits.shape != (b, length, p.vocabulary) or not np.isfinite(logits).all():
            raise HFLlamaContractError("Nonfinite or malformed Llama output")
        return logits


def _check_config(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 16_384:
        raise HFLlamaContractError("Missing, linked or oversized config")
    def reject_duplicate(pairs: list[tuple[str, Any]]) -> dict:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise HFLlamaContractError("Duplicate config JSON key")
            result[key] = value
        return result
    try:
        config = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicate)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise HFLlamaContractError("Invalid model config JSON") from exc
    if not isinstance(config, dict):
        raise HFLlamaContractError("Config must be an object")
    return config


def _profile(doc: dict[str, Any]) -> LlamaProfile:
    def integer(name: str, low: int, high: int) -> int:
        value = doc.get(name)
        if type(value) is not int or not low <= value <= high:
            raise HFLlamaContractError("Unsupported Llama config: " + name)
        return value

    if (
        doc.get("model_type") != "llama"
        or doc.get("architectures") != ["LlamaForCausalLM"]
        or doc.get("hidden_act") != "silu"
        or doc.get("rope_scaling") is not None
        or doc.get("attention_bias", False) is not False
        or doc.get("mlp_bias", False) is not False
        or doc.get("attention_dropout", 0.0) != 0
    ):
        raise HFLlamaContractError("Unsupported variant of Llama architecture")
    hidden = integer("hidden_size", 8, 384)
    layers = integer("num_hidden_layers", 1, 8)
    heads = integer("num_attention_heads", 1, 16)
    kv = integer("num_key_value_heads", 1, 16)
    intermediate = integer("intermediate_size", 8, 2048)
    vocab = integer("vocab_size", 2, 32768)
    context = integer("max_position_embeddings", 1, 4096)
    if hidden % heads or heads % kv or (hidden // heads) % 2:
        raise HFLlamaContractError("Incompatible attention grouping/head dimension")
    explicit_dim = doc.get("head_dim", hidden // heads)
    if type(explicit_dim) is not int or explicit_dim != hidden // heads:
        raise HFLlamaContractError("Custom attention head dimension not admitted")
    eps = doc.get("rms_norm_eps", 1e-6)
    theta = doc.get("rope_theta", 10000.0)
    if (
        type(eps) not in (int, float) or not 0 < eps <= 1e-3
        or type(theta) not in (int, float) or not 1 <= theta <= 1e7
    ):
        raise HFLlamaContractError("Invalid RMSNorm or rotary parameters")
    tied = doc.get("tie_word_embeddings", False)
    if type(tied) is not bool:
        raise HFLlamaContractError("Invalid embedding-tie metadata")
    return LlamaProfile(
        hidden, layers, heads, kv, hidden // heads,
        intermediate, vocab, context, float(eps), float(theta), tied,
    )


def _expected_shapes(p: LlamaProfile) -> dict[str, tuple[int, ...]]:
    table = {
        "model.embed_tokens.weight": (p.vocabulary, p.hidden),
        "model.norm.weight": (p.hidden,),
    }
    if not p.tied_head:
        table["lm_head.weight"] = (p.vocabulary, p.hidden)
    for layer in range(p.layers):
        prefix = "model.layers." + str(layer) + "."
        for key, shape in {
            "input_layernorm.weight": (p.hidden,),
            "post_attention_layernorm.weight": (p.hidden,),
            "self_attn.q_proj.weight": (p.heads * p.head_dim, p.hidden),
            "self_attn.k_proj.weight": (p.kv_heads * p.head_dim, p.hidden),
            "self_attn.v_proj.weight": (p.kv_heads * p.head_dim, p.hidden),
            "self_attn.o_proj.weight": (p.hidden, p.heads * p.head_dim),
            "mlp.gate_proj.weight": (p.intermediate, p.hidden),
            "mlp.up_proj.weight": (p.intermediate, p.hidden),
            "mlp.down_proj.weight": (p.hidden, p.intermediate),
        }.items():
            table[prefix + key] = shape
    return table


def load_hf_llama_local(
    directory: str | Path, *, declared_revision: str,
    expected_weights_sha256: str | None = None,
) -> LlamaLocalCheckpoint:
    """Read a pinned local directory without fetching or executing model code."""
    if not isinstance(declared_revision, str) or not re.fullmatch(r"[a-f0-9]{40}", declared_revision):
        raise HFLlamaContractError("Declare a full 40-character model revision")
    root = Path(directory)
    doc = _check_config(root / "config.json")
    profile = _profile(doc)
    weight_file = root / "model.safetensors"
    if (
        weight_file.is_symlink() or not weight_file.is_file()
        or not 0 < weight_file.stat().st_size <= 128 * 1024 * 1024
    ):
        raise HFLlamaContractError("Missing, linked or oversized safetensors file")
    with weight_file.open("rb") as source:
        fingerprint = hashlib.file_digest(source, "sha256").hexdigest()
    if expected_weights_sha256 is not None and fingerprint != expected_weights_sha256:
        raise HFLlamaContractError("Checkpoint SHA-256 differs from pinned expectation")
    expected = _expected_shapes(profile)
    arrays: dict[str, np.ndarray] = {}
    try:
        with safe_open(weight_file, framework="pt", device="cpu") as reader:
            if set(reader.keys()) != set(expected):
                raise HFLlamaContractError("Checkpoint tensor keys do not match exact architecture")
            for name, shape in expected.items():
                tensor = reader.get_tensor(name)
                if (
                    tuple(tensor.shape) != shape
                    or tensor.dtype not in (torch.float32, torch.float16, torch.bfloat16)
                ):
                    raise HFLlamaContractError("Checkpoint tensor shape or dtype mismatch: " + name)
                array = tensor.to(torch.float32).numpy().copy()
                if not np.isfinite(array).all():
                    raise HFLlamaContractError("Checkpoint contains nonfinite tensor: " + name)
                array.setflags(write=False)
                arrays[name] = array
    except (OSError, RuntimeError, ValueError) as exc:
        if isinstance(exc, HFLlamaContractError):
            raise
        raise HFLlamaContractError("Could not safely read safetensors") from exc
    tokenizer_file = root / "tokenizer.json"
    tokenizer_sha = None
    if tokenizer_file.is_file() and not tokenizer_file.is_symlink():
        if tokenizer_file.stat().st_size > 32 * 1024 * 1024:
            raise HFLlamaContractError("Tokenizer JSON too large")
        with tokenizer_file.open("rb") as source:
            tokenizer_sha = hashlib.file_digest(source, "sha256").hexdigest()
    with (root / "config.json").open("rb") as source:
        config_sha = hashlib.file_digest(source, "sha256").hexdigest()
    return LlamaLocalCheckpoint(
        profile, arrays, fingerprint, config_sha, declared_revision, tokenizer_sha
    )
