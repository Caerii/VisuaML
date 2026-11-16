"""Safe HF-style local checkpoint fixture and independent eager-Torch oracle."""

from __future__ import annotations

import json
from math import sqrt

import numpy as np
import pytest
import torch
from safetensors.torch import save_file
from visuaml.compiler_lab.hf_llama_checkpoint_v19 import (
    HFLlamaContractError,
    load_hf_llama_local,
)


def fixture(tmp_path, *, layers=2, hidden=32, heads=4, kv=2, vocab=64, middle=80):
    config = {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama", "hidden_act": "silu",
        "hidden_size": hidden, "intermediate_size": middle,
        "num_hidden_layers": layers, "num_attention_heads": heads,
        "num_key_value_heads": kv, "max_position_embeddings": 64,
        "vocab_size": vocab, "rope_theta": 10000.0,
        "rms_norm_eps": 1e-5, "tie_word_embeddings": False,
        "rope_scaling": None, "attention_bias": False,
        "attention_dropout": 0.0, "mlp_bias": False,
    }
    gen = torch.Generator().manual_seed(layers + hidden)
    dim = hidden // heads
    arrays = {
        "model.embed_tokens.weight": torch.randn(vocab, hidden, generator=gen) * .03,
        "model.norm.weight": torch.ones(hidden),
        "lm_head.weight": torch.randn(vocab, hidden, generator=gen) * .03,
    }
    for layer in range(layers):
        prefix = "model.layers." + str(layer) + "."
        shapes = {
            "input_layernorm.weight": (hidden,),
            "post_attention_layernorm.weight": (hidden,),
            "self_attn.q_proj.weight": (hidden, hidden),
            "self_attn.k_proj.weight": (kv * dim, hidden),
            "self_attn.v_proj.weight": (kv * dim, hidden),
            "self_attn.o_proj.weight": (hidden, hidden),
            "mlp.gate_proj.weight": (middle, hidden),
            "mlp.up_proj.weight": (middle, hidden),
            "mlp.down_proj.weight": (hidden, middle),
        }
        for name, shape in shapes.items():
            arrays[prefix + name] = (
                torch.ones(shape) if name.endswith("norm.weight")
                else torch.randn(shape, generator=gen) * .03
            )
    (tmp_path / "config.json").write_text(json.dumps(config))
    save_file(arrays, str(tmp_path / "model.safetensors"))
    return arrays, config


def torch_oracle(tokens, weights, config):
    heads, kv = config["num_attention_heads"], config["num_key_value_heads"]
    hidden = config["hidden_size"]
    dim = hidden // heads
    length = tokens.shape[1]
    x = torch.nn.functional.embedding(tokens, weights["model.embed_tokens.weight"])
    inv = 1.0 / (config["rope_theta"] ** (
        torch.arange(0, dim, 2, dtype=torch.float64) / dim
    ))
    theta = torch.outer(torch.arange(length, dtype=torch.float64), inv)
    freqs = torch.cat((theta, theta), dim=-1).to(torch.float32)
    cosine = freqs.cos()[None, :, None, :]
    sine = freqs.sin()[None, :, None, :]

    def rms(value, w):
        return value * torch.rsqrt(
            torch.mean(value * value, dim=-1, keepdim=True) + config["rms_norm_eps"]
        ) * w

    def rope(value):
        halfway = dim // 2
        return value * cosine + torch.cat(
            (-value[..., halfway:], value[..., :halfway]), dim=-1
        ) * sine

    for layer in range(config["num_hidden_layers"]):
        p = "model.layers." + str(layer) + "."
        norm = rms(x, weights[p + "input_layernorm.weight"])
        q = torch.nn.functional.linear(norm, weights[p + "self_attn.q_proj.weight"])
        k = torch.nn.functional.linear(norm, weights[p + "self_attn.k_proj.weight"])
        v = torch.nn.functional.linear(norm, weights[p + "self_attn.v_proj.weight"])
        q = rope(q.reshape(tokens.shape[0], length, heads, dim))
        k = rope(k.reshape(tokens.shape[0], length, kv, dim))
        v = v.reshape(tokens.shape[0], length, kv, dim)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3).repeat_interleave(heads // kv, dim=1)
        v = v.permute(0, 2, 1, 3).repeat_interleave(heads // kv, dim=1)
        scores = torch.matmul(q, k.transpose(-1, -2)) / sqrt(dim)
        causal = torch.ones((length, length), dtype=torch.bool).tril()
        scores = scores.masked_fill(~causal[None, None, :, :], -1e9)
        prob = torch.softmax(scores, dim=-1)
        attn = torch.matmul(prob, v).transpose(1, 2).reshape(tokens.shape[0], length, hidden)
        x = x + torch.nn.functional.linear(attn, weights[p + "self_attn.o_proj.weight"])
        normal = rms(x, weights[p + "post_attention_layernorm.weight"])
        gate = torch.nn.functional.linear(normal, weights[p + "mlp.gate_proj.weight"])
        up = torch.nn.functional.linear(normal, weights[p + "mlp.up_proj.weight"])
        x = x + torch.nn.functional.linear(
            torch.nn.functional.silu(gate) * up, weights[p + "mlp.down_proj.weight"]
        )
    x = rms(x, weights["model.norm.weight"])
    return torch.nn.functional.linear(x, weights["lm_head.weight"])


@pytest.mark.parametrize("layers", (1, 2, 4))
def test_local_llama_safetensors_runs_against_independent_torch(tmp_path, layers):
    arrays, config = fixture(tmp_path, layers=layers)
    model = load_hf_llama_local(tmp_path, declared_revision="a" * 40)
    assert model.profile.layers == layers
    tokens = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]])
    np_logits = model.forward(tokens.numpy().astype(np.int64))
    torch_logits = torch_oracle(tokens, arrays, config)
    np.testing.assert_allclose(np_logits, torch_logits.numpy(), atol=5e-5, rtol=5e-5)


def test_file_hash_and_unsupported_config_fail_closed(tmp_path):
    fixture(tmp_path)
    with pytest.raises(HFLlamaContractError):
        load_hf_llama_local(tmp_path, declared_revision="main")
    with pytest.raises(HFLlamaContractError):
        load_hf_llama_local(
            tmp_path, declared_revision="a" * 40, expected_weights_sha256="0" * 64
        )
    config_file = tmp_path / "config.json"
    doc = json.loads(config_file.read_text())
    doc["rope_scaling"] = {"rope_type": "llama3"}
    config_file.write_text(json.dumps(doc))
    with pytest.raises(HFLlamaContractError, match="Unsupported variant"):
        load_hf_llama_local(tmp_path, declared_revision="a" * 40)


def test_corrupt_or_missing_checkpoint_tensor_fails(tmp_path):
    arrays, _ = fixture(tmp_path)
    arrays.pop("model.norm.weight")
    save_file(arrays, str(tmp_path / "model.safetensors"))
    with pytest.raises(HFLlamaContractError, match="keys"):
        load_hf_llama_local(tmp_path, declared_revision="a" * 40)


def test_forward_rejects_tokens_outside_vocab(tmp_path):
    fixture(tmp_path)
    model = load_hf_llama_local(tmp_path, declared_revision="a" * 40)
    with pytest.raises(HFLlamaContractError):
        model.forward(np.array([[64]], dtype=np.int64))
    with pytest.raises(HFLlamaContractError):
        model.forward(np.array([[1]], dtype=np.int32))
