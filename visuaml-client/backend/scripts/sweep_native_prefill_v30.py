"""v30 synthetic-geometry prefill/decode sweep, never a trained-model claim.

Each case generates a deterministic bounded Llama-style checkpoint with known
weight shapes, checks independent NumPy parity before timing, and records
per-backend source hashes and raw latency samples. Pretrained-model parity
is separately checked in probe_native_llama_v28.py.
"""

from __future__ import annotations

import argparse
import json
import platform
import tempfile
from pathlib import Path

import torch
from safetensors.torch import save_file
from examples.native_cpu_prefill_bench_v30 import benchmark
from visuaml.compiler_lab.hf_llama_checkpoint_v19 import load_hf_llama_local


CASES = (
    ("small", 32, 4, 2, 80, 64, 8),
    ("middle", 96, 6, 3, 256, 512, 8),
    ("wide", 192, 8, 4, 512, 2048, 8),
    ("wide_long_prefill", 192, 8, 4, 512, 2048, 16),
    ("vocab_heavy", 128, 8, 4, 384, 4096, 8),
)


def make_checkpoint(
    root: Path, *, hidden: int, heads: int, kv_heads: int,
    intermediate: int, vocabulary: int,
) -> None:
    if hidden % heads or heads % kv_heads or hidden // heads % 2:
        raise ValueError("Invalid head geometry")
    config = {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "hidden_act": "silu",
        "hidden_size": hidden,
        "intermediate_size": intermediate,
        "num_hidden_layers": 1,
        "num_attention_heads": heads,
        "num_key_value_heads": kv_heads,
        "max_position_embeddings": 64,
        "vocab_size": vocabulary,
        "rope_theta": 10000.0,
        "rms_norm_eps": 1e-5,
        "tie_word_embeddings": False,
        "rope_scaling": None,
        "attention_bias": False,
        "attention_dropout": 0.0,
        "mlp_bias": False,
    }
    generator = torch.Generator().manual_seed(hidden + vocabulary)
    dim = hidden // heads

    def matrix(n: int, m: int) -> torch.Tensor:
        return torch.randn((n, m), generator=generator) * 0.03

    arrays = {
        "model.embed_tokens.weight": matrix(vocabulary, hidden),
        "model.norm.weight": torch.ones(hidden),
        "lm_head.weight": matrix(vocabulary, hidden),
    }
    prefix = "model.layers.0."
    arrays.update({
        prefix + "input_layernorm.weight": torch.ones(hidden),
        prefix + "post_attention_layernorm.weight": torch.ones(hidden),
        prefix + "self_attn.q_proj.weight": matrix(hidden, hidden),
        prefix + "self_attn.k_proj.weight": matrix(kv_heads * dim, hidden),
        prefix + "self_attn.v_proj.weight": matrix(kv_heads * dim, hidden),
        prefix + "self_attn.o_proj.weight": matrix(hidden, hidden),
        prefix + "mlp.gate_proj.weight": matrix(intermediate, hidden),
        prefix + "mlp.up_proj.weight": matrix(intermediate, hidden),
        prefix + "mlp.down_proj.weight": matrix(hidden, intermediate),
    })
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text(
        json.dumps(config, sort_keys=True), encoding="utf-8"
    )
    save_file(arrays, str(root / "model.safetensors"))


def run(*, trials: int = 5) -> dict:
    if type(trials) is not int or not 2 <= trials <= 100:
        raise ValueError("Trials must be in 2..100")
    entries = []
    with tempfile.TemporaryDirectory(prefix="visuaml-v30-shapes-") as tmp:
        for name, hidden, heads, kv_heads, middle, vocabulary, prompt in CASES:
            checkpoint = Path(tmp) / name
            make_checkpoint(
                checkpoint, hidden=hidden, heads=heads, kv_heads=kv_heads,
                intermediate=middle, vocabulary=vocabulary,
            )
            model = load_hf_llama_local(
                checkpoint, declared_revision="f" * 40
            )
            record = benchmark(
                model, prompt=prompt, decode=(1, 1), trials=trials
            )
            entries.append({
                "name": name,
                "geometry": {
                    "hidden": hidden, "heads": heads, "kv_heads": kv_heads,
                    "intermediate": middle, "vocabulary": vocabulary,
                    "prefill": prompt, "decode_chunks": [1, 1],
                },
                "prefill_ratio_avx2_over_batched": record[
                    "prefill_ratio_v29_avx2_over_batched"
                ],
                "decode_ratio_avx2_over_batched": record[
                    "decode_ratio_v29_avx2_over_batched"
                ],
                "evidence": record,
            })
    return {
        "schema": "visuaml.native-prefill-geometry-sweep.v30",
        "source": "deterministic synthetic Llama geometry, not pretrained",
        "platform": platform.platform(),
        "trials_per_case": trials,
        "cases": entries,
        "limitations": [
            "Only synthetic bounded shapes and one shared runner.",
            "Results may depend on CPU and cache geometry; not a universal dispatch rule.",
            "Native and NumPy numerical comparisons are finite evidence only.",
            "Separate pinned real pretrained Llama verification is mandatory.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    record = run(trials=args.trials)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "schema": record["schema"],
        "cases": [
            {
                "name": case["name"],
                "prefill_ratio_avx2_over_batched": case[
                    "prefill_ratio_avx2_over_batched"
                ],
                "decode_ratio_avx2_over_batched": case[
                    "decode_ratio_avx2_over_batched"
                ],
            }
            for case in record["cases"]
        ],
        "output": str(args.output),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
