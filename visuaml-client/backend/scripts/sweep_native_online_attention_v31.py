"""Deterministic synthetic geometry sweep for optional online C++ attention.

This is **not** real pretrained architecture or production serving evidence.
The pinned pretrained witness is checked separately by the native parity CI.
"""

from __future__ import annotations

import argparse
import json
import platform
import tempfile
from pathlib import Path

from examples.native_online_attention_bench_v31 import benchmark
from sweep_native_prefill_v30 import CASES, make_checkpoint
from visuaml.compiler_lab.hf_llama_checkpoint_v19 import load_hf_llama_local

# Exercise distinct dimensions / vocabulary costs without claiming exhaustive coverage.
CASE_NAMES = ("small", "middle", "wide_long_prefill")


def run(*, trials: int = 3) -> dict:
    if type(trials) is not int or not 2 <= trials <= 25:
        raise ValueError("Trials must be bounded to 2..25")
    entries = []
    with tempfile.TemporaryDirectory(prefix="visuaml-v31-attn-shapes-") as tmp:
        for name, hidden, heads, kv, inner, vocab, _ in CASES:
            if name not in CASE_NAMES:
                continue
            checkpoint = Path(tmp) / name
            make_checkpoint(
                checkpoint, hidden=hidden, heads=heads,
                kv_heads=kv, intermediate=inner, vocabulary=vocab,
            )
            weights = load_hf_llama_local(
                checkpoint, declared_revision="f" * 40
            )
            record = benchmark(
                weights, prefixes=(4, 16, 32, 48),
                trials=trials,
            )
            entries.append({
                "geometry": {
                    "name": name, "hidden": hidden, "heads": heads,
                    "kv_heads": kv, "intermediate": inner,
                    "vocabulary": vocab, "layers": 1,
                },
                "decode_ratios_by_context": [
                    {
                        "past_tokens": case["past_tokens"],
                        "materialized_over_online": case[
                            "online_vs_materialized_decode_ratio"
                        ],
                    }
                    for case in record["cases"]
                ],
                "evidence": record,
            })
    if len(entries) != len(CASE_NAMES):
        raise RuntimeError("Synthetic geometry cases changed unexpectedly")
    return {
        "schema": "visuaml.native-online-attention-geometry.v31",
        "trials_per_case": trials,
        "platform": platform.platform(),
        "cases": entries,
        "limitations": [
            "Synthetic tiny Llama-shaped weights, not pretrained models.",
            "One runner, 1 layer, at most 49 context tokens in this sweep.",
            "No native allocator-only memory or physical energy measurement.",
            "A faster synthetic case does not justify global dispatch changes.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=3)
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
                "geometry": case["geometry"]["name"],
                "ratios": case["decode_ratios_by_context"],
            } for case in record["cases"]
        ],
        "output": str(args.output),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
