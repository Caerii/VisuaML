"""Matched v16/v17 CPU benchmark of semantic IR-backed KV effects.

The two modes run the same synthetic model mathematics; the v17 variant changes
cache allocation, write ordering and cursor commit management, not the whole
decoder kernel. No inference-serving, GPU, or real checkpoint conclusions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from visuaml.compiler_lab.stacked_decoder_v14 import StackedWeights
from visuaml.compiler_lab.streaming_decoder_v16 import StreamingStackedDecoder


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summary(data: list[float]) -> dict:
    return {
        "median_ms": round(statistics.median(data), 6),
        "p95_ms": round(float(np.percentile(data, 95)), 6),
        "minimum_ms": round(min(data), 6),
        "maximum_ms": round(max(data), 6),
        "mad_ms": round(statistics.median(abs(x - statistics.median(data)) for x in data), 6),
    }


def bench(*, layers: int, prompt_len: int, steps: int, trials: int, seed: int) -> dict:
    weights = StackedWeights.seeded(seed, layers=layers)
    a = StreamingStackedDecoder(weights)
    b = StreamingStackedDecoder(weights, semantic_kv=True)
    prompt = np.arange(1, prompt_len + 1, dtype=np.int64)[None, :] % a.vocab

    def execute(engine: StreamingStackedDecoder):
        return engine.generate(prompt, steps=steps)

    ids_a, state_a = execute(a)
    ids_b, state_b = execute(b)
    np.testing.assert_array_equal(ids_a, ids_b)
    for old, new in zip(state_a.layers, state_b.layers):
        np.testing.assert_allclose(old.keys, new.keys, atol=3e-5, rtol=3e-5)
        np.testing.assert_allclose(old.values, new.values, atol=3e-5, rtol=3e-5)
    # Alternate measurement order between repeat groups to reduce thermal bias.
    a_times, b_times = [], []
    for i in range(trials):
        first, second = (a, b) if i % 2 == 0 else (b, a)
        for engine in (first, second):
            start = time.perf_counter_ns()
            execute(engine)
            duration = (time.perf_counter_ns() - start) / 1e6
            (a_times if engine is a else b_times).append(duration)
    med_a, med_b = statistics.median(a_times), statistics.median(b_times)
    return {
        "layers": layers,
        "prompt_tokens": prompt_len,
        "generated_tokens": steps,
        "trials_per_mode": trials,
        "v16_direct": summary(a_times),
        "v17_semantic": summary(b_times),
        "speed_ratio_v16_over_v17": round(med_a / med_b, 5),
        "v17_semantic_sha256": b._kv_session.plan.semantic_sha256,
        "parity": "generated IDs and every layer's complete KV values",
        "v16_reserved_bytes": a.report().reserved_bytes,
        "v17_reserved_bytes": b.report().reserved_bytes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layers", nargs="+", type=int, default=[1, 2, 4])
    parser.add_argument("--prompt-len", type=int, default=4)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--trials", type=int, default=25)
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if (
        not args.layers or any(n not in (1, 2, 3, 4) for n in args.layers)
        or not 1 <= args.prompt_len <= 32
        or not 1 <= args.steps <= 24
        or args.prompt_len + args.steps > 64
        or not 5 <= args.trials <= 200
    ):
        parser.error("Inadmissible decoder shape, generation length, or sample count")
    root = Path(__file__).resolve().parents[1]
    sources = (
        "visuaml/compiler_lab/kv_lowering_v17.py",
        "visuaml/compiler_lab/streaming_decoder_v16.py",
        "examples/streaming_kv_bench_v17.py",
    )
    output = {
        "schema": "visuaml.streaming-semantic-benchmark.v17",
        "scope": "small seeded CPU models; identical NumPy math, distinct KV state effects",
        "sources_sha256": {name: sha(root / name) for name in sources},
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "processor": platform.processor() or platform.machine(),
        "openblas_num_threads_env": os.environ.get("OPENBLAS_NUM_THREADS", "unset"),
        "results": [
            bench(
                layers=n, prompt_len=args.prompt_len, steps=args.steps,
                trials=args.trials, seed=args.seed,
            )
            for n in args.layers
        ],
        "caveats": [
            "No compiled attention/MLP kernel or standalone AOT performance comparison.",
            "No physical device, pretrained checkpoint, or steady-state serving throughput.",
            "Python timing and BLAS threading settings materially affect these values.",
        ],
    }
    wire = json.dumps(output, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(wire, encoding="utf-8")
    print(wire)


if __name__ == "__main__":
    main()
