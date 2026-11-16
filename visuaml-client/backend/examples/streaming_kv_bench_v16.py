"""Reproducible warm CPU KV adapter comparison, not a device benchmark.

Run with OPENBLAS_NUM_THREADS=1 (set BEFORE importing NumPy) and an isolated CPU.
Measures end-to-end autoregressive decode after warming both adapters; the AOT
baseline shape-specializes once per cache length during warmup. Setup and
compilation are not part of the timed section. Reports medians with trial counts.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import time
from pathlib import Path

import numpy as np
import torch

from visuaml.compiler_lab.stacked_decoder_v14 import (
    CompiledStackedDecoder,
    StackedWeights,
)
from visuaml.compiler_lab.streaming_decoder_v16 import StreamingStackedDecoder


def _median_ms(fn, trials: int) -> float:
    durations = []
    for _ in range(trials):
        start = time.perf_counter_ns()
        fn()
        durations.append((time.perf_counter_ns() - start) / 1e6)
    return statistics.median(durations)


def bench(layers: int, *, seed: int, steps: int, trials: int) -> dict:
    weights = StackedWeights.seeded(seed, layers=layers)
    tokens = np.array([[1, 2, 3, 4]], np.int64)
    compiled = CompiledStackedDecoder(weights)
    streaming = StreamingStackedDecoder(weights, batch=1)
    # First warm both. In particular, compile every fixed-past AOT specialization.
    id_a, _ = compiled.generate(tokens, steps=steps)
    id_b, _ = streaming.generate(tokens, steps=steps)
    np.testing.assert_array_equal(id_a, id_b)
    aot = _median_ms(lambda: compiled.generate(tokens, steps=steps), trials)
    resident = _median_ms(lambda: streaming.generate(tokens, steps=steps), trials)
    if not aot > 0 or not resident > 0:
        raise ValueError("Invalid benchmark duration")
    return {
        "layers": layers,
        "prompt_tokens": tokens.shape[1],
        "new_tokens": steps,
        "trials": trials,
        "aot_warm_median_ms": round(aot, 5),
        "resident_warm_median_ms": round(resident, 5),
        "speedup_vs_shape_specialized_aot": round(aot / resident, 3),
        "aot_shape_specializations": len(compiled._compiled),
        "resident_cache_allocations": len(streaming._keys) * 2,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layers", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--trials", type=int, default=9)
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not 2 <= args.trials <= 50 or not 1 <= args.steps <= 12:
        parser.error("Requires 2..50 trials and 1..12 generated tokens")
    if not args.layers or any(n not in (1, 2, 3, 4) for n in args.layers):
        parser.error("Layer count must be between 1 and 4")
    results = [bench(n, seed=args.seed, steps=args.steps, trials=args.trials) for n in args.layers]
    payload = {
        "schema": "visuaml.streaming-kv-benchmark.v16",
        "scope": "Warm synthetic CPU decoder generation; no physical device measurements",
        "backend_a": "v15 prepared NumPy-AOT fixed-shape driver",
        "backend_b": "v16 resident KV NumPy CPU adapter (not AOT compiler)",
        "numerical_parity": "Exact generated token IDs after warmup; full tensor parity in separate tests",
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "openblas_num_threads_env": os.getenv("OPENBLAS_NUM_THREADS", "unset"),
        "cpu": platform.processor() or platform.machine(),
        "results": results,
        "caveats": [
            "Warm runs exclude first-time compilation and optimizer startup.",
            "Results depend strongly on Python/BLAS/CPU versions and thread counts.",
            "Small synthetic models are not predictive of production LLM throughput.",
            "v16 is a bounded stateful software runtime, not portable NumPy-AOT lowering.",
        ],
    }
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
