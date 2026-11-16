"""Reproducible opt-in CPU AOT microbenchmarks, NOT accelerator or SOTA rankings.

Run from visuaml-client with PYTHONPATH=backend OPENBLAS_NUM_THREADS=1
  python backend/examples/compiler_perf_bench_v15.py --trials 50 --output perf.json
Results depend on NumPy BLAS, machine load, interpreter version, and CPU.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
from pathlib import Path
from statistics import median
from time import perf_counter_ns

import numpy as np
import torch
from torch.nn import functional as F

from visuaml.compiler_lab.numpy_runtime import _execute
from visuaml.compiler_lab.stacked_decoder_v14 import (
    CompiledStackedDecoder, StackedWeights, empty_stack, reference_stack_step,
)


def _bench(fn, runs: int) -> dict[str, float]:
    for _ in range(5):
        fn()
    times = []
    for _ in range(runs):
        start = perf_counter_ns()
        fn()
        times.append((perf_counter_ns() - start) / 1e6)
    series = sorted(times)
    return {
        "median_ms": round(median(times), 5),
        "p95_ms": round(series[min(len(series) - 1, int(0.95 * len(series)))], 5),
        "min_ms": round(series[0], 5),
    }


def benchmark(*, runs: int = 50) -> dict:
    if type(runs) is not int or not 5 <= runs <= 10000:
        raise ValueError("Need 5..10000 bounded benchmark repetitions")
    cases = []
    for layers in (1, 2, 4):
        w = StackedWeights.seeded(19, layers=layers)
        driver = CompiledStackedDecoder(w)
        state = empty_stack(w, 1)
        tokens = np.array([[1]], dtype=np.int64)
        actual, _ = driver.step(tokens, state)
        expected, _ = reference_stack_step(w, tokens, state)
        np.testing.assert_allclose(actual, expected, rtol=5e-5, atol=5e-5)
        cases.append({"name": f"decoder_{layers}_layer", **_bench(lambda driver=driver, tokens=tokens, state=state: driver.step(tokens, state), runs)})

    generator = np.random.default_rng(21)
    conv = generator.normal(size=(1, 16, 48, 48)).astype(np.float32)
    filt = generator.normal(size=(32, 16, 3, 3)).astype(np.float32)
    bias = generator.normal(size=(32,)).astype(np.float32)
    reference = F.conv2d(torch.from_numpy(conv), torch.from_numpy(filt), torch.from_numpy(bias), padding=1).numpy()
    np.testing.assert_allclose(_execute('conv2d', (conv, filt, bias, 1, 1, 1, 1), {}), reference, rtol=2e-5, atol=2e-5)
    cases.append({"name": "conv2d_16to32_48x48", **_bench(lambda: _execute('conv2d', (conv, filt, bias, 1, 1, 1, 1), {}), runs)})
    return {
        "schema": "visuaml.cpu-performance-benchmark.v15",
        "source": "synthetic-seeded-CPU-fixtures",
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__,
                        "openblas_threads": os.environ.get('OPENBLAS_NUM_THREADS', 'unknown'),
                        "platform": platform.platform()},
        "trials_per_case": runs,
        "cases": cases,
        "warnings": ["Warm fixed-shape CPU inference, compilation time excluded.",
                     "No real checkpoint, GPU, photonic hardware, or end-to-end system claim.",
                     "Compare only under matched platform and BLAS thread settings."],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--trials',type=int,default=50)
    parser.add_argument('--output',type=Path)
    args = parser.parse_args()
    document = benchmark(runs=args.trials)
    serialized = json.dumps(document, sort_keys=True, indent=2, allow_nan=False)
    if args.output:
        args.output.write_text(serialized + '\n', encoding='utf-8')
    print(serialized)


if __name__ == '__main__':
    main()
