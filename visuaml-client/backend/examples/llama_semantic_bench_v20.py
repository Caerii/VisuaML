"""Matched local Llama forward benchmark for v19 reference and v20 IR backends.

The input is an already downloaded, strict local safetensors checkpoint. No
network fetches, arbitrary Python checkpoint code, or hardware speedup claims.
Numerical parity is checked before reporting timings.
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

from visuaml.compiler_lab.hf_llama_checkpoint_v19 import (
    LlamaLocalCheckpoint,
    load_hf_llama_local,
)
from visuaml.compiler_lab.llama_semantic_v20 import (
    lower_llama_semantic_program,
    make_llama_semantic_program,
)
from visuaml.compiler_lab.memory_planner_v19 import plan_decoder_memory


def _summary(samples: list[float]) -> dict:
    center = statistics.median(samples)
    return {
        "median_ms": round(center, 6),
        "p95_ms": round(float(np.percentile(samples, 95)), 6),
        "mad_ms": round(statistics.median(abs(x - center) for x in samples), 6),
        "minimum_ms": round(min(samples), 6),
    }


def benchmark(
    model: LlamaLocalCheckpoint, *, lengths: tuple[int, ...] = (4, 8, 16),
    trials: int = 9,
    include_hybrid: bool = True,
) -> dict:
    if (
        type(trials) is not int or not 2 <= trials <= 100
        or not lengths or any(
            type(n) is not int or not 1 <= n <= min(32, model.profile.context, 128)
            for n in lengths
        )
    ):
        raise ValueError("Benchmark workloads must be small, finite and admitted")
    started = time.perf_counter_ns()
    program = make_llama_semantic_program(model)
    plan = lower_llama_semantic_program(program, model)
    compilation_ms = (time.perf_counter_ns() - started) / 1e6
    backends = ("numpy", "torchscript_hybrid") if include_hybrid else ("numpy",)
    results = []
    for n in lengths:
        tokens = (np.arange(1, n + 1, dtype=np.int64) % model.profile.vocabulary)[None, :]
        oracle = model.forward(tokens)
        engines = {name: plan.new_session(backend=name) for name in backends}
        for engine in engines.values():
            np.testing.assert_allclose(engine.step(tokens), oracle, rtol=5e-4, atol=5e-4)
        measured = {"reference_v19": [], **{name: [] for name in backends}}
        for trial in range(trials):
            options = [("reference_v19", None)]
            options.extend((name, engine) for name, engine in engines.items())
            if trial % 2:
                options.reverse()
            for name, engine in options:
                start = time.perf_counter_ns()
                if engine is None:
                    model.forward(tokens)
                else:
                    engine.reset()
                    engine.step(tokens)
                measured[name].append((time.perf_counter_ns() - start) / 1e6)
        results.append({
            "tokens": n,
            "latency": {key: _summary(samples) for key, samples in measured.items()},
            "torchscript_specializations": (
                engines["torchscript_hybrid"].report()["compiled_primitive_specializations"]
                if include_hybrid else 0
            ),
            "parity": "full logits verified against independent local reference",
        })
    return {
        "schema": "visuaml.llama-semantic-cpu-benchmark.v20",
        "model_revision": model.declared_revision,
        "checkpoint_sha256": model.checkpoint_sha256,
        "config_sha256": model.config_sha256,
        "semantic_sha256": plan.semantic_sha256,
        "graph_setup_ms": round(compilation_ms, 6),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "cpu": platform.processor() or platform.machine(),
        "openblas_num_threads": os.environ.get("OPENBLAS_NUM_THREADS", "unset"),
        "logical_liveness": plan_decoder_memory(
            program, maximum_query=min(32, plan.capacity),
            maximum_context=min(64, plan.capacity),
        ).as_dict(),
        "results": results,
        "limitations": [
            "CPU only, small bounded prompts, no serving concurrency.",
            "TorchScript is a hybrid primitive compiler, not a full compiled model.",
            "Physical RSS, GPU time, compile cost of individual TorchScript kernels and energy not measured.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--lengths", type=int, nargs="+", default=[4, 8, 16])
    parser.add_argument("--trials", type=int, default=9)
    parser.add_argument("--no-hybrid", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    model = load_hf_llama_local(args.checkpoint, declared_revision=args.revision)
    result = benchmark(
        model, lengths=tuple(args.lengths), trials=args.trials,
        include_hybrid=not args.no_hybrid,
    )
    wire = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(wire, encoding="utf-8")
    print(wire)


if __name__ == "__main__":
    main()
