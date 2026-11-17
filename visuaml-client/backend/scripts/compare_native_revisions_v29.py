"""Matched-runner v28-v29 comparison; exploratory, not a serving benchmark.

Fetch one pinned checkpoint, verify exact hashes, and benchmark the v28 source
and current branch in alternating process order on one runner. Every individual
benchmark verifies its logits and final KV state against independent NumPy.
Even paired runs do not eliminate CI scheduling, DVFS and thermal noise.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

from huggingface_hub import snapshot_download

BASE_COMMIT = "a2e2042cca69028bce528bef86246ec7f387bd5b"
MODEL_ID = "manojredhat/tiny-llama"
REVISION = "756e9746eb801752d69771ef3e9bc433b21d853d"
WEIGHTS_SHA = "d910edc7031721de23a7af111e355e6c09d5f1b3a2c43c0b6b0aa6a9135ffe03"
CONFIG_SHA = "b53aec3cd6cfb1be4b96cadf8ca0c5b9e0279d751100dbebd8f93ff6564c0e02"
SCRIPT = Path("visuaml-client/backend/examples/native_cpu_bench_v28.py")
BACKEND = Path("visuaml-client/backend")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def benchmark_source(
    root: Path, checkpoint: Path, *, trials: int, simd: bool
) -> dict:
    argv = [
        sys.executable, str(root / SCRIPT), str(checkpoint),
        "--revision", REVISION, "--prompt", "4", "--chunks",
        "1", "1", "2", "--trials", str(trials),
    ]
    if simd:
        argv.extend(("--compare-native-reference", "--compare-simd"))
    env = os.environ.copy()
    env.update({
        "PYTHONPATH": str(root / BACKEND),
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "PYTHONNOUSERSITE": "1",
    })
    completed = subprocess.run(
        argv, cwd=root, env=env,
        check=True, capture_output=True, text=True, timeout=240,
    )
    result = json.loads(completed.stdout)
    if result["checkpoint_sha256"] != WEIGHTS_SHA:
        raise RuntimeError("Benchmark did not use the pinned checkpoint")
    if result["source_model_revision"] != REVISION:
        raise RuntimeError("Benchmark model revision mismatch")
    if result["trials"] != trials or result["chunks"] != [4, 1, 1, 2]:
        raise RuntimeError("Benchmark workload changed")
    if simd and len(result.get("raw_samples_ms", {}).get("native_cpp", [])) != trials:
        raise RuntimeError("Missing current-branch individual native samples")
    if result["latency"]["native_cpp"]["median_ms"] <= 0:
        raise RuntimeError("Invalid measured native latency")
    if simd and "native_avx2" not in result["latency"]:
        raise RuntimeError("Current branch lacks guarded SIMD comparator")
    return result


def cpu_label() -> str:
    info = Path("/proc/cpuinfo")
    if info.exists():
        for line in info.read_text(encoding="utf-8").splitlines():
            if line.startswith("model name"):
                return line.partition(":")[2].strip()[:160]
    return platform.processor() or platform.machine()


def run(*, pairs: int, trials: int) -> dict:
    if type(pairs) is not int or not 1 <= pairs <= 5:
        raise ValueError("pairs must be in 1..5")
    if type(trials) is not int or not 2 <= trials <= 100:
        raise ValueError("trials must be in 2..100")
    root = Path(__file__).resolve().parents[3]
    if not (root / ".git").exists():
        raise RuntimeError("Expected a Git checkout to compare source revisions")
    with tempfile.TemporaryDirectory(prefix="visuaml-v29-paired-") as temp:
        location = Path(temp)
        baseline = location / "baseline"
        subprocess.run(
            ["git", "worktree", "add", "--detach", str(baseline), BASE_COMMIT],
            cwd=root, check=True, capture_output=True, text=True, timeout=90,
        )
        try:
            checkpoint = Path(snapshot_download(
                MODEL_ID, revision=REVISION,
                allow_patterns=("config.json", "model.safetensors"),
                local_dir=location / "checkpoint",
            ))
            if sha256(checkpoint / "model.safetensors") != WEIGHTS_SHA:
                raise RuntimeError("Pinned safetensors hash mismatch")
            if sha256(checkpoint / "config.json") != CONFIG_SHA:
                raise RuntimeError("Pinned model configuration hash mismatch")
            observations = []
            for index in range(pairs):
                order = ("v28", "v29") if index % 2 == 0 else ("v29", "v28")
                results = {}
                for version in order:
                    results[version] = benchmark_source(
                        baseline if version == "v28" else root,
                        checkpoint, trials=trials, simd=version == "v29",
                    )
                a, b = results["v28"], results["v29"]
                if a["semantic_sha256"] != b["semantic_sha256"]:
                    raise RuntimeError("Semantic program changed across versions")
                old_ms = a["latency"]["native_cpp"]["median_ms"]
                avx_ms = b["latency"]["native_avx2"]["median_ms"]
                saved_ms = old_ms - avx_ms
                incremental_compile_ms = (
                    b["compilation"]["native_avx2_ms"]
                    - a["compilation"]["native_cpp_ms"]
                )
                break_even = (
                    math.ceil(max(0.0, incremental_compile_ms) / saved_ms)
                    if saved_ms > 0 else None
                )
                observations.append({
                    "estimated_warm_runs_to_amortize_extra_avx_compile": break_even,
                    "incremental_avx_compile_ms": round(incremental_compile_ms, 6),
                    "warm_time_saved_per_run_ms": round(saved_ms, 6),
                    "order": list(order),
                    "baseline_native_ms": a["latency"]["native_cpp"]["median_ms"],
                    "v29_four_row_ms": b["latency"]["native_cpp"]["median_ms"],
                    "v29_avx2_ms": b["latency"]["native_avx2"]["median_ms"],
                    "baseline_to_four_row_ratio": round(
                        a["latency"]["native_cpp"]["median_ms"]
                        / b["latency"]["native_cpp"]["median_ms"], 6
                    ),
                    "baseline_to_avx2_ratio": round(
                        a["latency"]["native_cpp"]["median_ms"]
                        / b["latency"]["native_avx2"]["median_ms"], 6
                    ),
                    "baseline": a,
                    "current": b,
                })
            memory_record = None
            if Path("/proc/self/status").is_file():
                env = os.environ.copy()
                env["PYTHONPATH"] = str(root / BACKEND)
                env["OPENBLAS_NUM_THREADS"] = "1"
                memory_probe = root / "visuaml-client/backend/examples/native_cpu_memory_v29.py"
                measurement = subprocess.run(
                    (
                        sys.executable, str(memory_probe), str(checkpoint),
                        "--revision", REVISION, "--repetitions", "2",
                    ),
                    cwd=root, env=env, capture_output=True,
                    text=True, check=True, timeout=360,
                )
                memory_record = json.loads(measurement.stdout)
                if any(
                    item["checkpoint_sha256"] != WEIGHTS_SHA
                    for item in memory_record["samples"]
                ):
                    raise RuntimeError("RSS probe checkpoint mismatch")
            return {
                "schema": "visuaml.paired-native-v28-v29.v1",
                "physical_memory_probe": memory_record,
                "base_commit": BASE_COMMIT,
                "model": MODEL_ID,
                "model_revision": REVISION,
                "weights_sha256": WEIGHTS_SHA,
                "cpu": cpu_label(),
                "platform": platform.platform(),
                "pairs": pairs,
                "trials_per_backend_per_pair": trials,
                "observations": observations,
                "median_baseline_to_four_row_ratio": statistics.median(
                    p["baseline_to_four_row_ratio"] for p in observations
                ),
                "median_baseline_to_avx2_ratio": statistics.median(
                    p["baseline_to_avx2_ratio"] for p in observations
                ),
                "limitations": [
                    "One CI runner; subject to contention, clock scaling and thermal drift.",
                    "Short pinned Llama workload, not production token throughput.",
                    "The baseline is a separate process but uses the same checkpoint and toolchain.",
                    "Compile amortization uses per-process build and median warm latency only; it omits startup, concurrent work, thermal shifts and serving overhead.",
                    "Baseline v28 retains aggregate timings; v29 retains raw timing samples.",
                    "This is a comparison experiment, not a general speedup guarantee.",
                ],
            }
        finally:
            subprocess.run(
                ["git", "worktree", "remove", "--force", str(baseline)],
                cwd=root, check=False, capture_output=True, timeout=30,
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=int, default=2)
    parser.add_argument("--trials", type=int, default=7)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    record = run(pairs=args.pairs, trials=args.trials)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "schema": record["schema"],
        "base_commit": record["base_commit"],
        "pairs": record["pairs"],
        "median_baseline_to_four_row_ratio": record[
            "median_baseline_to_four_row_ratio"
        ],
        "median_baseline_to_avx2_ratio": record[
            "median_baseline_to_avx2_ratio"
        ],
        "artifact": str(args.output),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
