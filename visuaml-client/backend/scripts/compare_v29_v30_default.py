"""Same-runner v29-versus-v30 default CPU regression experiment.

Both builds use the same short pinned pretrained Llama checkpoint, workload,
compiler toolchain, and non-batched AVX2 mode. This detects v30 refactoring
regressions independently of the experimental batched variant. Comparisons
are paired and alternate process order. Not a production serving benchmark.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

from huggingface_hub import snapshot_download

BASE_COMMIT = "bd3086c8e7f3ea35f588a290bdeb4f149ec2b8d4"
MODEL_ID = "manojredhat/tiny-llama"
REVISION = "756e9746eb801752d69771ef3e9bc433b21d853d"
WEIGHTS_SHA = "d910edc7031721de23a7af111e355e6c09d5f1b3a2c43c0b6b0aa6a9135ffe03"
CONFIG_SHA = "b53aec3cd6cfb1be4b96cadf8ca0c5b9e0279d751100dbebd8f93ff6564c0e02"
BENCHMARK = Path("visuaml-client/backend/examples/native_cpu_bench_v28.py")
BACKEND = Path("visuaml-client/backend")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def timed(root: Path, checkpoint: Path, trials: int) -> dict:
    env = os.environ.copy()
    env.update({
        "PYTHONPATH": str(root / BACKEND),
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "PYTHONNOUSERSITE": "1",
    })
    command = (
        sys.executable, str(root / BENCHMARK), str(checkpoint),
        "--revision", REVISION, "--prompt", "8", "--chunks", "1", "1",
        "--trials", str(trials), "--compare-simd",
    )
    child = subprocess.run(
        command, cwd=root, env=env, capture_output=True,
        text=True, check=True, timeout=240,
    )
    record = json.loads(child.stdout)
    if (
        record["checkpoint_sha256"] != WEIGHTS_SHA
        or record["source_model_revision"] != REVISION
        or record["chunks"] != [8, 1, 1]
        or record["trials"] != trials
        or len(record["raw_samples_ms"]["native_avx2"]) != trials
    ):
        raise RuntimeError("Incompatible native AVX2 benchmark evidence")
    return record


def run(*, pairs: int = 2, trials: int = 7) -> dict:
    if (
        type(pairs) is not int or not 1 <= pairs <= 5
        or type(trials) is not int or not 2 <= trials <= 100
    ):
        raise ValueError("Invalid bounded paired comparison")
    root = Path(__file__).resolve().parents[3]
    if not (root / ".git").exists():
        raise RuntimeError("Expected Git checkout with pinned parent revision")
    with tempfile.TemporaryDirectory(prefix="visuaml-v29-v30-") as temporary:
        directory = Path(temporary)
        baseline = directory / "baseline"
        subprocess.run(
            ("git", "worktree", "add", "--detach", str(baseline), BASE_COMMIT),
            cwd=root, capture_output=True, text=True, check=True, timeout=90,
        )
        try:
            checkpoint = Path(snapshot_download(
                repo_id=MODEL_ID, revision=REVISION,
                allow_patterns=("config.json", "model.safetensors"),
                local_dir=directory / "checkpoint",
            ))
            if (
                sha256(checkpoint / "model.safetensors") != WEIGHTS_SHA
                or sha256(checkpoint / "config.json") != CONFIG_SHA
            ):
                raise RuntimeError("Pinned trained model changed")
            observations = []
            for index in range(pairs):
                order = ("v29", "v30") if index % 2 == 0 else ("v30", "v29")
                records = {}
                for version in order:
                    records[version] = timed(
                        baseline if version == "v29" else root,
                        checkpoint, trials,
                    )
                old, new = records["v29"], records["v30"]
                if old["semantic_sha256"] != new["semantic_sha256"]:
                    raise RuntimeError("Semantic identity differed between revisions")
                before = old["latency"]["native_avx2"]["median_ms"]
                after = new["latency"]["native_avx2"]["median_ms"]
                observations.append({
                    "order": list(order),
                    "old_avx2_ms": before,
                    "new_default_avx2_ms": after,
                    "baseline_to_current_ratio": round(before / after, 6),
                    "v29": old,
                    "v30": new,
                })
            return {
                "schema": "visuaml.native-v29-to-v30-default.v1",
                "base_commit": BASE_COMMIT,
                "checkpoint_sha256": WEIGHTS_SHA,
                "model_revision": REVISION,
                "platform": platform.platform(),
                "pairs": pairs,
                "trials_per_backend_per_pair": trials,
                "observations": observations,
                "median_paired_ratio": round(statistics.median(
                    observation["baseline_to_current_ratio"]
                    for observation in observations
                ), 6),
                "limitations": [
                    "One GitHub shared runner and one tiny pretrained Llama workload.",
                    "Default AVX2 path only; does not measure optional batched path.",
                    "Shared runner DVFS, contention, and temperature introduce noise.",
                    "A ratio near unity is not evidence of a robust throughput win.",
                ],
            }
        finally:
            subprocess.run(
                ("git", "worktree", "remove", "--force", str(baseline)),
                cwd=root, capture_output=True, check=False, timeout=30,
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=int, default=2)
    parser.add_argument("--trials", type=int, default=7)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(pairs=args.pairs, trials=args.trials)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "schema": result["schema"],
        "base_commit": result["base_commit"],
        "pairs": result["pairs"],
        "median_paired_ratio": result["median_paired_ratio"],
        "output": str(args.output),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
