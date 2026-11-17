"""Process-isolated Linux RSS probe for native Llama execution variants.

Each sample uses a fresh process. VmRSS is instantaneous and VmHWM is the
process high-water mark, NOT a native allocator-only peak. Both include Python,
checkpoint loading, compiler and shared libraries. This is an observation,
not proof of reduced physical allocations or an energy/serving benchmark.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
from visuaml.compiler_lab.hf_llama_checkpoint_v19 import load_hf_llama_local
from visuaml.compiler_lab.llama_semantic_v20 import (
    lower_llama_semantic_program,
    make_llama_semantic_program,
)
from visuaml.compiler_lab.native_cpu_v28 import NativeCpuLlamaCompiler


def linux_memory_kib() -> dict[str, int]:
    """Read Linux /proc process resident memory and peak, not virtual size."""
    status = Path("/proc/self/status")
    if not status.is_file():
        raise RuntimeError("This physical memory probe currently requires Linux /proc")
    metrics = {}
    for line in status.read_text(encoding="utf-8").splitlines():
        name, sep, value = line.partition(":")
        if sep and name in ("VmRSS", "VmHWM"):
            parts = value.strip().split()
            if len(parts) != 2 or parts[1] != "kB":
                raise RuntimeError("Unexpected Linux proc memory unit")
            metrics[name + "_kib"] = int(parts[0])
    if set(metrics) != {"VmRSS_kib", "VmHWM_kib"}:
        raise RuntimeError("Incomplete physical resident-memory fields")
    return metrics


def worker(checkpoint: Path, revision: str, variant: str) -> dict:
    if variant not in ("four_row", "scalar_reference", "avx2"):
        raise ValueError("Unknown native kernel variant")
    before = linux_memory_kib()
    weights = load_hf_llama_local(checkpoint, declared_revision=revision)
    plan = lower_llama_semantic_program(
        make_llama_semantic_program(weights), weights
    )
    after_checkpoint = linux_memory_kib()
    compiler = NativeCpuLlamaCompiler(
        plan, reference_kernels=variant == "scalar_reference",
        enable_avx2=variant == "avx2",
    )
    after_compilation = linux_memory_kib()
    session = compiler.new_session()
    tokens = np.array([[1, 2, 3, 4]], dtype=np.int64)
    if weights.profile.vocabulary <= 4:
        tokens %= weights.profile.vocabulary
    logits = session.step(tokens)
    assert np.isfinite(logits).all()
    # One additional streaming step covers live KV ownership.
    session.step(tokens[:, :1])
    after_execution = linux_memory_kib()
    return {
        "schema": "visuaml.native-process-rss.v29",
        "requested_variant": variant,
        "effective_variant": compiler.report()["linear_kernel"],
        "avx2_active": compiler.report()["avx2_active"],
        "checkpoint_sha256": weights.checkpoint_sha256,
        "native_source_sha256": compiler.report()["cpp_source_sha256"],
        "compiler_identity": compiler.report()["compiler_identity"],
        "platform": platform.platform(),
        "before_load": before,
        "after_checkpoint": after_checkpoint,
        "after_compilation": after_compilation,
        "after_execution": after_execution,
        "limitations": (
            "Linux process VmRSS/VmHWM include Python, weights, compiler, "
            "runtime and shared libraries; peak RSS cannot be attributed "
            "to native scratch allocation in isolation."
        ),
    }


def compare(checkpoint: Path, revision: str, repetitions: int) -> dict:
    if type(repetitions) is not int or not 1 <= repetitions <= 20:
        raise ValueError("Expected 1..20 fresh-process repeats")
    if not Path("/proc/self/status").is_file():
        raise RuntimeError("RSS comparison requires Linux /proc")
    samples = []
    script = Path(__file__).resolve()
    for run in range(repetitions):
        order = ("four_row", "scalar_reference", "avx2")
        if run % 2:
            order = tuple(reversed(order))
        for variant in order:
            child = subprocess.run(
                (
                    sys.executable, str(script), str(checkpoint),
                    "--revision", revision, "--worker", variant,
                ),
                capture_output=True, text=True, check=True, timeout=180,
            )
            samples.append(json.loads(child.stdout))
    hashes = {sample["checkpoint_sha256"] for sample in samples}
    if len(hashes) != 1:
        raise RuntimeError("Checkpoint identity changed between processes")
    return {
        "schema": "visuaml.native-process-rss-comparison.v29",
        "repetitions": repetitions,
        "samples": samples,
        "interpretation": (
            "Compare process RSS and high-water distributions only; "
            "this experiment cannot isolate native C++ allocations or "
            "establish a reliable speedup."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument(
        "--worker", choices=("four_row", "scalar_reference", "avx2")
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    record = (
        worker(args.checkpoint, args.revision, args.worker)
        if args.worker else compare(
            args.checkpoint, args.revision, args.repetitions
        )
    )
    serialized = json.dumps(record, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
    print(serialized)


if __name__ == "__main__":
    main()
