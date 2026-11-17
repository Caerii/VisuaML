"""v36: independently replay the original pinned Llama numerical experiment.

The candidate historical commit is admitted by exact source *byte* fingerprints,
not by its convenient branch name or commit date. The old source is executed in
a detached worktree by a trusted GitHub runner; the original historical CI result
is never substituted for a fresh run. This script does NOT import model Python.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import tempfile
from pathlib import Path

CANDIDATE_COMMIT = "a2e2042cca69028bce528bef86246ec7f387bd5b"
ARCHIVE = "benchmarks/pretrained_llama_native_cpu_benchmark_v28.json"
BACKEND = "visuaml-client/backend"
PROBE = "visuaml-client/backend/scripts/probe_native_llama_v28.py"
SOURCE_KEYS = (
    "examples/native_cpu_bench_v28.py",
    "visuaml/compiler_lab/native_cpu_v28.cpp",
    "visuaml/compiler_lab/native_cpu_v28.py",
)
ATOL = 3e-4
RTOL = 3e-4


class ReplayError(ValueError):
    """A historical source or independently measured result is not admitted."""


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def git(repo: Path, *args: str, timeout: int = 30) -> bytes:
    try:
        return subprocess.run(
            ("git", *args),
            cwd=repo,
            check=True,
            capture_output=True,
            timeout=timeout,
        ).stdout
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise ReplayError("Pinned historical Git source is unavailable") from exc


def load_archive(repo: Path) -> tuple[dict, str]:
    source = repo / ARCHIVE
    raw = source.read_bytes()
    if not 1 <= len(raw) <= 1024 * 1024:
        raise ReplayError("Historical evidence artifact exceeds byte budget")
    try:
        record = json.loads(raw)
    except (ValueError, UnicodeDecodeError) as exc:
        raise ReplayError("Historical evidence is not strict JSON") from exc
    if (
        not isinstance(record, dict)
        or record.get("schema") != "visuaml.pinned-native-llama-v28"
        or record.get("source") != "manojredhat/tiny-llama"
        or record.get("chunked_KV_and_greedy_parity") is not True
        or record.get("evidence_provenance", {}).get("conclusion") != "success"
    ):
        raise ReplayError("Unadmitted historical pretrained experiment")
    return record, sha(raw)


def discover_historical_sources(repo: Path) -> dict:
    """Locate exact observed source-byte versions across full Git history.

    A version matching every tracked source digest is a *candidate checkout*,
    not proof of the source commit of the original GitHub Actions run.
    If no common commit exists, report each verifiable file version instead of
    inventing a checkout that never existed.
    """
    archive, archived_sha = load_archive(repo)
    expected = archive["matched_cpu_benchmark"]["source_sha256"]
    if set(expected) != set(SOURCE_KEYS):
        raise ReplayError("Unrecognized historical source inventory")
    matching: dict[str, list[str]] = {}
    counts: dict[str, int] = {}
    for relative in SOURCE_KEYS:
        path = f"{BACKEND}/{relative}"
        output = git(
            repo, "log", "--all", "--format=%H", "--", path, timeout=120,
        )
        candidates = list(dict.fromkeys(
            line for line in output.decode("ascii").splitlines() if line
        ))
        if not candidates or len(candidates) > 5000:
            raise ReplayError("No bounded historical source history")
        counts[relative] = len(candidates)
        found = []
        for commit in candidates:
            try:
                original = git(repo, "show", f"{commit}:{path}", timeout=20)
            except ReplayError:
                continue
            if sha(original) == expected[relative]:
                found.append(commit)
        matching[relative] = found
    common = set.intersection(*(set(matches) for matches in matching.values()))
    return {
        "schema": "visuaml.historical-source-discovery.v36",
        "archive_sha256": archived_sha,
        "candidate_checked": CANDIDATE_COMMIT,
        "candidate_is_complete_match": CANDIDATE_COMMIT in common,
        "complete_matching_commits": sorted(common),
        "file_matching_commits": matching,
        "file_revision_counts": counts,
        "found_all_files": all(matching.values()),
        "limitations": (
            "A source file digest match verifies exact bytes in a Git object. "
            "It does not identify which Git commit the original CI executed."
        ),
    }


def verify_source(repo: Path, commit: str = CANDIDATE_COMMIT) -> dict:
    if commit != CANDIDATE_COMMIT:
        raise ReplayError("Historical commit is not on the source allowlist")
    record, archive_hash = load_archive(repo)
    resolved = git(repo, "rev-parse", "--verify", f"{commit}^{{commit}}").decode().strip()
    if resolved != commit:
        raise ReplayError("Historical source Git object identity mismatch")
    legacy = record["matched_cpu_benchmark"]["source_sha256"]
    if set(legacy) != set(SOURCE_KEYS):
        raise ReplayError("Unexpected source file contract in historical archive")
    source_hashes = {}
    for relative in SOURCE_KEYS:
        historical_bytes = git(repo, "show", f"{commit}:{BACKEND}/{relative}")
        digest = sha(historical_bytes)
        if digest != legacy[relative]:
            raise ReplayError(f"Historical source digest mismatch: {relative}")
        source_hashes[relative] = digest
    if source_hashes["visuaml/compiler_lab/native_cpu_v28.cpp"] != record[
        "backend"
    ]["cpp_source_sha256"]:
        raise ReplayError("Native C++ file fingerprint conflicts with archive")
    return {
        "schema": "visuaml.historical-source-recovery.v36",
        "candidate_commit": commit,
        "source_verified": True,
        "archive_sha256": archive_hash,
        "source_sha256": source_hashes,
        "original_ci": record["evidence_provenance"]["workflow_url"],
        "statement": (
            "Candidate Git commit matches every source file byte fingerprint; "
            "this does not prove the historical CI ran on this specific commit."
        ),
    }


def _finite(value: object, label: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ReplayError(f"Invalid nonfinite {label}")
    return float(value)


def verify_fresh(fresh: dict, historic: dict, source: dict) -> dict:
    if (
        not isinstance(fresh, dict)
        or fresh.get("schema") != "visuaml.pinned-native-llama-v28"
        or fresh.get("source") != historic["source"]
        or fresh.get("revision") != historic["revision"]
        or fresh.get("checkpoint_sha256") != historic["checkpoint_sha256"]
        or fresh.get("config_sha256") != historic["config_sha256"]
        or fresh.get("semantic_sha256") != historic["semantic_sha256"]
        or fresh.get("chunked_KV_and_greedy_parity") is not True
    ):
        raise ReplayError("Fresh numerical experiment violates pinned model contract")
    a, b = fresh["backend"], historic["backend"]
    if (
        a["cpp_source_sha256"] != b["cpp_source_sha256"]
        or a["checkpoint_sha256"] != b["checkpoint_sha256"]
        or a["semantic_sha256"] != b["semantic_sha256"]
    ):
        raise ReplayError("Fresh native experiment changed source or semantic identity")
    benchmark = fresh["matched_cpu_benchmark"]
    historical_benchmark = historic["matched_cpu_benchmark"]
    if (
        benchmark["schema"] != historical_benchmark["schema"]
        or benchmark["source_sha256"] != source["source_sha256"]
        or benchmark["checkpoint_sha256"] != historic["checkpoint_sha256"]
        or benchmark["semantic_sha256"] != historic["semantic_sha256"]
        or benchmark["source_model_revision"] != historic["revision"]
        or benchmark["chunks"] != historical_benchmark["chunks"]
        or benchmark["trials"] != historical_benchmark["trials"]
        or benchmark["numerical_certificate"]
           != "all chunked logits and final layer KV match NumPy IR"
    ):
        raise ReplayError("Fresh benchmark changed workload or numerical evidence")
    expected = historic["witnesses"]
    observed = fresh["witnesses"]
    if not isinstance(observed, list) or len(observed) != len(expected):
        raise ReplayError("Fresh experiment omitted historical witnesses")
    max_errors = {"hf": 0.0, "semantic": 0.0}
    for expected_witness, actual in zip(expected, observed, strict=True):
        if (
            actual["tokens"] != expected_witness["tokens"]
            or actual["logit_shape"] != expected_witness["logit_shape"]
        ):
            raise ReplayError("Fresh witness tokens or logits shape changed")
        hf = _finite(actual["max_abs_error_against_HF"], "HF logits")
        ir = _finite(actual["max_abs_error_against_semantic_IR"], "semantic logits")
        if hf < 0 or ir < 0 or hf > ATOL or ir > ATOL:
            raise ReplayError("Fresh pretrained numerical tolerance exceeded")
        max_errors["hf"] = max(max_errors["hf"], hf)
        max_errors["semantic"] = max(max_errors["semantic"], ir)
    if _finite(a["independent_max_abs_error"], "native independent error") > ATOL:
        raise ReplayError("Native compiler reported excessive error")
    for item in ("native_cpp", "numpy_semantic", "torchscript_whole"):
        value = _finite(benchmark["latency"][item]["median_ms"], "warm median")
        if not value > 0:
            raise ReplayError("Missing positive warm timing")
    return {
        "passed": True,
        "witness_count": len(observed),
        "max_abs_error_hf": max_errors["hf"],
        "max_abs_error_semantic": max_errors["semantic"],
        "native_cpp_source_sha256": a["cpp_source_sha256"],
        "native_executable_sha256": a["executable_sha256"],
        "native_compiler_identity": a["compiler_identity"],
        "native_compile_ms": a["compilation_ms"],
        "warm_native_median_ms": benchmark["latency"]["native_cpp"]["median_ms"],
        "tolerance_atol": ATOL,
        "tolerance_rtol": RTOL,
    }


def execute(repo: Path) -> dict:
    recovery = verify_source(repo)
    historical, _ = load_archive(repo)
    with tempfile.TemporaryDirectory(prefix="visuaml-v36-historical-") as directory:
        checkout = Path(directory) / "source"
        git(repo, "worktree", "add", "--detach", str(checkout), CANDIDATE_COMMIT, timeout=90)
        try:
            environment = os.environ.copy()
            environment.update({
                "PYTHONPATH": str(checkout / BACKEND),
                "PYTHONNOUSERSITE": "1",
                "HF_HUB_DISABLE_TELEMETRY": "1",
                "HF_HUB_DISABLE_XET": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "TOKENIZERS_PARALLELISM": "false",
            })
            try:
                run = subprocess.run(
                    (sys.executable, str(checkout / PROBE)),
                    cwd=checkout,
                    env=environment,
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=900,
                )
                fresh = json.loads(run.stdout)
            except (OSError, ValueError, subprocess.CalledProcessError,
                    subprocess.TimeoutExpired) as exc:
                raise ReplayError("Trusted historical pretrained replay failed") from exc
            admission = verify_fresh(fresh, historical, recovery)
            record = {
                "schema": "visuaml.pinned-pretrained-independent-replay.v36",
                "status": "fresh_independent_numerical_witness_passed",
                "historical_source_recovery": recovery,
                "checkpoint_sha256": historical["checkpoint_sha256"],
                "config_sha256": historical["config_sha256"],
                "model_revision": historical["revision"],
                "semantic_sha256": historical["semantic_sha256"],
                "fresh_result_sha256": sha(json.dumps(
                    fresh, indent=2, sort_keys=True, allow_nan=False
                ).encode()),
                "fresh_numerical_result": fresh,
                "verification": admission,
                "environment": {
                    "python": platform.python_version(),
                    "os": platform.platform(),
                    "machine": platform.machine(),
                },
                "limitations": [
                    "Fresh finite numerical witness, not formal all-input equivalence.",
                    "Checkpoint comes from a pinned Hugging Face revision; network download is limited to two named files.",
                    "Exact historical source bytes verified against the old source manifest, not proof that the original CI ran on that commit.",
                    "Original legacy benchmark has summary statistics, not per-trial raw latency samples.",
                    "GitHub runner timing does not establish dedicated-server throughput.",
                    "No signed reviewer authority or production deployment claim.",
                ],
            }
            record["record_sha256"] = sha(json.dumps(
                record, sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode())
            return record
        finally:
            git(repo, "worktree", "remove", "--force", str(checkout), timeout=60)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo", type=Path, default=Path(__file__).resolve().parents[3],
    )
    parser.add_argument("--verify-source", action="store_true")
    parser.add_argument("--discover-source", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if sum((args.verify_source, args.discover_source, args.run)) != 1:
        parser.error("Choose exactly one of --verify-source, --discover-source, or --run")
    repo = args.repo.resolve()
    if args.discover_source:
        result = discover_historical_sources(repo)
    elif args.verify_source:
        result = verify_source(repo)
    else:
        result = execute(repo)
    wire = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(wire, encoding="utf-8")
    print(wire if (args.verify_source or args.discover_source) else json.dumps({
        "schema": result["schema"],
        "status": result["status"],
        "record_sha256": result["record_sha256"],
        "numerical": result["verification"],
        "output": str(args.output) if args.output else "stdout-only",
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
