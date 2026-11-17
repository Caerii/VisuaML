"""v36: archive and verify the exact independently executed CI replay bytes.

This is a source-bound archival validator, NOT a fresh numerical executor.
A passing GitHub Actions job and original full JSON are separate from a
cryptographic signature; no signer authority is inferred from a SHA digest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from scripts.replay_pretrained_v36 import (
    CANDIDATE_COMMIT,
    SOURCE_BLOB_COMMITS,
    load_archive,
    verify_fresh,
)

SCHEMA = "visuaml.pinned-pretrained-independent-replay.v36"
ORIGIN_SCHEMA = "visuaml.pretrained-replay-origin.v36"
CI_RUN = "https://github.com/Caerii/VisuaML/actions/runs/44193012326"
CI_HEAD = "2013b52299abb048959752d9a8136611a62a56d4"
EVIDENCE_BYTES_SHA256 = "a6b06210cc37a78ded00ab52c201c2c96eea0f4f754b427437b6ded3acce48f0"
NUMERICAL_RECORD_SHA256 = "0e5f11a929e9a594f10233bc7e0027af0e86fc8ab2bc487ab13cab7c802da674"
BUNDLE_SHA256 = "9d06f35b8ffd17a5e3786a756c2e87feca25faf4017fd7dab93a563ecc8b8167"
ARCHIVE = "benchmarks/pretrained_llama_independent_replay_v36.json"
ORIGIN = "benchmarks/pretrained_llama_independent_replay_v36.origin.json"


class ArchivalError(ValueError):
    """Historical source, fresh numerical result or CI origin did not match."""


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical(value: dict) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()


def strict_json(data: bytes) -> dict:
    if not isinstance(data, bytes) or not 1 <= len(data) <= 1024 * 1024:
        raise ArchivalError("Replay artifact exceeds byte budget")
    try:
        value = json.loads(
            data, parse_constant=lambda _: (_ for _ in ()).throw(
                ArchivalError("Nonfinite JSON constant"),
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ArchivalError("Replay artifact is not strictly valid JSON") from exc
    if not isinstance(value, dict):
        raise ArchivalError("Expected JSON object")
    return value


def verify_artifact(raw: bytes, repo: Path) -> dict:
    if digest(raw) != EVIDENCE_BYTES_SHA256:
        raise ArchivalError("Bytes differ from the independently verified CI artifact")
    result = strict_json(raw)
    if result.get("schema") != SCHEMA or result.get("status") != (
        "fresh_independent_numerical_witness_passed"
    ):
        raise ArchivalError("Replay success status was not confirmed")
    if result.get("record_sha256") != NUMERICAL_RECORD_SHA256:
        raise ArchivalError("Unexpected fresh numerical record identity")
    body = {k: v for k, v in result.items() if k != "record_sha256"}
    if digest(canonical(body)) != result["record_sha256"]:
        raise ArchivalError("Fresh numerical record digest mismatch")
    source = result.get("historical_source_recovery")
    if not isinstance(source, dict) or (
        source.get("schema") != "visuaml.historical-source-bundle.v36"
        or source.get("base_checkout_commit") != CANDIDATE_COMMIT
        or source.get("source_file_commits") != SOURCE_BLOB_COMMITS
        or source.get("reconstructed_source_bundle_sha256") != BUNDLE_SHA256
        or source.get("source_verified") is not True
        or source.get("single_commit_contains_all_recorded_sources") is not False
        or source.get("original_ci_source_commit") is not None
    ):
        raise ArchivalError("Reconstructed multi-commit source provenance changed")
    historical, archive_sha = load_archive(repo)
    if source["archive_sha256"] != archive_sha:
        raise ArchivalError("Historical record changed since fresh replay")
    if source["source_sha256"] != historical["matched_cpu_benchmark"]["source_sha256"]:
        raise ArchivalError("Recovered file hashes differ from old benchmark")
    fresh = result.get("fresh_numerical_result")
    if not isinstance(fresh, dict):
        raise ArchivalError("Fresh model execution result missing")
    full_hash = digest(json.dumps(
        fresh, indent=2, sort_keys=True, allow_nan=False,
    ).encode())
    if full_hash != result.get("fresh_result_sha256"):
        raise ArchivalError("Embedded numerical JSON hash mismatch")
    admitted = verify_fresh(fresh, historical, source)
    if admitted != result.get("verification") or admitted["passed"] is not True:
        raise ArchivalError("Fresh numerical admission record is inconsistent")
    for metric in ("max_abs_error_hf", "max_abs_error_semantic"):
        value = admitted[metric]
        if type(value) is not float or not math.isfinite(value) or value > 3e-4:
            raise ArchivalError("Fresh numerical error exceeded tolerance")
    return result


def origin_manifest(raw: bytes) -> dict:
    return {
        "schema": ORIGIN_SCHEMA,
        "artifact_path": ARCHIVE,
        "record_bytes_sha256": digest(raw),
        "fresh_numerical_record_sha256": NUMERICAL_RECORD_SHA256,
        "reconstructed_source_bundle_sha256": BUNDLE_SHA256,
        "producing_ci_run": CI_RUN,
        "producing_branch_head": CI_HEAD,
        "status": "source_bound_ci_witness_archived_not_signed",
        "limitations": (
            "GitHub Actions produced this historical run record; the origin "
            "manifest preserves source identity but is not an authenticated "
            "independent reviewer signature."
        ),
    }


def verify_checked_in_archive(repo: Path) -> dict:
    root = Path(repo)
    raw = (root / ARCHIVE).read_bytes()
    verify_artifact(raw, root)
    origin = strict_json((root / ORIGIN).read_bytes())
    if origin != origin_manifest(raw):
        raise ArchivalError("Archived origin manifest is altered or mismatched")
    return origin


def prepare(repo: Path, raw: bytes) -> dict:
    root = Path(repo)
    verify_artifact(raw, root)
    origin = origin_manifest(raw)
    outputs = (
        (root / ARCHIVE, raw),
        (root / ORIGIN, (json.dumps(
            origin, indent=2, sort_keys=True, allow_nan=False,
        ) + "\n").encode()),
    )
    for path, content in outputs:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            if path.read_bytes() != content:
                raise ArchivalError("Cannot overwrite an existing historical source")
        else:
            path.write_bytes(content)
    if verify_checked_in_archive(root) != origin:
        raise ArchivalError("Post-write provenance verification failed")
    return origin


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo", type=Path, default=Path(__file__).resolve().parents[3],
    )
    parser.add_argument("--input", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check == (args.input is not None):
        parser.error("Provide exactly one of --check or --input")
    result = (
        verify_checked_in_archive(args.repo)
        if args.check else prepare(args.repo, args.input.read_bytes())
    )
    print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
