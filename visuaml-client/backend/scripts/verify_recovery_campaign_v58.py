"""Independently audit a failure-complete blinded neural recovery campaign.

Recomputes source-WASM identities, per-case status, failure denominators,
source-tree provenance, and the complete SMT/Wasmtime/v58 execution evidence
for all cases reported verified. Rejected attempts remain in the archive;
they are never silently dropped or counted as success.

No GitHub Actions, remote API, or inference about arbitrary learned weights.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from scripts.benchmark_recovery_failures_v58 import (
    SCHEMA,
    eligible_workloads,
    summarize,
)
from scripts.verify_neural_roundtrip_evidence_v58 import verify as verify_successes
from visuaml.compiler_lab.neural_symbolic_roundtrip_v58 import SCHEMA as ROUNDTRIP
from visuaml.compiler_lab.wasm_state_vm_v40 import encode_program


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(
        ("git", "-C", str(root), *args), text=True,
        stderr=subprocess.DEVNULL,
    ).strip()


def review(path: Path, root: Path) -> dict:
    raw = path.read_bytes()
    archive = json.loads(raw)
    if archive.get("schema") != SCHEMA:
        raise ValueError("Unknown failure-complete recovery campaign")
    source = archive.get("source", {})
    if (source.get("commit") != _git(root, "rev-parse", "HEAD")
            or source.get("tree") != _git(root, "rev-parse", "HEAD^{tree}")):
        raise ValueError("Recovery ledger does not match current Git source")
    policy = archive.get("policy", {})
    if (policy.get("all_i32_neural_weights_proved")
            or policy.get("arbitrary_learned_weight_decompilation")
            or not policy.get("failures_in_denominator")):
        raise ValueError("Invalid scientific proof or denominator policy")
    expected = eligible_workloads(
        holdout=bool(policy.get("holdout_enabled")),
    )
    attempts = archive.get("attempts")
    if not isinstance(attempts, list) or len(attempts) != len(expected):
        raise ValueError("Recovery campaign case inventory incomplete")
    successes = []
    for item, workload in zip(attempts, expected, strict=True):
        original = encode_program(workload.source)
        if (item["name"] != workload.name
                or item["split"] != workload.split
                or bytes.fromhex(item["source_wasm_hex"]) != original
                or item["source_wasm_sha256"] != hashlib.sha256(
                    original
                ).hexdigest()):
            raise ValueError("Recovery result not bound to canonical source")
        if item["status"] not in ("verified", "rejected"):
            raise ValueError("Unknown recovery status")
        audit = item.get("audit", {})
        if any(type(audit.get(key, 0)) is not int
               or audit.get(key, 0) < 0
               for key in (
                   "neural_oracle_queries",
                   "independent_verifier_invocations",
               )):
            raise ValueError("Invalid recorded neural query or solver counts")
        if item["status"] == "verified":
            if (item.get("failure_class") is not None
                    or not item.get("all_i32_typed_source_equivalence_proved")
                    or item.get("all_i32_neural_weights_equivalence_proved")
                    or item.get("receipt", {}).get("original_wasm_sha256")
                    != item["source_wasm_sha256"]):
                raise ValueError("Verified case has missing or false proof claim")
            successes.append(item["receipt"])
        else:
            if (item.get("receipt") is not None
                    or item.get("all_i32_typed_source_equivalence_proved")
                    or item.get("all_i32_neural_weights_equivalence_proved")
                    or not item.get("failure_class")
                    or not item.get("exception_type")):
                raise ValueError("Rejected case has missing or false rejection")
    recomputed = summarize(attempts)
    if archive.get("summary") != recomputed:
        raise ValueError("Recovery success/failure denominator modified")
    negative = recomputed["by_split"]["negative_control"]
    if negative["verified"]:
        raise ValueError("Outside-grammar cubic was incorrectly accepted")
    # Reuse the independent strict receipt verifier, which reruns Z3 and
    # independently executes the original/recovered WASM and v58 traces.
    validated = None
    if successes:
        strict = {
            "schema": ROUNDTRIP,
            "source": source,
            "policy": {
                "prove_all_inputs_through_constructed_neural_weights": False,
                "arbitrary_learned_weight_decompilation": False,
            },
            "inventory": {
                "source_programs": len(successes),
                "finite_wasm_replays": sum(
                    result["finite_replay_inputs"] for result in successes
                ),
            },
            "results": successes,
        }
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "strict-subset.json"
            artifact.write_text(
                json.dumps(strict, sort_keys=True, allow_nan=False) + "\n",
                encoding="utf-8",
            )
            validated = verify_successes(artifact, root)
    return {
        "schema": "visuaml.failure-complete-recovery-audit.v58",
        "artifact_sha256": hashlib.sha256(raw).hexdigest(),
        "source_commit": source["commit"],
        "source_tree": source["tree"],
        "full_denominator_verified": True,
        "source_bytes_verified": True,
        "rejected_attempts_preserved": recomputed["total_rejected"],
        "verified_attempts_replayed": recomputed["total_verified"],
        "independent_success_receipt_replay": validated,
        "summary": recomputed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    print(json.dumps(review(args.input, root), sort_keys=True))


if __name__ == "__main__":
    main()
