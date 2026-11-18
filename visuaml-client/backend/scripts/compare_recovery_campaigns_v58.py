"""Compare two source-matched, failure-complete blinded recovery campaigns.

The independent campaign auditor must be run on each report at its original
source revision first. This comparator is metadata-only: it does not re-run
the historical code from two commits in one checkout. It never declares
an unverified campaign scientifically accepted merely from a status flag.

Require identical canonical source WASM, split, input inventory and solver
budget. Then report transitions, valid denominator and oracle/SMT query cost
without hiding regressions or invalid experiments.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path

from scripts.benchmark_recovery_failures_v58 import (
    SCHEMA, summarize,
)


def load(path: Path) -> dict:
    raw = path.read_bytes()
    report = json.loads(raw)
    if report.get("schema") != SCHEMA:
        raise ValueError("Unknown recovery-campaign source schema")
    attempts = report.get("attempts")
    if not isinstance(attempts, list) or not attempts:
        raise ValueError("Incomplete campaign outcomes")
    if report.get("summary") != summarize(attempts):
        raise ValueError("Unbound or inconsistent campaign denominator")
    report["_file_sha256"] = hashlib.sha256(raw).hexdigest()
    return report


def _cases(report: dict) -> dict[str, dict]:
    cases = {}
    for item in report["attempts"]:
        key = item.get("name")
        if not isinstance(key, str) or not key or key in cases:
            raise ValueError("Duplicate or missing recovery source identity")
        source = bytes.fromhex(item["source_wasm_hex"])
        if hashlib.sha256(source).hexdigest() != item["source_wasm_sha256"]:
            raise ValueError("Tampered source WASM content in campaign")
        if item["status"] not in ("verified", "rejected", "invalid"):
            raise ValueError("Unexpected recovery status")
        cases[key] = item
    return cases


def compare(before: dict, after: dict) -> dict:
    first, second = _cases(before), _cases(after)
    if set(first) != set(second):
        raise ValueError("Recovery program inventory changed between campaigns")
    keys = (
        "holdout_enabled", "solver_timeout_ms",
        "failures_in_denominator",
    )
    for name in keys:
        if before["policy"].get(name) != after["policy"].get(name):
            raise ValueError(f"Recovery campaign policy changed: {name}")
    transitions = {}
    improvements, regressions = [], []
    paired_costs = []
    rows = []
    for name in sorted(first):
        original, candidate = first[name], second[name]
        if (
            original["source_wasm_sha256"] != candidate["source_wasm_sha256"]
            or original["source_wasm_hex"] != candidate["source_wasm_hex"]
            or original["split"] != candidate["split"]
        ):
            raise ValueError(f"Recovery source/split changed: {name}")
        status_before = original["status"]
        status_after = candidate["status"]
        edge = f"{status_before}->{status_after}"
        transitions[edge] = transitions.get(edge, 0) + 1
        if status_before == "rejected" and status_after == "verified":
            improvements.append(name)
        if status_before == "verified" and status_after != "verified":
            regressions.append(name)
        if status_before == "invalid" or status_after == "invalid":
            # An invalid run does not count as a verified improvement.
            pass
        previous_audit = original.get("audit", {})
        current_audit = candidate.get("audit", {})
        old_queries = previous_audit.get("neural_oracle_queries")
        new_queries = current_audit.get("neural_oracle_queries")
        if (type(old_queries) is int and old_queries >= 0
                and type(new_queries) is int and new_queries >= 0):
            paired_costs.append(new_queries - old_queries)
        rows.append({
            "program": name,
            "split": original["split"],
            "source_wasm_sha256": original["source_wasm_sha256"],
            "baseline_status": status_before,
            "candidate_status": status_after,
            "baseline_failure_class": original.get("failure_class"),
            "candidate_failure_class": candidate.get("failure_class"),
            "baseline_neural_oracle_queries": old_queries,
            "candidate_neural_oracle_queries": new_queries,
        })
    return {
        "schema": "visuaml.failure-complete-recovery-comparison.v58",
        "baseline_artifact_sha256": before["_file_sha256"],
        "candidate_artifact_sha256": after["_file_sha256"],
        "baseline_commit": before["source"]["commit"],
        "candidate_commit": after["source"]["commit"],
        "source_programs_match": True,
        "policy_match": True,
        "baseline_summary": before["summary"],
        "candidate_summary": after["summary"],
        "status_transitions": transitions,
        "newly_verified_from_rejected": improvements,
        "regressed_from_verified": regressions,
        "paired_oracle_query_delta_median": (
            statistics.median(paired_costs) if paired_costs else None
        ),
        "rows": rows,
        "limitation": (
            "Metadata comparison only. Each archive must first be checked "
            "at its own Git source revision by verify_recovery_campaign_v58; "
            "a changed status cannot substitute for independent proof replay."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    out = args.output.resolve()
    if out.exists():
        parser.error("Refusing to overwrite recovery comparison")
    result = compare(load(args.baseline), load(args.candidate))
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({
        "output": str(out),
        "baseline": result["baseline_commit"],
        "candidate": result["candidate_commit"],
        "newly_verified": result["newly_verified_from_rejected"],
        "regressions": result["regressed_from_verified"],
        "status_transitions": result["status_transitions"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
