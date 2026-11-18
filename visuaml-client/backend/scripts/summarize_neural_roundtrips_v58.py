"""Measured-only scorecard for independently verified neural-symbolic rounds.

Consumes a completed, source-bound v58 round-trip benchmark artifact.
Does not invent performance or interpret typed-source SMT as neural-circuit
all-i32 proof. It reports exact program inventory and per-split costs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path

SCHEMA = "visuaml.cross-representation-verified-roundtrip.v58"


def summarize(raw: bytes) -> dict:
    report = json.loads(raw)
    if report.get("schema") != SCHEMA:
        raise ValueError("Unknown roundtrip evidence schema")
    results = report.get("results")
    if not isinstance(results, list) or not results:
        raise ValueError("No completed roundtrip outcomes")
    inventory = report.get("inventory", {})
    if inventory.get("source_programs") != len(results):
        raise ValueError("Roundtrip inventory inconsistent with outcomes")
    if len({row["original_wasm_sha256"] for row in results}) != len(results):
        raise ValueError("Duplicate original WASM source identifiers")
    if any(
        not row.get("all_i32_typed_source_equivalence")
        or row.get("all_i32_neural_weights_equivalence")
        or row.get("arbitrary_learned_neural_model_decompiled")
        or len(row.get("finite_results", [])) != row["finite_replay_inputs"]
        for row in results
    ):
        raise ValueError("Roundtrip archive contains invalid proof-scope claims")
    splits = {}
    for name in sorted({row["split"] for row in results}):
        cases = [row for row in results if row["split"] == name]
        def median(field):
            return statistics.median(row[field] for row in cases)
        splits[name] = {
            "verified_programs": len(cases),
            "finite_execution_replays": sum(
                row["finite_replay_inputs"] for row in cases
            ),
            "median_cegis_refinements": median("cegis_refinements"),
            "median_cegis_observations": median("cegis_observation_count"),
            "median_recovery_and_smt_ns": median(
                "cegis_recovery_and_smt_wall_ns"
            ),
            "median_original_wasm_bytes": median("original_wasm_bytes"),
            "median_recovered_wasm_bytes": median("recovered_wasm_bytes"),
            "median_recovered_to_original_size_ratio": statistics.median(
                row["recovered_wasm_bytes"] / row["original_wasm_bytes"]
                for row in cases
            ),
            "structurally_distinct_recovered_binaries": sum(
                not row["same_source_binary"] for row in cases
            ),
        }
    return {
        "schema": "visuaml.cross-representation-scorecard.v58",
        "input_evidence_sha256": hashlib.sha256(raw).hexdigest(),
        "source_commit": report["source"]["commit"],
        "source_tree": report["source"]["tree"],
        "source_programs": len(results),
        "finite_execution_replays": sum(
            row["finite_replay_inputs"] for row in results
        ),
        "verified_typed_source_to_candidate_z3": len(results),
        "claimed_full_domain_constructed_neural_weight_proofs": 0,
        "claimed_unknown_learned_weight_decompilations": 0,
        "measured_among_completed_strict_runs_only": True,
        "attempted_failed_recoveries_not_scored_by_this_strict_archive": True,
        "splits": splits,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    summary = summarize(args.input.read_bytes())
    if args.output is not None:
        destination = args.output.resolve()
        if destination.exists():
            parser.error(f"Refusing to overwrite scorecard: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("x", encoding="utf-8") as stream:
            json.dump(summary, stream, indent=2, sort_keys=True,
                      allow_nan=False)
            stream.write("\n")
    print(json.dumps(summary, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
