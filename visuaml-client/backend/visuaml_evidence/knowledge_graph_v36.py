"""v36: append a genuinely new real-model replay to the immutable v35 graph.

The historical v28 witness and the new v36 rerun are separate experiments.
An identical native/checkpoint digest establishes an exact *identity edge*,
not a general equivalence theorem or portable performance result.
"""

from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path
from typing import Any

from scripts.archive_pretrained_replay_v36 import (
    ARCHIVE,
    verify_checked_in_archive,
)
from .knowledge_graph_v35 import knowledge_graph as historical_graph

SCHEMA = "visuaml.knowledge-graph.v36"


def canonical(value: dict) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()


def sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def knowledge_graph(repo: Path) -> dict[str, Any]:
    root = Path(repo)
    previous = historical_graph(root)
    origin = verify_checked_in_archive(root)
    fresh_raw = (root / ARCHIVE).read_bytes()
    replay = json.loads(fresh_raw)
    native = replay["fresh_numerical_result"]["backend"]
    source = replay["historical_source_recovery"]
    verify = replay["verification"]
    fresh_id = sha(canonical({
        "kind": "pinned_pretrained_independent_replay",
        "record_bytes_sha256": origin["record_bytes_sha256"],
        "model_revision": replay["model_revision"],
        "source_bundle_sha256": source["reconstructed_source_bundle_sha256"],
        "semantic_sha256": replay["semantic_sha256"],
    }))
    new = {
        "entry_id": fresh_id,
        "kind": "pinned_pretrained_independent_replay",
        "label": "Tiny Llama — independently reconstructed C++ numerical replay (v36)",
        "evidence_class": "fresh_source_bound_finite_pretrained_numerical_replay",
        "status": "fresh_independent_numerical_witness_passed",
        "artifact_sha256": origin["record_bytes_sha256"],
        "source_path": ARCHIVE,
        "source_commit": None,
        "workflow_url": origin["producing_ci_run"],
        "semantic_sha256": replay["semantic_sha256"],
        "generated_cpp_sha256": native["cpp_source_sha256"],
        "native_binary_sha256": native["executable_sha256"],
        "checkpoint_sha256": replay["checkpoint_sha256"],
        "model_revision": replay["model_revision"],
        "op_sequence": ["llama_decoder"],
        "context": {
            "model": replay["fresh_numerical_result"]["source"],
            "precision": "float32",
            "native_target": native["target"],
            "support_base_git_commit": source["base_checkout_commit"],
            "reconstructed_bundle_sha256": source[
                "reconstructed_source_bundle_sha256"
            ],
            "original_ci_git_checkout": "unknown",
            "verification_scope": "independent_hf_numpy_kv_greedy",
        },
        "result": {
            "max_abs_error_against_hf": verify["max_abs_error_hf"],
            "max_abs_error_against_semantic_ir": verify["max_abs_error_semantic"],
            "native_warm_median_ms": verify["warm_native_median_ms"],
            "native_compile_ms": verify["native_compile_ms"],
            "numerical_witness_count": verify["witness_count"],
        },
        "limitations": replay["limitations"],
        "replay_plan_id": None,
    }
    originals = list(previous["nodes"])
    if fresh_id in {entry["entry_id"] for entry in originals}:
        raise ValueError("New replay identity overwrites an existing study")
    nodes = sorted([*originals, new], key=lambda item: item["entry_id"])
    edges = list(previous["edges"])
    for other in originals:
        for key, relation in (
            ("native_binary_sha256", "identical_native_binary_bytes"),
            ("generated_cpp_sha256", "identical_generated_cpp_bytes"),
            ("semantic_sha256", "identical_semantic_digest"),
            ("checkpoint_sha256", "identical_checkpoint_fingerprint"),
        ):
            value = new.get(key)
            if value and value == other.get(key):
                edges.append({
                    "from": min(fresh_id, other["entry_id"]),
                    "to": max(fresh_id, other["entry_id"]),
                    "relation": relation,
                    "sha256": value,
                    "scope": "identity_only_not_transferable_correctness_or_performance",
                })
    edges.sort(key=lambda e: (e["relation"], e["from"], e["to"]))
    if len({
        (edge["from"], edge["to"], edge["relation"])
        for edge in edges
    }) != len(edges):
        raise ValueError("Duplicated exact identity relation")
    replay_summary = {
        "schema": "visuaml.independent-pretrained-replay-summary.v36",
        "study_id": fresh_id,
        "record_bytes_sha256": origin["record_bytes_sha256"],
        "numerical_record_sha256": replay["record_sha256"],
        "source_bundle_sha256": source["reconstructed_source_bundle_sha256"],
        "source_file_commits": source["source_file_commits"],
        "producing_ci_run": origin["producing_ci_run"],
        "producing_branch_head": origin["producing_branch_head"],
        "original_experiment_commit": None,
        "numerical_replay_performed": True,
        "passed": verify["passed"],
        "limits": (
            "A fresh finite pretrained inference witness on a reconstructed "
            "three-blob source bundle; no formal proof or universal speedup."
        ),
    }
    body = {
        "schema": SCHEMA,
        "parent_graph_sha256": previous["knowledge_graph_sha256"],
        "parent_catalog_sha256": previous["parent_catalog_sha256"],
        "reviewed_artifacts": {
            **previous["reviewed_artifacts"],
            ARCHIVE: origin["record_bytes_sha256"],
        },
        "nodes": nodes,
        "edges": edges,
        # Do not mutate or relabel the pre-existing v35 preflight-only plan.
        "plans": previous["plans"],
        "fresh_replays": [replay_summary],
        "counts": {
            "studies": len(nodes),
            "source_artifacts": len(previous["reviewed_artifacts"]) + 1,
            "identity_edges": len(edges),
            "pinned_pretrained_witnesses": 2,
            "distinct_pinned_pretrained_models": 1,
            "fresh_independent_replays": 1,
        },
        "limitations": [
            *previous["limitations"],
            "v36 is one fresh finite rerun of the same Tiny Llama checkpoint, not a second pretrained architecture family.",
            "The original CI source checkout remains unverified; v36 uses a hash-bound reconstructed source bundle.",
            "Equal generated binaries/checkpoints support exact lineage, not all-input equivalence or production serving performance.",
            "GitHub Actions CI provenance is a recorded platform origin, not an authenticated human reviewer signature.",
        ],
    }
    return {**body, "knowledge_graph_sha256": sha(canonical(body))}
