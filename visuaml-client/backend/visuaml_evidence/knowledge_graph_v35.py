"""v35: source-bound knowledge graph and conservative replay admission.

Historical source/CI hashes can establish exact *recorded identity*, not a new
numerical run. Preflight reports blockers if native code or model files are
missing or changed. This module is standard-library only and NEVER downloads,
imports model Python, executes generated C++, or promotes a benchmark.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import re
from pathlib import Path
from typing import Any

from .library_v34 import EvidenceLibrary, EvidenceLibraryError

SCHEMA = "visuaml.knowledge-graph.v35"
PLAN_SCHEMA = "visuaml.replay-plan.v35"
SOURCE = "benchmarks/pretrained_llama_native_cpu_benchmark_v28.json"
HEX64 = re.compile(r"[a-f0-9]{64}\Z")
REV40 = re.compile(r"[a-f0-9]{40}\Z")
RUN = re.compile(r"https://github\.com/Caerii/VisuaML/actions/runs/[0-9]+\Z")
ALLOWED_SOURCE = {
    "examples/native_cpu_bench_v28.py",
    "visuaml/compiler_lab/native_cpu_v28.cpp",
    "visuaml/compiler_lab/native_cpu_v28.py",
}
LIMIT = 1024 * 1024


class KnowledgeGraphError(EvidenceLibraryError):
    """A reviewed historical observation fails admission or replay preflight."""


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical(record: dict) -> bytes:
    return json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _hash64(value: object, label: str) -> str:
    if not isinstance(value, str) or not HEX64.fullmatch(value):
        raise KnowledgeGraphError(f"Invalid SHA-256 identity for {label}")
    return value


def _numeric(value: object, label: str, *, positive: bool = False) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise KnowledgeGraphError(f"Nonfinite {label}")
    number = float(value)
    if not 0 <= number < 1e10 or (positive and number <= 0):
        raise KnowledgeGraphError(f"Invalid measured {label}")
    return number


def _load_v28(repo: Path) -> tuple[dict[str, Any], str]:
    path = Path(repo) / SOURCE
    raw = path.read_bytes()
    if not 1 <= len(raw) <= LIMIT:
        raise KnowledgeGraphError("Pinned v28 observation exceeds size limit")
    try:
        record = json.loads(raw, parse_constant=lambda _: (_ for _ in ()).throw(
            KnowledgeGraphError("Nonfinite historical JSON")
        ))
    except (json.JSONDecodeError, UnicodeDecodeError, RecursionError) as exc:
        raise KnowledgeGraphError("Malformed historical v28 JSON") from exc
    if not isinstance(record, dict) or record.get("schema") != "visuaml.pinned-native-llama-v28":
        raise KnowledgeGraphError("Unsupported pinned v28 evidence schema")
    backend = record["backend"]
    benchmark = record["matched_cpu_benchmark"]
    provenance = record["evidence_provenance"]
    if (
        backend.get("schema") != "visuaml.native-cpu-llama.v28"
        or benchmark.get("schema") != "visuaml.native-cpu-benchmark.v28"
        or record["chunked_KV_and_greedy_parity"] is not True
        or provenance.get("conclusion") != "success"
    ):
        raise KnowledgeGraphError("Historical v28 parity/provenance is not admitted")
    if not isinstance(provenance.get("workflow_url"), str) or not RUN.fullmatch(
        provenance["workflow_url"]
    ):
        raise KnowledgeGraphError("Missing approved CI provenance")
    source_revision = record["revision"]
    if not isinstance(source_revision, str) or not REV40.fullmatch(source_revision):
        raise KnowledgeGraphError("Missing pinned 40-character model revision")
    if record["source"] != "manojredhat/tiny-llama":
        raise KnowledgeGraphError("Unapproved historical model source")
    _hash64(record["checkpoint_sha256"], "checkpoint")
    _hash64(record["config_sha256"], "config")
    semantic = _hash64(record["semantic_sha256"], "semantic")
    if (
        record["checkpoint_sha256"] != backend["checkpoint_sha256"]
        or record["checkpoint_sha256"] != benchmark["checkpoint_sha256"]
        or record["revision"] != benchmark["source_model_revision"]
        or semantic != backend["semantic_sha256"]
        or semantic != benchmark["semantic_sha256"]
        or backend["cpp_source_sha256"] != benchmark["source_sha256"][
            "visuaml/compiler_lab/native_cpu_v28.cpp"
        ]
    ):
        raise KnowledgeGraphError("Historical v28 source identities disagree")
    _hash64(backend["cpp_source_sha256"], "compiler source")
    _hash64(backend["executable_sha256"], "executable")
    source_hashes = benchmark["source_sha256"]
    if not isinstance(source_hashes, dict) or set(source_hashes) != ALLOWED_SOURCE:
        raise KnowledgeGraphError("Unexpected historical source inventory")
    for name, digest in source_hashes.items():
        _hash64(digest, name)
    witnesses = record["witnesses"]
    if not isinstance(witnesses, list) or not 1 <= len(witnesses) <= 16:
        raise KnowledgeGraphError("Missing bounded pretrained numerical witnesses")
    for witness in witnesses:
        for key in ("max_abs_error_against_HF", "max_abs_error_against_semantic_IR"):
            if _numeric(witness[key], key) > 3e-4:
                raise KnowledgeGraphError("Historical Llama tolerance violation")
        shape = witness["logit_shape"]
        tokens = witness["tokens"]
        if (
            not isinstance(shape, list) or len(shape) != 3
            or any(type(n) is not int or not 1 <= n <= 4096 for n in shape)
            or not isinstance(tokens, list) or len(tokens) != shape[0]
            or any(not isinstance(row, list) or len(row) != shape[1]
                   or any(type(value) is not int or not 0 <= value < shape[2]
                          for value in row)
                   for row in tokens)
        ):
            raise KnowledgeGraphError("Malformed historical witness geometry")
    if (
        type(benchmark["trials"]) is not int
        or not 1 <= benchmark["trials"] <= 100
        or benchmark["chunks"] != [4, 1, 1, 2]
    ):
        raise KnowledgeGraphError("Historical benchmark workload mismatched")
    for variant in ("native_cpp", "numpy_semantic", "torchscript_whole"):
        metrics = benchmark["latency"][variant]
        for name in ("median_ms", "minimum_ms", "p95_ms"):
            _numeric(metrics[name], f"{variant}.{name}", positive=True)
    if _numeric(backend["independent_max_abs_error"], "native error") > 3e-4:
        raise KnowledgeGraphError("Native historical numerical tolerance exceeded")
    _numeric(backend["compilation_ms"], "native compile", positive=True)
    _numeric(benchmark["compilation"]["native_cpp_ms"], "benchmark compile", positive=True)
    if benchmark["numerical_certificate"] != (
        "all chunked logits and final layer KV match NumPy IR"
    ):
        raise KnowledgeGraphError("Missing recorded KV numerical certificate")
    return record, _sha(raw)


def replay_plan_v28(record: dict[str, Any], artifact_sha256: str) -> dict:
    """Exact preflight requirements; no automatic download or execution."""
    benchmark = record["matched_cpu_benchmark"]
    backend = record["backend"]
    return {
        "schema": PLAN_SCHEMA,
        "plan_id": _sha(_canonical({
            "artifact_sha256": artifact_sha256,
            "model_revision": record["revision"],
            "semantic_sha256": record["semantic_sha256"],
        })),
        "source_artifact_path": SOURCE,
        "source_artifact_sha256": artifact_sha256,
        "evidence_class": "historical_ci_numerical_witness_not_replayed_here",
        "ci_run": record["evidence_provenance"]["workflow_url"],
        "model": record["source"],
        "model_revision": record["revision"],
        "checkpoint_files_sha256": {
            "config.json": record["config_sha256"],
            "model.safetensors": record["checkpoint_sha256"],
        },
        "semantic_sha256": record["semantic_sha256"],
        "native_compiler_cpp_sha256": backend["cpp_source_sha256"],
        "source_files_sha256": dict(sorted(benchmark["source_sha256"].items())),
        "witness_count": len(record["witnesses"]),
        "tolerance_atol": 3e-4,
        "tolerance_rtol": 3e-4,
        "steps": [
            "verify_archive_bytes",
            "verify_exact_historical_source_files",
            "verify_local_pinned_checkpoint",
            "trusted_cpu_native_numerical_replay_requires_separate_runner",
            "compare_replayed_logits_and_final_kv_before_latency",
        ],
        "historical_source_commit": None,
        "status": "preflight_only_no_executable_replay",
        "limitations": [
            "A historical Git source commit is not established by this record.",
            "Existing current main native source may differ from the old baseline.",
            "Preflight never downloads checkpoint or executes model/native code.",
            "Recorded CI numerical agreement is not a fresh independent replay.",
        ],
    }


def _file_hash(path: Path, *, size_limit: int = 1_000_000_000) -> str:
    if not path.is_file() or path.stat().st_size > size_limit:
        raise KnowledgeGraphError("Missing or oversized local replay input")
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def preflight_replay(
    plan: dict[str, Any], repo: Path, *, checkpoint: Path | None = None,
) -> dict[str, Any]:
    """Check original evidence and source bytes; DO NOT claim numerical replay.

    A filename alone is never an authenticated source revision. Even if every
    hash matches, a trusted isolated runner must execute the independent
    numerical witnesses before recording any new passed result.
    """
    if not isinstance(plan, dict) or plan.get("schema") != PLAN_SCHEMA:
        raise KnowledgeGraphError("Unsupported replay plan schema")
    record, artifact_hash = _load_v28(repo)
    expected = replay_plan_v28(record, artifact_hash)
    if _canonical(plan) != _canonical(expected):
        raise KnowledgeGraphError("Replay plan is not bound to reviewed source evidence")
    failures = []
    root = Path(repo) / "visuaml-client" / "backend"
    for name, digest in sorted(plan["source_files_sha256"].items()):
        if name not in ALLOWED_SOURCE:
            raise KnowledgeGraphError("Unexpected source path in replay plan")
        path = root / name
        try:
            observed = _file_hash(path, size_limit=LIMIT)
        except KnowledgeGraphError:
            observed = None
        if observed != digest:
            failures.append({"kind": "source_sha256_mismatch", "path": name})
    if checkpoint is None:
        failures.append({
            "kind": "pinned_checkpoint_not_provided",
            "path": "config.json + model.safetensors",
        })
    else:
        for name, digest in sorted(plan["checkpoint_files_sha256"].items()):
            try:
                observed = _file_hash(Path(checkpoint) / name)
            except KnowledgeGraphError:
                observed = None
            if observed != digest:
                failures.append({"kind": "checkpoint_sha256_mismatch", "path": name})
    return {
        "schema": "visuaml.replay-preflight-result.v35",
        "plan_id": plan["plan_id"],
        "historical_artifact_sha256": artifact_hash,
        "static_source_and_checkpoint_ready": not failures,
        "blockers": failures,
        "numerical_replay_performed": False,
        "performance_replay_performed": False,
        "result_class": (
            "static_preflight_passed_pending_trusted_execution"
            if not failures else "replay_blocked"
        ),
    }


def knowledge_graph(repo: Path) -> dict[str, Any]:
    """Versioned overlay; never mutates existing v34 catalog or entry IDs."""
    old = EvidenceLibrary.from_repo(repo).catalog()
    original, artifact_sha = _load_v28(repo)
    plan = replay_plan_v28(original, artifact_sha)
    backend = original["backend"]
    historical_id = _sha(_canonical({
        "kind": "pinned_pretrained_native_witness_v28",
        "artifact_sha256": artifact_sha,
        "semantic_sha256": original["semantic_sha256"],
        "checkpoint_sha256": original["checkpoint_sha256"],
    }))
    historical = {
        "entry_id": historical_id,
        "kind": "pinned_pretrained_native_witness",
        "label": "Tiny Llama scalar native C++17 (v28)",
        "evidence_class": "historical_full_ci_witness_not_replayed_v35",
        "status": "historically_passed_numerical_witness",
        "artifact_sha256": artifact_sha,
        "source_path": SOURCE,
        "source_commit": None,
        "workflow_url": original["evidence_provenance"]["workflow_url"],
        "semantic_sha256": original["semantic_sha256"],
        "generated_cpp_sha256": backend["cpp_source_sha256"],
        "native_binary_sha256": backend["executable_sha256"],
        "checkpoint_sha256": original["checkpoint_sha256"],
        "model_revision": original["revision"],
        "op_sequence": ["llama_decoder"],
        "context": {
            "model": original["source"], "precision": "float32",
            "native_target": backend["target"],
            "historical_source_commit": "unknown",
        },
        "result": {
            "max_abs_error_against_semantic_ir": max(
                item["max_abs_error_against_semantic_IR"]
                for item in original["witnesses"]
            ),
            "native_warm_median_ms": original["matched_cpu_benchmark"]["latency"][
                "native_cpp"
            ]["median_ms"],
            "numerical_witness_count": len(original["witnesses"]),
        },
        "limitations": original["limitations"],
        "replay_plan_id": plan["plan_id"],
    }
    existing = [
        {
            **record,
            "model_revision": dict(record["context"]).get("revision"),
            "replay_plan_id": None,
        } for record in old["entries"]
    ]
    nodes = sorted([*existing, historical], key=lambda entry: entry["entry_id"])
    if len({node["entry_id"] for node in nodes}) != len(nodes):
        raise KnowledgeGraphError("Knowledge node identity collision")
    edges = []
    # Multiple independent relations may exist between a pair. None implies
    # observational correctness transfers to a new model/hardware/input.
    for left, right in itertools.combinations(nodes, 2):
        for attribute, relation in (
            ("native_binary_sha256", "identical_native_binary_bytes"),
            ("generated_cpp_sha256", "identical_generated_cpp_bytes"),
            ("semantic_sha256", "identical_semantic_digest"),
            ("checkpoint_sha256", "identical_checkpoint_fingerprint"),
        ):
            digest = left.get(attribute)
            if digest and digest == right.get(attribute):
                edges.append({
                    "from": left["entry_id"],
                    "to": right["entry_id"],
                    "relation": relation,
                    "sha256": digest,
                    "scope": "identity_only_not_transferable_correctness_or_performance",
                })
    edges.sort(key=lambda x: (x["relation"], x["from"], x["to"]))
    body = {
        "schema": SCHEMA,
        "parent_catalog_sha256": old["catalog_sha256"],
        "reviewed_artifacts": {
            **old["artifacts"],
            SOURCE: artifact_sha,
        },
        "nodes": nodes,
        "edges": edges,
        "plans": [plan],
        "counts": {
            "studies": len(nodes),
            "source_artifacts": len(old["artifacts"]) + 1,
            "identity_edges": len(edges),
            "pinned_pretrained_witnesses": 1,
        },
        "limitations": [
            "v28 witness is historical, not freshly numerically replayed by v35.",
            "The old source checkout SHA is not recorded; source preflight must refuse drift.",
            "Identity edges are byte/model identity, not formal proof or speedup portability.",
            "Unreviewed proposals are excluded; no roadmap gate is promoted.",
            "Other synthetic or curated v31-v34 studies retain their original evidence class.",
        ],
    }
    return {**body, "knowledge_graph_sha256": _sha(_canonical(body))}
