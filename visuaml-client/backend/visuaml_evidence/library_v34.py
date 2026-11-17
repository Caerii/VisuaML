"""v34: append-only research evidence library with exact provenance and matching.

The library indexes *historical observations*, not mathematical proofs.
It discovers safe reuse candidates by identity (IR/source/binary/shape), but
never marks an optimization accepted merely because a digest matches.

Source data is plain JSON, never executed; unknown schemas fail admission.
Only reviewed, fixed repository paths may enter the public index. Future
uploads are held in a separate content-addressed unreviewed proposal store.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

SCHEMA = "visuaml.evidence-library.v34"
_PROPOSAL_SCHEMA = "visuaml.evidence-proposal.v34"
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_HEX40 = re.compile(r"[0-9a-f]{40}\Z")
_RUN = re.compile(r"https://github\.com/Caerii/VisuaML/actions/runs/[0-9]+\Z")
_MAX_BLOB = 1024 * 1024
_REVIEWED = (
    "benchmarks/native_online_attention_observation_v31.json",
    "benchmarks/native_semantic_codegen_observation_v32.json",
    "benchmarks/export_evidence_observation_v33.json",
)


class EvidenceLibraryError(ValueError):
    """Invalid observation, identity, provenance, or attempted promotion."""


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _digest(value: Any) -> str:
    return _sha(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8"))


def _hash(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _HEX64.fullmatch(value):
        raise EvidenceLibraryError(f"Invalid {label} SHA-256")
    return value


def _commit(value: Any) -> str:
    if not isinstance(value, str) or not _HEX40.fullmatch(value):
        raise EvidenceLibraryError("Invalid source commit")
    return value


def _run(value: Any) -> str:
    if not isinstance(value, str) or not _RUN.fullmatch(value):
        raise EvidenceLibraryError("CI provenance must be an exact VisuaML run URL")
    return value


def _bounded_real(value: Any) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise EvidenceLibraryError("Nonfinite or nonnumeric observation")
    number = float(value)
    if not 0 <= number < 1e12:
        raise EvidenceLibraryError("Numerical observation outside admission bound")
    return number


def _blob(data: bytes, expected_schema: str) -> dict[str, Any]:
    if not isinstance(data, bytes) or not 1 <= len(data) <= _MAX_BLOB:
        raise EvidenceLibraryError("Evidence exceeds safe byte admission")
    try:
        obj = json.loads(data, parse_constant=lambda _: (_ for _ in ()).throw(
            EvidenceLibraryError("Nonfinite JSON numerical constant")
        ))
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise EvidenceLibraryError("Invalid JSON evidence") from exc
    if not isinstance(obj, dict) or obj.get("schema") != expected_schema:
        raise EvidenceLibraryError("Unknown evidence schema")
    return obj


@dataclass(frozen=True)
class EvidenceEntry:
    entry_id: str
    artifact_sha256: str
    source_path: str
    schema: str
    kind: str
    label: str
    evidence_class: str
    status: str
    workflow_url: str
    source_commit: str | None
    semantic_sha256: str | None
    generated_cpp_sha256: str | None
    native_binary_sha256: str | None
    checkpoint_sha256: str | None
    op_sequence: tuple[str, ...]
    context: tuple[tuple[str, str], ...]
    result: tuple[tuple[str, float], ...]
    limitations: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["op_sequence"] = list(self.op_sequence)
        data["context"] = dict(self.context)
        data["result"] = dict(self.result)
        data["limitations"] = list(self.limitations)
        return data


def _entry(
    *,
    sha: str, source: str, schema: str, kind: str, label: str,
    evidence_class: str, status: str, run: str,
    source_commit: str | None = None,
    semantic: str | None = None,
    cpp: str | None = None,
    binary: str | None = None,
    checkpoint: str | None = None,
    ops: tuple[str, ...] = (),
    context: dict[str, str] | None = None,
    result: dict[str, float] | None = None,
    limitations: tuple[str, ...] = (),
) -> EvidenceEntry:
    if not isinstance(label, str) or not 1 <= len(label) <= 96:
        raise EvidenceLibraryError("Invalid study label")
    if not isinstance(status, str) or not status:
        raise EvidenceLibraryError("Missing evidence status")
    if any(not isinstance(op, str) or not op or len(op) > 64 for op in ops):
        raise EvidenceLibraryError("Invalid operation signature")
    if len(ops) > 128:
        raise EvidenceLibraryError("Operator signature too large")
    ctx = context or {}
    if (
        len(ctx) > 16 or any(
            not isinstance(k, str) or not isinstance(v, str)
            or not 1 <= len(k) <= 48 or len(v) > 160
            for k, v in ctx.items()
        )
    ):
        raise EvidenceLibraryError("Invalid context identity")
    metrics = {k: _bounded_real(v) for k, v in (result or {}).items()}
    for name in (semantic, cpp, binary, checkpoint):
        if name is not None:
            _hash(name, "identity")
    if source_commit is not None:
        _commit(source_commit)
    _hash(sha, "artifact")
    _run(run)
    identity = _digest({
        "artifact": sha, "kind": kind, "label": label,
        "context": ctx, "semantic": semantic, "binary": binary,
    })
    return EvidenceEntry(
        identity, sha, source, schema, kind, label, evidence_class,
        status, run, source_commit, semantic, cpp, binary, checkpoint,
        ops, tuple(sorted(ctx.items())), tuple(sorted(metrics.items())),
        limitations,
    )


def _v31(data: dict, sha: str, source: str) -> list[EvidenceEntry]:
    run = _run(data["evidenceOrigin"]["nativeWorkflowUrl"])
    pretrained = data["pretrained"]
    checkpoint = _hash(pretrained["checkpointSha256"], "checkpoint")
    ratios = pretrained["ratios"]
    if len(ratios) != len(pretrained["pastTokens"]) or not ratios:
        raise EvidenceLibraryError("Malformed context/ratio observations")
    entries = []
    for name, shape, trial, source_name, revision, digest in (
        ("pinned-tiny-llama", pretrained, pretrained["trials"],
         pretrained["source"], pretrained["revision"], checkpoint),
        *(
            (
                "synthetic-" + item["name"], item, data["synthetic"]["trials"],
                "synthetic", "synthetic-fixture", None,
            )
            for item in data["synthetic"]["shapes"]
        ),
    ):
        if type(trial) is not int or not 1 <= trial <= 100:
            raise EvidenceLibraryError("Invalid observation trial count")
        checked_ratios = shape["ratios"]
        if not isinstance(checked_ratios, list) or len(checked_ratios) != len(
            pretrained["pastTokens"] if digest else data["synthetic"]["pastTokens"]
        ):
            raise EvidenceLibraryError("Inconsistent context study geometry")
        for ratio in checked_ratios:
            if not 0 < _bounded_real(ratio) <= 1000:
                raise EvidenceLibraryError("Invalid attention speed ratio")
        entries.append(_entry(
            sha=sha, source=source, schema=data["schema"],
            kind="attention_context_sweep", label=name,
            evidence_class="curated_historical_ci_observation",
            status="mixed_not_promoted", run=run, checkpoint=digest,
            ops=("attention", "softmax", "value_accumulation"),
            context={"model": source_name, "revision": revision,
                     "max_past": str(max(pretrained["pastTokens"]
                                      if digest else data["synthetic"]["pastTokens"])),
                     "shape": name},
            result={"min_speed_ratio": min(checked_ratios),
                    "max_speed_ratio": max(checked_ratios),
                    "trials": trial},
            limitations=tuple(data["limitations"]),
        ))
    return entries


def _v32(data: dict, sha: str, source: str) -> list[EvidenceEntry]:
    run = _run(data["workflow_url"])
    commit = _commit(data["source_commit"])
    entries = []
    for source_kind, array in (
        ("declared_ir", data["manual_fixtures"]),
        ("trusted_fx", data["trusted_fx_fixtures"]),
    ):
        if not isinstance(array, list) or len(array) != 2:
            raise EvidenceLibraryError("Expected exactly two historical fixtures")
        for fixture in array:
            cpp = _hash(fixture["generated_cpp_sha256"], "generated C++")
            binary = _hash(fixture["binary_sha256"], "native binary")
            metrics = {
                "max_abs_error_numpy": fixture["maximum_abs_error_numpy"],
                "max_abs_error_torch": fixture["maximum_abs_error_torch"],
                "compile_ms": fixture["compile_ms"],
            }
            if "warm_median_ms" in fixture:
                metrics["warm_median_ms"] = fixture["warm_median_ms"]
            entries.append(_entry(
                sha=sha, source=source, schema=data["schema"],
                kind="native_generated_graph", label=fixture["family"],
                evidence_class="curated_historical_ci_observation",
                status="finite_numeric_witnesses", run=run,
                source_commit=commit, cpp=cpp, binary=binary,
                ops=(
                    ("matmul", "relu", "matmul", "add")
                    if "mlp" in fixture["family"]
                    else ("conv2d", "relu", "add")
                ),
                context={"frontend": source_kind, "model": "synthetic"},
                result=metrics, limitations=tuple(data["limitations"]),
            ))
    return entries


def _v33(data: dict, sha: str, source: str, origin: dict) -> list[EvidenceEntry]:
    if origin["schema"] != "visuaml.durable-evidence-origin.v1":
        raise EvidenceLibraryError("Unknown v33 evidence origin")
    run = _run(origin["ci_run"])
    commit = _commit(origin["source_commit"])
    if origin["evidence_sha256"] != sha:
        raise EvidenceLibraryError("Archived original CI byte digest mismatch")
    entries = []
    models = data["model_witnesses"]
    if not isinstance(models, list) or len(models) != 2:
        raise EvidenceLibraryError("Malformed v33 model witnesses")
    for fixture in models:
        compiler = fixture["compiler"]
        cpp = _hash(compiler["generated_cpp_sha256"], "generated C++")
        binary = _hash(compiler["native_binary_sha256"], "native binary")
        if _sha(fixture["generated_cpp"].encode("utf-8")) != cpp:
            raise EvidenceLibraryError("Generated C++ bytes diverged from receipt")
        semantic = _hash(compiler["semantic_sha256"], "semantic IR")
        entries.append(_entry(
            sha=sha, source=source, schema=data["schema"],
            kind="native_exported_graph", label=fixture["family"],
            evidence_class="source_bound_finite_numeric_witnesses",
            status="finite_numeric_witnesses", run=run,
            source_commit=commit, semantic=semantic, cpp=cpp, binary=binary,
            ops=tuple(compiler["op_sequence"]),
            context={"frontend": "trusted_static_torch_export",
                     "model": "synthetic", "precision": "float32",
                     "native_target": str(compiler["target"]),
                     "compiler": str(compiler["compiler_identity"])[:160]},
            result={"max_abs_error_eager_torch": fixture["max_abs_error_eager_torch"],
                    "max_abs_error_second_input": fixture["max_abs_error_second_input"],
                    "warm_median_ms": fixture["warm_native_ms"]["median"]},
            limitations=tuple(data["limitations"]),
        ))
    rewrite = data["rewrite_witness"]
    optimizer = data["optimization_experiment"]
    repeated = optimizer["numerical_evidence"]
    # These are TWO independently produced receipts for the same computation.
    # Different compile timings create different record hashes legitimately.
    for receipt in (rewrite, repeated):
        digest = _hash(receipt["record_sha256"], "rewrite")
        unsigned = {k: v for k, v in receipt.items() if k != "record_sha256"}
        if _digest(unsigned) != digest:
            raise EvidenceLibraryError("Rewrite numerical record digest mismatch")
        if receipt["status"] not in ("passed_finite_witnesses", "rejected"):
            raise EvidenceLibraryError("Unknown numerical rewrite result")
        for side in ("before", "after"):
            for key in (
                "semantic_sha256", "generated_cpp_sha256",
                "native_binary_sha256",
            ):
                _hash(receipt[side][key], "rewrite program")
    for side in ("before", "after"):
        for key in ("semantic_sha256", "generated_cpp_sha256", "native_binary_sha256"):
            if rewrite[side][key] != repeated[side][key]:
                raise EvidenceLibraryError("Optimization changed rewrite identity")
    if rewrite["rule"] != repeated["rule"] or rewrite["status"] != repeated["status"]:
        raise EvidenceLibraryError("Optimization changed rewrite rule or status")
    first_witnesses = [x["input_sha256"] for x in rewrite["comparison"]["witnesses"]]
    second_witnesses = [x["input_sha256"] for x in repeated["comparison"]["witnesses"]]
    if first_witnesses != second_witnesses:
        raise EvidenceLibraryError("Optimization used different untracked witnesses")
    identity_context = {
        "rule": rewrite["rule"],
        "precision": "float32",
        "native_target": str(rewrite["before"]["target"]),
        "compiler": str(rewrite["before"]["compiler_identity"])[:160],
    }
    entries.append(_entry(
        sha=sha, source=source, schema=data["schema"], kind="rewrite_evaluation",
        label=rewrite["rule"], evidence_class="source_bound_finite_numeric_witnesses",
        status=rewrite["status"], run=run, source_commit=commit,
        semantic=_hash(rewrite["before"]["semantic_sha256"], "rewrite source IR"),
        cpp=_hash(rewrite["after"]["generated_cpp_sha256"], "rewrite generated C++"),
        binary=_hash(rewrite["after"]["native_binary_sha256"], "rewrite binary"),
        ops=("relu", "relu"),
        context={**identity_context, "numerical_receipt": rewrite["record_sha256"]},
        result={"witness_count": len(first_witnesses)},
        limitations=tuple(data["limitations"]),
    ))
    ratio = _bounded_real(optimizer["median_ratio_before_over_after"])
    if (
        ratio == 0
        or optimizer["status"] != "exploratory_timing_after_finite_numerical_witnesses"
        or repeated["status"] != "passed_finite_witnesses"
    ):
        raise EvidenceLibraryError("Unverified or invalid optimization performance claim")
    entries.append(_entry(
        sha=sha, source=source, schema=data["schema"],
        kind="optimization_measurement", label=rewrite["rule"] + " warm CPU comparison",
        evidence_class="exploratory_timing_after_finite_witnesses",
        status="measured_not_promoted", run=run, source_commit=commit,
        semantic=_hash(repeated["before"]["semantic_sha256"], "optimizer source IR"),
        cpp=_hash(repeated["after"]["generated_cpp_sha256"], "optimizer C++"),
        binary=_hash(repeated["after"]["native_binary_sha256"], "optimizer binary"),
        ops=("relu", "relu"),
        context={**identity_context, "numerical_receipt": repeated["record_sha256"]},
        result={
            "before_over_after_latency_ratio": ratio,
            "witness_count": len(second_witnesses),
        }, limitations=tuple(data["limitations"]),
    ))
    return entries


def _extract(data: dict, sha: str, path: str, origin: dict | None) -> list[EvidenceEntry]:
    kind = data["schema"]
    if kind == "visuaml.native-online-attention-observation.v31":
        return _v31(data, sha, path)
    if kind == "visuaml.native-semantic-observation.v32":
        return _v32(data, sha, path)
    if kind == "visuaml.export-evidence-demonstration.v33" and origin is not None:
        return _v33(data, sha, path, origin)
    raise EvidenceLibraryError("Evidence schema has no approved adapter")


class EvidenceLibrary:
    """Immutable, sorted in-memory index over reviewed artifacts.

    A matching entry is a discovery hit, NEVER a claim of all-input validity,
    independent reviewer approval, or hardware portability.
    """

    def __init__(self, entries: list[EvidenceEntry], artifacts: dict[str, str]):
        identifiers = [item.entry_id for item in entries]
        if len(set(identifiers)) != len(identifiers):
            raise EvidenceLibraryError("Duplicate semantic library entries")
        self._entries = tuple(sorted(entries, key=lambda x: (x.kind, x.label, x.entry_id)))
        self._artifacts = dict(sorted(artifacts.items()))

    @classmethod
    def from_repo(cls, repo: Path) -> EvidenceLibrary:
        source = Path(repo)
        entries: list[EvidenceEntry] = []
        artifacts = {}
        origin_path = source / "benchmarks/export_evidence_observation_v33.origin.json"
        origin = _blob(
            origin_path.read_bytes(), "visuaml.durable-evidence-origin.v1"
        )
        for path in _REVIEWED:
            raw = (source / path).read_bytes()
            if len(raw) > _MAX_BLOB:
                raise EvidenceLibraryError("Historical evidence artifact over budget")
            sha = _sha(raw)
            obj = _blob(raw, {
                _REVIEWED[0]: "visuaml.native-online-attention-observation.v31",
                _REVIEWED[1]: "visuaml.native-semantic-observation.v32",
                _REVIEWED[2]: "visuaml.export-evidence-demonstration.v33",
            }[path])
            entries.extend(_extract(
                obj, sha, path, origin if path == _REVIEWED[2] else None
            ))
            artifacts[path] = sha
        return cls(entries, artifacts)

    @property
    def entries(self) -> tuple[EvidenceEntry, ...]:
        return self._entries

    def query(
        self, *, kind: str | None = None,
        op_sequence: tuple[str, ...] | None = None,
        semantic_sha256: str | None = None,
        native_binary_sha256: str | None = None,
        checkpoint_sha256: str | None = None,
        context: dict[str, str] | None = None,
    ) -> tuple[EvidenceEntry, ...]:
        if semantic_sha256 is not None:
            _hash(semantic_sha256, "semantic")
        if native_binary_sha256 is not None:
            _hash(native_binary_sha256, "binary")
        if checkpoint_sha256 is not None:
            _hash(checkpoint_sha256, "checkpoint")
        required = context or {}
        if any(not isinstance(k, str) or not isinstance(v, str)
               for k, v in required.items()):
            raise EvidenceLibraryError("Context query must be exact string pairs")
        return tuple(
            item for item in self._entries
            if (kind is None or item.kind == kind)
            and (op_sequence is None or item.op_sequence == op_sequence)
            and (semantic_sha256 is None or item.semantic_sha256 == semantic_sha256)
            and (native_binary_sha256 is None
                 or item.native_binary_sha256 == native_binary_sha256)
            and (checkpoint_sha256 is None or item.checkpoint_sha256 == checkpoint_sha256)
            and all(dict(item.context).get(k) == v for k, v in required.items())
        )

    def neighbors(self, entry_id: str) -> tuple[dict[str, str], ...]:
        """Group exact binary/source identities without inferring equivalence."""
        anchor = next((entry for entry in self._entries
                       if entry.entry_id == entry_id), None)
        if anchor is None:
            raise EvidenceLibraryError("Unknown evidence entry")
        found = []
        for other in self._entries:
            if other.entry_id == entry_id:
                continue
            if (
                anchor.native_binary_sha256 is not None
                and anchor.native_binary_sha256 == other.native_binary_sha256
            ):
                relation = "same_native_binary_hash"
            elif (
                anchor.generated_cpp_sha256 is not None
                and anchor.generated_cpp_sha256 == other.generated_cpp_sha256
            ):
                relation = "same_generated_source_hash"
            else:
                continue
            found.append({"entry_id": other.entry_id, "relation": relation})
        return tuple(sorted(found, key=lambda item: item["entry_id"]))

    def catalog(self) -> dict[str, Any]:
        data = {
            "schema": SCHEMA,
            "entries": [record.as_dict() for record in self._entries],
            "artifacts": self._artifacts,
            "counts": {
                "studies": len(self._entries),
                "source_artifacts": len(self._artifacts),
                "kinds": {
                    kind: sum(entry.kind == kind for entry in self._entries)
                    for kind in sorted({entry.kind for entry in self._entries})
                },
            },
            "limitations": [
                "Discovery and exact fingerprint grouping, not proof of all-input equivalence.",
                "All observations are historical and bound to particular machines/workloads.",
                "A content hash alone is not an authenticated signer approval.",
                "No status/roadmap promotion or optimization selection occurs automatically.",
                "Curated v31/v32 summaries cannot be upgraded into full raw CI evidence.",
                "New proposals remain unreviewed until independently replayed and accepted.",
            ],
        }
        return {**data, "catalog_sha256": _digest(data)}


def _write_exclusive(path: Path, data: bytes) -> None:
    """Never replace a prior content-addressed payload or its origin claim."""
    try:
        with path.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError:
        if path.read_bytes() != data:
            raise EvidenceLibraryError("Content-address collision or tampering") from None


def _proposal_origin(path: Path) -> Path:
    return path.with_name(path.stem + ".origin.json")


def submit_unreviewed_proposal(
    directory: Path, data: bytes, *, source_url: str,
) -> Path:
    """Add bounded content AND immutable run provenance, never promote it.

    A process interrupted between the two exclusive writes leaves an invalid
    incomplete proposal which cannot be accepted without deliberate recovery.
    This is a local staging API, not a safe multi-tenant remote upload service.
    """
    if not isinstance(data, bytes) or not 1 <= len(data) <= _MAX_BLOB:
        raise EvidenceLibraryError("Proposal exceeds byte budget")
    run = _run(source_url)
    _blob(data, "visuaml.export-evidence-demonstration.v33")
    digest = _sha(data)
    destination = Path(directory) / "proposals"
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / (digest + ".json")
    origin = {
        "schema": _PROPOSAL_SCHEMA,
        "artifact_sha256": digest,
        "evidence_schema": "visuaml.export-evidence-demonstration.v33",
        "source_url": run,
        "status": "unreviewed",
    }
    origin_bytes = (
        json.dumps(origin, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")
    _write_exclusive(target, data)
    _write_exclusive(_proposal_origin(target), origin_bytes)
    if not verify_unreviewed_proposal(target):
        raise EvidenceLibraryError("Incomplete or invalid staged proposal provenance")
    return target


def verify_unreviewed_proposal(path: Path) -> bool:
    selected = Path(path)
    if not _HEX64.fullmatch(selected.stem) or selected.suffix != ".json":
        return False
    try:
        data = selected.read_bytes()
        _blob(data, "visuaml.export-evidence-demonstration.v33")
        origin = _blob(
            _proposal_origin(selected).read_bytes(), _PROPOSAL_SCHEMA
        )
        if set(origin) != {
            "schema", "artifact_sha256", "evidence_schema",
            "source_url", "status",
        }:
            return False
        _run(origin["source_url"])
    except (OSError, ValueError):
        return False
    return (
        _sha(data) == selected.stem == origin["artifact_sha256"]
        and origin["evidence_schema"] == "visuaml.export-evidence-demonstration.v33"
        and origin["status"] == "unreviewed"
    )
