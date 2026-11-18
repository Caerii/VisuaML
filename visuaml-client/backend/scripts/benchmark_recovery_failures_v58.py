"""Failure-complete v58 neural-to-symbolic recovery evaluation.

Unlike the strict proof-only artifact, this campaign preserves every
attempt, including failed CEGIS searches, counterexamples, solver timeouts,
and rejected behaviors. An exception is never counted as recovery success.

This is a *separate* scoreboard. Successful recoveries still require
typed-source all-i32 Z3 unsat and independent source/target execution;
failures do not imply absence of an equivalent algorithm in another grammar.
No GitHub Actions, network access or learned-weight decompilation claims.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path

from visuaml.compiler_lab.blinded_behavioral_synthesis_v50 import (
    BehavioralSynthesisError,
)
from visuaml.compiler_lab.neural_symbolic_roundtrip_v58 import (
    RoundtripWorkload,
    source_workloads,
    verify_roundtrip,
)
from visuaml.compiler_lab.wasm_state_vm_v40 import (
    I32Program,
    VMOperation,
    encode_program,
)

SCHEMA = "visuaml.failure-complete-blinded-recovery.v58"


def _digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _stable(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()


def negative_cubic() -> RoundtripWorkload:
    return RoundtripWorkload("outside_grammar_cubic", I32Program(0, (
        VMOperation("local.get", 0),
        VMOperation("local.get", 0),
        VMOperation("i32.mul"),
        VMOperation("local.get", 0),
        VMOperation("i32.mul"),
        VMOperation("end"),
    )), "negative_control")


def eligible_workloads(*, holdout: bool) -> tuple[RoundtripWorkload, ...]:
    return tuple(x for x in source_workloads()
                 if holdout or x.split == "historical") + (negative_cubic(),)


def classify_failure(exc: BaseException) -> str:
    if isinstance(exc, BehavioralSynthesisError):
        message = str(exc)
        if "No surviving candidate" in message:
            return "grammar_rejection"
        if "fuel exhausted" in message:
            return "cegis_budget_exhausted"
        if "unknown" in message or "timeout" in message:
            return "solver_unknown_or_timeout"
        return "synthesis_protocol_error"
    if isinstance(exc, AssertionError):
        return "semantic_or_replay_mismatch"
    return "execution_or_dependency_error"


def attempt(workload: RoundtripWorkload, *,
            timeout_ms: int = 15000) -> dict:
    """Produce a per-source outcome; never suppress unexpected failures."""
    original = encode_program(workload.source)
    start = time.perf_counter_ns()
    audit: dict = {}
    try:
        result = verify_roundtrip(
            workload, timeout_ms=timeout_ms, audit=audit,
        )
    except (BehavioralSynthesisError, AssertionError) as exc:
        return {
            "name": workload.name,
            "split": workload.split,
            "source_wasm_sha256": _digest(original),
            "source_wasm_hex": original.hex(),
            "audit": audit,
            "status": "rejected",
            "failure_class": classify_failure(exc),
            "exception_type": type(exc).__name__,
            "message": str(exc),
            "elapsed_ns": time.perf_counter_ns() - start,
            "all_i32_typed_source_equivalence_proved": False,
            "all_i32_neural_weights_equivalence_proved": False,
        }
    return {
        "name": workload.name,
        "split": workload.split,
        "source_wasm_sha256": _digest(original),
        "source_wasm_hex": original.hex(),
        "audit": audit,
        "status": "verified",
        "failure_class": None,
        "elapsed_ns": time.perf_counter_ns() - start,
        "all_i32_typed_source_equivalence_proved": True,
        "all_i32_neural_weights_equivalence_proved": False,
        "receipt": result,
    }


def summarize(attempts: list[dict]) -> dict:
    if not attempts:
        raise ValueError("Cannot summarize empty recovery experiment")
    if len({x["source_wasm_sha256"] for x in attempts}) != len(attempts):
        raise ValueError("Duplicate source programs in recovery evaluation")
    per_split = {}
    for split in sorted({x["split"] for x in attempts}):
        cases = [x for x in attempts if x["split"] == split]
        verified = sum(x["status"] == "verified" for x in cases)
        rejected = len(cases) - verified
        classes = {}
        for item in cases:
            if item["status"] == "rejected":
                kind = item["failure_class"]
                classes[kind] = classes.get(kind, 0) + 1
        per_split[split] = {
            "attempted": len(cases),
            "verified": verified,
            "rejected": rejected,
            "verified_fraction": verified / len(cases),
            "failure_classes": classes,
        }
    return {
        "total_attempted": len(attempts),
        "total_verified": sum(x["status"] == "verified" for x in attempts),
        "total_rejected": sum(x["status"] == "rejected" for x in attempts),
        "by_split": per_split,
        "definition": (
            "Finite attempted source programs within an explicit bounded "
            "synthesis grammar, not a universal neural decompilation rate"
        ),
    }


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(
        ("git", "-C", str(root), *args),
        text=True, stderr=subprocess.DEVNULL,
    ).strip()


def run(*, holdout: bool, timeout_ms: int, root: Path) -> dict:
    if not 1 <= timeout_ms <= 60000:
        raise ValueError("Unsupported SMT timeout policy")
    if _git(root, "status", "--porcelain", "--untracked-files=normal"):
        raise RuntimeError("Recovery experiments require clean source checkout")
    source = {
        "commit": _git(root, "rev-parse", "HEAD"),
        "tree": _git(root, "rev-parse", "HEAD^{tree}"),
    }
    attempts = []
    for workload in eligible_workloads(holdout=holdout):
        attempts.append(attempt(workload, timeout_ms=timeout_ms))
    summary = summarize(attempts)
    negative = summary["by_split"]["negative_control"]
    if negative["verified"] != 0:
        raise AssertionError("Nonadmitted negative-control behavior accepted")
    return {
        "schema": SCHEMA,
        "source": source,
        "environment": {"python": platform.python_version()},
        "policy": {
            "holdout_enabled": holdout,
            "solver_timeout_ms": timeout_ms,
            "negative_control_must_reject": True,
            "failures_in_denominator": True,
            "all_i32_neural_weights_proved": False,
            "arbitrary_learned_weight_decompilation": False,
            "original_source_uniqueness_claim": False,
        },
        "summary": summary,
        "attempts": attempts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--holdout", action="store_true")
    parser.add_argument("--timeout-ms", type=int, default=15000)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        parser.error("Refusing to overwrite recovery campaign")
    root = Path(__file__).resolve().parents[3]
    if root == output or root in output.parents:
        parser.error("Choose an evidence path outside the checkout")
    result = run(
        holdout=args.holdout,
        timeout_ms=args.timeout_ms, root=root,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    raw = _stable(result) + b"\n"
    with output.open("xb") as handle:
        handle.write(raw)
    print(json.dumps({
        "file": str(output),
        "sha256": _digest(raw),
        "source_commit": result["source"]["commit"],
        "summary": result["summary"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
