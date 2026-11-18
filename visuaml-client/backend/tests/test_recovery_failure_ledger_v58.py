"""Failure-complete blinded-recovery benchmark accounting tests.

Synthetic fixtures test denominators without claiming successful real
recoveries. Real integration tests stay behind explicit offline Z3/Wasmtime.
"""
from __future__ import annotations

import pytest

from scripts import benchmark_recovery_failures_v58 as benchmark
from visuaml.compiler_lab.blinded_behavioral_synthesis_v50 import (
    BehavioralSynthesisError,
)
from visuaml.compiler_lab.neural_symbolic_roundtrip_v58 import (
    source_workloads,
)


def test_recovery_inventory_includes_negative_control_and_heldout():
    items = benchmark.eligible_workloads(holdout=True)
    assert len(items) == 14
    assert sum(x.split == "historical" for x in items) == 5
    assert sum(x.split == "heldout" for x in items) == 8
    assert sum(x.split == "negative_control" for x in items) == 1
    assert len(benchmark.eligible_workloads(holdout=False)) == 6
    assert items[-1].name == "outside_grammar_cubic"


def test_failed_attempt_cannot_disappear_from_denominator(monkeypatch):
    source = source_workloads()[0]

    def rejected(*args, **kwargs):
        raise BehavioralSynthesisError("No surviving candidate in admitted grammar")

    monkeypatch.setattr(benchmark, "verify_roundtrip", rejected)
    record = benchmark.attempt(source)
    assert record["status"] == "rejected"
    assert record["failure_class"] == "grammar_rejection"
    assert not record["all_i32_typed_source_equivalence_proved"]
    score = benchmark.summarize([record])
    assert score["total_attempted"] == 1
    assert score["total_verified"] == 0
    assert score["total_rejected"] == 1
    assert score["by_split"]["historical"]["verified_fraction"] == 0.


def test_verified_attempt_requires_actual_receipt(monkeypatch):
    source = source_workloads()[0]
    monkeypatch.setattr(benchmark, "verify_roundtrip", lambda *_args, **_kwargs: {
        "all_i32_typed_source_equivalence": True,
        "finite_results": [{"input_i32": 0}],
    })
    record = benchmark.attempt(source)
    assert record["status"] == "verified"
    score = benchmark.summarize([record])
    assert score["total_verified"] == 1


def test_timeout_and_fuel_have_separate_failure_classes():
    assert benchmark.classify_failure(
        BehavioralSynthesisError("CEGIS refinement fuel exhausted")
    ) == "cegis_budget_exhausted"
    assert benchmark.classify_failure(
        BehavioralSynthesisError("SMT verifier returned unknown: timeout")
    ) == "solver_unknown_or_timeout"
    assert benchmark.classify_failure(
        AssertionError("trace mismatch")
    ) == "semantic_or_replay_mismatch"


def test_duplicate_source_programs_rejected():
    program = source_workloads()[0]
    record = {
        "name": "a",
        "split": "historical",
        "source_wasm_sha256": program.source.digest(),
        "status": "rejected",
        "failure_class": "grammar_rejection",
    }
    with pytest.raises(ValueError, match="Duplicate"):
        benchmark.summarize([record, record])


def test_out_of_grammar_neural_cubic_is_not_accepted():
    pytest.importorskip("wasmtime")
    pytest.importorskip("z3")
    outcome = benchmark.attempt(benchmark.negative_cubic())
    assert outcome["status"] == "rejected"
    assert outcome["failure_class"] == "grammar_rejection"
