"""v58 paired blinded-recovery campaign comparator contract tests."""
from __future__ import annotations

import copy
import hashlib

import pytest

from scripts.benchmark_recovery_failures_v58 import summarize
from scripts.compare_recovery_campaigns_v58 import compare


def campaign(statuses=("rejected", "verified")):
    attempts = []
    for i, status in enumerate(statuses):
        source = bytes((0x00, 0x61, 0x73, 0x6D, i))
        attempts.append({
            "name": f"fixture_{i}",
            "split": "historical",
            "status": status,
            "source_wasm_hex": source.hex(),
            "source_wasm_sha256": hashlib.sha256(source).hexdigest(),
            "failure_class": (
                "grammar_rejection" if status == "rejected"
                else "semantic_or_replay_mismatch" if status == "invalid"
                else None
            ),
            "audit": {"neural_oracle_queries": 7 + i},
        })
    return {
        "_file_sha256": "f" * 64,
        "source": {"commit": "a" * 40},
        "policy": {
            "holdout_enabled": False,
            "solver_timeout_ms": 15000,
            "failures_in_denominator": True,
        },
        "attempts": attempts,
        "summary": summarize(attempts),
    }


def test_new_recovery_and_regression_both_remain_visible():
    first = campaign(("rejected", "verified"))
    second = campaign(("verified", "rejected"))
    result = compare(first, second)
    assert result["newly_verified_from_rejected"] == ["fixture_0"]
    assert result["regressed_from_verified"] == ["fixture_1"]
    assert result["status_transitions"]["rejected->verified"] == 1
    assert result["status_transitions"]["verified->rejected"] == 1
    assert result["paired_oracle_query_delta_median"] == 0
    assert result["source_programs_match"]


def test_invalid_outcome_is_not_scored_as_newly_verified():
    before = campaign(("invalid", "rejected"))
    after = campaign(("verified", "invalid"))
    result = compare(before, after)
    assert result["newly_verified_from_rejected"] == []
    assert result["regressed_from_verified"] == []
    assert result["status_transitions"]["invalid->verified"] == 1
    assert result["status_transitions"]["rejected->invalid"] == 1


def test_modified_source_program_must_not_be_compared():
    baseline = campaign()
    changed = copy.deepcopy(baseline)
    changed["attempts"][0]["source_wasm_hex"] = b"other".hex()
    changed["attempts"][0]["source_wasm_sha256"] = hashlib.sha256(
        b"other"
    ).hexdigest()
    with pytest.raises(ValueError, match="source/split"):
        compare(baseline, changed)


def test_changed_solver_policy_must_not_be_compared():
    baseline = campaign()
    changed = copy.deepcopy(baseline)
    changed["policy"]["solver_timeout_ms"] = 1000
    with pytest.raises(ValueError, match="policy changed"):
        compare(baseline, changed)


def test_missing_witness_cannot_be_scored_as_recovered():
    baseline = campaign()
    changed = copy.deepcopy(baseline)
    changed["attempts"].pop()
    with pytest.raises(ValueError, match="inventory changed"):
        compare(baseline, changed)
