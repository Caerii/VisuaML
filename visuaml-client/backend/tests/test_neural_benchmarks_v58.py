"""Tests for reproducible benchmark protocol, not fabricated timings."""
from __future__ import annotations

from copy import deepcopy

import pytest

from scripts.benchmark_neural_vm_v58 import (
    SCHEMA, summarize_ns, workloads,
)
from scripts.compare_neural_benchmarks_v58 import (
    compare, quantile, speedup_stats,
)


def tiny_report(ns=100):
    case = {
        "input_i32": -17,
        "expected_i32": 4,
        "steps": 3,
        "latencies": {
            "host_v51": {"wall": {"median_ns": ns}},
            "microstep_v58": {"wall": {"median_ns": ns}},
        },
    }
    return {
        "schema": SCHEMA,
        "_raw_sha256": str(ns).zfill(64),
        "environment": {
            "platform": "fixture", "machine": "synthetic",
            "processor": "no-real-measurement", "cpu_count": 1,
            "python": "3.12", "numpy": "test", "wasmtime": "test",
            "thread_controls": {},
        },
        "policy": {
            "repeats": 5, "warmups": 1,
            "fuel": 1500, "holdout_programs": 0,
        },
        "engines": ["host_v51", "microstep_v58"],
        "rows": [{
            "name": "synthetic_ONLY_not_measurement",
            "split": "public",
            "family": "synthetic",
            "source_wasm_sha256": "a" * 64,
            "cases": [case],
        }],
    }


def test_nearest_rank_p95_and_median_are_explicit():
    out = summarize_ns([12, 4, 7, 50, 6])
    assert out["median_ns"] == 7
    assert out["p95_ns"] == 50
    assert out["min_ns"] == 4
    assert out["max_ns"] == 50
    with pytest.raises(ValueError):
        summarize_ns([])


def test_baseline_comparison_reports_paired_ratio_not_aggregate_time():
    a = tiny_report(200)
    b = tiny_report(100)
    result = compare(a, b, engine="microstep_v58", require_compatible=True)
    assert result["all"]["geometric_mean_speedup"] == 2.
    assert result["comparable_witnesses"] == 1
    assert result["environment_compatible"]
    assert not result["missing_engine_witnesses"]
    assert quantile([1., 2., 3., 4.], .5) == 2.5
    assert speedup_stats([.5, 2.])["fraction_slower"] == .5


def test_changed_original_wasm_or_inputs_must_fail_comparison():
    baseline = tiny_report()
    modified = deepcopy(baseline)
    modified["rows"][0]["source_wasm_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="Source/split"):
        compare(baseline, modified, engine="host_v51")
    modified = deepcopy(baseline)
    modified["rows"][0]["cases"][0]["input_i32"] = 21
    with pytest.raises(ValueError, match="Input inventory"):
        compare(baseline, modified, engine="host_v51")


def test_hardware_mismatch_must_fail_strict_comparison():
    baseline = tiny_report()
    modified = deepcopy(baseline)
    modified["environment"]["machine"] = "other"
    result = compare(baseline, modified, engine="host_v51")
    assert not result["environment_compatible"]
    assert "environment.machine" in result["compatibility_mismatches"]
    with pytest.raises(ValueError, match="Incompatible"):
        compare(
            baseline, modified, engine="host_v51",
            require_compatible=True,
        )


def test_missing_engine_not_padded_with_fake_zero():
    a = tiny_report()
    b = deepcopy(a)
    b["rows"][0]["cases"][0]["latencies"].pop("microstep_v58")
    with pytest.raises(ValueError, match="No comparable"):
        compare(a, b, engine="microstep_v58")


def test_smoke_inventory_is_small_and_explicitly_nonrepresentative():
    cases = workloads(holdout=8, scaling=True, smoke=True)
    assert [item.name for item in cases] == ["countdown", "generated_00"]
    assert sum(len(item.arguments) for item in cases) == 4
    assert all(item.split == "smoke" for item in cases)


def test_timed_measurement_checks_correctness_even_after_warmup():
    from scripts.benchmark_neural_vm_v58 import measure

    assert measure(lambda: 17, warmups=1, repeats=3, expected=17)[
        "wall"
    ]["count"] == 3
    with pytest.raises(AssertionError, match="Timed engine diverged"):
        measure(lambda: 18, warmups=1, repeats=1, expected=17)


def test_workload_discovery_has_disjoint_holdout_source_programs():
    work = workloads(holdout=8)
    assert len(work) == 27
    assert len({w.name for w in work}) == len(work)
    assert sum(w.split == "heldout" for w in work) == 8
    assert sum(len(w.arguments) for w in work) == 194
    with pytest.raises(ValueError):
        workloads(holdout=-1)


def test_system_benchmark_registry_is_accurate_and_nonfabricated():
    from pathlib import Path

    from scripts.list_system_benchmarks_v58 import read_registry, summary

    root = Path(__file__).resolve().parents[3]
    record = read_registry(root)
    values = summary(record)
    assert values["suite_count"] >= 10
    assert values["implemented_unmeasured"] >= 2
    assert values["planned"] >= 5
    assert values["measured_result_count"] == 0
    assert {s["id"] for s in values["suites"]} >= {
        "neural-vm", "compiler-ir-roundtrip", "collaborative-graph",
        "neural-symbolic-recovery", "native-backend-parity",
    }
