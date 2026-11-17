"""Numerical admission and evidence-schema tests for v31 attention sweeps."""

from __future__ import annotations

import pytest
from examples.native_online_attention_bench_v31 import benchmark
from test_hf_llama_checkpoint_v19 import fixture
from visuaml.compiler_lab.hf_llama_checkpoint_v19 import load_hf_llama_local
from visuaml.compiler_lab.native_cpu_v28 import compiler_available

pytestmark = pytest.mark.skipif(
    not compiler_available(), reason="Native C++17 compiler unavailable"
)


def test_attention_context_sweep_checks_oracle_and_retains_samples(tmp_path):
    fixture(
        tmp_path, layers=1, hidden=48, heads=6,
        kv=3, middle=101, vocab=67,
    )
    weights = load_hf_llama_local(tmp_path, declared_revision="a" * 40)
    result = benchmark(weights, prefixes=(1, 8, 31, 48), trials=2)
    assert result["schema"] == "visuaml.native-online-attention-benchmark.v31"
    assert [case["past_tokens"] for case in result["cases"]] == [1, 8, 31, 48]
    assert result["certificate"].startswith("Every prefill")
    assert result["compilers"]["online"]["attention_kernel"] == "online_stable_softmax"
    assert result["compilers"]["materialized"]["attention_kernel"] == "materialized_scores"
    assert result["compilers"]["online"]["executable_sha256"] != (
        result["compilers"]["materialized"]["executable_sha256"]
    )
    for case in result["cases"]:
        assert case["online_vs_materialized_decode_ratio"] > 0
        for name in ("materialized", "online"):
            for phase in ("prefix_ms", "decode_ms"):
                samples = case["timing"][name][phase]["raw_samples_ms"]
                assert len(samples) == 2
                assert all(value > 0 for value in samples)
    assert len(result["source_sha256"]) == 3


def test_attention_sweep_rejects_invalid_past(tmp_path):
    fixture(tmp_path, layers=1)
    weights = load_hf_llama_local(tmp_path, declared_revision="a" * 40)
    for prefixes in ((), (64,), (0,), (8, 8), (4.5,)):
        with pytest.raises(ValueError, match="Invalid bounded"):
            benchmark(weights, prefixes=prefixes, trials=2)
