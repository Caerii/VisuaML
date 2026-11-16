"""v20 latency harness must verify output correctness before timing."""

from __future__ import annotations

import pytest
from test_hf_llama_checkpoint_v19 import fixture
from visuaml.compiler_lab.hf_llama_checkpoint_v19 import load_hf_llama_local
from examples.llama_semantic_bench_v20 import benchmark


@pytest.mark.parametrize("hybrid", (False, True))
def test_small_real_shaped_checkpoint_benchmark(hybrid, tmp_path):
    fixture(tmp_path, layers=1)
    checkpoint = load_hf_llama_local(tmp_path, declared_revision="a" * 40)
    report = benchmark(
        checkpoint, lengths=(3, 5), trials=2, include_hybrid=hybrid
    )
    assert report["schema"] == "visuaml.llama-semantic-cpu-benchmark.v20"
    assert report["graph_setup_ms"] > 0
    assert len(report["results"]) == 2
    assert report["logical_liveness"]["peak_logical_bytes"] > 0
    for row in report["results"]:
        assert row["latency"]["reference_v19"]["median_ms"] > 0
        assert row["latency"]["numpy"]["median_ms"] > 0
        if hybrid:
            assert row["latency"]["torchscript_hybrid"]["median_ms"] > 0
            assert row["torchscript_specializations"] > 0


def test_benchmark_rejects_unadmitted_lengths(tmp_path):
    fixture(tmp_path, layers=1)
    checkpoint = load_hf_llama_local(tmp_path, declared_revision="a" * 40)
    with pytest.raises(ValueError):
        benchmark(checkpoint, lengths=(99,), trials=2)
    with pytest.raises(ValueError):
        benchmark(checkpoint, lengths=(4,), trials=1)
