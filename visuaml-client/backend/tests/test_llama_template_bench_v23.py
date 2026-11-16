"""v23 benchmark measures separate model compilation versus shared template."""

from __future__ import annotations

import pytest
from examples.llama_template_bench_v23 import benchmark
from test_hf_llama_checkpoint_v19 import fixture
from visuaml.compiler_lab.hf_llama_checkpoint_v19 import load_hf_llama_local


@pytest.mark.parametrize("clients", (2, 3))
def test_template_benchmark_checks_numerical_parity_and_one_compile(
    tmp_path, clients
):
    fixture(tmp_path, layers=1)
    weights = load_hf_llama_local(tmp_path, declared_revision="a" * 40)
    report = benchmark(weights, clients=clients, trials=2, prompt_len=3)
    assert report["schema"] == "visuaml.llama-compiled-template-benchmark.v23"
    assert report["results"]["per_session_retrace"]["graph_counts"] == [clients] * 2
    assert report["results"]["shared_template"]["graph_counts"] == [1] * 2
    assert report["results"]["shared_template"]["trace_median_ms"] > 0
    assert report["results"]["per_session_retrace"]["wall_median_ms"] > 0
    assert len(report["source_files_sha256"]) == 3


@pytest.mark.parametrize("bad", (1, 17))
def test_benchmark_rejects_invalid_client_count(tmp_path, bad):
    fixture(tmp_path, layers=1)
    weights = load_hf_llama_local(tmp_path, declared_revision="a" * 40)
    with pytest.raises(ValueError):
        benchmark(weights, clients=bad, trials=2)
