"""Measure v23 versus v24 module duplication with independent output parity."""

from __future__ import annotations

import pytest

from examples.shared_weights_bench_v24 import benchmark
from test_hf_llama_checkpoint_v19 import fixture
from visuaml.compiler_lab.hf_llama_checkpoint_v19 import load_hf_llama_local


@pytest.mark.parametrize("clients", (2, 4))
def test_benchmark_confirms_one_module_for_many_sessions(tmp_path, clients):
    fixture(tmp_path, layers=1)
    checkpoint = load_hf_llama_local(
        tmp_path, declared_revision="a" * 40
    )
    result = benchmark(checkpoint, clients=clients, trials=2, prompt_len=3)
    assert result["schema"] == "visuaml.shared-weights-benchmark.v24"
    old = result["results"]["v23_graph_only"]
    new = result["results"]["v24_graph_and_weights"]
    assert old["unique_weight_module_counts"] == [clients, clients]
    assert new["unique_weight_module_counts"] == [1, 1]
    assert old["compiled_graph_counts"] == [1, 1]
    assert new["compiled_graph_counts"] == [1, 1]
    assert old["weight_module_logical_bytes"][0] == (
        new["weight_module_logical_bytes"][0] * clients
    )
    assert old["kv_storage_bytes"] == new["kv_storage_bytes"]
    assert old["cold_wall_median_ms"] > 0
    assert new["cold_wall_median_ms"] > 0
    assert len(result["source_sha256"]) == 4


@pytest.mark.parametrize("bad", (1, 17))
def test_benchmark_rejects_invalid_clients(tmp_path, bad):
    fixture(tmp_path, layers=1)
    checkpoint = load_hf_llama_local(
        tmp_path, declared_revision="a" * 40
    )
    with pytest.raises(ValueError):
        benchmark(checkpoint, clients=bad, trials=2)
