"""v22 warm decode and graph-specialization benchmark has numerical parity gates."""

from __future__ import annotations

import pytest
from examples.llama_polymorphic_bench_v22 import benchmark
from test_hf_llama_checkpoint_v19 import fixture
from visuaml.compiler_lab.hf_llama_checkpoint_v19 import load_hf_llama_local


@pytest.mark.parametrize("layers", (1, 2))
def test_polymorphic_benchmark_has_one_graph_for_irregular_chunks(tmp_path, layers):
    fixture(tmp_path, layers=layers)
    model = load_hf_llama_local(tmp_path, declared_revision="a" * 40)
    result = benchmark(model, chunks=(3, 1, 5, 2), trials=3)
    assert result["schema"] == "visuaml.llama-polymorphic-cache-benchmark.v22"
    assert result["compilation"]["torchscript_polymorphic_v22"]["specializations"] == 1
    assert result["compilation"]["torchscript_specialized_v21"]["specializations"] == 4
    assert result["compilation"]["torchscript_polymorphic_v22"]["total_trace_ms"] > 0
    assert result["latency"]["torchscript_polymorphic_v22"]["median_ms"] > 0
    assert result["latency"]["torchscript_specialized_v21"]["median_ms"] > 0
    assert result["resident_cache_reserved_bytes"]["torchscript_polymorphic_v22"] > 0
    traffic = result["cumulative_kv_tensor_accounting"]
    assert traffic["torchscript_polymorphic_v22"]["returned_kv_tensor_bytes"] < (
        traffic["torchscript_specialized_v21"]["returned_kv_tensor_bytes"]
    )
    assert traffic["torchscript_polymorphic_v22"]["input_kv_tensor_bytes"] == (
        traffic["torchscript_specialized_v21"]["input_kv_tensor_bytes"]
    )
    assert len(result["source_files_sha256"]) == 4


@pytest.mark.parametrize("invalid", ((33,), (0,), (1, 1, 63)))
def test_benchmark_rejects_invalid_shapes(tmp_path, invalid):
    fixture(tmp_path, layers=1)
    model = load_hf_llama_local(tmp_path, declared_revision="a" * 40)
    with pytest.raises(ValueError):
        benchmark(model, chunks=invalid, trials=2)
