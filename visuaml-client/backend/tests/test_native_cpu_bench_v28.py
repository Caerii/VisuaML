"""Parity-gated native C++ benchmark, including honest compilation accounting."""

from __future__ import annotations

import pytest
from examples.native_cpu_bench_v28 import benchmark
from test_hf_llama_checkpoint_v19 import fixture
from visuaml.compiler_lab.hf_llama_checkpoint_v19 import load_hf_llama_local
from visuaml.compiler_lab.native_cpu_v28 import compiler_available

pytestmark = pytest.mark.skipif(
    not compiler_available(), reason="Native C++17 compiler not installed"
)


def test_native_cpu_benchmark_executes_and_checks_two_backends(tmp_path):
    fixture(tmp_path, layers=1)
    weights = load_hf_llama_local(
        tmp_path, declared_revision="a" * 40
    )
    result = benchmark(
        weights, prompt=3, chunk_sizes=(1, 2), trials=2
    )
    assert result["schema"] == "visuaml.native-cpu-benchmark.v28"
    assert result["numerical_certificate"].startswith("all chunked logits")
    assert result["latency"]["native_cpp"]["median_ms"] > 0
    assert result["latency"]["numpy_semantic"]["median_ms"] > 0
    assert result["latency"]["torchscript_whole"]["median_ms"] > 0
    assert result["compilation"]["native_cpp_ms"] > 0
    assert result["timing_order"] == "forward on even trials; reverse on odd trials"
    assert set(result["raw_samples_ms"]) == set(result["latency"])
    assert all(
        len(samples) == 2 and all(value > 0 for value in samples)
        for samples in result["raw_samples_ms"].values()
    )
    assert len(result["source_sha256"]) == 3


def test_native_cpu_benchmark_rejects_oversized_context(tmp_path):
    fixture(tmp_path, layers=1)
    weights = load_hf_llama_local(
        tmp_path, declared_revision="a" * 40
    )
    with pytest.raises(ValueError):
        benchmark(weights, prompt=32, chunk_sizes=(32, 1), trials=2)


def test_native_cpu_benchmark_compares_scalar_and_four_row_binaries(tmp_path):
    fixture(tmp_path, layers=1)
    weights = load_hf_llama_local(tmp_path, declared_revision="a" * 40)
    result = benchmark(
        weights, prompt=3, chunk_sizes=(1, 2), trials=2,
        compare_native_reference=True,
    )
    assert result["native_linear_variants"] == {
        "native_cpp": "four_row",
        "native_scalar_reference": "scalar_reference",
    }
    assert result["compilation"]["native_scalar_reference_ms"] > 0
    assert result["warm_native_speedup_vs_scalar"] > 0
    assert len(result["raw_samples_ms"]["native_scalar_reference"]) == 2
