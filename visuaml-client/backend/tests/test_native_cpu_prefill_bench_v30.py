"""Bounded prefill/decode benchmark schema and independent-parity checks."""

from __future__ import annotations

import pytest
from examples.native_cpu_prefill_bench_v30 import benchmark
from test_hf_llama_checkpoint_v19 import fixture
from visuaml.compiler_lab.hf_llama_checkpoint_v19 import load_hf_llama_local
from visuaml.compiler_lab.native_cpu_v28 import compiler_available

pytestmark = pytest.mark.skipif(
    not compiler_available(), reason="Native C++17 compiler not installed"
)


@pytest.mark.parametrize("batch,prompt", ((1, 6), (2, 7)))
def test_native_prefill_benchmark_separates_prefill_from_decode(
    tmp_path, batch, prompt
):
    fixture(tmp_path, layers=1, hidden=48, heads=6, kv=3, middle=101, vocab=67)
    model = load_hf_llama_local(tmp_path, declared_revision="a" * 40)
    record = benchmark(
        model, prompt=prompt, decode=(1, 2), batch=batch, trials=2
    )
    assert record["schema"] == "visuaml.native-prefill-benchmark.v30"
    assert record["numerical_certificate"].startswith("all prefill")
    assert record["prompt_tokens"] == prompt
    assert record["decode_chunks"] == [1, 2]
    assert record["batch"] == batch
    assert record["prefill_ratio_v29_avx2_over_batched"] > 0
    assert record["decode_ratio_v29_avx2_over_batched"] > 0
    for name in ("native_scalar", "native_avx2", "native_batched"):
        for phase in ("prefill_ms", "decode_ms", "total_ms"):
            samples = record["timing"][name][phase]["raw_samples_ms"]
            assert len(samples) == 2
            assert all(sample > 0 for sample in samples)
    a = record["compiler_evidence"]["native_avx2"]
    b = record["compiler_evidence"]["native_batched"]
    assert a["batched_prefill_requested"] is False
    assert b["batched_prefill_requested"] is True
    assert a["executable_sha256"] != b["executable_sha256"]
    assert record["prefill_dispatch"]["native_batched"]["query_tokens"] == prompt
    assert record["prefill_dispatch"]["native_batched"]["native_batched_selected"] is False
    assert record["single_token_decode_dispatch"]["native_batched"]["execution"] == "v29_native"
    assert len(record["source_sha256"]) == 3


def test_native_prefill_benchmark_rejects_oversized_workload(tmp_path):
    fixture(tmp_path, layers=1)
    model = load_hf_llama_local(tmp_path, declared_revision="a" * 40)
    with pytest.raises(ValueError, match="Invalid bounded"):
        benchmark(model, prompt=32, decode=(33,), trials=2)
