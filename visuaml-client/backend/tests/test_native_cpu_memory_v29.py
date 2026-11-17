"""Linux physical RSS probe structural checks (no performance assertion)."""

from __future__ import annotations

import platform

import pytest
from examples.native_cpu_memory_v29 import linux_memory_kib, worker
from test_hf_llama_checkpoint_v19 import fixture
from visuaml.compiler_lab.native_cpu_v28 import compiler_available


@pytest.mark.skipif(platform.system() != "Linux", reason="Linux /proc only")
def test_linux_memory_probe_reports_physical_resident_fields():
    sample = linux_memory_kib()
    assert set(sample) == {"VmRSS_kib", "VmHWM_kib"}
    assert sample["VmRSS_kib"] > 0
    assert sample["VmHWM_kib"] >= sample["VmRSS_kib"]


@pytest.mark.skipif(
    platform.system() != "Linux" or not compiler_available(),
    reason="Linux /proc and native C++ compiler required",
)
@pytest.mark.parametrize("variant", ("scalar_reference", "four_row", "avx2"))
def test_native_process_memory_worker_reports_real_variant(tmp_path, variant):
    fixture(tmp_path, layers=1)
    record = worker(tmp_path, "a" * 40, variant)
    assert record["requested_variant"] == variant
    assert len(record["checkpoint_sha256"]) == 64
    assert len(record["native_source_sha256"]) == 64
    assert record["after_execution"]["VmRSS_kib"] > 0
    assert (
        record["after_execution"]["VmHWM_kib"]
        >= record["after_execution"]["VmRSS_kib"]
    )
    if variant == "avx2":
        assert record["effective_variant"] in (
            "avx2_runtime_with_short_width_fallback", "four_row",
        )
