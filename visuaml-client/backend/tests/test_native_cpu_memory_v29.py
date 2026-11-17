"""Linux physical RSS probe structural checks (no performance assertion)."""

from __future__ import annotations

import platform

import pytest
from examples.native_cpu_memory_v29 import linux_memory_kib


@pytest.mark.skipif(platform.system() != "Linux", reason="Linux /proc only")
def test_linux_memory_probe_reports_physical_resident_fields():
    sample = linux_memory_kib()
    assert set(sample) == {"VmRSS_kib", "VmHWM_kib"}
    assert sample["VmRSS_kib"] > 0
    assert sample["VmHWM_kib"] >= sample["VmRSS_kib"]
