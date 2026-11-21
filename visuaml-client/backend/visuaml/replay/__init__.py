"""Portable source-boundary replay for the supported VisuaML WASM profile."""
from .divergence import first_divergence
from .reports import REPORT_KIND, capture, check_report, compare, execute_report, make_bundle, open_bundle, seal_report
from .runtime import PAGE_BYTES, Replay
from .window import inspect_window

__all__ = [
    "PAGE_BYTES", "REPORT_KIND", "Replay", "capture", "check_report", "compare",
    "execute_report", "first_divergence", "inspect_window", "make_bundle", "open_bundle", "seal_report",
]
