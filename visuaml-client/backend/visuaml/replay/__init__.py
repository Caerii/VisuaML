"""Portable source-boundary replay for the supported VisuaML WASM profile."""
from importlib import import_module

_EXPORTS = {
    "PAGE_BYTES": "runtime", "Replay": "runtime", "first_divergence": "divergence",
    "inspect_window": "window", "REPORT_KIND": "reports", "capture": "reports",
    "check_report": "reports", "compare": "reports", "execute_report": "reports",
    "make_bundle": "reports", "open_bundle": "reports", "seal_report": "reports",
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    value = getattr(import_module(f".{_EXPORTS[name]}", __name__), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | _EXPORTS.keys())
