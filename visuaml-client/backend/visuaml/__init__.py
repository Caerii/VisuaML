"""VisuaML: PyTorch model visualization and graph export tools."""
from importlib import import_module

__version__ = "0.1.0"

_EXPORTS = {
    "FilterConfig": ("filters", "FilterConfig"),
    "ModelLoadError": ("model_loader", "ModelLoadError"),
    "NodeVisualProperties": ("visualization", "NodeVisualProperties"),
    "export_model": ("graph_export", "export_model_graph_with_fallback"),
    "export_model_graph": ("graph_export", "export_model_graph"),
    "export_model_graph_with_fallback": ("graph_export", "export_model_graph_with_fallback"),
    "export_model_open_hypergraph": ("openhypergraph_export", "export_model_open_hypergraph"),
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    module, symbol = _EXPORTS[name]
    value = getattr(import_module(f".{module}", __name__), symbol)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | _EXPORTS.keys())
