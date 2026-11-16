"""VisuaML: PyTorch model visualization and graph export tools."""

__version__ = "0.1.0"

from .filters import FilterConfig
from .graph_export import export_model_graph, export_model_graph_with_fallback
from .model_loader import ModelLoadError
from .openhypergraph_export import export_model_open_hypergraph
from .visualization import NodeVisualProperties

# Make the fallback function the primary export
export_model = export_model_graph_with_fallback

__all__ = [
    "FilterConfig",
    "ModelLoadError",
    "NodeVisualProperties",
    "export_model",  # Primary export with fallback behavior
    "export_model_graph",  # Legacy function for backward compatibility
    "export_model_graph_with_fallback",  # Explicit fallback function
    "export_model_open_hypergraph",
]
