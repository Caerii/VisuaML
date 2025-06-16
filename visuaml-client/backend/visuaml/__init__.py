"""VisuaML: PyTorch model visualization and graph export tools."""

__version__ = "0.1.0"

from .graph_export import export_model_graph_with_fallback, export_model_graph
from .openhypergraph_export import export_model_open_hypergraph
from .filters import FilterConfig
from .model_loader import ModelLoadError
from .visualization import NodeVisualProperties

# Make the fallback function the primary export
export_model = export_model_graph_with_fallback

__all__ = [
    "export_model",  # Primary export with fallback behavior
    "export_model_graph_with_fallback",  # Explicit fallback function
    "export_model_graph",  # Legacy function for backward compatibility
    "export_model_open_hypergraph",
    "FilterConfig",
    "ModelLoadError",
    "NodeVisualProperties",
] 