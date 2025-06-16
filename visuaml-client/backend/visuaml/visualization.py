"""Module for determining visual properties of graph nodes."""

from dataclasses import dataclass
from typing import Dict, Optional
from torch.fx import Node, GraphModule


@dataclass
class NodeVisualProperties:
    """Visual properties for a graph node."""
    layer_type: str
    color: str
    
    
class NodeColorScheme:
    """Manages color assignments for different node types."""
    
    # Default colors for different layer types
    DEFAULT_COLORS: Dict[str, str] = {
        # Input/Output
        'Input': '#adebad',  # Light green
        'Output': '#ffb3b3',  # Light red
        
        # Linear layers
        'Linear': '#add8e6',  # Light blue
        'Dense': '#add8e6',   # Light blue (alias)
        
        # Convolutional layers
        'Conv1d': '#90ee90',  # Light green
        'Conv2d': '#90ee90',  # Light green
        'Conv3d': '#90ee90',  # Light green
        'ConvTranspose1d': '#98fb98',  # Pale green
        'ConvTranspose2d': '#98fb98',  # Pale green
        'ConvTranspose3d': '#98fb98',  # Pale green
        
        # Normalization layers
        'BatchNorm1d': '#ffd700',  # Gold
        'BatchNorm2d': '#ffd700',  # Gold
        'BatchNorm3d': '#ffd700',  # Gold
        'LayerNorm': '#ffda44',    # Lighter gold
        'GroupNorm': '#ffda44',    # Lighter gold
        
        # Activation functions
        'ReLU': '#ffcb6b',     # Orange
        'LeakyReLU': '#ffcb6b', # Orange
        'PReLU': '#ffcb6b',    # Orange
        'GELU': '#ffcb6b',     # Orange
        'Sigmoid': '#ffa500',   # Darker orange
        'Tanh': '#ffa500',      # Darker orange
        'Softmax': '#e6e6fa',   # Lavender
        
        # Dropout layers
        'Dropout': '#d3d3d3',   # Light grey
        'Dropout2d': '#d3d3d3', # Light grey
        'Dropout3d': '#d3d3d3', # Light grey
        
        # Recurrent layers
        'LSTM': '#b19cd9',      # Light purple
        'GRU': '#b19cd9',       # Light purple
        'RNN': '#b19cd9',       # Light purple
        
        # Attention layers
        'MultiheadAttention': '#ffb6c1',  # Light pink
        'Attention': '#ffb6c1',           # Light pink
        
        # Embedding layers
        'Embedding': '#dda0dd',  # Plum
        
        # Pooling layers
        'MaxPool1d': '#87ceeb',  # Sky blue
        'MaxPool2d': '#87ceeb',  # Sky blue
        'MaxPool3d': '#87ceeb',  # Sky blue
        'AvgPool1d': '#87ceeb',  # Sky blue
        'AvgPool2d': '#87ceeb',  # Sky blue
        'AvgPool3d': '#87ceeb',  # Sky blue
        'AdaptiveMaxPool1d': '#87ceeb',  # Sky blue
        'AdaptiveMaxPool2d': '#87ceeb',  # Sky blue
        'AdaptiveMaxPool3d': '#87ceeb',  # Sky blue
        'AdaptiveAvgPool1d': '#87ceeb',  # Sky blue
        'AdaptiveAvgPool2d': '#87ceeb',  # Sky blue
        'AdaptiveAvgPool3d': '#87ceeb',  # Sky blue
        
        # Default colors for operation types
        'call_function': '#b0e0e6',  # Powder blue
        'call_method': '#b0e0e6',    # Powder blue
        'get_attr': '#fffacd',       # Lemon chiffon
        'unknown': '#f0e68c',        # Khaki
    }
    
    def get_color_for_layer(self, layer_type: str) -> str:
        """Get color for a specific layer type."""
        # Check exact match first
        if layer_type in self.DEFAULT_COLORS:
            return self.DEFAULT_COLORS[layer_type]
        
        # Check partial matches (e.g., CustomLinear matches Linear)
        for key, color in self.DEFAULT_COLORS.items():
            if key in layer_type:
                return color
                
        # Return default color
        return self.DEFAULT_COLORS.get('unknown', '#cccccc')


class NodeVisualizer:
    """Determines visual properties for graph nodes."""
    
    def __init__(self, color_scheme: Optional[NodeColorScheme] = None):
        self.color_scheme = color_scheme or NodeColorScheme()
    
    def get_node_visual_properties(
        self, 
        node: Node, 
        graph_module: GraphModule
    ) -> NodeVisualProperties:
        """
        Determine visual properties (layer type, color) for a graph node.
        
        Args:
            node: The FX node
            graph_module: The graph module containing the node
            
        Returns:
            NodeVisualProperties with layer type and color
        """
        op_type = node.op
        layer_type_str = op_type  # Default layer type to op code
        color = "#cccccc"  # Default grey
        
        if op_type == 'placeholder':
            layer_type_str = 'Input'
            color = self.color_scheme.get_color_for_layer('Input')
            
        elif op_type == 'output':
            layer_type_str = 'Output'
            color = self.color_scheme.get_color_for_layer('Output')
            
        elif op_type == 'call_module':
            try:
                # Get the class name of the module being called
                module_instance = graph_module.get_submodule(node.target)
                layer_type_str = type(module_instance).__name__
                color = self.color_scheme.get_color_for_layer(layer_type_str)
            except AttributeError:
                # Target might not be a submodule or not exist
                layer_type_str = str(node.target)
                color = self.color_scheme.get_color_for_layer('unknown')
                
        elif op_type in ('call_function', 'call_method'):
            if hasattr(node.target, '__name__'):
                layer_type_str = str(node.target.__name__)
            else:
                layer_type_str = str(node.target)
            color = self.color_scheme.get_color_for_layer(op_type)
            
        elif op_type == 'get_attr':
            layer_type_str = f"GetAttr: {node.target}"
            color = self.color_scheme.get_color_for_layer('get_attr')
            
        else:
            # Unknown operation type
            color = self.color_scheme.get_color_for_layer('unknown')
            
        return NodeVisualProperties(layer_type=layer_type_str, color=color)
    
    def get_node_visual_properties_from_module(
        self, 
        module
    ) -> NodeVisualProperties:
        """
        Determine visual properties for a PyTorch module (used in hook-based tracing).
        
        Args:
            module: The PyTorch module
            
        Returns:
            NodeVisualProperties with layer type and color
        """
        layer_type_str = type(module).__name__
        color = self.color_scheme.get_color_for_layer(layer_type_str)
        
        return NodeVisualProperties(layer_type=layer_type_str, color=color) 