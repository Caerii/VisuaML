"""Main module for exporting PyTorch models to VisuaML graph format."""

from typing import Dict, List, Any, Optional, Set
from torch.fx import symbolic_trace, GraphModule, Node
from torch.fx.passes.shape_prop import ShapeProp
import torch
import torch.nn as nn

from .model_loader import load_model_class, instantiate_model
from .filters import GraphFilter, FilterConfig
from .visualization import NodeVisualizer


def _format_arg(
    arg: Any, 
    name_map: Dict[Node, str], 
    included_fx_nodes: Set[Node], 
    graph_filter: GraphFilter # Pass GraphFilter instance
) -> Any:
    """Helper to format a single argument for JSON output, respecting filters."""
    if isinstance(arg, Node):
        if arg in included_fx_nodes:
            return {"source_node": name_map.get(arg, str(arg))}
        else:
            # Node is filtered, find its transitive (visible) inputs
            transitive_sources = graph_filter.get_transitive_inputs(arg, included_fx_nodes)
            if transitive_sources:
                # If multiple transitive sources map to this single arg slot, list them all.
                # This could happen if a filtered node took multiple inputs that are now bypassed.
                # The frontend will need to handle an array here if that's the case.
                # For simplicity now, let's assume most common case is one or a primary one.
                # A more robust solution might always return a list for transitive sources.
                if len(transitive_sources) == 1:
                    return {"source_node": name_map.get(transitive_sources[0], str(transitive_sources[0])),
                            "transitive": True}
                else:
                    return {
                        "source_nodes": [name_map.get(ts, str(ts)) for ts in transitive_sources],
                        "transitive": True
                    }
            else:
                # Should not happen if graph is connected and inputs are processed, but as a fallback:
                return {"source_node": f"{name_map.get(arg, str(arg))} (filtered)", "transitive": False}
    elif isinstance(arg, (str, int, float, bool, type(None))):
        return arg
    elif isinstance(arg, torch.Tensor):
        return f"Tensor(shape={list(arg.shape)}, dtype={arg.dtype})"
    elif isinstance(arg, list):
        return [_format_arg(x, name_map, included_fx_nodes, graph_filter) for x in arg]
    elif isinstance(arg, tuple):
        return tuple(_format_arg(x, name_map, included_fx_nodes, graph_filter) for x in arg)
    elif isinstance(arg, dict):
        return {k: _format_arg(v, name_map, included_fx_nodes, graph_filter) for k, v in arg.items()}
    return str(arg)


def create_graph_json(
    graph_module: GraphModule,
    filter_config: Optional[FilterConfig] = None,
) -> Dict[str, Any]:
    """
    Convert a PyTorch FX GraphModule to VisuaML JSON format.
    Includes shape information and humanized args/kwargs.
    
    Args:
        graph_module: The FX GraphModule to convert
        filter_config: Optional filter configuration
        
    Returns:
        Dictionary with 'nodes' and 'edges' lists
    """
    graph_filter = GraphFilter(filter_config)
    node_visualizer = NodeVisualizer()
    
    nodes_json = []
    edges_json = []
    
    # Build name_map for all original nodes (before filtering)
    # This ensures that even if an arg node is filtered out, we can still refer to its original name.
    name_map: Dict[Node, str] = {node: node.name for node in graph_module.graph.nodes}

    all_fx_nodes = list(graph_module.graph.nodes)
    included_fx_nodes = graph_filter.filter_graph(all_fx_nodes, graph_module)
    
    for node in all_fx_nodes:
        if node not in included_fx_nodes:
            continue
            
        visual_props = node_visualizer.get_node_visual_properties(node, graph_module)
        
        # Extract shape information if available
        output_shape_str = None
        if 'tensor_meta' in node.meta:
            tensor_meta = node.meta['tensor_meta']
            # tensor_meta can be a list of TensorsMetadata or a single TensorsMetadata
            if isinstance(tensor_meta, list):
                output_shape_str = ", ".join([str(list(tm.shape)) for tm in tensor_meta])
            elif hasattr(tensor_meta, 'shape'): # Check if it has shape attribute
                output_shape_str = str(list(tensor_meta.shape))
            # Consider other structures if fx changes or for different node types

        # Humanize args and kwargs, passing necessary context to _format_arg
        resolved_args = [_format_arg(arg, name_map, included_fx_nodes, graph_filter) for arg in node.args]
        resolved_kwargs = {k: _format_arg(v, name_map, included_fx_nodes, graph_filter) for k, v in node.kwargs.items()}

        nodes_json.append({
            "id": node.name,
            "type": "transformer",
            "data": {
                "label": node.name, # label will be refined by frontend
                "target": str(node.target),
                "name": node.name,
                "op": node.op,
                "args": resolved_args, # Use resolved args
                "kwargs": resolved_kwargs, # Use resolved kwargs
                "layerType": visual_props.layer_type,
                "color": visual_props.color,
                "outputShape": output_shape_str, # New field
            },
            "position": {"x": 0, "y": 0}
        })
        
        processed_edges = set()
        
        for input_node_obj in node.all_input_nodes: # Ensure this is List[Node]
            # Get output shape of the source node for edge data
            source_output_shape_str = None
            if 'tensor_meta' in input_node_obj.meta:
                source_tensor_meta = input_node_obj.meta['tensor_meta']
                if isinstance(source_tensor_meta, list):
                    source_output_shape_str = ", ".join([str(list(tm.shape)) for tm in source_tensor_meta])
                elif hasattr(source_tensor_meta, 'shape'):
                    source_output_shape_str = str(list(source_tensor_meta.shape))

            if input_node_obj in included_fx_nodes:
                edge_id = f"{input_node_obj.name}->{node.name}"
                if edge_id not in processed_edges:
                    edges_json.append({
                        "id": edge_id,
                        "source": input_node_obj.name,
                        "target": node.name,
                        "data": {"sourceOutputShape": source_output_shape_str} # Add shape to edge
                    })
                    processed_edges.add(edge_id)
            else:
                transitive_inputs = graph_filter.get_transitive_inputs(
                    input_node_obj, included_fx_nodes
                )
                for trans_input in transitive_inputs:
                    # Get output shape of the true transitive source node
                    trans_source_output_shape_str = None
                    if 'tensor_meta' in trans_input.meta:
                        ts_tensor_meta = trans_input.meta['tensor_meta']
                        if isinstance(ts_tensor_meta, list):
                             trans_source_output_shape_str = ", ".join([str(list(tm.shape)) for tm in ts_tensor_meta])
                        elif hasattr(ts_tensor_meta, 'shape'):
                            trans_source_output_shape_str = str(list(ts_tensor_meta.shape))
                            
                    edge_id = f"{trans_input.name}->{node.name}"
                    if edge_id not in processed_edges:
                        edges_json.append({
                            "id": edge_id,
                            "source": trans_input.name,
                            "target": node.name,
                            "data": {
                                "transitive": True,
                                "sourceOutputShape": trans_source_output_shape_str # Add shape to edge
                            }
                        })
                        processed_edges.add(edge_id)
    
    return {"nodes": nodes_json, "edges": edges_json}


def export_model_graph(
    model_path: str,
    filter_config: Optional[FilterConfig] = None,
    model_args: Optional[tuple] = None,
    model_kwargs: Optional[dict] = None,
    sample_input_args: Optional[tuple] = None,
    sample_input_kwargs: Optional[dict] = None,
    sample_input_dtypes: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Export a PyTorch model to VisuaML graph format.
    
    Args:
        model_path: Module path to the model (e.g., 'models.MyModel')
        filter_config: Optional filter configuration
        model_args: Optional arguments for model constructor
        model_kwargs: Optional keyword arguments for model constructor
        sample_input_args: Optional sample positional inputs for shape propagation
        sample_input_kwargs: Optional sample keyword inputs for shape propagation
        sample_input_dtypes: Optional list of sample input dtypes for shape propagation
        
    Returns:
        Dictionary with 'nodes' and 'edges' lists
        
    Raises:
        ModelLoadError: If model cannot be loaded
        Exception: If tracing or shape propagation fails
    """
    # Load and instantiate the model
    model_class, class_name = load_model_class(model_path)
    model = instantiate_model(model_class, class_name, model_args=model_args, model_kwargs=model_kwargs)
    model.eval()
    
    # Perform symbolic tracing
    try:
        graph_module = symbolic_trace(model)
    except Exception as e:
        raise Exception(
            f"Error during symbolic tracing of '{class_name}': {e}"
        )
    
    # Perform shape propagation if sample inputs are provided
    if sample_input_args is not None:
        try:
            if not isinstance(sample_input_args, tuple):
                sample_input_args = (sample_input_args,)
            
            # Ensure dtypes list matches args length if provided
            if sample_input_dtypes and len(sample_input_dtypes) != len(sample_input_args):
                raise ValueError(
                    "Length of sample_input_dtypes must match length of sample_input_args."
                )

            actual_sample_inputs = []
            for i, arg_shape_or_tensor in enumerate(sample_input_args):
                dtype_str = sample_input_dtypes[i] if sample_input_dtypes else 'float32'
                if dtype_str == 'long':
                    current_dtype = torch.long
                elif dtype_str == 'float16':
                    current_dtype = torch.float16
                # Add more dtypes as needed
                else:
                    current_dtype = torch.float32 # Default
                
                if isinstance(arg_shape_or_tensor, tuple) and all(isinstance(x, int) for x in arg_shape_or_tensor):
                    actual_sample_inputs.append(torch.ones(arg_shape_or_tensor, dtype=current_dtype))
                else:
                    # If it's not a shape tuple, assume it's a pre-constructed tensor or other input type
                    # This path might need more robust handling if users can provide actual tensors directly.
                    actual_sample_inputs.append(arg_shape_or_tensor) 
            
            sp = ShapeProp(graph_module)
            sp.propagate(*actual_sample_inputs, **(sample_input_kwargs or {}))
            # Shape info is now in node.meta['tensor_meta'] for each node
        except Exception as e:
            error_message = f"Error during shape propagation for '{class_name}': {e}. "
            error_message += f"Sample args: {sample_input_args}, dtypes: {sample_input_dtypes}, kwargs: {sample_input_kwargs}."
            raise Exception(error_message)
            
    return create_graph_json(graph_module, filter_config) 