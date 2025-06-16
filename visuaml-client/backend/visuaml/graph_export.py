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
    graph_filter: GraphFilter, # Pass GraphFilter instance
    raw: bool = False
) -> Any:
    """Helper to format a single argument for JSON output, respecting filters."""
    if raw:
        return arg
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
        return [_format_arg(x, name_map, included_fx_nodes, graph_filter, raw) for x in arg]
    elif isinstance(arg, tuple):
        return tuple(_format_arg(x, name_map, included_fx_nodes, graph_filter, raw) for x in arg)
    elif isinstance(arg, dict):
        return {k: _format_arg(v, name_map, included_fx_nodes, graph_filter, raw) for k, v in arg.items()}
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
        resolved_args = [_format_arg(arg, name_map, included_fx_nodes, graph_filter, False) for arg in node.args]
        resolved_kwargs = {k: _format_arg(v, name_map, included_fx_nodes, graph_filter, False) for k, v in node.kwargs.items()}

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


def export_model_graph_with_fallback(
    model_path: str,
    filter_config: Optional[FilterConfig] = None,
    model_args: Optional[tuple] = None,
    model_kwargs: Optional[dict] = None,
    sample_input_args: Optional[tuple] = None,
    sample_input_kwargs: Optional[dict] = None,
    sample_input_dtypes: Optional[List[str]] = None,
    tracing_method: str = "auto",  # "auto", "fx", "hooks", "torchscript"
    export_format: str = "visuaml-json",
) -> Dict[str, Any]:
    """
    Export a PyTorch model to VisuaML graph format with multiple tracing methods.
    
    Args:
        model_path: Module path to the model (e.g., 'models.MyModel')
        filter_config: Optional filter configuration
        model_args: Optional arguments for model constructor
        model_kwargs: Optional keyword arguments for model constructor
        sample_input_args: Optional sample positional inputs for shape propagation
        sample_input_kwargs: Optional sample keyword inputs for shape propagation
        sample_input_dtypes: Optional list of sample input dtypes for shape propagation
        tracing_method: Tracing method to use ("auto", "fx", "hooks", "torchscript")
        export_format: Output format for the model graph
        
    Returns:
        Dictionary with 'nodes' and 'edges' lists
        
    Raises:
        ModelLoadError: If model cannot be loaded
        Exception: If all tracing methods fail
    """
    # Load and instantiate the model
    model_class, class_name = load_model_class(model_path)
    model = instantiate_model(model_class, class_name, model_args=model_args, model_kwargs=model_kwargs)
    model.eval()
    
    # Prepare sample inputs if provided
    actual_sample_inputs = None
    if sample_input_args is not None:
        actual_sample_inputs = []
        if not isinstance(sample_input_args, tuple):
            sample_input_args = (sample_input_args,)
        
        for i, arg_shape_or_tensor in enumerate(sample_input_args):
            dtype_str = sample_input_dtypes[i] if sample_input_dtypes and i < len(sample_input_dtypes) else 'float32'
            if dtype_str == 'long':
                current_dtype = torch.long
            elif dtype_str == 'float16':
                current_dtype = torch.float16
            else:
                current_dtype = torch.float32
            
            if isinstance(arg_shape_or_tensor, tuple) and all(isinstance(x, int) for x in arg_shape_or_tensor):
                actual_sample_inputs.append(torch.ones(arg_shape_or_tensor, dtype=current_dtype))
            else:
                actual_sample_inputs.append(arg_shape_or_tensor)
    
    # Try different tracing methods based on preference
    tracing_methods = []
    if tracing_method == "auto":
        tracing_methods = ["fx", "hooks"]  # Could add "torchscript", "onnx" later
    elif tracing_method in ["fx", "hooks", "torchscript"]:
        tracing_methods = [tracing_method]
    else:
        raise ValueError(f"Unknown tracing method: {tracing_method}")
    
    last_error = None
    
    # Early-out for open-hypergraph path
    if export_format.startswith("openhg"):
        from .openhypergraph_export import export_model_open_hypergraph
        return export_model_open_hypergraph(
            model_path,
            filter_config,
            model_args,
            model_kwargs,
            sample_input_args,
            sample_input_dtypes=sample_input_dtypes,
            sample_dtypes=sample_input_dtypes,
            out_format=export_format.split("-", 1)[-1] if "-" in export_format else "json",
        )
    
    for method in tracing_methods:
        try:
            if method == "fx":
                # Use existing FX symbolic tracing
                return export_model_graph(
                    model_path, filter_config, model_args, model_kwargs,
                    sample_input_args, sample_input_kwargs, sample_input_dtypes
                )
            
            elif method == "hooks":
                return trace_with_hooks(model, actual_sample_inputs, filter_config)
            
            elif method == "torchscript":
                return trace_with_torchscript(model, actual_sample_inputs, filter_config)
                
        except Exception as e:
            last_error = e
            # Don't print error messages for individual methods - only if all fail
            continue
    
    # If all methods failed, raise the last error with details about what was tried
    method_names = ", ".join(tracing_methods)
    raise Exception(f"All tracing methods ({method_names}) failed. Last error from '{tracing_methods[-1]}': {last_error}")


def trace_with_hooks(
    model: nn.Module, 
    sample_inputs: Optional[List[torch.Tensor]], 
    filter_config: Optional[FilterConfig] = None
) -> Dict[str, Any]:
    """
    Trace model execution using forward hooks.
    This works with any PyTorch model, including those with dynamic control flow.
    """
    from .visualization import NodeVisualizer
    
    nodes_json = []
    edges_json = []
    execution_order = []
    module_outputs = {}
    
    node_visualizer = NodeVisualizer()
    
    def forward_hook(module, input, output):
        module_id = id(module)
        module_name = None
        
        # Find the module name
        for name, mod in model.named_modules():
            if mod is module:
                module_name = name or f"module_{module_id}"
                break
        
        if module_name is None:
            module_name = f"module_{module_id}"
        
        # Record execution
        execution_order.append((module_name, module))
        module_outputs[module_name] = output
        
        # Get input shapes
        input_shapes = []
        if isinstance(input, (list, tuple)):
            for inp in input:
                if hasattr(inp, 'shape'):
                    input_shapes.append(list(inp.shape))
        elif hasattr(input, 'shape'):
            input_shapes = [list(input.shape)]
        
        # Get output shape
        output_shape = None
        if hasattr(output, 'shape'):
            output_shape = list(output.shape)
        elif isinstance(output, (list, tuple)) and len(output) > 0 and hasattr(output[0], 'shape'):
            output_shape = list(output[0].shape)
        
        # Create node
        visual_props = node_visualizer.get_node_visual_properties_from_module(module)
        
        nodes_json.append({
            "id": module_name,
            "type": "transformer",
            "data": {
                "label": module_name,
                "target": type(module).__name__,
                "name": module_name,
                "op": "call_module",
                "args": [],  # Hook-based tracing doesn't capture detailed args
                "kwargs": {},
                "layerType": type(module).__name__,
                "color": visual_props.color if hasattr(visual_props, 'color') else "#add8e6",
                "outputShape": str(output_shape) if output_shape else None,
                "inputShapes": input_shapes,
            },
            "position": {"x": 0, "y": 0}
        })
    
    # Register hooks on all leaf modules (modules with no children)
    hooks = []
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf module
            hook = module.register_forward_hook(forward_hook)
            hooks.append(hook)
    
    try:
        # Generate default sample inputs if none provided
        if not sample_inputs:
            sample_inputs = _generate_default_sample_inputs(model)
            if sample_inputs:
                import sys
                print(f"Generated default sample inputs with shapes: {[list(inp.shape) for inp in sample_inputs]}", file=sys.stderr)
            else:
                import sys
                print("Warning: Could not generate default sample inputs for hook-based tracing", file=sys.stderr)
                return _static_module_analysis(model, filter_config)
        
        # Run forward pass
        with torch.no_grad():
            model(*sample_inputs)
        
        # Create edges based on execution order
        for i in range(1, len(execution_order)):
            prev_name, _ = execution_order[i-1]
            curr_name, _ = execution_order[i]
            
            edges_json.append({
                "id": f"{prev_name}->{curr_name}",
                "source": prev_name,
                "target": curr_name,
                "data": {}
            })
    
    finally:
        # Clean up hooks
        for hook in hooks:
            hook.remove()
    
    return {"nodes": nodes_json, "edges": edges_json}


def _generate_default_sample_inputs(model: nn.Module) -> Optional[List[torch.Tensor]]:
    """
    Generate reasonable default sample inputs for common model types.
    """
    try:
        # Try to infer input requirements from the first layer
        first_module = None
        for module in model.modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.RNN, nn.LSTM, nn.GRU)):
                first_module = module
                break
        
        if first_module is None:
            # If no recognizable layer found, try to infer from model name or structure
            model_name = type(model).__name__.lower()
            if 'transformer' in model_name:
                # For transformer models, assume [batch_size, seq_len, d_model]
                # Common transformer dimensions
                return [torch.randn(1, 10, 512)]
            elif 'rnn' in model_name or 'lstm' in model_name or 'gru' in model_name:
                # For RNN-like models, assume [batch_size, seq_len, input_size]
                return [torch.randn(1, 10, 10)]
            else:
                # Generic fallback
                return [torch.randn(1, 10)]
        
        # Generate inputs based on module type
        if isinstance(first_module, nn.Linear):
            input_features = first_module.in_features
            
            # Check if this might be a transformer model by looking at the model structure
            model_name = type(model).__name__.lower()
            if 'transformer' in model_name or input_features >= 256:
                # Likely a transformer - add sequence dimension
                return [torch.randn(1, 10, input_features)]
            else:
                # Regular linear layer
                return [torch.randn(1, input_features)]
        
        elif isinstance(first_module, nn.Conv2d):
            # Conv2d: (batch_size, channels, height, width)
            return [torch.randn(1, first_module.in_channels, 32, 32)]
        
        elif isinstance(first_module, nn.Conv1d):
            # Conv1d: (batch_size, channels, length)
            return [torch.randn(1, first_module.in_channels, 100)]
        
        elif isinstance(first_module, (nn.RNN, nn.LSTM, nn.GRU)):
            # RNN-like: (batch_size, seq_len, input_size)
            return [torch.randn(1, 10, first_module.input_size)]
        
        # Default fallback
        return [torch.randn(1, 10)]
        
    except Exception:
        return None


def _static_module_analysis(model: nn.Module, filter_config: Optional[FilterConfig] = None) -> Dict[str, Any]:
    """
    Fallback static analysis when forward pass is not possible.
    Creates nodes based on module structure without execution.
    """
    from .visualization import NodeVisualizer
    
    nodes_json = []
    edges_json = []
    node_visualizer = NodeVisualizer()
    
    # Get all leaf modules
    leaf_modules = []
    for name, module in model.named_modules():
        if len(list(module.children())) == 0 and name:  # Leaf module with a name
            leaf_modules.append((name, module))
    
    # Create nodes for each leaf module
    for name, module in leaf_modules:
        visual_props = node_visualizer.get_node_visual_properties_from_module(module)
        
        nodes_json.append({
            "id": name,
            "type": "transformer",
            "data": {
                "label": name,
                "target": type(module).__name__,
                "name": name,
                "op": "call_module",
                "args": [],
                "kwargs": {},
                "layerType": type(module).__name__,
                "color": visual_props.color if hasattr(visual_props, 'color') else "#add8e6",
                "outputShape": None,  # Can't determine without forward pass
                "inputShapes": [],
            },
            "position": {"x": 0, "y": 0}
        })
    
    # Create simple sequential edges (best guess without execution)
    for i in range(1, len(leaf_modules)):
        prev_name, _ = leaf_modules[i-1]
        curr_name, _ = leaf_modules[i]
        
        edges_json.append({
            "id": f"{prev_name}->{curr_name}",
            "source": prev_name,
            "target": curr_name,
            "data": {}
        })
    
    return {"nodes": nodes_json, "edges": edges_json}


def trace_with_torchscript(
    model: nn.Module, 
    sample_inputs: Optional[List[torch.Tensor]], 
    filter_config: Optional[FilterConfig] = None
) -> Dict[str, Any]:
    """
    Trace model using TorchScript tracing.
    This can handle some dynamic behavior but may fail on complex control flow.
    """
    if not sample_inputs:
        raise ValueError("TorchScript tracing requires sample inputs")
    
    try:
        _traced_model = torch.jit.trace(model, sample_inputs)
        # TODO: Parse TorchScript graph and convert to VisuaML format
        # For now, fall back to hook-based tracing
        return trace_with_hooks(model, sample_inputs, filter_config)
    except Exception as e:
        raise Exception(f"TorchScript tracing failed: {e}") 