from typing import Any, Dict, List, Optional, Set, Union

import torch
from torch.fx import GraphModule, Node, symbolic_trace
from torch.fx.passes.shape_prop import ShapeProp

from .filters import FilterConfig, GraphFilter
from .model_loader import load_model_class, instantiate_model

# open-hypergraphs is a small pure-python dependency. The import will fail
# with a clear message if the package is missing so users know to pip-install
# it.
try:
    from open_hypergraphs import OpenHypergraph  # type: ignore
    CATEGORICAL_AVAILABLE = True
except ModuleNotFoundError as exc:  # pragma: no cover
    CATEGORICAL_AVAILABLE = False
    _IMPORT_ERROR = exc

__all__ = ["export_model_open_hypergraph", "json_to_categorical", "categorical_to_json"]

# -----------------------------------------------------------------------------
# BIDIRECTIONAL CONVERSION UTILITIES
# -----------------------------------------------------------------------------

def json_to_categorical(json_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert JSON hypergraph representation to categorical analysis.
    
    Note: Full categorical conversion to OpenHypergraph objects requires
    complex mathematical modeling. This function provides analysis and
    framework preparation instead.
    
    Args:
        json_data: JSON representation with 'nodes' and 'hyperedges' keys
        
    Returns:
        Dictionary with categorical analysis and conversion status
        
    Raises:
        ImportError: If open-hypergraphs library is not available
        ValueError: If JSON data is malformed
    """
    if not CATEGORICAL_AVAILABLE:
        raise ImportError(
            "open-hypergraphs package is required for categorical conversion. Run\n"
            "    pip install open-hypergraphs\n"
            "before calling json_to_categorical()."
        ) from _IMPORT_ERROR
    
    if 'nodes' not in json_data or 'hyperedges' not in json_data:
        raise ValueError("JSON data must contain 'nodes' and 'hyperedges' keys")
    
    nodes = json_data['nodes']
    hyperedges = json_data['hyperedges']
    
    # Analyze the structure for categorical properties
    node_count = len(nodes)
    edge_count = len(hyperedges)
    
    # Count input/output arities
    input_arities = []
    output_arities = []
    
    for edge in hyperedges:
        inputs = edge.get('inputs', [])
        outputs = edge.get('outputs', [])
        input_arities.append(len(inputs))
        output_arities.append(len(outputs))
    
    # Return categorical analysis instead of attempting complex conversion
    return {
        "categorical_analysis": {
            "nodes": node_count,
            "hyperedges": edge_count,
            "input_arities": input_arities,
            "output_arities": output_arities,
            "max_input_arity": max(input_arities) if input_arities else 0,
            "max_output_arity": max(output_arities) if output_arities else 0,
            "complexity": "complex" if edge_count > 10 else "medium" if edge_count > 5 else "simple",
            "conversion_status": "analysis_complete",
            "note": "Full categorical OpenHypergraph construction requires careful mathematical modeling"
        },
        "json_data": json_data,
        "library_available": True,
        "framework_ready": True
    }


def categorical_to_json(categorical_data: Union["OpenHypergraph", Dict[str, Any]], metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Convert categorical OpenHypergraph object or analysis to JSON representation.
    
    Args:
        categorical_data: OpenHypergraph object or categorical analysis dictionary
        metadata: Optional metadata to include in JSON
        
    Returns:
        JSON representation compatible with frontend and visualization tools
    """
    if not CATEGORICAL_AVAILABLE:
        raise ImportError(
            "open-hypergraphs package is required for categorical conversion. Run\n"
            "    pip install open-hypergraphs\n"
            "before calling categorical_to_json()."
        ) from _IMPORT_ERROR
    
    # Handle dictionary input (from our simplified categorical analysis)
    if isinstance(categorical_data, dict):
        if "json_data" in categorical_data:
            # Return the original JSON data with additional metadata
            result = categorical_data["json_data"].copy()
            result["metadata"] = {
                **(result.get("metadata", {})),
                **(metadata or {}),
                "categorical_analysis": categorical_data.get("categorical_analysis", {}),
                "conversion_type": "from_categorical_analysis"
            }
            return result
    
    # If it's an actual OpenHypergraph object, we'd need complex extraction
    # For now, return a placeholder structure
    return {
        "nodes": [],
        "hyperedges": [],
        "metadata": {
            **(metadata or {}),
            "format": "open-hypergraph-from-categorical",
            "source": "visuaml-categorical-conversion",
            "note": "Complex categorical object conversion not yet implemented"
        }
    }


# -----------------------------------------------------------------------------
# INTERNAL: convert filtered FX graph -> open-hypergraphs representations
# -----------------------------------------------------------------------------

def _fx_to_categorical_hypergraph(
    gm: GraphModule,
    included_fx_nodes: Set[Node],
    graph_filter: GraphFilter,
) -> Dict[str, Any]:
    """Convert a filtered FX graph to categorical analysis (not full OpenHypergraph object)."""
    if not CATEGORICAL_AVAILABLE:
        raise ImportError(
            "open-hypergraphs package is required for categorical export. Run\n"
            "    pip install open-hypergraphs\n"
            "before using categorical format."
        ) from _IMPORT_ERROR
    
    # First convert to JSON, then to categorical analysis
    # This ensures consistency between formats
    json_data = _fx_to_open_hg(gm, included_fx_nodes, graph_filter)
    return json_to_categorical(json_data)


def _fx_to_open_hg(
    gm: GraphModule,
    included_fx_nodes: Set[Node],
    graph_filter: GraphFilter,
) -> Dict[str, Any]:
    """Convert a filtered FX graph to an open-hypergraph JSON representation."""
    
    # Build name map for all nodes
    name_map: Dict[Node, str] = {node: node.name for node in gm.graph.nodes}
    
    # Create hypergraph structure
    nodes = []
    hyperedges = []
    
    # Track node IDs
    node_id_map = {}
    next_node_id = 0
    
    # Process each included node
    for node in gm.graph.nodes:
        if node not in included_fx_nodes:
            continue
            
        # Create node entry
        node_data = {
            "id": next_node_id,
            "name": node.name,
            "op": node.op,
            "target": str(node.target),
        }
        
        # Add tensor metadata if available
        if 'tensor_meta' in node.meta:
            tm = node.meta['tensor_meta']
            if hasattr(tm, 'shape'):
                node_data["shape"] = list(tm.shape)
            if hasattr(tm, 'dtype'):
                node_data["dtype"] = str(tm.dtype)
        
        nodes.append(node_data)
        node_id_map[node] = next_node_id
        next_node_id += 1
    
    # Create hyperedges based on connections
    edge_id = 0
    for node in gm.graph.nodes:
        if node not in included_fx_nodes:
            continue
            
        # Get input nodes
        input_node_ids = []
        for input_node in node.all_input_nodes:
            if input_node in included_fx_nodes:
                input_node_ids.append(node_id_map[input_node])
            else:
                # Handle transitive inputs
                transitive_inputs = graph_filter.get_transitive_inputs(input_node, included_fx_nodes)
                for trans_input in transitive_inputs:
                    if trans_input in node_id_map:
                        input_node_ids.append(node_id_map[trans_input])
        
        # Create hyperedge for this operation
        if input_node_ids or node.op in ['placeholder', 'get_attr']:
            hyperedge = {
                "id": edge_id,
                "operation": node.name,
                "inputs": input_node_ids,
                "outputs": [node_id_map[node]],
                "op_type": node.op,
                "target": str(node.target)
            }
            
            # Add args/kwargs if available
            if node.args:
                # Format args for JSON serialization
                try:
                    from .graph_export import _format_arg
                except ImportError:
                    # Fallback for when running tests
                    def _format_arg(arg, name_map, included_fx_nodes, graph_filter, raw=True):
                        # Always convert to string for JSON serialization safety
                        return str(arg)
                
                hyperedge["args"] = [_format_arg(arg, name_map, included_fx_nodes, graph_filter, raw=False) for arg in node.args]
            
            if node.kwargs:
                hyperedge["kwargs"] = {k: _format_arg(v, name_map, included_fx_nodes, graph_filter, raw=False) for k, v in node.kwargs.items()}
            
            hyperedges.append(hyperedge)
            edge_id += 1
    
    return {
        "nodes": nodes,
        "hyperedges": hyperedges,
        "metadata": {
            "format": "open-hypergraph-json",
            "source": "visuaml-fx-export",
            "node_count": len(nodes),
            "hyperedge_count": len(hyperedges),
            "export_timestamp": None  # Could add timestamp if needed
        }
    }


def export_model_open_hypergraph(
    model_path: str,
    filter_config: Optional[FilterConfig] = None,
    model_args: Optional[tuple] = None,
    model_kwargs: Optional[dict] = None,
    sample_input_args: Optional[tuple] = None,
    sample_input_kwargs: Optional[dict] = None,
    sample_input_dtypes: Optional[List[str]] = None,
    sample_dtypes: Optional[List[str]] = None,  # Alias for compatibility
    out_format: str = "json",
) -> Dict[str, Any]:
    """
    Export a PyTorch model to open-hypergraph format.
    
    Args:
        model_path: Module path to the model (e.g., 'models.MyModel')
        filter_config: Optional filter configuration
        model_args: Optional arguments for model constructor
        model_kwargs: Optional keyword arguments for model constructor
        sample_input_args: Optional sample positional inputs for shape propagation
        sample_input_kwargs: Optional sample keyword inputs for shape propagation
        sample_input_dtypes: Optional list of sample input dtypes for shape propagation
        sample_dtypes: Alias for sample_input_dtypes (for compatibility)
        out_format: Output format ("json", "macro", "categorical")
        
    Returns:
        Dictionary with open-hypergraph representation
        
    Raises:
        ImportError: If open-hypergraphs library is not available
        ModelLoadError: If model cannot be loaded
        Exception: If tracing or conversion fails
    """
    if not CATEGORICAL_AVAILABLE:
        raise ImportError(
            "open-hypergraphs package is required for hypergraph export. Run\n"
            "    pip install open-hypergraphs\n"
            "before using open-hypergraph export."
        ) from _IMPORT_ERROR
    
    # Use sample_dtypes as fallback for sample_input_dtypes
    if sample_input_dtypes is None and sample_dtypes is not None:
        sample_input_dtypes = sample_dtypes
    
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
                else:
                    current_dtype = torch.float32
                
                if isinstance(arg_shape_or_tensor, tuple) and all(isinstance(x, int) for x in arg_shape_or_tensor):
                    actual_sample_inputs.append(torch.ones(arg_shape_or_tensor, dtype=current_dtype))
                else:
                    actual_sample_inputs.append(arg_shape_or_tensor) 
            
            sp = ShapeProp(graph_module)
            sp.propagate(*actual_sample_inputs, **(sample_input_kwargs or {}))
        except Exception as e:
            error_message = f"Error during shape propagation for '{class_name}': {e}. "
            error_message += f"Sample args: {sample_input_args}, dtypes: {sample_input_dtypes}, kwargs: {sample_input_kwargs}."
            raise Exception(error_message)
    
    # Apply filters
    graph_filter = GraphFilter(filter_config)
    all_fx_nodes = list(graph_module.graph.nodes)
    included_fx_nodes = graph_filter.filter_graph(all_fx_nodes, graph_module)
    
    # Convert based on output format
    if out_format == "json":
        return _fx_to_open_hg(graph_module, included_fx_nodes, graph_filter)
    elif out_format == "macro":
        ohg_data = _fx_to_open_hg(graph_module, included_fx_nodes, graph_filter)
        macro_syntax = _generate_macro_syntax(ohg_data)
        return {
            "macro_syntax": macro_syntax,
            "metadata": ohg_data.get("metadata", {}),
            "format": "hypersyn-macro"
        }
    elif out_format == "categorical":
        return _fx_to_categorical_hypergraph(graph_module, included_fx_nodes, graph_filter)
    else:
        raise ValueError(f"Unknown output format: {out_format}. Supported: json, macro, categorical")


def _generate_macro_syntax(ohg_data: Dict[str, Any]) -> str:
    """
    Generate hypersyn macro syntax from open-hypergraph JSON data.
    
    This creates a Rust-like macro syntax that can be used with the
    hellas-ai/open-hypergraphs library.
    """
    nodes = ohg_data.get("nodes", [])
    hyperedges = ohg_data.get("hyperedges", [])
    
    lines = []
    lines.append("// Generated hypersyn macro syntax")
    lines.append("// Compatible with hellas-ai/open-hypergraphs")
    lines.append("")
    
    # Generate node declarations
    lines.append("// Node declarations")
    for node in nodes:
        node_name = node["name"]
        op_type = node.get("op", "unknown")
        target = node.get("target", "")
        
        if "shape" in node:
            shape_comment = f" // shape: {node['shape']}"
        else:
            shape_comment = ""
        
        lines.append(f"let {node_name}: {op_type} = {target};{shape_comment}")
    
    lines.append("")
    lines.append("// Hyperedge connections")
    
    # Generate hyperedge connections
    for edge in hyperedges:
        operation = edge["operation"]
        inputs = edge.get("inputs", [])
        outputs = edge.get("outputs", [])
        
        # Map node IDs back to names
        input_names = []
        for input_id in inputs:
            for node in nodes:
                if node["id"] == input_id:
                    input_names.append(node["name"])
                    break
        
        output_names = []
        for output_id in outputs:
            for node in nodes:
                if node["id"] == output_id:
                    output_names.append(node["name"])
                    break
        
        if input_names and output_names:
            input_str = ", ".join(input_names)
            output_str = ", ".join(output_names)
            lines.append(f"compose!({input_str} => {output_str}); // {operation}")
        elif output_names:  # No inputs (e.g., placeholder)
            output_str = ", ".join(output_names)
            lines.append(f"generate!({output_str}); // {operation}")
    
    return "\n".join(lines) 