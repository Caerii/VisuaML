import torch, json, importlib, sys
import os
from torch.fx import symbolic_trace

# Add the current working directory (which execa sets to projectRoot) to sys.path
# to help Python find the 'models' module.
current_working_directory = os.getcwd()
if current_working_directory not in sys.path:
    sys.path.insert(0, current_working_directory)

# For debugging, print sys.path and cwd (can be enabled if issues persist)
# print(f"DEBUG: fx_export.py - sys.path: {sys.path}", file=sys.stderr)
# print(f"DEBUG: fx_export.py - cwd: {current_working_directory}", file=sys.stderr)

# Check if a module path is provided
if len(sys.argv) < 2:
    print("Usage: python fx_export.py my_module.submodule.ModelClass")
    sys.exit(1)

mod_path_argument = sys.argv[1]  # e.g. models.MyTinyGPT

try:
    # The module path to import is the full path provided by the user.
    # The class name is the last component of this path.
    module_to_import_str = mod_path_argument 
    class_name_str = mod_path_argument.split(".")[-1]
    
    # print(f"DEBUG: Attempting to import module: {module_to_import_str}", file=sys.stderr)
    # print(f"DEBUG: Attempting to get class: {class_name_str}", file=sys.stderr)

    imported_target_module = importlib.import_module(module_to_import_str)
    ModelClass = getattr(imported_target_module, class_name_str)

except ImportError:
    # This error means the entire module path (e.g., models.MyTinyGPT) could not be found/imported.
    # This could be due to missing __init__.py in 'models' or 'MyTinyGPT.py' not found in 'models'.
    print(f"Error: Could not import module '{module_to_import_str}'. Check path and ensure all necessary __init__.py files exist.")
    sys.exit(1)
except AttributeError:
    # This error means the module was imported, but the class was not found within it.
    print(f"Error: Class '{class_name_str}' not found in module '{module_to_import_str}'.")
    sys.exit(1)
except Exception as e:
    print(f"An unexpected error occurred during module/class loading: {e}")
    sys.exit(1)

# Instantiate the model and perform symbolic tracing
try:
    # Ensure the model can be instantiated without arguments for this script
    # If your model requires arguments, this script will need modification
    # or a wrapper model that instantiates it with defaults.
    model_instance = ModelClass()
    gm = symbolic_trace(model_instance)
except Exception as e:
    print(f"Error during model instantiation or symbolic tracing: {e}")
    print("Please ensure your model '{class_name_str}' can be instantiated with no arguments for this script.")
    sys.exit(1)

nodes = []
edges = []
name_map = {}

# Simple color and layerType mapping (extend as needed)
def get_node_visual_properties(fx_node, graph_module):
    op_type = fx_node.op
    layer_type_str = op_type # Default layer type to op code
    color = "#cccccc"  # Default color

    if op_type == 'placeholder':
        layer_type_str = 'Input'
        color = "#adebad"
    elif op_type == 'output':
        layer_type_str = 'Output'
        color = "#ffb3b3"
    elif op_type == 'call_module':
        try:
            # Get the class name of the module being called
            module_instance = graph_module.get_submodule(fx_node.target)
            layer_type_str = type(module_instance).__name__
            # Assign colors based on common PyTorch layer types
            if "Linear" in layer_type_str: color = "#add8e6" # Light blue for Linear
            elif "Conv" in layer_type_str: color = "#90ee90" # Light green for Conv
            elif "BatchNorm" in layer_type_str: color = "#ffd700" # Gold for BatchNorm
            elif "ReLU" in layer_type_str or "GELU" in layer_type_str: color = "#ffcb6b" # Orange for Activations
            elif "Dropout" in layer_type_str: color = "#d3d3d3" # Light grey for Dropout
            elif "Embedding" in layer_type_str: color = "#dda0dd" # Plum for Embedding
            elif "Attention" in layer_type_str: color = "#ffb6c1" # Light pink for Attention
            else: color = "#e6e6fa" # Lavender for other modules
        except AttributeError: # Target might not be a submodule or not exist
            layer_type_str = str(fx_node.target) # Fallback to target string
            color = "#f0e68c" # Khaki for unknown call_module targets
    elif op_type == 'call_function' or op_type == 'call_method':
        layer_type_str = str(fx_node.target.__name__ if hasattr(fx_node.target, '__name__') else fx_node.target)
        color = "#b0e0e6" # Powder blue for functions/methods
    elif op_type == 'get_attr':
        layer_type_str = f"GetAttr: {fx_node.target}"
        color = "#fffacd" # Lemon chiffon for get_attr

    return layer_type_str, color

# Process graph nodes
for n in gm.graph.nodes:
    name_map[n] = n.name
    layer_type, node_color = get_node_visual_properties(n, gm)
    
    nodes.append({
        "id": n.name,
        "type": "transformer", # Use the custom node type key registered in React Flow
        "data": {
            "label": n.name, # Use node name as a default label in data
            "target": str(n.target),
            "name": n.name,
            "op": n.op,
            "args": str(n.args),
            "kwargs": str(n.kwargs),
            "layerType": layer_type,
            "color": node_color
        },
        "position": {"x": 0, "y": 0}
    })
    
    # Process input nodes to create edges
    # n.all_input_nodes includes all direct data dependencies
    for inp_node in n.all_input_nodes:
        if inp_node.name in name_map.values(): # Ensure the source node is part of our graph representation
            edges.append({
                "id": f"{inp_node.name}->{n.name}",
                "source": inp_node.name,
                "target": n.name
            })

# Output JSON
output_json = {"nodes": nodes, "edges": edges}
try:
    print(json.dumps(output_json, indent=2)) # Added indent for readability
except TypeError as e:
    print(f"Error serializing graph to JSON: {e}")
    sys.exit(1) 