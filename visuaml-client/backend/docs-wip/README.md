# VisuaML Backend Architecture

The VisuaML backend is organized into a modular architecture for processing and visualizing PyTorch neural networks.

## Directory Structure

```
backend/
├── visuaml/              # Main package
│   ├── __init__.py       # Package initialization
│   ├── model_loader.py   # Model loading and instantiation
│   ├── filters.py        # Graph filtering logic
│   ├── visualization.py  # Visual properties and colors
│   └── graph_export.py   # Main export functionality
├── scripts/
│   └── fx_export.py      # Export script using modular architecture
└── tests/                # Unit tests
```

## Modules

### `model_loader.py`

Handles loading PyTorch models from module paths:

- `load_model_class()`: Dynamically imports model classes
- `instantiate_model()`: Creates model instances
- `ModelLoadError`: Custom exception for load failures

### `filters.py`

Manages graph node filtering:

- `FilterConfig`: Configuration for different abstraction levels
- `GraphFilter`: Implements node filtering and smart edge routing
- Supports multiple abstraction levels:
  - Level 0: Show everything
  - Level 1: Hide low-level ops (getitem, getattr)
  - Level 2: Also hide tensor operations
  - Level 3: Show only core ML layers

### `visualization.py`

Determines visual properties for nodes:

- `NodeColorScheme`: Manages colors for different layer types
- `NodeVisualizer`: Assigns colors and labels to nodes
- Extensive color palette for all PyTorch layer types

### `graph_export.py`

Main export functionality:

- `export_model_graph()`: High-level API for model export
- `create_graph_json()`: Converts FX graphs to VisuaML format
- Handles smart edge routing through filtered nodes

## Usage

### Command Line

```bash
# Basic usage (with default filtering)
python backend/scripts/fx_export.py models.MyModel

# Show all nodes (no filtering)
python backend/scripts/fx_export.py models.MyModel --no-filter

# Different abstraction levels
python backend/scripts/fx_export.py models.MyModel --abstraction-level 2
```

### Python API

```python
from visuaml import export_model_graph, FilterConfig

# Export with default filtering
graph_data = export_model_graph("models.MyModel")

# Export with custom filtering
config = FilterConfig(abstraction_level=2)
graph_data = export_model_graph("models.MyModel", config)

# Access nodes and edges
nodes = graph_data["nodes"]
edges = graph_data["edges"]
```

## Extending the System

### Adding New Filters

1. Update `FILTERED_OPS` in `FilterConfig`
2. Add logic to `GraphFilter.should_include_node()`
3. Update operation categories for abstraction levels

### Adding New Colors

1. Update `NodeColorScheme.DEFAULT_COLORS`
2. Add layer type mappings

### Adding Analysis Features

Create new modules in the `visuaml/` package:

- `analysis.py`: For architectural analysis
- `patterns.py`: For pattern detection
- `suggestions.py`: For improvement suggestions

## Future Enhancements

1. **Pattern Detection**: Identify common architectural patterns
2. **Performance Analysis**: Estimate FLOPs, parameters
3. **Optimization Suggestions**: Recommend improvements
4. **Export Formats**: Support ONNX, TensorFlow
5. **Caching**: Cache traced graphs for faster exports
