# Open-Hypergraph Export for VisuaML

This document describes the new open-hypergraph export functionality added to VisuaML, which allows you to export PyTorch models to the open-hypergraph format for advanced analysis and visualization.

## Overview

The open-hypergraph export feature converts PyTorch FX graphs into the open-hypergraph representation, which is useful for:

- **String diagram visualization**: View neural networks as categorical diagrams
- **Compositional analysis**: Understand how operations compose together
- **Mathematical reasoning**: Apply category theory concepts to neural networks
- **Advanced optimization**: Use hypergraph-based optimization techniques

## Installation

First, install the required dependency:

```bash
pip install open-hypergraphs
```

## Usage

### Basic Export

```python
from visuaml import export_model_open_hypergraph

# Export to JSON format (default)
result = export_model_open_hypergraph(
    'models.SimpleNN',
    sample_input_args=((10,),)
)

print(f"Exported {len(result['nodes'])} nodes and {len(result['hyperedges'])} hyperedges")
```

### Macro Format Export

```python
# Export to macro format (text-based representation)
result = export_model_open_hypergraph(
    'models.SimpleNN',
    sample_input_args=((10,),),
    out_format="macro"
)

print(result['macro'])
```

### With Shape Propagation

```python
# Include tensor shape and dtype information
result = export_model_open_hypergraph(
    'models.SimpleCNN',
    sample_input_args=((1, 28, 28),),
    sample_input_dtypes=['float32']
)
```

### With Filtering

```python
from visuaml.filters import FilterConfig

# Apply filters to focus on specific operations
filter_config = FilterConfig(
    exclude_ops=['relu', 'dropout']  # Example filter
)

result = export_model_open_hypergraph(
    'models.SimpleNN',
    sample_input_args=((10,),),
    filter_config=filter_config
)
```

## Output Format

### JSON Format

The JSON output contains:

```json
{
  "nodes": [
    {
      "id": 0,
      "name": "x",
      "op": "placeholder",
      "target": "x",
      "shape": [10],
      "dtype": "torch.float32"
    }
  ],
  "hyperedges": [
    {
      "id": 0,
      "nodes": [0, 1],
      "source_nodes": [0],
      "target_node": 1,
      "operation": "fc1"
    }
  ],
  "metadata": {
    "format": "open-hypergraph",
    "source": "visuaml-pytorch-fx",
    "model_class": "SimpleNN"
  }
}
```

### Macro Format

The macro format provides a text-based representation:

```
// Open-hypergraph macro syntax
// Generated from PyTorch FX graph

node 0 : x (placeholder)
node 1 : fc1 (call_module)
node 2 : relu1 (call_module)

edge 0 : [0] -> 1 (fc1)
edge 1 : [1] -> 2 (relu1)
```

## API Reference

### `export_model_open_hypergraph()`

```python
def export_model_open_hypergraph(
    model_path: str,
    filter_config: Optional[FilterConfig] = None,
    model_args: Optional[tuple] = None,
    model_kwargs: Optional[dict] = None,
    sample_input_args: Optional[tuple] = None,
    sample_input_kwargs: Optional[dict] = None,
    sample_input_dtypes: Optional[List[str]] = None,
    sample_dtypes: Optional[List[str]] = None,
    out_format: str = "json",
) -> Dict[str, Any]:
```

**Parameters:**

- `model_path`: Module path to the model (e.g., 'models.MyModel')
- `filter_config`: Optional filter configuration
- `model_args`: Optional arguments for model constructor
- `model_kwargs`: Optional keyword arguments for model constructor
- `sample_input_args`: Optional sample positional inputs for shape propagation
- `sample_input_kwargs`: Optional sample keyword inputs for shape propagation
- `sample_input_dtypes`: Optional list of sample input dtypes for shape propagation
- `sample_dtypes`: Alias for sample_input_dtypes (for backward compatibility)
- `out_format`: Output format ("json" or "macro")

**Returns:**

Dictionary with open-hypergraph representation

## Supported Models

The open-hypergraph export works with models that are compatible with PyTorch FX tracing:

✅ **Supported:**

- Feedforward networks (SimpleNN, Autoencoder)
- Convolutional networks (with proper input shapes)
- Most standard PyTorch operations

❌ **Limitations:**

- Dynamic control flow (if/else statements)
- Dynamic tensor shapes
- Some RNN architectures with complex state handling
- Models with unsupported operations

## Integration with Open-Hypergraph Ecosystem

The exported format is compatible with:

- [open-hypergraphs](https://github.com/statusfailed/open-hypergraphs) - Python library for hypergraph manipulation
- [visualising-llms-diagrams](https://github.com/statusfailed/visualising-llms-diagrams) - WASM-based visualization
- [catgrad](https://github.com/hellas-ai/catgrad) - Categorical gradient computation
- [hypersyn](https://github.com/hellas-ai/hypersyn) - Macro syntax for hypergraphs

## Testing

Run the demonstration script to test the functionality:

```bash
python backend/test_ohg_final.py
```

## Implementation Details

The export process:

1. **FX Tracing**: Converts PyTorch model to FX graph
2. **Shape Propagation**: Optionally propagates tensor shapes and dtypes
3. **Filtering**: Applies user-specified filters
4. **Conversion**: Transforms FX nodes and edges to hypergraph representation
5. **Serialization**: Outputs in JSON or macro format

The implementation preserves:

- Operation types and targets
- Tensor metadata (shapes, dtypes)
- Graph connectivity
- Node attributes

## Future Enhancements

Potential improvements:

- Support for more complex control flow
- Integration with visualization tools
- Advanced filtering options
- Performance optimizations
- Support for dynamic models

## Related Work

This implementation is inspired by:

- [Open Hypergraphs for String Diagrams](https://arxiv.org/pdf/2305.01041) paper
- Category theory applications to machine learning
- String diagram representations of neural networks
