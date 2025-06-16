# VisuaML Categorical System Guide

## Overview

The VisuaML Categorical System provides a mathematically sound foundation for working with open hypergraphs, following the patterns established by [catgrad](https://github.com/statusfailed/catgrad) and category theory.

## Key Concepts

### 1. Categorical Foundation

Unlike traditional approaches that treat neural networks as computational graphs, our system treats them as **categorical morphisms** with proper mathematical structure:

- **Morphisms**: Categorical arrows with well-defined input/output types
- **Composition**: Mathematical composition following categorical laws
- **Types**: Proper type system ensuring compositional safety
- **Open Hypergraphs**: Hypergraphs with input/output boundaries

### 2. Comparison with Catgrad

| Aspect          | Catgrad                        | VisuaML Categorical                        |
| --------------- | ------------------------------ | ------------------------------------------ |
| **Purpose**     | Deep learning compiler         | Model visualization + categorical analysis |
| **Input**       | Categorical layer definitions  | PyTorch FX graphs                          |
| **Output**      | Static compiled code           | Open hypergraph representations            |
| **Composition** | Direct categorical composition | Bridge-based conversion                    |
| **Backends**    | Python, C++, CUDA              | JSON, Rust macros, OpenHypergraph          |

## Architecture

```
PyTorch Model → FX Graph → Categorical Morphisms → Open Hypergraphs → Export Formats
                    ↑              ↑                    ↑              ↑
                Bridge         Composition         Conversion      JSON/Rust/Python
```

### Directory Structure

```
visuaml/
├── categorical/           # Core categorical implementation
│   ├── types.py          # Type system (ArrayType, TensorType)
│   ├── morphisms.py      # Morphism classes (Linear, Activation, etc.)
│   ├── composition.py    # Composition operations
│   └── hypergraph.py     # Categorical hypergraph representation
├── bridges/              # Bridge between systems
│   ├── pytorch_bridge.py # PyTorch FX → Categorical
│   └── hypergraph_bridge.py # Categorical → OpenHypergraph
└── export/               # Export to different formats
    ├── json_exporter.py  # JSON export
    ├── macro_exporter.py # Rust macro export
    └── categorical_exporter.py # Categorical export
```

## Usage Examples

### Basic Morphism Creation

```python
from visuaml.categorical import ArrayType, Dtype, LinearMorphism, ActivationMorphism

# Define types
input_type = ArrayType((784,), Dtype.FLOAT32)
hidden_type = ArrayType((128,), Dtype.FLOAT32)
output_type = ArrayType((10,), Dtype.FLOAT32)

# Create morphisms
layer1 = LinearMorphism(784, 128, name="hidden_layer")
activation = ActivationMorphism((128,), "relu", name="relu")
layer2 = LinearMorphism(128, 10, name="output_layer")
```

### Categorical Composition

```python
from visuaml.categorical import compose

# Sequential composition (mathematical composition)
model = compose(layer1, activation, layer2, name="mnist_classifier")

print(f"Model type: {model.input_type} → {model.output_type}")
# Output: Array[784:float32] → Array[10:float32]
```

### Builder Pattern

```python
from visuaml.categorical.composition import CompositionBuilder

# Build complex models
builder = CompositionBuilder()
model = (builder
    .add_linear(784, 128, name="layer1")
    .add_relu((128,), name="relu1")
    .add_linear(128, 64, name="layer2")
    .add_relu((64,), name="relu2")
    .add_linear(64, 10, name="output")
    .build("deep_classifier"))
```

### Parallel Composition

```python
from visuaml.categorical import parallel

# Create parallel pathways
pathway1 = compose(LinearMorphism(5, 3), ActivationMorphism((3,), "relu"))
pathway2 = compose(LinearMorphism(7, 4), ActivationMorphism((4,), "sigmoid"))

# Parallel composition (tensor product)
parallel_model = parallel(pathway1, pathway2, name="dual_pathway")
```

### Hypergraph Conversion

```python
from visuaml.categorical import CategoricalHypergraph

# Convert morphism to hypergraph
hypergraph = CategoricalHypergraph.from_morphism(model)

# Validate structure
is_valid, errors = hypergraph.validate()
print(f"Valid: {is_valid}")

# Get statistics
stats = hypergraph.get_statistics()
print(f"Hyperedges: {stats['num_hyperedges']}")
print(f"Wires: {stats['num_wires']}")
```

## Type System

### ArrayType

Represents typed arrays with shape and dtype:

```python
from visuaml.categorical.types import ArrayType, Dtype

# Static shapes
vector = ArrayType((10,), Dtype.FLOAT32)
matrix = ArrayType((5, 10), Dtype.FLOAT32)

# Dynamic shapes (None for batch dimension)
batch_vector = ArrayType((None, 10), Dtype.FLOAT32)

# Type compatibility
print(vector.is_compatible_with(vector))  # True
print(vector.is_compatible_with(matrix))  # False
```

### TensorType

Represents morphism types (arrows):

```python
from visuaml.categorical.types import TensorType

# Morphism type: input → output
morphism_type = TensorType(
    input_type=ArrayType((784,), Dtype.FLOAT32),
    output_type=ArrayType((10,), Dtype.FLOAT32)
)

print(morphism_type)  # Array[784:float32] → Array[10:float32]
```

## Morphism Types

### LinearMorphism

Matrix multiplication with optional bias:

```python
linear = LinearMorphism(
    input_dim=784,
    output_dim=128,
    use_bias=True,
    name="hidden_layer"
)

# Get hypergraph representation
hg_data = linear.to_hypergraph()
```

### ActivationMorphism

Element-wise activation functions:

```python
relu = ActivationMorphism((128,), "relu", name="activation")
sigmoid = ActivationMorphism((64,), "sigmoid", name="sigmoid")
```

### ComposedMorphism

Sequential composition of morphisms:

```python
# Automatic creation via compose()
composed = compose(linear, relu)

# Manual creation
composed = ComposedMorphism(linear, relu)
```

### ParallelMorphism

Parallel composition (tensor product):

```python
# Automatic creation via parallel()
parallel_morph = parallel(morphism1, morphism2)

# Manual creation
parallel_morph = ParallelMorphism([morphism1, morphism2])
```

## Composition Rules

### Sequential Composition

For morphisms `f: A → B` and `g: B → C`, the composition `g ∘ f: A → C` is valid if:

1. `f.output_type.is_compatible_with(g.input_type)`
2. The composition follows categorical associativity laws

```python
# Valid composition
f = LinearMorphism(10, 5)    # Array[10:float32] → Array[5:float32]
g = LinearMorphism(5, 3)     # Array[5:float32] → Array[3:float32]
composed = g @ f             # Array[10:float32] → Array[3:float32]

# Invalid composition (type mismatch)
h = LinearMorphism(7, 3)     # Array[7:float32] → Array[3:float32]
# g @ h  # Would raise ValueError
```

### Parallel Composition

For morphisms `f: A → B` and `g: C → D`, the tensor product `f ⊗ g: A⊗C → B⊗D`:

```python
f = LinearMorphism(5, 3)     # Array[5:float32] → Array[3:float32]
g = LinearMorphism(7, 4)     # Array[7:float32] → Array[4:float32]
parallel_fg = parallel(f, g) # Array[12:float32] → Array[7:float32]
```

## Hypergraph Structure

### Components

1. **HyperEdges**: Operations with multiple inputs/outputs
2. **Wires**: Connections between hyperedges
3. **Boundaries**: Input/output interfaces

### Example Structure

```python
hypergraph = CategoricalHypergraph.from_morphism(model)
hg_dict = hypergraph.to_dict()

# Structure:
{
    "name": "model_name",
    "hyperedges": [
        {
            "id": "layer1_matmul",
            "type": "matmul",
            "inputs": ["input", "weight"],
            "outputs": ["output"]
        }
    ],
    "wires": [
        {
            "id": "input",
            "type": "Array[784:float32]",
            "source": None,
            "targets": ["layer1_matmul"]
        }
    ],
    "input_boundary": {
        "wires": ["input"],
        "types": ["Array[784:float32]"]
    },
    "output_boundary": {
        "wires": ["output"],
        "types": ["Array[10:float32]"]
    }
}
```

## Export Formats

### JSON Export

```python
# Get JSON representation
json_data = hypergraph.to_json(indent=2)

# Or from morphism directly
hg_data = model.to_hypergraph()
```

### Rust Macro Export

```python
# TODO: Implement Rust macro exporter
from visuaml.export import MacroExporter

exporter = MacroExporter()
rust_code = exporter.export(model)
```

### OpenHypergraph Export

```python
# TODO: Implement proper OpenHypergraph bridge
from visuaml.bridges import CategoricalToOpenHypergraph

bridge = CategoricalToOpenHypergraph()
ohg = bridge.convert_morphism(model)
```

## Integration with Existing System

### PyTorch Bridge

```python
# TODO: Implement PyTorch FX bridge
from visuaml.bridges import PyTorchToCategorical
from torch.fx import symbolic_trace

# Convert PyTorch model to categorical
pytorch_model = MyModel()
fx_graph = symbolic_trace(pytorch_model)

bridge = PyTorchToCategorical()
categorical_model = bridge.convert_graph(fx_graph)
```

### Backward Compatibility

The categorical system is designed to work alongside the existing VisuaML system:

```python
# Existing export still works
from visuaml.openhypergraph_export import export_model_open_hypergraph

# New categorical export
from visuaml.categorical import CategoricalHypergraph
```

## Mathematical Foundations

### Category Theory Concepts

1. **Objects**: Types (ArrayType, TensorType)
2. **Morphisms**: Transformations between types
3. **Composition**: Associative composition of morphisms
4. **Identity**: Identity morphisms for each type
5. **Functors**: Structure-preserving mappings

### Laws

1. **Associativity**: `(h ∘ g) ∘ f = h ∘ (g ∘ f)`
2. **Identity**: `id_B ∘ f = f = f ∘ id_A` for `f: A → B`
3. **Functoriality**: Composition preserves structure

## Best Practices

### 1. Type Safety

Always ensure type compatibility:

```python
# Good: Check compatibility
if f.can_compose_with(g):
    composed = g @ f

# Better: Use validation
from visuaml.categorical.composition import validate_composition
if validate_composition([f, g]):
    composed = compose(f, g)
```

### 2. Naming Conventions

Use descriptive names for morphisms:

```python
# Good
input_layer = LinearMorphism(784, 128, name="input_layer")
hidden_activation = ActivationMorphism((128,), "relu", name="hidden_relu")

# Better: Include type information
input_layer = LinearMorphism(784, 128, name="input_784_to_128")
```

### 3. Composition Patterns

Use the builder pattern for complex models:

```python
# Good for simple models
model = compose(layer1, relu, layer2)

# Better for complex models
builder = CompositionBuilder()
model = builder.add_linear(784, 128).add_relu((128,)).build()
```

### 4. Validation

Always validate hypergraphs:

```python
hypergraph = CategoricalHypergraph.from_morphism(model)
is_valid, errors = hypergraph.validate()
if not is_valid:
    print(f"Validation errors: {errors}")
```

## Troubleshooting

### Common Issues

1. **Type Mismatch**: Ensure output type of first morphism matches input type of second
2. **Invalid Composition**: Check that morphisms can be composed using `can_compose_with()`
3. **Hypergraph Validation**: Use `validate()` to check for structural issues

### Debugging

```python
# Check morphism types
print(f"Morphism: {morphism.input_type} → {morphism.output_type}")

# Validate composition
print(f"Can compose: {g.can_compose_with(f)}")

# Check hypergraph structure
stats = hypergraph.get_statistics()
print(f"Hypergraph stats: {stats}")
```

## Future Directions

1. **Full OpenHypergraph Integration**: Complete bridge to open-hypergraphs library
2. **Catgrad Compilation**: Add catgrad-style compilation to static code
3. **More Morphism Types**: Convolution, attention, normalization layers
4. **Optimization**: Categorical optimization and simplification rules
5. **Visualization**: Interactive categorical diagram visualization

## References

- [Catgrad Repository](https://github.com/statusfailed/catgrad)
- [Open Hypergraphs Library](https://github.com/statusfailed/open-hypergraphs)
- [Category Theory for Programmers](https://bartoszmilewski.com/2014/10/28/category-theory-for-programmers-the-preface/)
- [Reverse Derivative Ascent Paper](http://catgrad.com/p/reverse-derivative-ascent/)
