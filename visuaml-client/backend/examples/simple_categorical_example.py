"""
Simple example demonstrating the categorical hypergraph system.

This example shows how to:
1. Create categorical morphisms
2. Compose them mathematically
3. Convert to hypergraph representation
4. Export in different formats

This follows the catgrad pattern but adapted for VisuaML.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from visuaml.categorical import (
    ArrayType, Dtype, 
    LinearMorphism, ActivationMorphism,
    compose, parallel,
    CategoricalHypergraph
)
from visuaml.categorical.composition import CompositionBuilder


def simple_linear_example():
    """Create a simple linear model like catgrad's iris example."""
    print("=== Simple Linear Model ===")
    
    # Define types (similar to catgrad)
    INPUT_TYPE = ArrayType((4,), Dtype.FLOAT32)  # Iris features
    HIDDEN_TYPE = ArrayType((10,), Dtype.FLOAT32)
    OUTPUT_TYPE = ArrayType((3,), Dtype.FLOAT32)  # Iris classes
    
    print(f"Input type: {INPUT_TYPE}")
    print(f"Hidden type: {HIDDEN_TYPE}")
    print(f"Output type: {OUTPUT_TYPE}")
    
    # Create morphisms
    layer1 = LinearMorphism(4, 10, name="input_layer")
    activation = ActivationMorphism((10,), "relu", name="relu_activation")
    layer2 = LinearMorphism(10, 3, name="output_layer")
    
    print("\nMorphisms:")
    print(f"  {layer1}")
    print(f"  {activation}")
    print(f"  {layer2}")
    
    # Compose them (like catgrad composition)
    model = compose(layer1, activation, layer2, name="iris_classifier")
    
    print(f"\nComposed model: {model}")
    print(f"Model type: {model.input_type} → {model.output_type}")
    
    return model


def composition_builder_example():
    """Demonstrate the composition builder pattern."""
    print("\n=== Composition Builder Example ===")
    
    # Build a more complex model using the builder pattern
    builder = CompositionBuilder()
    
    model = (builder
             .add_linear(784, 128, name="layer1")
             .add_relu((128,), name="relu1")
             .add_linear(128, 64, name="layer2")
             .add_relu((64,), name="relu2")
             .add_linear(64, 10, name="output")
             .build("mnist_classifier"))
    
    print(f"Built model: {model}")
    print(f"Type signature: {builder.get_signature()}")
    print(f"Composition valid: {builder.validate()}")
    
    return model


def parallel_composition_example():
    """Demonstrate parallel composition."""
    print("\n=== Parallel Composition Example ===")
    
    # Create two separate pathways
    pathway1 = compose(
        LinearMorphism(5, 3, name="path1_layer1"),
        ActivationMorphism((3,), "relu", name="path1_relu"),
        name="pathway1"
    )
    
    pathway2 = compose(
        LinearMorphism(7, 4, name="path2_layer1"),
        ActivationMorphism((4,), "sigmoid", name="path2_sigmoid"),
        name="pathway2"
    )
    
    # Compose them in parallel
    parallel_model = parallel(pathway1, pathway2, name="dual_pathway")
    
    print(f"Pathway 1: {pathway1}")
    print(f"Pathway 2: {pathway2}")
    print(f"Parallel model: {parallel_model}")
    
    return parallel_model


def hypergraph_conversion_example(model):
    """Convert morphism to categorical hypergraph."""
    print("\n=== Hypergraph Conversion ===")
    
    # Convert to categorical hypergraph
    hypergraph = CategoricalHypergraph.from_morphism(model)
    
    print(f"Hypergraph: {hypergraph}")
    print(f"Statistics: {hypergraph.get_statistics()}")
    
    # Validate the hypergraph
    is_valid, errors = hypergraph.validate()
    print(f"Valid: {is_valid}")
    if errors:
        print(f"Errors: {errors}")
    
    # Show hypergraph structure
    print("\nHypergraph structure:")
    hg_dict = hypergraph.to_dict()
    print(f"  Hyperedges: {len(hg_dict['hyperedges'])}")
    print(f"  Wires: {len(hg_dict['wires'])}")
    print(f"  Input boundary: {hg_dict['input_boundary']}")
    print(f"  Output boundary: {hg_dict['output_boundary']}")
    
    return hypergraph


def export_formats_example(model):
    """Demonstrate different export formats."""
    print("\n=== Export Formats ===")
    
    # Get hypergraph representation
    hg_data = model.to_hypergraph()
    
    print("JSON representation (truncated):")
    import json
    json_str = json.dumps(hg_data, indent=2)
    print(json_str[:500] + "..." if len(json_str) > 500 else json_str)
    
    # Show parameters
    print("\nModel parameters:")
    params = model.get_parameters()
    for key, value in params.items():
        print(f"  {key}: {value}")
    
    return hg_data


def catgrad_style_demo():
    """
    Demonstrate catgrad-style usage.
    
    This shows how our system follows catgrad patterns:
    1. Define types
    2. Create morphisms
    3. Compose categorically
    4. Export/compile
    """
    print("🐱 VisuaML Categorical Demo (Catgrad-style)")
    print("=" * 50)
    
    # Simple linear model (like catgrad iris example)
    simple_model = simple_linear_example()
    
    # Builder pattern for complex models
    complex_model = composition_builder_example()
    
    # Parallel composition
    parallel_model = parallel_composition_example()
    
    # Convert to hypergraph
    hypergraph = hypergraph_conversion_example(simple_model)
    
    # Export formats
    export_data = export_formats_example(simple_model)
    
    print("\n🎉 Demo completed successfully!")
    print("This demonstrates proper categorical foundations for open hypergraphs.")
    
    return {
        "simple_model": simple_model,
        "complex_model": complex_model,
        "parallel_model": parallel_model,
        "hypergraph": hypergraph,
        "export_data": export_data
    }


if __name__ == "__main__":
    try:
        results = catgrad_style_demo()
        print("\n✅ All examples completed successfully!")
        print(f"Created {len(results)} different models/representations")
    except Exception as e:
        print(f"\n❌ Error in demo: {e}")
        import traceback
        traceback.print_exc() 