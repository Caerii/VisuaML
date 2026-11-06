"""
Categorical implementation for VisuaML.

This module provides a proper categorical foundation for open hypergraphs,
following the patterns established by catgrad and category theory.

Key concepts:
- Morphisms: Categorical arrows with input/output types
- Composition: Mathematical composition of morphisms
- Types: Proper type system for categorical structures
- Hypergraphs: Open hypergraph representations
"""

from .composition import compose, parallel, tensor_product
from .hypergraph import CategoricalHypergraph
from .morphisms import ActivationMorphism, ComposedMorphism, LinearMorphism, Morphism
from .types import ArrayType, Dtype, TensorType

__all__ = [
    # Types
    "Dtype",
    "ArrayType",
    "TensorType",
    # Morphisms
    "Morphism",
    "LinearMorphism",
    "ActivationMorphism",
    "ComposedMorphism",
    # Composition
    "compose",
    "parallel",
    "tensor_product",
    # Hypergraphs
    "CategoricalHypergraph",
]

__version__ = "0.1.0"
