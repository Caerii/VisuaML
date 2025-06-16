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

from .types import Dtype, ArrayType, TensorType
from .morphisms import Morphism, LinearMorphism, ActivationMorphism, ComposedMorphism
from .composition import compose, parallel, tensor_product
from .hypergraph import CategoricalHypergraph

__all__ = [
    # Types
    'Dtype', 'ArrayType', 'TensorType',
    # Morphisms
    'Morphism', 'LinearMorphism', 'ActivationMorphism', 'ComposedMorphism',
    # Composition
    'compose', 'parallel', 'tensor_product',
    # Hypergraphs
    'CategoricalHypergraph'
]

__version__ = "0.1.0" 