"""
Composition operations for categorical morphisms.

This module provides functions for composing morphisms in various ways,
following categorical composition rules.
"""

from typing import List, Optional
from .morphisms import Morphism, ComposedMorphism, ParallelMorphism, IdentityMorphism
from .types import ArrayType


def compose(*morphisms: Morphism, name: Optional[str] = None) -> Morphism:
    """
    Compose multiple morphisms sequentially (left to right).
    
    Args:
        *morphisms: Morphisms to compose (f1, f2, f3, ...) -> f3 ∘ f2 ∘ f1
        name: Optional name for the composed morphism
        
    Returns:
        Composed morphism
        
    Example:
        >>> f = linear(10, 5)
        >>> g = relu((5,))
        >>> h = linear(5, 1)
        >>> model = compose(f, g, h)  # h ∘ g ∘ f
    """
    if not morphisms:
        raise ValueError("Cannot compose empty list of morphisms")
    
    if len(morphisms) == 1:
        return morphisms[0]
    
    # Compose from left to right: f1, f2, f3 -> f3 ∘ f2 ∘ f1
    result = morphisms[0]
    for morphism in morphisms[1:]:
        if not morphism.can_compose_with(result):
            raise ValueError(
                f"Cannot compose {result.output_type} with {morphism.input_type}"
            )
        result = ComposedMorphism(result, morphism)
    
    if name:
        result.name = name
    
    return result


def parallel(*morphisms: Morphism, name: Optional[str] = None) -> ParallelMorphism:
    """
    Compose morphisms in parallel (tensor product).
    
    Args:
        *morphisms: Morphisms to compose in parallel
        name: Optional name for the parallel morphism
        
    Returns:
        Parallel morphism
        
    Example:
        >>> f = linear(5, 3)
        >>> g = linear(7, 2)
        >>> parallel_fg = parallel(f, g)  # f ⊗ g
    """
    if not morphisms:
        raise ValueError("Cannot create parallel composition with no morphisms")
    
    return ParallelMorphism(list(morphisms), name)


def tensor_product(left: Morphism, right: Morphism, 
                  name: Optional[str] = None) -> ParallelMorphism:
    """
    Tensor product of two morphisms (f ⊗ g).
    
    Args:
        left: Left morphism
        right: Right morphism
        name: Optional name for the tensor product
        
    Returns:
        Tensor product morphism
    """
    return ParallelMorphism([left, right], name)


def sequential(*morphisms: Morphism, name: Optional[str] = None) -> Morphism:
    """
    Alias for compose() for clarity.
    
    Args:
        *morphisms: Morphisms to compose sequentially
        name: Optional name for the composed morphism
        
    Returns:
        Sequentially composed morphism
    """
    return compose(*morphisms, name=name)


def chain(*morphisms: Morphism, name: Optional[str] = None) -> Morphism:
    """
    Another alias for compose() for clarity.
    
    Args:
        *morphisms: Morphisms to chain together
        name: Optional name for the chained morphism
        
    Returns:
        Chained morphism
    """
    return compose(*morphisms, name=name)


def identity_for(array_type: ArrayType, name: Optional[str] = None) -> IdentityMorphism:
    """
    Create identity morphism for a given type.
    
    Args:
        array_type: Type for the identity morphism
        name: Optional name for the identity morphism
        
    Returns:
        Identity morphism
    """
    return IdentityMorphism(array_type, name)


def validate_composition(morphisms: List[Morphism]) -> bool:
    """
    Validate that a list of morphisms can be composed.
    
    Args:
        morphisms: List of morphisms to validate
        
    Returns:
        True if composition is valid, False otherwise
    """
    if len(morphisms) < 2:
        return True
    
    for i in range(len(morphisms) - 1):
        current = morphisms[i]
        next_morphism = morphisms[i + 1]
        if not next_morphism.can_compose_with(current):
            return False
    
    return True


def get_composition_type_signature(morphisms: List[Morphism]) -> str:
    """
    Get the type signature of a composition.
    
    Args:
        morphisms: List of morphisms
        
    Returns:
        String representation of the composition type
    """
    if not morphisms:
        return "Empty"
    
    if len(morphisms) == 1:
        return str(morphisms[0].tensor_type)
    
    input_type = morphisms[0].input_type
    output_type = morphisms[-1].output_type
    return f"{input_type} → {output_type}"


class CompositionBuilder:
    """
    Builder pattern for creating complex compositions.
    
    Example:
        >>> builder = CompositionBuilder()
        >>> model = (builder
        ...     .add(linear(784, 128))
        ...     .add(relu((128,)))
        ...     .add(linear(128, 64))
        ...     .add(relu((64,)))
        ...     .add(linear(64, 10))
        ...     .build("mnist_classifier"))
    """
    
    def __init__(self):
        self.morphisms: List[Morphism] = []
    
    def add(self, morphism: Morphism) -> 'CompositionBuilder':
        """Add a morphism to the composition."""
        if self.morphisms and not morphism.can_compose_with(self.morphisms[-1]):
            raise ValueError(
                f"Cannot add {morphism.input_type} after {self.morphisms[-1].output_type}"
            )
        self.morphisms.append(morphism)
        return self
    
    def add_linear(self, input_dim: int, output_dim: int, 
                  use_bias: bool = True, name: Optional[str] = None) -> 'CompositionBuilder':
        """Add a linear layer."""
        from .morphisms import linear
        return self.add(linear(input_dim, output_dim, use_bias, name))
    
    def add_relu(self, shape: tuple, name: Optional[str] = None) -> 'CompositionBuilder':
        """Add a ReLU activation."""
        from .morphisms import relu
        return self.add(relu(shape, name))
    
    def add_sigmoid(self, shape: tuple, name: Optional[str] = None) -> 'CompositionBuilder':
        """Add a sigmoid activation."""
        from .morphisms import sigmoid
        return self.add(sigmoid(shape, name))
    
    def build(self, name: Optional[str] = None) -> Morphism:
        """Build the final composed morphism."""
        if not self.morphisms:
            raise ValueError("Cannot build empty composition")
        return compose(*self.morphisms, name=name)
    
    def validate(self) -> bool:
        """Validate the current composition."""
        return validate_composition(self.morphisms)
    
    def get_signature(self) -> str:
        """Get the type signature of the current composition."""
        return get_composition_type_signature(self.morphisms)
    
    def clear(self) -> 'CompositionBuilder':
        """Clear the builder."""
        self.morphisms.clear()
        return self
    
    def copy(self) -> 'CompositionBuilder':
        """Create a copy of the builder."""
        new_builder = CompositionBuilder()
        new_builder.morphisms = self.morphisms.copy()
        return new_builder 