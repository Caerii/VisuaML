"""
Morphism system for categorical structures.

This module defines the base morphism classes and specific morphism types
that represent categorical arrows in our hypergraph system.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import uuid
from .types import ArrayType, TensorType, Dtype


class Morphism(ABC):
    """
    Base class for categorical morphisms (arrows).
    
    A morphism represents a categorical arrow with well-defined input and output types.
    Morphisms can be composed using the @ operator (categorical composition).
    """
    
    def __init__(self, input_type: ArrayType, output_type: ArrayType, name: Optional[str] = None):
        self.input_type = input_type
        self.output_type = output_type
        self.name = name or f"{self.__class__.__name__}_{uuid.uuid4().hex[:8]}"
        self.tensor_type = TensorType(input_type, output_type)
    
    @abstractmethod
    def forward(self, inputs: List[Any]) -> List[Any]:
        """
        Forward computation of the morphism.
        
        Args:
            inputs: List of input arrays
            
        Returns:
            List of output arrays
        """
        pass
    
    @abstractmethod
    def to_hypergraph(self) -> Dict[str, Any]:
        """
        Convert morphism to open hypergraph representation.
        
        Returns:
            Dictionary representing the hypergraph structure
        """
        pass
    
    def __matmul__(self, other: 'Morphism') -> 'Morphism':
        """
        Categorical composition via @ operator (self ∘ other).
        
        Args:
            other: Morphism to compose with
            
        Returns:
            Composed morphism
        """
        return ComposedMorphism(other, self)
    
    def __str__(self):
        return f"{self.name}: {self.input_type} → {self.output_type}"
    
    def __repr__(self):
        return f"{self.__class__.__name__}({self.input_type}, {self.output_type}, name='{self.name}')"
    
    def can_compose_with(self, other: 'Morphism') -> bool:
        """Check if this morphism can compose with another."""
        return self.input_type.is_compatible_with(other.output_type)
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get morphism parameters for serialization."""
        return {
            "name": self.name,
            "input_type": str(self.input_type),
            "output_type": str(self.output_type),
            "type": self.__class__.__name__
        }


class LinearMorphism(Morphism):
    """
    Linear transformation morphism (matrix multiplication).
    
    Represents f(x) = Wx + b where W is a weight matrix and b is bias.
    """
    
    def __init__(self, input_dim: int, output_dim: int, 
                 use_bias: bool = True, name: Optional[str] = None):
        input_type = ArrayType((input_dim,), Dtype.FLOAT32)
        output_type = ArrayType((output_dim,), Dtype.FLOAT32)
        super().__init__(input_type, output_type, name)
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = use_bias
        self.weight_shape = (output_dim, input_dim)
        self.bias_shape = (output_dim,) if use_bias else None
    
    def forward(self, inputs: List[Any]) -> List[Any]:
        """Forward pass: y = Wx + b"""
        # This would be implemented with actual tensor operations
        # For now, we just return the structure
        return [f"linear({inputs[0]}, weight={self.weight_shape}, bias={self.bias_shape})"]
    
    def to_hypergraph(self) -> Dict[str, Any]:
        """Convert to hypergraph representation."""
        return {
            "type": "linear",
            "name": self.name,
            "input_type": str(self.input_type),
            "output_type": str(self.output_type),
            "parameters": {
                "input_dim": self.input_dim,
                "output_dim": self.output_dim,
                "weight_shape": self.weight_shape,
                "bias_shape": self.bias_shape,
                "use_bias": self.use_bias
            },
            "hyperedges": [
                {
                    "id": f"{self.name}_weight",
                    "type": "parameter",
                    "shape": self.weight_shape,
                    "inputs": [],
                    "outputs": [f"{self.name}_weight_out"]
                },
                {
                    "id": f"{self.name}_matmul",
                    "type": "matmul",
                    "inputs": [f"{self.name}_input", f"{self.name}_weight_out"],
                    "outputs": [f"{self.name}_matmul_out"]
                }
            ] + ([{
                "id": f"{self.name}_bias",
                "type": "parameter", 
                "shape": self.bias_shape,
                "inputs": [],
                "outputs": [f"{self.name}_bias_out"]
            }, {
                "id": f"{self.name}_add",
                "type": "add",
                "inputs": [f"{self.name}_matmul_out", f"{self.name}_bias_out"],
                "outputs": [f"{self.name}_output"]
            }] if self.use_bias else [{
                "id": f"{self.name}_identity",
                "type": "identity",
                "inputs": [f"{self.name}_matmul_out"],
                "outputs": [f"{self.name}_output"]
            }]),
            "inputs": [f"{self.name}_input"],
            "outputs": [f"{self.name}_output"]
        }


class ActivationMorphism(Morphism):
    """
    Activation function morphism.
    
    Represents element-wise activation functions like ReLU, sigmoid, etc.
    """
    
    def __init__(self, shape: tuple, activation: str = "relu", name: Optional[str] = None):
        array_type = ArrayType(shape, Dtype.FLOAT32)
        super().__init__(array_type, array_type, name)
        self.activation = activation
        self.shape = shape
    
    def forward(self, inputs: List[Any]) -> List[Any]:
        """Forward pass: y = activation(x)"""
        return [f"{self.activation}({inputs[0]})"]
    
    def to_hypergraph(self) -> Dict[str, Any]:
        """Convert to hypergraph representation."""
        return {
            "type": "activation",
            "name": self.name,
            "input_type": str(self.input_type),
            "output_type": str(self.output_type),
            "parameters": {
                "activation": self.activation,
                "shape": self.shape
            },
            "hyperedges": [{
                "id": f"{self.name}_activation",
                "type": self.activation,
                "inputs": [f"{self.name}_input"],
                "outputs": [f"{self.name}_output"]
            }],
            "inputs": [f"{self.name}_input"],
            "outputs": [f"{self.name}_output"]
        }


class ComposedMorphism(Morphism):
    """
    Composition of two morphisms (first ∘ second).
    
    Represents the categorical composition of morphisms.
    """
    
    def __init__(self, first: Morphism, second: Morphism, name: Optional[str] = None):
        if not second.can_compose_with(first):
            raise ValueError(
                f"Cannot compose morphisms: {second.output_type} ≠ {first.input_type}"
            )
        
        super().__init__(first.input_type, second.output_type, name)
        self.first = first
        self.second = second
    
    def forward(self, inputs: List[Any]) -> List[Any]:
        """Forward pass: second(first(inputs))"""
        intermediate = self.first.forward(inputs)
        return self.second.forward(intermediate)
    
    def to_hypergraph(self) -> Dict[str, Any]:
        """Convert to hypergraph representation."""
        first_hg = self.first.to_hypergraph()
        second_hg = self.second.to_hypergraph()
        
        # Merge hypergraphs and connect intermediate outputs to inputs
        return {
            "type": "composition",
            "name": self.name,
            "input_type": str(self.input_type),
            "output_type": str(self.output_type),
            "components": {
                "first": first_hg,
                "second": second_hg
            },
            "hyperedges": first_hg["hyperedges"] + second_hg["hyperedges"] + [{
                "id": f"{self.name}_connection",
                "type": "wire",
                "inputs": first_hg["outputs"],
                "outputs": second_hg["inputs"]
            }],
            "inputs": first_hg["inputs"],
            "outputs": second_hg["outputs"]
        }


class IdentityMorphism(Morphism):
    """Identity morphism (no-op)."""
    
    def __init__(self, array_type: ArrayType, name: Optional[str] = None):
        super().__init__(array_type, array_type, name)
    
    def forward(self, inputs: List[Any]) -> List[Any]:
        """Forward pass: y = x"""
        return inputs
    
    def to_hypergraph(self) -> Dict[str, Any]:
        """Convert to hypergraph representation."""
        return {
            "type": "identity",
            "name": self.name,
            "input_type": str(self.input_type),
            "output_type": str(self.output_type),
            "hyperedges": [{
                "id": f"{self.name}_identity",
                "type": "identity",
                "inputs": [f"{self.name}_input"],
                "outputs": [f"{self.name}_output"]
            }],
            "inputs": [f"{self.name}_input"],
            "outputs": [f"{self.name}_output"]
        }


class ParallelMorphism(Morphism):
    """
    Parallel composition of morphisms (tensor product).
    
    Represents f ⊗ g where inputs and outputs are combined.
    """
    
    def __init__(self, morphisms: List[Morphism], name: Optional[str] = None):
        if not morphisms:
            raise ValueError("Cannot create parallel morphism with no components")
        
        # For simplicity, we'll concatenate the shapes
        # In a full implementation, this would use proper tensor products
        input_shapes = [m.input_type.shape for m in morphisms]
        output_shapes = [m.output_type.shape for m in morphisms]
        
        # Concatenate along first dimension
        input_dim = sum(shape[0] or 0 for shape in input_shapes)
        output_dim = sum(shape[0] or 0 for shape in output_shapes)
        
        input_type = ArrayType((input_dim,), Dtype.FLOAT32)
        output_type = ArrayType((output_dim,), Dtype.FLOAT32)
        
        super().__init__(input_type, output_type, name)
        self.morphisms = morphisms
    
    def forward(self, inputs: List[Any]) -> List[Any]:
        """Forward pass: parallel application"""
        results = []
        for i, morphism in enumerate(self.morphisms):
            result = morphism.forward([f"split_{i}({inputs[0]})"])
            results.extend(result)
        return [f"concat({results})"]
    
    def to_hypergraph(self) -> Dict[str, Any]:
        """Convert to hypergraph representation."""
        component_hgs = [m.to_hypergraph() for m in self.morphisms]
        
        return {
            "type": "parallel",
            "name": self.name,
            "input_type": str(self.input_type),
            "output_type": str(self.output_type),
            "components": {f"component_{i}": hg for i, hg in enumerate(component_hgs)},
            "hyperedges": [
                # Split input
                {
                    "id": f"{self.name}_split",
                    "type": "split",
                    "inputs": [f"{self.name}_input"],
                    "outputs": [f"{self.name}_split_{i}" for i in range(len(self.morphisms))]
                }
            ] + [
                # Component hyperedges
                edge for hg in component_hgs for edge in hg["hyperedges"]
            ] + [
                # Concat outputs
                {
                    "id": f"{self.name}_concat",
                    "type": "concat",
                    "inputs": [hg["outputs"][0] for hg in component_hgs],
                    "outputs": [f"{self.name}_output"]
                }
            ],
            "inputs": [f"{self.name}_input"],
            "outputs": [f"{self.name}_output"]
        }


# Convenience functions for creating common morphisms
def linear(input_dim: int, output_dim: int, use_bias: bool = True, 
          name: Optional[str] = None) -> LinearMorphism:
    """Create a linear morphism."""
    return LinearMorphism(input_dim, output_dim, use_bias, name)


def relu(shape: tuple, name: Optional[str] = None) -> ActivationMorphism:
    """Create a ReLU activation morphism."""
    return ActivationMorphism(shape, "relu", name)


def sigmoid(shape: tuple, name: Optional[str] = None) -> ActivationMorphism:
    """Create a sigmoid activation morphism."""
    return ActivationMorphism(shape, "sigmoid", name)


def identity(array_type: ArrayType, name: Optional[str] = None) -> IdentityMorphism:
    """Create an identity morphism."""
    return IdentityMorphism(array_type, name) 