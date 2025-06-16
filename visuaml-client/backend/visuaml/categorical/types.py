"""
Type system for categorical structures.

This module defines the type system used throughout the categorical implementation,
providing proper mathematical foundations for morphisms and compositions.
"""

from dataclasses import dataclass
from typing import Tuple, Optional, Union, List
from enum import Enum


class Dtype(Enum):
    """Data types for categorical arrays."""
    FLOAT32 = "float32"
    FLOAT64 = "float64"
    INT32 = "int32"
    INT64 = "int64"
    BOOL = "bool"
    
    def __str__(self):
        return self.value


@dataclass(frozen=True)
class ArrayType:
    """
    Type for categorical arrays with shape and dtype.
    
    Args:
        shape: Tuple of dimensions, None for dynamic dimensions
        dtype: Data type of the array
    """
    shape: Tuple[Optional[int], ...]
    dtype: Dtype
    
    def __str__(self):
        shape_str = "×".join(str(d) if d is not None else "?" for d in self.shape)
        return f"Array[{shape_str}:{self.dtype.value}]"
    
    def __eq__(self, other):
        if not isinstance(other, ArrayType):
            return False
        return self.shape == other.shape and self.dtype == other.dtype
    
    def __hash__(self):
        return hash((self.shape, self.dtype))
    
    @property
    def ndim(self) -> int:
        """Number of dimensions."""
        return len(self.shape)
    
    @property
    def size(self) -> Optional[int]:
        """Total size if all dimensions are known."""
        if any(d is None for d in self.shape):
            return None
        return int(np.prod(self.shape))
    
    def is_compatible_with(self, other: 'ArrayType') -> bool:
        """Check if this type is compatible with another for composition."""
        if self.dtype != other.dtype:
            return False
        
        if len(self.shape) != len(other.shape):
            return False
        
        for s1, s2 in zip(self.shape, other.shape):
            if s1 is not None and s2 is not None and s1 != s2:
                return False
        
        return True


@dataclass(frozen=True)
class TensorType:
    """
    Type for categorical morphisms (arrows).
    
    Args:
        input_type: Input array type
        output_type: Output array type
    """
    input_type: ArrayType
    output_type: ArrayType
    
    def __str__(self):
        return f"{self.input_type} → {self.output_type}"
    
    def __eq__(self, other):
        if not isinstance(other, TensorType):
            return False
        return self.input_type == other.input_type and self.output_type == other.output_type
    
    def __hash__(self):
        return hash((self.input_type, self.output_type))
    
    def can_compose_with(self, other: 'TensorType') -> bool:
        """Check if this morphism can compose with another (self ∘ other)."""
        return self.input_type.is_compatible_with(other.output_type)


@dataclass(frozen=True)
class ProductType:
    """
    Product type for parallel composition.
    
    Args:
        types: List of array types in the product
    """
    types: Tuple[ArrayType, ...]
    
    def __str__(self):
        return " ⊗ ".join(str(t) for t in self.types)
    
    def __eq__(self, other):
        if not isinstance(other, ProductType):
            return False
        return self.types == other.types
    
    def __hash__(self):
        return hash(self.types)


# Type aliases for convenience
Shape = Tuple[Optional[int], ...]
TypeLike = Union[ArrayType, TensorType, ProductType]


def make_array_type(shape: Shape, dtype: Union[Dtype, str] = Dtype.FLOAT32) -> ArrayType:
    """Convenience function to create ArrayType."""
    if isinstance(dtype, str):
        dtype = Dtype(dtype)
    return ArrayType(shape, dtype)


def make_tensor_type(input_shape: Shape, output_shape: Shape, 
                    dtype: Union[Dtype, str] = Dtype.FLOAT32) -> TensorType:
    """Convenience function to create TensorType."""
    if isinstance(dtype, str):
        dtype = Dtype(dtype)
    input_type = ArrayType(input_shape, dtype)
    output_type = ArrayType(output_shape, dtype)
    return TensorType(input_type, output_type)


def make_product_type(*types: ArrayType) -> ProductType:
    """Convenience function to create ProductType."""
    return ProductType(tuple(types))


# Common types for convenience
SCALAR_F32 = ArrayType((), Dtype.FLOAT32)
VECTOR_F32 = lambda n: ArrayType((n,), Dtype.FLOAT32)
MATRIX_F32 = lambda m, n: ArrayType((m, n), Dtype.FLOAT32)
BATCH_VECTOR_F32 = lambda n: ArrayType((None, n), Dtype.FLOAT32)
BATCH_MATRIX_F32 = lambda m, n: ArrayType((None, m, n), Dtype.FLOAT32)


# Import numpy for size calculation
try:
    import numpy as np
except ImportError:
    # Fallback for size calculation without numpy
    def _prod(iterable):
        result = 1
        for x in iterable:
            result *= x
        return result
    
    class _np:
        @staticmethod
        def prod(x):
            return _prod(x)
    
    np = _np() 