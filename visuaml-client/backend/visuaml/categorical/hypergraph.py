"""
Categorical hypergraph implementation.

This module provides the CategoricalHypergraph class that represents
open hypergraphs with proper categorical structure.
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import json
from .morphisms import Morphism
from .types import ArrayType


@dataclass
class HyperEdge:
    """
    Represents a hyperedge in the categorical hypergraph.
    
    Args:
        id: Unique identifier for the hyperedge
        type: Type of operation (e.g., 'linear', 'relu', 'add')
        inputs: List of input wire IDs
        outputs: List of output wire IDs
        parameters: Optional parameters for the operation
    """
    id: str
    type: str
    inputs: List[str]
    outputs: List[str]
    parameters: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.parameters is None:
            self.parameters = {}


@dataclass
class Wire:
    """
    Represents a wire in the categorical hypergraph.
    
    Args:
        id: Unique identifier for the wire
        type: Type information for the wire
        source: Optional source hyperedge ID
        targets: List of target hyperedge IDs
    """
    id: str
    type: str
    source: Optional[str] = None
    targets: Optional[List[str]] = None
    
    def __post_init__(self):
        if self.targets is None:
            self.targets = []


class CategoricalHypergraph:
    """
    Categorical hypergraph with proper input/output boundaries.
    
    This class represents an open hypergraph that can be composed
    with other hypergraphs following categorical rules.
    """
    
    def __init__(self, name: Optional[str] = None):
        self.name = name or "CategoricalHypergraph"
        self.hyperedges: Dict[str, HyperEdge] = {}
        self.wires: Dict[str, Wire] = {}
        self.input_wires: List[str] = []
        self.output_wires: List[str] = []
        self.input_types: List[ArrayType] = []
        self.output_types: List[ArrayType] = []
    
    def add_hyperedge(self, hyperedge: HyperEdge) -> None:
        """Add a hyperedge to the graph."""
        if hyperedge.id in self.hyperedges:
            raise ValueError(f"Hyperedge {hyperedge.id} already exists")
        self.hyperedges[hyperedge.id] = hyperedge
        
        # Add wires for inputs and outputs if they don't exist
        for wire_id in hyperedge.inputs + hyperedge.outputs:
            if wire_id not in self.wires:
                self.wires[wire_id] = Wire(wire_id, "unknown")
    
    def add_wire(self, wire: Wire) -> None:
        """Add a wire to the graph."""
        if wire.id in self.wires:
            raise ValueError(f"Wire {wire.id} already exists")
        self.wires[wire.id] = wire
    
    def set_input_boundary(self, wire_ids: List[str], types: List[ArrayType]) -> None:
        """Set the input boundary of the hypergraph."""
        if len(wire_ids) != len(types):
            raise ValueError("Number of input wires must match number of types")
        
        for wire_id in wire_ids:
            if wire_id not in self.wires:
                self.wires[wire_id] = Wire(wire_id, str(types[wire_ids.index(wire_id)]))
        
        self.input_wires = wire_ids
        self.input_types = types
    
    def set_output_boundary(self, wire_ids: List[str], types: List[ArrayType]) -> None:
        """Set the output boundary of the hypergraph."""
        if len(wire_ids) != len(types):
            raise ValueError("Number of output wires must match number of types")
        
        for wire_id in wire_ids:
            if wire_id not in self.wires:
                self.wires[wire_id] = Wire(wire_id, str(types[wire_ids.index(wire_id)]))
        
        self.output_wires = wire_ids
        self.output_types = types
    
    def compose_sequential(self, other: 'CategoricalHypergraph') -> 'CategoricalHypergraph':
        """
        Compose this hypergraph with another sequentially (self ∘ other).
        
        Args:
            other: Hypergraph to compose with
            
        Returns:
            New composed hypergraph
        """
        if len(self.input_types) != len(other.output_types):
            raise ValueError("Cannot compose: incompatible boundaries")
        
        for i, (self_type, other_type) in enumerate(zip(self.input_types, other.output_types)):
            if not self_type.is_compatible_with(other_type):
                raise ValueError(f"Incompatible types at position {i}: {self_type} vs {other_type}")
        
        # Create new hypergraph
        result = CategoricalHypergraph(f"{self.name}_compose_{other.name}")
        
        # Add all hyperedges from both graphs
        for edge in other.hyperedges.values():
            result.add_hyperedge(edge)
        
        for edge in self.hyperedges.values():
            result.add_hyperedge(edge)
        
        # Add all wires from both graphs
        for wire in other.wires.values():
            result.add_wire(wire)
        
        for wire in self.wires.values():
            result.add_wire(wire)
        
        # Connect the boundaries
        for i, (self_input, other_output) in enumerate(zip(self.input_wires, other.output_wires)):
            # Create connection wire
            connection_id = f"connection_{i}"
            connection_wire = Wire(connection_id, str(self.input_types[i]))
            result.add_wire(connection_wire)
            
            # Update wire connections
            result.wires[other_output].targets.append(connection_id)
            result.wires[self_input].source = connection_id
        
        # Set boundaries
        result.set_input_boundary(other.input_wires, other.input_types)
        result.set_output_boundary(self.output_wires, self.output_types)
        
        return result
    
    def compose_parallel(self, other: 'CategoricalHypergraph') -> 'CategoricalHypergraph':
        """
        Compose this hypergraph with another in parallel (self ⊗ other).
        
        Args:
            other: Hypergraph to compose with
            
        Returns:
            New parallel composed hypergraph
        """
        # Create new hypergraph
        result = CategoricalHypergraph(f"{self.name}_parallel_{other.name}")
        
        # Add all hyperedges from both graphs
        for edge in self.hyperedges.values():
            result.add_hyperedge(edge)
        
        for edge in other.hyperedges.values():
            result.add_hyperedge(edge)
        
        # Add all wires from both graphs
        for wire in self.wires.values():
            result.add_wire(wire)
        
        for wire in other.wires.values():
            result.add_wire(wire)
        
        # Combine boundaries
        combined_input_wires = self.input_wires + other.input_wires
        combined_output_wires = self.output_wires + other.output_wires
        combined_input_types = self.input_types + other.input_types
        combined_output_types = self.output_types + other.output_types
        
        result.set_input_boundary(combined_input_wires, combined_input_types)
        result.set_output_boundary(combined_output_wires, combined_output_types)
        
        return result
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert hypergraph to dictionary representation."""
        return {
            "name": self.name,
            "hyperedges": [
                {
                    "id": edge.id,
                    "type": edge.type,
                    "inputs": edge.inputs,
                    "outputs": edge.outputs,
                    "parameters": edge.parameters
                }
                for edge in self.hyperedges.values()
            ],
            "wires": [
                {
                    "id": wire.id,
                    "type": wire.type,
                    "source": wire.source,
                    "targets": wire.targets
                }
                for wire in self.wires.values()
            ],
            "input_boundary": {
                "wires": self.input_wires,
                "types": [str(t) for t in self.input_types]
            },
            "output_boundary": {
                "wires": self.output_wires,
                "types": [str(t) for t in self.output_types]
            }
        }
    
    def to_json(self, indent: Optional[int] = 2) -> str:
        """Convert hypergraph to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)
    
    @classmethod
    def from_morphism(cls, morphism: Morphism) -> 'CategoricalHypergraph':
        """
        Create a categorical hypergraph from a morphism.
        
        Args:
            morphism: Morphism to convert
            
        Returns:
            Categorical hypergraph representation
        """
        hg_data = morphism.to_hypergraph()
        
        # Create hypergraph
        hypergraph = cls(hg_data.get("name", morphism.name))
        
        # Add hyperedges
        for edge_data in hg_data.get("hyperedges", []):
            edge = HyperEdge(
                id=edge_data["id"],
                type=edge_data["type"],
                inputs=edge_data["inputs"],
                outputs=edge_data["outputs"],
                parameters=edge_data.get("parameters", {})
            )
            hypergraph.add_hyperedge(edge)
        
        # Set boundaries
        input_wires = hg_data.get("inputs", [])
        output_wires = hg_data.get("outputs", [])
        
        if input_wires:
            hypergraph.set_input_boundary(input_wires, [morphism.input_type])
        
        if output_wires:
            hypergraph.set_output_boundary(output_wires, [morphism.output_type])
        
        return hypergraph
    
    def validate(self) -> Tuple[bool, List[str]]:
        """
        Validate the hypergraph structure.
        
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        # Check that all wire references in hyperedges exist
        for edge in self.hyperedges.values():
            for wire_id in edge.inputs + edge.outputs:
                if wire_id not in self.wires:
                    errors.append(f"Wire {wire_id} referenced in edge {edge.id} does not exist")
        
        # Check that input/output boundaries reference existing wires
        for wire_id in self.input_wires:
            if wire_id not in self.wires:
                errors.append(f"Input boundary wire {wire_id} does not exist")
        
        for wire_id in self.output_wires:
            if wire_id not in self.wires:
                errors.append(f"Output boundary wire {wire_id} does not exist")
        
        # Check for cycles (simplified check)
        visited = set()
        rec_stack = set()
        
        def has_cycle(wire_id: str) -> bool:
            if wire_id in rec_stack:
                return True
            if wire_id in visited:
                return False
            
            visited.add(wire_id)
            rec_stack.add(wire_id)
            
            wire = self.wires.get(wire_id)
            if wire and wire.targets:
                for target_id in wire.targets:
                    if has_cycle(target_id):
                        return True
            
            rec_stack.remove(wire_id)
            return False
        
        for wire_id in self.wires:
            if wire_id not in visited:
                if has_cycle(wire_id):
                    errors.append(f"Cycle detected involving wire {wire_id}")
                    break
        
        return len(errors) == 0, errors
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the hypergraph."""
        edge_types = {}
        for edge in self.hyperedges.values():
            edge_types[edge.type] = edge_types.get(edge.type, 0) + 1
        
        return {
            "num_hyperedges": len(self.hyperedges),
            "num_wires": len(self.wires),
            "num_inputs": len(self.input_wires),
            "num_outputs": len(self.output_wires),
            "edge_types": edge_types,
            "is_valid": self.validate()[0]
        }
    
    def __str__(self):
        return f"CategoricalHypergraph({self.name}, {len(self.hyperedges)} edges, {len(self.wires)} wires)"
    
    def __repr__(self):
        return f"CategoricalHypergraph(name='{self.name}', hyperedges={len(self.hyperedges)}, wires={len(self.wires)})" 