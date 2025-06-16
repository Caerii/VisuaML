"""Module for filtering graph nodes based on operation types and patterns."""

from typing import Set, List, Optional, Dict
from dataclasses import dataclass, field
from torch.fx import Node, GraphModule


@dataclass
class FilterConfig:
    """Configuration for graph filtering."""
    
    # Operations to filter out
    filtered_ops: Set[str] = field(default_factory=lambda: {
        'getitem',  # Indexing operations like tensor[0] or tuple unpacking
        'getattr',  # Attribute access (unless it's a parameter/buffer)
    })
    
    # Operation categories for different abstraction levels
    operation_categories: Dict[str, Set[str]] = field(default_factory=lambda: {
        'low_level': {'getitem', 'getattr', 'add', 'mul', 'sub', 'div'},
        'control_flow': {'if', 'while', 'for'},
        'tensor_ops': {'view', 'reshape', 'permute', 'transpose', 'contiguous'},
        'ml_core': {'Linear', 'Conv2d', 'LSTM', 'GRU', 'Attention', 'Embedding'},
    })
    
    # Abstraction level (0 = show all, higher = more filtered)
    abstraction_level: int = 1
    
    def should_filter_op(self, op_name: str) -> bool:
        """Check if an operation should be filtered based on config."""
        if self.abstraction_level == 0:
            return False
            
        # Level 1: Filter low-level ops
        if self.abstraction_level >= 1 and op_name.lower() in self.filtered_ops:
            return True
            
        # Level 2: Also filter tensor operations
        if self.abstraction_level >= 2 and op_name in self.operation_categories['tensor_ops']:
            return True
            
        return False


class GraphFilter:
    """Handles filtering of FX graph nodes."""
    
    def __init__(self, config: Optional[FilterConfig] = None):
        self.config = config or FilterConfig()
    
    def should_include_node(self, node: Node, graph_module: GraphModule) -> bool:
        """
        Determine if a node should be included in the visualization.
        
        Args:
            node: The FX node to check
            graph_module: The graph module containing the node
            
        Returns:
            bool: True if node should be included
        """
        # Always include inputs and outputs
        if node.op in ('placeholder', 'output'):
            return True
        
        # Always include module calls (these are the main ML operations)
        if node.op == 'call_module':
            return True
        
        # Check function/method calls against filter
        if node.op in ('call_function', 'call_method'):
            target_name = getattr(node.target, '__name__', str(node.target))
            if self.config.should_filter_op(target_name):
                return False
        
        # Filter get_attr based on config
        if node.op == 'get_attr':
            # TODO: Could check if target is a parameter/buffer name
            # For now, filter all getattr at level 1+
            return self.config.abstraction_level == 0
        
        # Include everything else by default
        return True
    
    def get_transitive_inputs(
        self, 
        node: Node, 
        included_nodes: Set[Node], 
        visited: Optional[Set[Node]] = None
    ) -> List[Node]:
        """
        Get all included input nodes, traversing through filtered nodes.
        
        This enables smart edge routing when intermediate nodes are filtered.
        
        Args:
            node: Starting node
            included_nodes: Set of nodes that are included in the graph
            visited: Set of already visited nodes (for cycle detection)
            
        Returns:
            List of included input nodes
        """
        if visited is None:
            visited = set()
        
        if node in visited:
            return []
        visited.add(node)
        
        # If this node is included, return it
        if node in included_nodes:
            return [node]
        
        # Otherwise, trace through its inputs
        result = []
        for inp in node.all_input_nodes:
            result.extend(self.get_transitive_inputs(inp, included_nodes, visited))
        
        return result
    
    def filter_graph(
        self, 
        nodes: List[Node], 
        graph_module: GraphModule
    ) -> Set[Node]:
        """
        Filter a list of nodes based on the current configuration.
        
        Args:
            nodes: List of FX nodes to filter
            graph_module: The graph module containing the nodes
            
        Returns:
            Set of nodes that should be included
        """
        included_nodes = set()
        
        for node in nodes:
            if self.should_include_node(node, graph_module):
                included_nodes.add(node)
                
        return included_nodes 