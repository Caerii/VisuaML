/** @fileoverview Defines shared TypeScript types and interfaces used by MLNode and its sub-components. This includes types for representing resolved node arguments (including links to source nodes) and the main data structure for MLNode (MLNodeData). */
// visuaml-client/src/ui/nodes/types.ts

// Defines the structure for a resolved argument that points to a source node (or nodes)
export interface ResolvedArgSourceNode {
  source_node?: string; // Name of the direct or single transitive source node
  source_nodes?: string[]; // Names of multiple transitive source nodes
  transitive?: boolean; // True if the source is via one or more filtered nodes
}

// Defines the possible types for a resolved argument value in the node's details
export type ResolvedArgValue =
  | string
  | number
  | boolean
  | null
  | ResolvedArgSourceNode
  | { [key: string]: ResolvedArgValue } // For nested objects
  | ResolvedArgValue[]; // For arrays of resolved arguments

// Defines the structure of the `data` prop for our custom machine learning nodes (MLNode)
// This comes from the backend and is specific to VisuaML's node representation.
export interface MLNodeData {
  label?: string; // Optional: Standard label, can be module name or a more user-friendly name
  target?: string; // From visuaml backend: The target of an FX node (e.g., function name, module path)
  name?: string; // From visuaml backend: The unique name of the FX node
  op?: string; // From visuaml backend: The operation type (e.g., 'call_module', 'placeholder', 'get_attr')
  args?: ResolvedArgValue[]; // From visuaml backend: Resolved positional arguments for the operation
  kwargs?: { [key: string]: ResolvedArgValue }; // From visuaml backend: Resolved keyword arguments
  layerType?: string; // From visuaml backend: A more specific layer type (e.g., 'Dense', 'Conv2d', 'MultiHeadAttn')
  color?: string; // From visuaml backend: An op-specific or layer-specific color for the node
  outputShape?: string; // From visuaml backend: String representation of the node's output tensor shape(s)
  [key: string]: unknown; // Use unknown for better type safety than any
}
