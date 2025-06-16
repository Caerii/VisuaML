/**
 * TypeScript interfaces for open-hypergraph API responses
 * These match the backend response formats from visuaml.openhypergraph_export
 */

// Base hypergraph node structure
export interface HypergraphNode {
  id: number;
  name: string;
  op: string;
  target: string;
  shape?: number[];
  dtype?: string;
}

// Hyperedge structure representing operations
export interface Hyperedge {
  id: number;
  operation: string;
  inputs: number[];
  outputs: number[];
  op_type: string;
  target: string;
  args?: unknown[];
  kwargs?: Record<string, unknown>;
}

// Metadata for hypergraph exports
export interface HypergraphMetadata {
  format: string;
  source: string;
  node_count: number;
  hyperedge_count: number;
  export_timestamp?: string | null;
}

// JSON format response
export interface OpenHypergraphJsonResponse {
  nodes: HypergraphNode[];
  hyperedges: Hyperedge[];
  metadata: HypergraphMetadata;
}

// Macro format response
export interface OpenHypergraphMacroResponse {
  macro_syntax: string;
  metadata: HypergraphMetadata;
  format: 'hypersyn-macro';
}

// Categorical analysis structure
export interface CategoricalAnalysis {
  nodes: number;
  hyperedges: number;
  input_arities: number[];
  output_arities: number[];
  max_input_arity: number;
  max_output_arity: number;
  complexity: 'simple' | 'medium' | 'complex';
  conversion_status: string;
  note: string;
}

// Categorical format response
export interface OpenHypergraphCategoricalResponse {
  categorical_analysis: CategoricalAnalysis;
  json_data: OpenHypergraphJsonResponse;
  library_available: boolean;
  framework_ready: boolean;
}

// Union type for all possible responses
export type OpenHypergraphResponse =
  | OpenHypergraphJsonResponse
  | OpenHypergraphMacroResponse
  | OpenHypergraphCategoricalResponse;

// Request schemas
export interface OpenHypergraphExportRequest {
  modelPath: string;
  format: 'json' | 'macro' | 'categorical';
  sampleInputArgs?: string;
  sampleInputDtypes?: string[];
}

// Enhanced import request with export format support
export interface EnhancedImportRequest {
  modelPath: string;
  exportFormat?: string;
  sampleInputArgs?: string;
  sampleInputDtypes?: string[];
}

// Error response structure
export interface HypergraphErrorResponse {
  error: string;
  success: false;
  details?: unknown;
}
