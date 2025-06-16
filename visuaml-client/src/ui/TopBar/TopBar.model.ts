/** @fileoverview Defines the data model for the TopBar component, including available models and API response types. */

import type { Node, Edge } from '@xyflow/react';

// Define structure for API response if not already globally available
export interface ImportApiResponse {
  nodes: Node[];
  edges: Edge[];
}

// Define export format options
export type ExportFormat = 'json' | 'macro' | 'categorical' | 'all';

// Define categorical morphism data structure
export interface CategoricalMorphism {
  id: string;
  name: string;
  type: string; // 'linear', 'activation', 'composed', etc.
  input_type: string;
  output_type: string;
  parameters?: Record<string, unknown>;
}

// Define categorical hypergraph structure
export interface CategoricalHypergraph {
  name: string;
  hyperedges: Record<
    string,
    {
      id: string;
      type: string;
      inputs: string[];
      outputs: string[];
      parameters?: Record<string, unknown>;
    }
  >;
  wires: Record<
    string,
    {
      id: string;
      type: string;
      source?: string;
      targets: string[];
    }
  >;
  input_wires: string[];
  output_wires: string[];
  input_types: string[];
  output_types: string[];
}

// Define structure for open-hypergraph export response
export interface ExportHypergraphResponse {
  // JSON format fields
  nodes?: Record<string, unknown>[];
  hyperedges?: Record<string, unknown>[];
  metadata?: Record<string, unknown>;

  // Macro format fields
  macro?: string;
  macro_syntax?: string; // Backend uses this field name
  json_representation?: Record<string, unknown>;

  // Categorical format fields (enhanced)
  open_hypergraph?: unknown; // OpenHypergraph object (not serializable to JSON)
  categorical_hypergraph?: CategoricalHypergraph;
  morphisms?: CategoricalMorphism[];
  composition_chain?: string[]; // List of morphism names in composition order
  type_signature?: string; // Overall input -> output type signature

  // Backend categorical response fields
  json_data?: {
    nodes: Record<string, unknown>[];
    hyperedges: Record<string, unknown>[];
    metadata: Record<string, unknown>;
  };
  library_available?: boolean;
  framework_ready?: boolean;

  // Common fields
  success: boolean;
  message?: string;
  categorical_available?: boolean;

  // Bidirectional conversion support
  bidirectional?: boolean;

  // Enhanced categorical analysis (frontend format)
  categorical_analysis?: {
    total_morphisms: number;
    composition_depth: number;
    type_safety_validated: boolean;
    hypergraph_statistics: {
      hyperedges: number;
      wires: number;
      input_boundary_size: number;
      output_boundary_size: number;
    };
    // Backend format fields
    nodes?: number;
    hyperedges?: number;
    input_arities?: number[];
    output_arities?: number[];
    max_input_arity?: number;
    max_output_arity?: number;
    complexity?: string;
    conversion_status?: string;
    note?: string;
  };
}

// Model configuration with export compatibility info
export interface ModelConfig {
  value: string;
  label: string;
  category: 'original' | 'fixed' | 'working';
  sampleInputArgs?: string;
  sampleInputDtypes?: string[];
  description?: string;
  exportCompatible: boolean;
}

// Hardcoded model options for testing
export const AVAILABLE_MODELS: ModelConfig[] = [
  // Working original models
  {
    value: 'models.SimpleNN',
    label: 'SimpleNN',
    category: 'working',
    sampleInputArgs: '((10,),)',
    exportCompatible: true,
    description: 'Simple feedforward neural network',
  },
  {
    value: 'models.Autoencoder',
    label: 'Autoencoder',
    category: 'working',
    sampleInputArgs: '((10,),)',
    exportCompatible: true,
    description: 'Basic autoencoder architecture',
  },
  {
    value: 'models.MyTinyGPT',
    label: 'MyTinyGPT',
    category: 'working',
    exportCompatible: true,
    description: 'Tiny GPT transformer model',
  },
  {
    value: 'models.TestModel',
    label: 'TestModel',
    category: 'working',
    exportCompatible: true,
    description: 'Test model for validation',
  },
  {
    value: 'models.TransformerBlock',
    label: 'TransformerBlock',
    category: 'working',
    exportCompatible: true,
    description: 'Single transformer block',
  },

  // Fixed models (these work with open-hypergraph export)
  {
    value: 'models.FixedSimpleCNN',
    label: 'FixedSimpleCNN ✅',
    category: 'fixed',
    sampleInputArgs: '((1, 1, 28, 28),)',
    exportCompatible: true,
    description: 'Fixed CNN with correct input shapes',
  },
  {
    value: 'models.FixedBasicRNN',
    label: 'FixedBasicRNN ✅',
    category: 'fixed',
    sampleInputArgs: '((1, 10, 10),)',
    exportCompatible: true,
    description: 'Fixed RNN without dynamic operations',
  },
  {
    value: 'models.FixedDemoNet',
    label: 'FixedDemoNet ✅',
    category: 'fixed',
    sampleInputArgs: '((1, 32),)',
    sampleInputDtypes: ['long'],
    exportCompatible: true,
    description: 'Fixed embedding model with correct input types',
  },

  // Original problematic models (for comparison)
  {
    value: 'models.SimpleCNN',
    label: 'SimpleCNN ❌',
    category: 'original',
    sampleInputArgs: '((1, 28, 28),)', // Wrong shape - will fail
    exportCompatible: false,
    description: 'Original CNN with shape issues',
  },
  {
    value: 'models.BasicRNN',
    label: 'BasicRNN ❌',
    category: 'original',
    sampleInputArgs: '((1, 10, 10),)',
    exportCompatible: false,
    description: 'Original RNN with dynamic operations',
  },
  {
    value: 'models.DemoNet',
    label: 'DemoNet ❌',
    category: 'original',
    sampleInputArgs: '((3, 32, 32),)', // Wrong type - will fail
    exportCompatible: false,
    description: 'Original embedding model with type issues',
  },
];
