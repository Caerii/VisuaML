/** @fileoverview Utility functions for processing and handling export data from different formats. */

import type { ArchiveFile } from './archiveUtils';

export interface ExportResult {
  format: string;
  name: string;
  success: boolean;
  error?: string;
}

export interface AllExportsData {
  modelPath: string;
  modelName: string;
  timestamp: string;
  exports: {
    json?: unknown;
    macro?: unknown;
    categorical?: unknown;
  };
  results: ExportResult[];
  success: boolean;
  readme: string;
}

interface CategoricalAnalysis {
  nodes: number;
  hyperedges: number;
  input_arities: number[];
  output_arities: number[];
  max_input_arity: number;
  max_output_arity: number;
  complexity: string;
  conversion_status: string;
}

/**
 * Processes export results and converts them to archive files.
 * 
 * @param allResults The complete export results from the API
 * @returns Array of archive files ready for packaging
 */
export const processExportResults = (allResults: AllExportsData): ArchiveFile[] => {
  const files: ArchiveFile[] = [];

  // Add JSON format
  if (allResults.exports.json) {
    files.push({
      name: `${allResults.modelName}_hypergraph.json`,
      content: JSON.stringify(allResults.exports.json, null, 2),
      type: 'application/json'
    });
  }

  // Add Macro format
  if (allResults.exports.macro && typeof allResults.exports.macro === 'object' && 'macro_syntax' in allResults.exports.macro) {
    const macroData = allResults.exports.macro as { macro_syntax?: string };
    if (macroData.macro_syntax) {
      files.push({
        name: `${allResults.modelName}_hypergraph.macro`,
        content: macroData.macro_syntax,
        type: 'text/plain'
      });
    }
  }

  // Add Categorical format with enhanced processing
  if (allResults.exports.categorical) {
    const categoricalContent = processCategoricalExport(allResults.exports.categorical, allResults.modelName);
    files.push({
      name: `${allResults.modelName}_categorical_analysis.json`,
      content: categoricalContent,
      type: 'application/json'
    });
  }

  // Add README
  files.push({
    name: 'README.md',
    content: allResults.readme,
    type: 'text/markdown'
  });

  return files;
};

/**
 * Type guard to check if data has the expected categorical analysis structure.
 */
const isCategoricalAnalysis = (data: unknown): data is CategoricalAnalysis => {
  return typeof data === 'object' && data !== null &&
    'nodes' in data && 'hyperedges' in data && 'input_arities' in data &&
    Array.isArray((data as Record<string, unknown>).input_arities);
};

/**
 * Processes and enhances categorical export data with meaningful analysis.
 * 
 * @param categoricalData Raw categorical export data
 * @param modelName Name of the model for context
 * @returns Enhanced categorical analysis as JSON string
 */
const processCategoricalExport = (categoricalData: unknown, modelName: string): string => {
  if (typeof categoricalData !== 'object' || !categoricalData) {
    return JSON.stringify({ error: 'Invalid categorical data' }, null, 2);
  }

  const data = categoricalData as Record<string, unknown>;
  const analysisData = data.categorical_analysis;
  
  if (!isCategoricalAnalysis(analysisData)) {
    return JSON.stringify({ error: 'No valid categorical analysis found' }, null, 2);
  }

  // Create enhanced analysis with meaningful descriptions
  const enhancedAnalysis = {
    model_info: {
      name: modelName,
      analysis_type: "Categorical Hypergraph Structure",
      timestamp: new Date().toISOString()
    },
    
    structural_summary: {
      total_operations: analysisData.nodes,
      total_connections: analysisData.hyperedges,
      complexity_classification: analysisData.complexity,
      conversion_status: analysisData.conversion_status
    },
    
    operation_analysis: {
      input_arity_distribution: analyzeArityDistribution(analysisData.input_arities),
      max_inputs_per_operation: analysisData.max_input_arity,
      max_outputs_per_operation: analysisData.max_output_arity,
      multi_input_operations: analysisData.input_arities.filter((arity: number) => arity > 1).length,
      interpretation: generateArityInterpretation(analysisData, modelName)
    },
    
    architectural_insights: generateArchitecturalInsights(analysisData, modelName),
    
    mathematical_properties: {
      hypergraph_type: analysisData.max_input_arity > 1 ? "Multi-input hypergraph" : "Simple graph",
      compositionality: analysisData.complexity === "complex" ? "High" : "Low",
      parallelism_potential: analysisData.input_arities.filter((a: number) => a === 0).length > 1 ? "Multiple entry points" : "Single entry point"
    },
    
    raw_analysis: analysisData // Keep original for reference, but move to end
  };

  return JSON.stringify(enhancedAnalysis, null, 2);
};

/**
 * Analyzes the distribution of input arities.
 */
const analyzeArityDistribution = (arities: number[]): Record<string, number> => {
  const distribution: Record<string, number> = {};
  arities.forEach(arity => {
    const key = `arity_${arity}`;
    distribution[key] = (distribution[key] || 0) + 1;
  });
  return distribution;
};

/**
 * Generates interpretation of arity patterns for the specific model.
 */
const generateArityInterpretation = (analysis: CategoricalAnalysis, modelName: string): string[] => {
  const interpretations: string[] = [];
  
  // Analyze input patterns
  const zeroInputs = analysis.input_arities.filter((a: number) => a === 0).length;
  const singleInputs = analysis.input_arities.filter((a: number) => a === 1).length;
  const multiInputs = analysis.input_arities.filter((a: number) => a > 1).length;
  
  if (zeroInputs > 0) {
    interpretations.push(`${zeroInputs} source operations (no inputs) - likely input placeholders or constants`);
  }
  
  if (singleInputs > 0) {
    interpretations.push(`${singleInputs} unary operations - transformations like activations, normalizations, reshaping`);
  }
  
  if (multiInputs > 0) {
    interpretations.push(`${multiInputs} multi-input operations - likely attention mechanisms, residual connections, or tensor operations`);
  }
  
  // Model-specific insights
  if (modelName.toLowerCase().includes('transformer')) {
    if (analysis.max_input_arity >= 3) {
      interpretations.push("High-arity operations suggest complex attention patterns with Q, K, V inputs");
    }
    if (multiInputs >= 3) {
      interpretations.push("Multiple multi-input operations indicate self-attention and feed-forward layers");
    }
  }
  
  return interpretations;
};

/**
 * Generates architectural insights based on the categorical analysis.
 */
const generateArchitecturalInsights = (analysis: CategoricalAnalysis, modelName: string): Record<string, string> => {
  const insights: Record<string, string> = {};
  
  // Complexity analysis
  if (analysis.complexity === "complex") {
    insights.complexity_reason = "Model contains operations with multiple inputs, indicating sophisticated data flow patterns";
  } else {
    insights.complexity_reason = "Model has primarily sequential operations with simple data flow";
  }
  
  // Architecture type detection
  if (modelName.toLowerCase().includes('transformer')) {
    insights.architecture_type = "Transformer-based architecture detected";
    insights.attention_pattern = analysis.max_input_arity >= 3 ? 
      "Multi-head attention with separate Q, K, V processing" : 
      "Simplified attention mechanism";
  } else if (modelName.toLowerCase().includes('cnn') || modelName.toLowerCase().includes('conv')) {
    insights.architecture_type = "Convolutional architecture detected";
    insights.convolution_pattern = analysis.max_input_arity > 1 ? 
      "Complex convolution operations with multiple inputs" : 
      "Standard convolution layers";
  } else {
    insights.architecture_type = "General neural network architecture";
  }
  
  // Data flow analysis
  const avgInputArity = analysis.input_arities.reduce((a: number, b: number) => a + b, 0) / analysis.input_arities.length;
  insights.data_flow_complexity = avgInputArity > 1.5 ? "High" : avgInputArity > 1.0 ? "Medium" : "Low";
  
  return insights;
};

/**
 * Generates a success message for export operations.
 * 
 * @param results Array of export results
 * @param isArchive Whether this is for an archive export
 * @returns Formatted success message
 */
export const generateExportSuccessMessage = (results: ExportResult[], isArchive: boolean = false): string => {
  const successCount = results.filter(r => r.success).length;
  const totalCount = results.length;
  const archiveText = isArchive ? ' as archive' : '';
  
  return `📦 Export All complete! ${successCount}/${totalCount} formats exported${archiveText}`;
};

/**
 * Generates an error message for failed exports.
 * 
 * @param results Array of export results
 * @returns Formatted error message
 */
export const generateExportErrorMessage = (results: ExportResult[]): string => {
  const failedFormats = results.filter(r => !r.success).map(r => r.format).join(', ');
  return `Export All failed for: ${failedFormats}`;
};

/**
 * Logs detailed export results to console.
 * 
 * @param allResults The complete export results
 * @param files Array of processed files
 */
export const logExportResults = (allResults: AllExportsData, files: ArchiveFile[]): void => {
  console.group('📦 Export All Results');
  console.log('🎯 Model:', allResults.modelPath);
  console.log('📊 Results:', allResults.results);
  console.log('📁 Files in archive:', files.map(f => f.name));
  console.log('✅ Success:', allResults.success);
  console.groupEnd();
}; 