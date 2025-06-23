/** @fileoverview Controller hook for the TopBar component. It manages UI state, triggers model imports, and coordinates updates to global state (Y.js, Zustand). */
import { useState, useCallback } from 'react';
import * as Y from 'yjs';
import { useYDoc } from '../../y/DocProvider';
import { autoLayout, type Node, type Edge } from '../../lib/autoLayout';
import { importModel, exportModelHypergraph, exportAllFormats, uploadModel } from '../../lib/api';
import { toast } from 'sonner';
import { AVAILABLE_MODELS, type ExportFormat, type ExportHypergraphResponse } from './TopBar.model';
import {
  createArchiveFile,
  downloadBlob,
  createArchiveFilename,
  type ArchiveMetadata,
} from '../../lib/archiveUtils';
import {
  processExportResults,
  generateExportSuccessMessage,
  generateExportErrorMessage,
  logExportResults,
} from '../../lib/exportUtils';

// Global categorical panel state (we'll use a simple event system)
let categoricalPanelCallback: ((data: ExportHypergraphResponse) => void) | null = null;

export const setCategoricalPanelCallback = (callback: (data: ExportHypergraphResponse) => void) => {
  categoricalPanelCallback = callback;
};

export const useTopBar = () => {
  const [modelPath, setModelPath] = useState(AVAILABLE_MODELS[0].value);
  const [isLoadingUI, setIsLoadingUI] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [isExporting, setIsExporting] = useState(false);
  const [exportFormat, setExportFormat] = useState<ExportFormat>('json');
  const { ydoc } = useYDoc();

  const processImportedData = useCallback(
    (
      importedData: { nodes: Node[]; edges: Edge[] },
      modelName: string,
    ) => {
      // --- ATOMIC GRAPH RESET AND UPDATE ---
      // This is the single source of truth for updating the graph.
      // It happens in one transaction to ensure consistency across all clients.
      ydoc.transact(() => {
        // 1. Clear all previous graph data
        const yNodes = ydoc.getArray<Y.Map<unknown>>('nodes');
        const yEdges = ydoc.getArray<Y.Map<unknown>>('edges');
        yNodes.delete(0, yNodes.length);
        yEdges.delete(0, yEdges.length);

        // 2. Add new graph data
        const laidOutData = autoLayout(importedData.nodes, importedData.edges);
        laidOutData.nodes.forEach(node => {
          yNodes.push([new Y.Map(Object.entries(node))]);
        });
        laidOutData.edges.forEach(edge => {
          yEdges.push([new Y.Map(Object.entries(edge))]);
        });
        
        // 3. Set the authoritative network facts
        const inputNodes = laidOutData.nodes.filter((n) => n.data.op === 'placeholder');
        const inputShapes = inputNodes.map(
          (n) => n.data.outputShape || 'N/A'
        );
        const componentTypes = [
          ...new Set(laidOutData.nodes.map((n) => n.data.layerType || 'unknown')),
        ];
        const ySharedFacts = ydoc.getMap('sharedNetworkFacts');
        ySharedFacts.set('networkName', modelName);
        ySharedFacts.set('numNodes', laidOutData.nodes.length);
        ySharedFacts.set('numEdges', laidOutData.edges.length);
        ySharedFacts.set('inputShapes', inputShapes);
        ySharedFacts.set('componentTypes', componentTypes);
        ySharedFacts.set('isLoadingGraph', false);
      });

      toast.success(`Model '${modelName}' imported successfully!`);
    },
    [ydoc],
  );

  const handleImportError = useCallback(
    (error: unknown) => {
      console.error('Failed to import model:', error);
      const message =
        error instanceof Error ? error.message : 'An unknown error occurred during import.';
      toast.error(message, { duration: 8000 }); // Increased duration for complex errors

      const ySharedFacts = ydoc.getMap('sharedNetworkFacts');
      ydoc.transact(() => {
        ySharedFacts.set('networkName', 'Failed to load');
        ySharedFacts.set('numNodes', 0);
        ySharedFacts.set('numEdges', 0);
        ySharedFacts.set('inputShapes', []);
        ySharedFacts.set('componentTypes', []);
        ySharedFacts.set('isLoadingGraph', false);

        const yNameToClear = ydoc.getText('networkNameShared');
        if (yNameToClear.length > 0) {
          yNameToClear.delete(0, yNameToClear.length);
        }
      });
    },
    [ydoc],
  );

  const handleImportClick = async () => {
    setIsLoadingUI(true);
    const modelDetails = AVAILABLE_MODELS.find((m) => m.value === modelPath);
    const localNetworkName = modelDetails ? modelDetails.label : modelPath;

    // Set loading state for all clients
    const ySharedFacts = ydoc.getMap('sharedNetworkFacts');
    ySharedFacts.set('isLoadingGraph', true);
    
    try {
      const importedData = await importModel(modelPath);
      // Let the processor handle the definitive state update
      processImportedData(importedData, localNetworkName);
    } catch (err) {
      handleImportError(err);
    } finally {
      setIsLoadingUI(false);
    }
  };

  const handleFileUpload = async (file: File) => {
    if (!file) return;
    if (!file.name.endsWith('.py')) {
      toast.error('Invalid file type. Please upload a .py file.');
      return;
    }

    setIsUploading(true);
    const localNetworkName = file.name;
    
    // Set loading state for all clients
    const ySharedFacts = ydoc.getMap('sharedNetworkFacts');
    ySharedFacts.set('isLoadingGraph', true);

    try {
      const importedData = await uploadModel(file);
      // Let the processor handle the definitive state update
      processImportedData(importedData, localNetworkName);
    } catch (err) {
      handleImportError(err);
    } finally {
      setIsUploading(false);
    }
  };

  const handleModelChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const value = e.target.value;
    if (value === 'add_new') {
      toast.info('Adding new models is coming soon!');
      return;
    }
    setModelPath(value);
  };

  // Utility function for downloading files
  const downloadFile = (content: string, filename: string, mimeType: string) => {
    const blob = new Blob([content], { type: mimeType });
    const url = URL.createObjectURL(blob);

    const link = document.createElement('a');
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  const handleExport = useCallback(async () => {
    const modelDetails = AVAILABLE_MODELS.find((m) => m.value === modelPath);

    if (!modelDetails) {
      toast.error('Please select a model first');
      return;
    }

    if (!modelDetails.exportCompatible) {
      toast.error(`${modelDetails.label} is not compatible with open-hypergraph export`);
      return;
    }

    setIsExporting(true);

    try {
      // Handle "Export All" format
      if (exportFormat === 'all') {
        const allResults = await exportAllFormats(
          modelPath,
          modelDetails.sampleInputArgs,
          modelDetails.sampleInputDtypes,
        );

        if (allResults.success) {
          // Process export results into archive files
          const files = processExportResults(allResults);

          // Create archive metadata
          const metadata: ArchiveMetadata = {
            modelName: allResults.modelName,
            modelPath: allResults.modelPath,
            timestamp: allResults.timestamp,
            totalFiles: files.length,
          };

          // Create and download archive file
          const archiveBlob = createArchiveFile(files, metadata);
          const filename = createArchiveFilename(allResults.modelName);
          downloadBlob(archiveBlob, filename);

          // Show success message
          const successMessage = generateExportSuccessMessage(allResults.results, true);
          toast.success(successMessage, { duration: 8000 });

          // Enhanced logging
          logExportResults(allResults, files);
        } else {
          const errorMessage = generateExportErrorMessage(allResults.results);
          toast.error(errorMessage);
        }

        return; // Early return for "all" format
      }

      // Handle individual format exports
      const result = await exportModelHypergraph(
        modelPath,
        exportFormat,
        modelDetails.sampleInputArgs,
        modelDetails.sampleInputDtypes,
      );

      if (result.success) {
        // Handle different export formats
        if (exportFormat === 'json') {
          // Download JSON file
          const jsonData = {
            nodes: result.nodes,
            hyperedges: result.hyperedges,
            metadata: result.metadata,
          };
          downloadFile(
            JSON.stringify(jsonData, null, 2),
            `${modelDetails.label.replace(/[^a-zA-Z0-9]/g, '_')}_hypergraph.json`,
            'application/json',
          );

          const nodeCount = result.nodes?.length || 0;
          const edgeCount = result.hyperedges?.length || 0;
          toast.success(`VisuaML Graph export complete: ${nodeCount} nodes, ${edgeCount} hyperedges`);
        } else if (exportFormat === 'macro') {
          // Download macro file - handle both 'macro' and 'macro_syntax' field names
          const macroContent = result.macro || result.macro_syntax;
          if (macroContent) {
            downloadFile(
              macroContent,
              `${modelDetails.label.replace(/[^a-zA-Z0-9]/g, '_')}_hypergraph.macro`,
              'text/plain',
            );
            toast.success('HyperSyn Macro export complete - compatible with Rust crate');
          } else {
            toast.error('Macro content not found in export result');
          }
        } else if (exportFormat === 'categorical') {
          // Handle categorical format - backend returns different structure
          const analysisData = result.categorical_analysis;
          const jsonData = result.json_data;

          if (jsonData) {
            // Download the JSON hypergraph data
            downloadFile(
              JSON.stringify(jsonData, null, 2),
              `${modelDetails.label.replace(/[^a-zA-Z0-9]/g, '_')}_hypergraph_data.json`,
              'application/json',
            );
          }

          if (analysisData) {
            // Download categorical analysis
            downloadFile(
              JSON.stringify(analysisData, null, 2),
              `${modelDetails.label.replace(/[^a-zA-Z0-9]/g, '_')}_categorical_analysis.json`,
              'application/json',
            );

            // Show enhanced success message with categorical analysis
            const message = `Open Hypergraph export complete! 
📊 ${analysisData.nodes} nodes, ${analysisData.hyperedges} hyperedges
🔗 Max arity: ${analysisData.max_input_arity} inputs, ${analysisData.max_output_arity} outputs
✅ Complexity: ${analysisData.complexity}`;
            toast.success(message, { duration: 8000 });
          } else {
            toast.success('Open Hypergraph export complete - analysis data generated');
          }

          // Enhanced logging for developers
          console.group('🔬 Open Hypergraph Export Results');
          console.log('📈 Categorical Analysis:', analysisData);
          console.log('📊 JSON Data:', jsonData);
          console.log('🔧 Library Available:', result.library_available);
          console.log('🚀 Framework Ready:', result.framework_ready);
          console.groupEnd();

          // Create a compatible structure for the categorical panel
          if (categoricalPanelCallback && analysisData) {
            const compatibleResult = {
              ...result,
              categorical_analysis: {
                total_morphisms: analysisData.hyperedges || 0,
                composition_depth: Math.max(...(analysisData.input_arities || [0])),
                type_safety_validated: analysisData.conversion_status === 'analysis_complete',
                hypergraph_statistics: {
                  hyperedges: analysisData.hyperedges || 0,
                  wires: analysisData.nodes || 0,
                  input_boundary_size: analysisData.max_input_arity || 0,
                  output_boundary_size: analysisData.max_output_arity || 0,
                },
              },
            };
            categoricalPanelCallback(compatibleResult);
          }
        }

        // Show categorical availability info
        if (result.categorical_available && exportFormat !== 'categorical') {
          console.log('Note: Categorical conversion is available for this export');
        }
      } else {
        toast.error(result.message || 'Export failed');
      }
    } catch (error) {
      console.error('Export error:', error);
      toast.error(error instanceof Error ? error.message : 'Export failed');
    } finally {
      setIsExporting(false);
    }
  }, [modelPath, exportFormat]);

  const handleExportFormatChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    setExportFormat(e.target.value as ExportFormat);
  };

  return {
    modelPath,
    isLoadingUI,
    isUploading,
    isExporting,
    exportFormat,
    handleImportClick,
    handleFileUpload,
    handleModelChange,
    handleExport,
    handleExportFormatChange,
  };
};
