import type { Node, Edge } from '@xyflow/react';

// Define the expected structure of the JSON from the Python script
interface FxGraphJSON {
  nodes: Node[]; // Assuming the Python script output matches React Flow Node structure
  edges: Edge[]; // Assuming the Python script output matches React Flow Edge structure
}

/**
 * Fetches graph data from the /api/import endpoint and commits it to Yjs.
 * @param modelPath The Python path to the model class (e.g., "models.MyModel").
 * @param commitNodes Function to commit nodes to Yjs (from useSyncedGraph).
 * @param commitEdges Function to commit edges to Yjs (from useSyncedGraph).
 */
export const importFxGraph = async (
  modelPath: string,
  commitNodes: (nodes: Node[]) => void,
  commitEdges: (edges: Edge[]) => void,
): Promise<void> => {
  try {
    // In a real setup, the Vite dev server would proxy this to a backend
    // that executes the visuaml backend export script.
    const response = await fetch(`/api/import?path=${encodeURIComponent(modelPath)}`);

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(
        `Failed to import FX graph: ${response.status} ${response.statusText}. ${errorText}`,
      );
    }

    const data = (await response.json()) as FxGraphJSON;

    if (!data.nodes || !data.edges) {
      throw new Error('Invalid FX graph data received from API: missing nodes or edges.');
    }

    // TODO: Implement auto-layouting (e.g., with Dagre) here before committing
    // For now, nodes will use the (0,0) positions from visuaml backend
    console.log('Imported FX data:', data);

    // Batch the updates into a single Yjs transaction if not already handled by commitNodes/Edges
    // (useSyncedGraph already wraps these in transactions)
    commitNodes(data.nodes);
    commitEdges(data.edges);

    console.log(`Successfully imported and committed FX graph for ${modelPath}`);
  } catch (error) {
    console.error('Error during FX graph import:', error);
    // Optionally, re-throw or handle error in UI
    throw error;
  }
};
