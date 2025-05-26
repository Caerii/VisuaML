/** @fileoverview Defines the TopBar component, which includes controls for selecting a model and initiating the import process. It interacts with the Yjs document for shared state and the Zustand store for network facts. */
import { useState } from 'react';
import { useSyncedGraphActions } from '../../y/useSyncedGraph'; // Adjusted path
import { useNetworkStore } from '../../store/networkStore'; // Adjusted path
import { useYDoc } from '../../y/DocProvider'; // Adjusted path
import { autoLayout } from '../../lib/autoLayout'; // Adjusted path, assuming .ts is resolved
import type { Node, Edge } from '@xyflow/react';
import { toast } from 'sonner';
import styles from './styles/TopBar.module.css'; // Adjusted path

// Define structure for API response if not already globally available
interface ImportApiResponse {
  nodes: Node[];
  edges: Edge[];
}

// Hardcoded model options for testing
const AVAILABLE_MODELS = [
  { value: 'models.MyTinyGPT', label: 'MyTinyGPT' },
  { value: 'models.SimpleNN', label: 'SimpleNN' },
  { value: 'models.TestModel', label: 'TestModel' },
  { value: 'models.DemoNet', label: 'DemoNet' },
];

const TopBar = () => {
  const [modelPath, setModelPath] = useState('models.MyTinyGPT');
  const [isLoadingUI, setIsLoadingUI] = useState(false); 
  const { commitNodes, commitEdges } = useSyncedGraphActions();
  const { setFacts: setNetworkFacts } = useNetworkStore();
  const { ydoc } = useYDoc();

  const handleImportClick = async () => {
    setIsLoadingUI(true); 
    const modelDetails = AVAILABLE_MODELS.find(m => m.value === modelPath);
    const localNetworkName = modelDetails ? modelDetails.label : modelPath;

    const ySharedName = ydoc.getText('networkNameShared');
    const ySharedAppStatus = ydoc.getMap('sharedAppStatus');

    ydoc.transact(() => {
      ySharedName.delete(0, ySharedName.length);
      ySharedName.insert(0, localNetworkName);
      ySharedAppStatus.set('isLoadingGraph', true);
    });
    
    try {
      const response = await fetch('/api/import', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ modelPath }), 
      });

      if (!response.ok) {
        const errData = await response.json().catch(() => ({ message: response.statusText }));
        const errorMessage = errData.message || errData.error || JSON.stringify(errData) || `API Error: ${response.status}`;
        throw new Error(errorMessage);
      }

      const importedData = (await response.json()) as ImportApiResponse;
      if (!importedData.nodes || !importedData.edges) {
        throw new Error('Invalid data format from API: nodes or edges missing.');
      }

      const laidOutData = autoLayout(importedData.nodes, importedData.edges);
      commitNodes(laidOutData.nodes); 
      commitEdges(laidOutData.edges); 

      ySharedAppStatus.set('isLoadingGraph', false);
      toast.success(`Model '${localNetworkName}' imported successfully!`); 

    } catch (err: unknown) {
      console.error('Failed to import model:', err);
      const message = err instanceof Error ? err.message : 'An unknown error occurred during import.';
      toast.error(message, { duration: 5000 });
      
      ydoc.transact(() => {
        const yNameToClear = ydoc.getText('networkNameShared');
        if (yNameToClear.length > 0) {
            yNameToClear.delete(0, yNameToClear.length);
        }
        ySharedAppStatus.set('isLoadingGraph', false);
      });
      
      const latestFacts = useNetworkStore.getState().facts;
      setNetworkFacts({
        networkName: undefined, 
        isLoadingGraph: false, 
        numNodes: latestFacts?.numNodes || 0,
        numEdges: latestFacts?.numEdges || 0,
        inputShapes: latestFacts?.inputShapes || [],
        componentTypes: latestFacts?.componentTypes || [],
      });
    } finally {
      setIsLoadingUI(false); 
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

  return (
    <header className={styles.header}>
      <label htmlFor="modelPathSelect" className={styles.label}>
        Model Path:
      </label>
      <select
        id="modelPathSelect"
        value={modelPath}
        onChange={handleModelChange}
        className={styles.select}
        disabled={isLoadingUI} 
      >
        <optgroup label="Available Models">
          {AVAILABLE_MODELS.map((model) => (
            <option key={model.value} value={model.value}>
              {model.label}
            </option>
          ))}
        </optgroup>
        <optgroup label="Actions">
          <option value="add_new" disabled>
            + Add New Model (Coming Soon)
          </option>
        </optgroup>
      </select>
      <button
        onClick={handleImportClick}
        className={styles.button}
        disabled={isLoadingUI} 
      >
        {isLoadingUI ? 'Importing...' : 'Import'}
      </button>
    </header>
  );
};

export default TopBar; 