import { create } from 'zustand';

interface NetworkFacts {
  numNodes: number;
  numEdges: number;
  inputShapes: string[];
  componentTypes: string[];
  networkName?: string; // Optional for now
  isLoadingGraph: boolean; // Added to track graph loading state
  // Add more facts as needed, e.g., model name, layers, parameters
}

interface NetworkState {
  facts: NetworkFacts | null;
  setFacts: (facts: NetworkFacts) => void;
  clearFacts: () => void;
}

export const useNetworkStore = create<NetworkState>((set) => ({
  facts: null,
  setFacts: (facts) => set({ facts }),
  clearFacts: () => set({ facts: null }), // Reset all facts to null
}));
