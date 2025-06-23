import { create } from 'zustand';

export interface NetworkFacts {
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
  isGraphInteractive: boolean;
  setFacts: (facts: NetworkFacts) => void;
  clearFacts: () => void;
  setIsGraphInteractive: (isInteractive: boolean) => void;
}

export const useNetworkStore = create<NetworkState>((set) => ({
  facts: null,
  isGraphInteractive: true,
  setFacts: (newFacts) => set({ facts: newFacts }),
  clearFacts: () => set({ facts: null, isGraphInteractive: true }), // Also reset interaction state
  setIsGraphInteractive: (isInteractive) => set({ isGraphInteractive: isInteractive }),
}));
