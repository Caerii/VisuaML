/** @fileoverview Hook for managing categorical panel state and data. */
import { useState, useCallback } from 'react';
import type { CategoricalMorphism, CategoricalHypergraph } from '../TopBar/TopBar.model';

interface CategoricalData {
  morphisms?: CategoricalMorphism[];
  hypergraph?: CategoricalHypergraph;
  compositionChain?: string[];
  typeSignature?: string;
  analysis?: {
    total_morphisms: number;
    composition_depth: number;
    type_safety_validated: boolean;
    hypergraph_statistics: {
      hyperedges: number;
      wires: number;
      input_boundary_size: number;
      output_boundary_size: number;
    };
  };
}

export const useCategoricalPanel = () => {
  const [isVisible, setIsVisible] = useState(false);
  const [categoricalData, setCategoricalData] = useState<CategoricalData | null>(null);

  const showPanel = useCallback((data: CategoricalData) => {
    setCategoricalData(data);
    setIsVisible(true);
  }, []);

  const hidePanel = useCallback(() => {
    setIsVisible(false);
  }, []);

  const togglePanel = useCallback(() => {
    setIsVisible((prev) => !prev);
  }, []);

  const updateData = useCallback((data: Partial<CategoricalData>) => {
    setCategoricalData((prev) => (prev ? { ...prev, ...data } : (data as CategoricalData)));
  }, []);

  const clearData = useCallback(() => {
    setCategoricalData(null);
    setIsVisible(false);
  }, []);

  return {
    isVisible,
    categoricalData,
    showPanel,
    hidePanel,
    togglePanel,
    updateData,
    clearData,
  };
};
