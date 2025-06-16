/** @fileoverview React context for managing categorical panel state across components. */
import React, { createContext, useContext, useState, useCallback } from 'react';
import type { ReactNode } from 'react';
import type { ExportHypergraphResponse } from '../TopBar/TopBar.model';

interface CategoricalContextType {
  isVisible: boolean;
  categoricalData: ExportHypergraphResponse | null;
  showPanel: (data: ExportHypergraphResponse) => void;
  hidePanel: () => void;
  togglePanel: () => void;
  clearData: () => void;
}

const CategoricalContext = createContext<CategoricalContextType | undefined>(undefined);

interface CategoricalProviderProps {
  children: ReactNode;
}

export const CategoricalProvider: React.FC<CategoricalProviderProps> = ({ children }) => {
  const [isVisible, setIsVisible] = useState(false);
  const [categoricalData, setCategoricalData] = useState<ExportHypergraphResponse | null>(null);

  const showPanel = useCallback((data: ExportHypergraphResponse) => {
    setCategoricalData(data);
    setIsVisible(true);
  }, []);

  const hidePanel = useCallback(() => {
    setIsVisible(false);
  }, []);

  const togglePanel = useCallback(() => {
    setIsVisible(prev => !prev);
  }, []);

  const clearData = useCallback(() => {
    setCategoricalData(null);
    setIsVisible(false);
  }, []);

  const value: CategoricalContextType = {
    isVisible,
    categoricalData,
    showPanel,
    hidePanel,
    togglePanel,
    clearData
  };

  return (
    <CategoricalContext.Provider value={value}>
      {children}
    </CategoricalContext.Provider>
  );
};

export const useCategoricalContext = (): CategoricalContextType => {
  const context = useContext(CategoricalContext);
  if (context === undefined) {
    throw new Error('useCategoricalContext must be used within a CategoricalProvider');
  }
  return context;
}; 