/** @fileoverview Main application component for the VisuaML visualization tool. */
import React, { useState } from 'react';
import { YDocProvider } from './y/DocProvider';
import { Canvas } from './ui/Canvas/Canvas';
import TopBar from './ui/TopBar/TopBar';
import CategoricalPanel from './ui/CategoricalPanel/CategoricalPanel';
import { setCategoricalPanelCallback } from './ui/TopBar/useTopBar';
import { ClerkProvider } from '@clerk/clerk-react';
import { Toaster } from 'sonner';
import type { ExportHypergraphResponse } from './ui/TopBar/TopBar.model';

// OPTIONAL: Create a .env.local file in the root of your visuaml-client project
// and add your Clerk Publishable Key for authentication features:
// VITE_CLERK_PUBLISHABLE_KEY="your_publishable_key_here"
// You can get your key from your Clerk dashboard.
// If not provided, the app will run without authentication.
const clerkPublishableKey = import.meta.env.VITE_CLERK_PUBLISHABLE_KEY;

export const MainApp = () => {
  const [categoricalData, setCategoricalData] = useState<ExportHypergraphResponse | null>(null);
  const [showCategoricalPanel, setShowCategoricalPanel] = useState(false);

  // Set up the callback for categorical exports
  React.useEffect(() => {
    setCategoricalPanelCallback((data: ExportHypergraphResponse) => {
      setCategoricalData(data);
      setShowCategoricalPanel(true);
    });
  }, []);

  const content = (
    <YDocProvider>
      <div className="flex-column" style={{ height: '100vh' }}>
        <TopBar />
        <div style={{ flexGrow: 1, position: 'relative' }}>
          <Canvas />
          {categoricalData && (
            <CategoricalPanel
              morphisms={categoricalData.morphisms}
              hypergraph={categoricalData.categorical_hypergraph}
              compositionChain={categoricalData.composition_chain}
              typeSignature={categoricalData.type_signature}
              analysis={categoricalData.categorical_analysis}
              isVisible={showCategoricalPanel}
              onClose={() => setShowCategoricalPanel(false)}
            />
          )}
        </div>
      </div>
      <Toaster richColors position="top-right" />
    </YDocProvider>
  );

  // Only wrap with ClerkProvider if we have a publishable key
  if (clerkPublishableKey) {
    return <ClerkProvider publishableKey={clerkPublishableKey}>{content}</ClerkProvider>;
  }

  return content;
};
