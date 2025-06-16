/** @fileoverview Main entry point for the VisuaML client application. Sets up the Clerk provider, Yjs document provider, MUI Theme provider and renders the main layout including TopBar and Canvas. */
import React, { useState } from 'react';
import ReactDOM from 'react-dom/client';
import { YDocProvider } from './y/DocProvider'; // (create shortly)
import { Canvas } from './ui/Canvas/Canvas';
import TopBar from './ui/TopBar/TopBar';
import CategoricalPanel from './ui/CategoricalPanel/CategoricalPanel';
import { setCategoricalPanelCallback } from './ui/TopBar/useTopBar';
import { ClerkProvider } from '@clerk/clerk-react';
import { Toaster } from 'sonner';
import { ThemeProvider, createTheme, CssBaseline } from '@mui/material';
import type { ExportHypergraphResponse } from './ui/TopBar/TopBar.model';
import './index.css';

// IMPORTANT: Create a .env.local file in the root of your visuaml-client project
// and add your Clerk Publishable Key:
// VITE_CLERK_PUBLISHABLE_KEY="your_publishable_key_here"
// You can get your key from your Clerk dashboard.
const clerkPublishableKey = import.meta.env.VITE_CLERK_PUBLISHABLE_KEY;

if (!clerkPublishableKey) {
  throw new Error(
    'Missing Clerk Publishable Key. Please set VITE_CLERK_PUBLISHABLE_KEY in .env.local',
  );
}

const theme = createTheme({
  // You can add basic theme customizations here later if you wish, e.g.:
  // palette: {
  //   primary: {
  //     main: '#1976d2',
  //   },
  // },
});

const MainApp = () => {
  const [categoricalData, setCategoricalData] = useState<ExportHypergraphResponse | null>(null);
  const [showCategoricalPanel, setShowCategoricalPanel] = useState(false);

  // Set up the callback for categorical exports
  React.useEffect(() => {
    setCategoricalPanelCallback((data: ExportHypergraphResponse) => {
      setCategoricalData(data);
      setShowCategoricalPanel(true);
    });
  }, []);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100vh' }}>
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
  );
};

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <ClerkProvider publishableKey={clerkPublishableKey}>
        <YDocProvider>
          <MainApp />
          <Toaster richColors position="top-right" />
        </YDocProvider>
      </ClerkProvider>
    </ThemeProvider>
  </React.StrictMode>,
);
