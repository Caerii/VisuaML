/** @fileoverview Main entry point for the VisuaML client application. Sets up the Clerk provider, Yjs document provider, and renders the main layout including TopBar and Canvas. */
import React from 'react';
import ReactDOM from 'react-dom/client';
import { YDocProvider } from './y/DocProvider'; // (create shortly)
import { Canvas } from './ui/Canvas/Canvas';
import TopBar from './ui/TopBar/TopBar';
import { ClerkProvider } from '@clerk/clerk-react';
import { Toaster } from 'sonner';
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

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <ClerkProvider publishableKey={clerkPublishableKey}>
      <YDocProvider>
        <div style={{ display: 'flex', flexDirection: 'column', height: '100vh' }}>
          <TopBar />
          <div style={{ flexGrow: 1 }}>
            <Canvas />
          </div>
        </div>
        <Toaster richColors position="top-right" />
      </YDocProvider>
    </ClerkProvider>
  </React.StrictMode>,
);
