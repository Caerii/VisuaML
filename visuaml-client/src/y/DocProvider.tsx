import { createContext, useEffect, useContext, useState } from 'react';
import * as Y from 'yjs';
import { WebsocketProvider } from 'y-websocket';
import { IndexeddbPersistence } from 'y-indexeddb'; // Import IndexeddbPersistence
import { useUser } from '@clerk/clerk-react';
import { stringToHSL } from '../lib/colors';
// Import the specific type for cursor data payload from usePresence.ts
import type { CursorBroadcastData } from './usePresence'; // Import for typing local state

export const ydoc = new Y.Doc();

// Initialize IndexedDB persistence
// Ensure this is only created once per ydoc instance.
// The room name should match the one used by WebsocketProvider for consistency if needed,
// or be unique if you want separate local persistence unrelated to the websocket room name.
const persistence = new IndexeddbPersistence('visuaml-local-room', ydoc);

persistence.on('synced', (isSynced: boolean) => {
  console.log('IndexedDB persistence synced:', isSynced);
});

// Define the shape of the context value
interface YContextValue {
  ydoc: Y.Doc;
  provider: WebsocketProvider | null;
  awareness: WebsocketProvider['awareness'] | null;
}

// Initialize context with ydoc and null for provider/awareness initially
const YContext = createContext<YContextValue>({
  ydoc,
  provider: null,
  awareness: null,
});

// For typing the local awareness state
interface LocalClientAwarenessState {
  cursor?: CursorBroadcastData; // Use the imported type
  user?: { id: string; name: string; color: string }; // Structure we set for user
  [key: string]: unknown; // Changed to unknown
}

export const YDocProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [providerState, setProviderState] = useState<WebsocketProvider | null>(null);
  const { user, isSignedIn } = useUser();

  useEffect(() => {
    // local dev ws on :1234 – we'll spin it up next commit
    const wsProvider = new WebsocketProvider('ws://localhost:1234', 'visuaml-room', ydoc);
    setProviderState(wsProvider);

    // Check if persistence is already established and synced
    if (persistence.synced) {
      console.log('IndexedDB already synced upon wsProvider creation.');
    }

    return () => {
      wsProvider.destroy();
      setProviderState(null); // Clean up provider state
      // Note: IndexeddbPersistence doesn't have a destroy method in the same way.
      // It stays active as long as the ydoc is in memory and the page is open.
      // To truly clear it, you might need to use its clearData() method or manage its lifecycle carefully.
    };
  }, []);

  // Effect to update awareness with user info when available
  useEffect(() => {
    if (providerState && providerState.awareness) {
      if (isSignedIn && user) {
      const userColor = stringToHSL(user.id);
      const userName = user.fullName || user.primaryEmailAddress?.emailAddress || 'Anonymous';

      providerState.awareness.setLocalStateField('user', {
        id: user.id,
        name: userName,
        color: userColor,
      });

      const localAwarenessState = providerState.awareness.getLocalState() as
        | LocalClientAwarenessState
        | undefined;
      const currentCursorData = localAwarenessState?.cursor;

      if (currentCursorData) {
        providerState.awareness.setLocalStateField('cursor', {
          ...currentCursorData,
          name: userName,
          color: userColor,
        });
      }
      } else {
        // Not signed in - set guest user info
        // Use modulo to get a simple guest number (1-999)
        const guestNumber = (ydoc.clientID % 999) + 1;
        const guestName = `Guest ${guestNumber}`;
        const guestColor = stringToHSL(`anonymous_${ydoc.clientID}`);
        
        providerState.awareness.setLocalStateField('user', {
          id: `guest_${ydoc.clientID}`,
          name: guestName,
          color: guestColor,
        });
        
      const localAwarenessState = providerState.awareness.getLocalState() as
        | LocalClientAwarenessState
        | undefined;
      const currentCursorData = localAwarenessState?.cursor;
        
      if (currentCursorData) {
        providerState.awareness.setLocalStateField('cursor', {
          ...currentCursorData,
            name: guestName,
            color: guestColor,
        });
        }
      }
    }
    // Dependencies: user object, isSignedIn status, and providerState.awareness
  }, [user, isSignedIn, providerState]); // providerState.awareness removed as it's covered by providerState

  const contextValue: YContextValue = {
    ydoc,
    provider: providerState,
    awareness: providerState?.awareness || null,
  };

  return <YContext.Provider value={contextValue}>{children}</YContext.Provider>;
};

// useYDoc now returns the full context value
export const useYDoc = (): YContextValue => useContext(YContext);
