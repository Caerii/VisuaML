import { useEffect, useState } from "react";
import { useYDoc } from "./DocProvider";

// Define the structure of the cursor data broadcasted
export interface CursorBroadcastData {
  x: number;
  y: number;
  // name and color might be deprecated here if primarily taken from user field
  name?: string; // Name set by the broadcasting client (e.g. guest name)
  color?: string; // Color set by the broadcasting client (e.g. guest color)
}

// Define the structure of the user data in awareness state
export interface UserAwarenessData {
  id: string;
  name: string;
  color: string;
}

// Combined structure representing a cursor to be rendered
export interface RenderableCursor {
  x: number;
  y: number;
  name: string;
  color: string;
  clientID: number;
  userID?: string; // Optional Clerk User ID
}

// The full awareness state for a client.
interface ClientAwarenessState {
  cursor?: CursorBroadcastData;
  user?: UserAwarenessData;
  [key: string]: unknown; 
}

export const useRemoteCursors = (): RenderableCursor[] => {
  const { provider, ydoc } = useYDoc();
  const [cursors, setCursors] = useState<RenderableCursor[]>([]);

  useEffect(() => {
    if (!provider || !provider.awareness) {
      setCursors([]);
      return;
    }

    const onChange = () => {
      const localClientID = ydoc.clientID;
      const newCursors: RenderableCursor[] = [];
      
      provider.awareness.getStates().forEach((state: ClientAwarenessState, clientID: number) => {
        if (clientID !== localClientID && state.cursor) {
          const cursorData = state.cursor;
          const userData = state.user;

          newCursors.push({
            x: cursorData.x,
            y: cursorData.y,
            // Prioritize user field for name and color, fallback to cursor field
            name: userData?.name || cursorData.name || "Anonymous",
            color: userData?.color || cursorData.color || "#888888", // Default fallback color
            clientID,
            userID: userData?.id,
          });
        }
      });
      setCursors(newCursors);
    };

    provider.awareness.on("change", onChange);
    onChange();

    return () => {
      if (provider && provider.awareness) {
        provider.awareness.off("change", onChange);
      }
    };
  }, [provider, provider?.awareness, ydoc.clientID]);

  return cursors;
}; 