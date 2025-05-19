declare module 'y-websocket' {
  import * as Y from 'yjs';

  export class WebsocketProvider {
    constructor(
      serverUrl: string,
      roomName: string,
      doc: Y.Doc,
      options?: {
        awareness?: Y.Awareness;
        params?: Record<string, string>;
        WebSocketPolyfill?: typeof WebSocket | undefined;
        connect?: boolean;
        maxBackoffTime?: number;
        disableBc?: boolean;
      }
    );
    
    on(event: 'status', callback: (status: { status: 'connected' | 'disconnected' | 'connecting' }) => void): void;
    on(event: 'synced', callback: (isSynced: boolean) => void): void;
    on(event: 'disconnect', callback: (event: { reason: string, code: number }) => void): void;
    on(event: string, callback: (...args: unknown[]) => void): void;

    off(event: string, callback: (...args: unknown[]) => void): void;
    destroy(): void;
    connect(): void;
    disconnect(): void;
    
    public readonly awareness: Y.Awareness;
    public readonly bcconnected: boolean;
    public readonly synced: boolean;
    public readonly wsconnected: boolean;
    public readonly wsconnecting: boolean;
  }
} 