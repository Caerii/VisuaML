import * as dotenv from 'dotenv';
import { WebSocketServer } from 'ws';
import * as Y from 'yjs';

dotenv.config();

const PORT = process.env.WS_PORT || 1234;

const wss = new WebSocketServer({ port: Number(PORT) });

// Store documents by room name
const docs = new Map<string, Y.Doc>();

wss.on('connection', (ws) => {
  const roomName = 'visuaml-room';

  // Get or create document for this room
  if (!docs.has(roomName)) {
    docs.set(roomName, new Y.Doc());
  }
  const doc = docs.get(roomName)!;

  // Handle incoming messages
  ws.on('message', (message) => {
    try {
      const data = new Uint8Array(message as ArrayBuffer);
      Y.applyUpdate(doc, data);

      // Broadcast to all other clients in the same room
      wss.clients.forEach((client) => {
        if (client !== ws && client.readyState === client.OPEN) {
          client.send(message);
        }
      });
    } catch (error) {
      console.error('Error handling message:', error);
    }
  });

  // Send current state to new client
  const state = Y.encodeStateAsUpdate(doc);
  ws.send(state);

  // Handle client disconnect
  ws.on('close', () => {
    console.log('Client disconnected');
  });
});

console.log(`WebSocket server is listening on port ${PORT}`);

// Handle errors
wss.on('error', (error: Error) => {
  console.error('WebSocket server error:', error);
});

// Cleanup on process termination
process.on('SIGTERM', () => {
  console.log('Received SIGTERM signal. Closing WebSocket server...');
  wss.close(() => {
    console.log('WebSocket server closed.');
    process.exit(0);
  });
});
