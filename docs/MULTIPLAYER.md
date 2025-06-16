# VisuaML Multiplayer System

VisuaML features a comprehensive real-time multiplayer system that allows multiple users to collaborate on neural network visualization simultaneously.

## 🌐 Architecture Overview

The multiplayer system is built on top of **Yjs** (a CRDT library) and **WebSockets** for real-time synchronization:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Client A      │    │  WebSocket      │    │   Client B      │
│   (Browser)     │◄──►│  Server         │◄──►│   (Browser)     │
│                 │    │  (port 1234)    │    │                 │
│ • Yjs Document  │    │ • y-websocket   │    │ • Yjs Document  │
│ • Awareness API │    │ • Room: visuaml │    │ • Awareness API │
│ • Live Cursors  │    │ • CRDT Sync     │    │ • Live Cursors  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🚀 Quick Start

### 1. Start the WebSocket Server

```bash
npm run ws
```

This starts the y-websocket server on port 1234 using the official y-websocket package.

### 2. Open Multiple Browser Windows

1. Start the frontend: `npm run dev`
2. Open `http://localhost:5173` in multiple browser tabs/windows
3. Load a model in one window
4. Watch it appear in real-time in all other windows!

## 🔧 Technical Implementation

### WebSocket Server (`npm run ws`)

```bash
# Uses the official y-websocket server
cross-env HOST=0.0.0.0 node ./node_modules/y-websocket/bin/server.js --port 1234
```

**Features:**
- Automatic room management
- CRDT-based conflict resolution
- Persistent document state
- Connection management and recovery

### Client-Side Integration

#### Yjs Document Provider (`src/y/DocProvider.tsx`)

```typescript
// Create shared document
export const ydoc = new Y.Doc();

// WebSocket connection
const wsProvider = new WebsocketProvider('ws://localhost:1234', 'visuaml-room', ydoc);

// IndexedDB persistence for offline capability
const persistence = new IndexeddbPersistence('visuaml-local-room', ydoc);
```

#### Shared State Structure

```typescript
// Shared arrays for graph data
const yNodes = ydoc.getArray<Node>('nodes');
const yEdges = ydoc.getArray<Edge>('edges');

// Awareness for user presence
interface ClientAwarenessState {
  cursor?: { x: number; y: number; name?: string; color?: string };
  user?: { id: string; name: string; color: string };
}
```

## 👥 User Presence System

### User Authentication Integration

The system integrates with **Clerk** for authenticated users and provides guest access:

```typescript
// Authenticated users
if (isSignedIn && user) {
  const userColor = stringToHSL(user.id);
  const userName = user.fullName || user.primaryEmailAddress?.emailAddress;
  
  awareness.setLocalStateField('user', {
    id: user.id,
    name: userName,
    color: userColor,
  });
}

// Guest users
else {
  const guestNumber = (ydoc.clientID % 999) + 1;
  const guestName = `Guest ${guestNumber}`;
  const guestColor = stringToHSL(`anonymous_${ydoc.clientID}`);
  
  awareness.setLocalStateField('user', {
    id: `guest_${ydoc.clientID}`,
    name: guestName,
    color: guestColor,
  });
}
```

### Live Cursor Tracking

Real-time cursor positions are tracked and displayed for all users:

```typescript
// Track mouse movement
const handleMouseMove = (event: React.MouseEvent) => {
  const bounds = reactFlowWrapper.current.getBoundingClientRect();
  const x = event.clientX - bounds.left;
  const y = event.clientY - bounds.top;
  
  // Throttled updates using requestAnimationFrame
  provider.awareness.setLocalStateField('cursor', { x, y });
};

// Render remote cursors
{remoteCursors.map((cursor) => (
  <RemoteCursor
    key={cursor.clientID}
    x={cursor.x}
    y={cursor.y}
    name={cursor.name}
    color={cursor.color}
  />
))}
```

## 🔄 Real-time Synchronization

### Graph State Synchronization

The graph state (nodes and edges) is automatically synchronized across all clients:

```typescript
// React → Yjs (when user makes changes)
const commitNodes = (nodes: Node[]) => {
  ydoc.transact(() => {
    yNodes.delete(0, yNodes.length);
    nodes.forEach((node) => {
      yNodes.push([new Y.Map(Object.entries(node))]);
    });
  });
};

// Yjs → React (when remote changes arrive)
useEffect(() => {
  const observer = () => {
    const newNodes = yNodes.toArray().map(yNode => 
      Object.fromEntries(yNode.entries())
    );
    setNodes(newNodes);
  };
  
  yNodes.observe(observer);
  return () => yNodes.unobserve(observer);
}, []);
```

### Conflict Resolution

Yjs uses **CRDTs** (Conflict-free Replicated Data Types) to automatically resolve conflicts:

- **Automatic merging**: Changes from multiple users are merged automatically
- **Causal ordering**: Operations maintain causal relationships
- **Eventual consistency**: All clients converge to the same state
- **No manual conflict resolution**: Developers don't need to handle conflicts

## 🎨 Visual Features

### Remote Cursors

Each user's cursor is displayed with:
- **Unique color**: Generated from user ID using HSL color space
- **User name**: Display name or guest identifier
- **Smooth animation**: Throttled updates for performance
- **Automatic cleanup**: Cursors disappear when users disconnect

### User Indicators

- **Color-coded presence**: Each user has a unique color
- **Guest numbering**: Anonymous users get sequential guest numbers
- **Authenticated names**: Clerk users show their real names
- **Connection status**: Visual feedback for connection state

## 🔧 Configuration

### Environment Variables

```bash
# WebSocket server URL (default: ws://localhost:1234)
VITE_WEBSOCKET_URL=ws://localhost:1234

# WebSocket server port (for npm run ws)
WS_PORT=1234
```

### Room Management

Currently uses a single room (`visuaml-room`) for all users. Future enhancements could include:
- **Project-based rooms**: Separate rooms for different projects
- **Private rooms**: Invitation-only collaboration
- **Room discovery**: Browse and join public rooms

## 🧪 Testing Multiplayer Features

### Local Testing

1. **Start servers**:
   ```bash
   npm run api    # Terminal 1
   npm run ws     # Terminal 2  
   npm run dev    # Terminal 3
   ```

2. **Open multiple windows**:
   - Open `http://localhost:5173` in 2+ browser windows
   - Use different browsers or incognito mode for different users

3. **Test scenarios**:
   - Load a model in one window → verify it appears in others
   - Move nodes in one window → verify positions sync
   - Move mouse → verify cursors appear in other windows
   - Disconnect/reconnect → verify state persistence

### Network Testing

Test with multiple devices on the same network:

```bash
# Start with host binding
npm run ws  # Binds to 0.0.0.0:1234

# Access from other devices
http://[your-ip]:5173
```

## 🚀 Production Deployment

### WebSocket Server Deployment

For production, deploy the WebSocket server separately:

```bash
# Using PM2
pm2 start "node ./node_modules/y-websocket/bin/server.js --port 1234" --name "visuaml-ws"

# Using Docker
docker run -p 1234:1234 -d y-websocket-server

# Using cloud providers
# Deploy to Heroku, Railway, or any WebSocket-compatible platform
```

### Scaling Considerations

- **Horizontal scaling**: Use Redis adapter for multi-instance sync
- **Load balancing**: Sticky sessions for WebSocket connections
- **Monitoring**: Track connection counts and message rates
- **Rate limiting**: Prevent abuse and ensure fair usage

## 🔒 Security Considerations

### Authentication

- **Clerk integration**: Secure user authentication
- **Guest access**: Limited permissions for anonymous users
- **Session validation**: Verify user sessions on connection

### Data Protection

- **Input validation**: Sanitize all shared data
- **Rate limiting**: Prevent spam and abuse
- **Room isolation**: Ensure users can only access authorized rooms
- **Audit logging**: Track user actions for security

## 🐛 Troubleshooting

### Common Issues

**WebSocket connection fails:**
```bash
# Check if WebSocket server is running
npm run ws

# Verify port is not in use
netstat -an | grep 1234

# Check firewall settings
```

**State not syncing:**
```bash
# Clear IndexedDB cache
# Open DevTools → Application → Storage → IndexedDB → Delete

# Restart all servers
npm run api && npm run ws && npm run dev
```

**Cursors not appearing:**
```bash
# Check awareness state in DevTools
console.log(provider.awareness.getStates());

# Verify mouse move events
# Check Canvas component mouse handlers
```

### Debug Mode

Enable debug logging:

```typescript
// In DocProvider.tsx
wsProvider.on('status', (event) => {
  console.log('WebSocket status:', event.status);
});

wsProvider.on('synced', (isSynced) => {
  console.log('Document synced:', isSynced);
});
```

## 🔮 Future Enhancements

### Planned Features

- **Voice chat integration**: WebRTC voice communication
- **Screen sharing**: Share model views between users
- **Collaborative annotations**: Add comments and notes
- **Version history**: Track and revert changes
- **Permissions system**: Role-based access control
- **Room management**: Create and manage collaboration spaces

### Technical Improvements

- **Performance optimization**: Reduce message frequency
- **Offline support**: Better offline/online synchronization
- **Mobile support**: Touch-friendly collaboration
- **Analytics**: Usage tracking and insights

---

The VisuaML multiplayer system provides a seamless collaborative experience for neural network visualization, enabling teams to work together in real-time while maintaining data consistency and user presence awareness. 