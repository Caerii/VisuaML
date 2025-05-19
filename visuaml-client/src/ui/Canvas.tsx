import {
  ReactFlow,
  Background,
  Controls,
  addEdge,
  ReactFlowProvider,
  applyNodeChanges,
  applyEdgeChanges,
} from "@xyflow/react";
import type { Connection, Edge, Node, NodeChange, EdgeChange } from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import { useSyncedGraph } from "../y/useSyncedGraph";
import { useYDoc } from "../y/DocProvider";
import { useRemoteCursors, type RenderableCursor } from "../y/usePresence";
import { useCallback, useRef, useEffect } from 'react';
import TransformerNode from './nodes/TransformerNode'; // Import custom node

// LOCAL_USER_NAME and LOCAL_USER_COLOR removed as they are now handled in DocProvider or by Clerk identity

// Define custom node types
const nodeTypes = { transformer: TransformerNode };

export const Canvas: React.FC = () => {
  const [nodes, setNodes, edges, setEdges] = useSyncedGraph();
  const { provider } = useYDoc();
  const remoteCursors = useRemoteCursors();
  const reactFlowWrapper = useRef<HTMLDivElement>(null);
  
  // Throttling for mouse move updates
  const animationFrameId = useRef<number | null>(null);
  const lastSentPosition = useRef<{ x: number; y: number } | null>(null);

  const onNodesChange = useCallback(
    (changes: NodeChange[]) => setNodes(applyNodeChanges(changes, nodes) as Node[]),
    [nodes, setNodes],
  );
  const onEdgesChange = useCallback(
    (changes: EdgeChange[]) => setEdges(applyEdgeChanges(changes, edges) as Edge[]),
    [edges, setEdges],
  );
  const onConnect = useCallback(
    (connection: Connection) => setEdges(addEdge(connection, edges) as Edge[]),
    [edges, setEdges],
  );

  const handleMouseMove = useCallback((event: React.MouseEvent<HTMLDivElement>) => {
    if (!provider || !provider.awareness || !reactFlowWrapper.current) return;

    const bounds = reactFlowWrapper.current.getBoundingClientRect();
    const x = event.clientX - bounds.left;
    const y = event.clientY - bounds.top;

    if (animationFrameId.current) {
      cancelAnimationFrame(animationFrameId.current);
    }

    animationFrameId.current = requestAnimationFrame(() => {
      if (!lastSentPosition.current || lastSentPosition.current.x !== x || lastSentPosition.current.y !== y) {
        provider.awareness.setLocalStateField("cursor", { x, y });
        lastSentPosition.current = { x, y };
      }
    });
  }, [provider]);

  // Cleanup animation frame on unmount
  useEffect(() => {
    return () => {
      if (animationFrameId.current) {
        cancelAnimationFrame(animationFrameId.current);
      }
    };
  }, []);

  return (
    <ReactFlowProvider>
      <div 
        className="w-screen h-screen relative" 
        ref={reactFlowWrapper} 
        onMouseMove={handleMouseMove}
      >
        <ReactFlow
          nodes={nodes}
          edges={edges}
          onNodesChange={onNodesChange}
          onEdgesChange={onEdgesChange}
          onConnect={onConnect}
          nodeTypes={nodeTypes} // Register custom node types
          fitView
        >
          <Background />
          <Controls />
        </ReactFlow>
        {remoteCursors.map((cursor: RenderableCursor) => (
          <div
            key={cursor.clientID}
            style={{
              position: 'absolute',
              left: `${cursor.x}px`,
              top: `${cursor.y}px`,
              // Adding a small offset so the dot is under the actual system cursor tip
              transform: 'translate(0px, 0px)', // Or translate(-50%, -50%) to center on point
              width: '8px',
              height: '8px',
              backgroundColor: cursor.color,
              borderRadius: '50%',
              pointerEvents: 'none',
              zIndex: 100, // Ensure cursors are above React Flow elements but below UI controls if any
            }}
            // Display name on hover or as a small label next to cursor
            title={`${cursor.name} (User: ${cursor.userID || 'N/A'}, Client: ${cursor.clientID})`} 
          >
            {/* Optional: render name next to cursor dot */}
            {/* <span style={{ marginLeft: '10px', fontSize: '10px', color: cursor.color }}>{cursor.name}</span> */}
          </div>
        ))}
      </div>
    </ReactFlowProvider>
  );
}; 