/** @fileoverview Defines the main interactive canvas for VisuaML, integrating React Flow for graph visualization, Yjs for real-time collaboration, and components for displaying network statistics and remote user cursors. */
import {
  ReactFlow,
  Background,
  BackgroundVariant,
  Controls,
  addEdge,
  ReactFlowProvider,
  applyNodeChanges,
  applyEdgeChanges,
} from '@xyflow/react';
import type { Connection, Edge, Node, NodeChange, EdgeChange } from '@xyflow/react';
import '@xyflow/react/dist/style.css';
import { useSyncedGraph } from '../../y/useSyncedGraph'; // Adjusted path
import { useYDoc } from '../../y/DocProvider'; // Adjusted path
import { useRemoteCursors, type RenderableCursor } from '../../y/usePresence'; // Adjusted path
import { useCallback, useRef, useEffect } from 'react';
import MLNode from '../nodes/MLNode/MLNode'; // Adjusted path
import { RemoteCursor } from '../RemoteCursor/RemoteCursor'; // Adjusted path
import styles from './styles/Canvas.module.css'; // Adjusted path
import { NetworkStatsDisplay } from '../NetworkStatsDisplay/NetworkStatsDisplay'; // Adjusted path

const nodeTypes = { transformer: MLNode };

export const Canvas: React.FC = () => {
  const [nodes, setNodes, edges, setEdges] = useSyncedGraph();
  const { provider } = useYDoc();
  const remoteCursors = useRemoteCursors();
  const reactFlowWrapper = useRef<HTMLDivElement>(null);
  
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
  
  const handleMouseMove = useCallback(
    (event: React.MouseEvent<HTMLDivElement>) => {
      if (!provider || !provider.awareness || !reactFlowWrapper.current) return;

      const bounds = reactFlowWrapper.current.getBoundingClientRect();
      const x = event.clientX - bounds.left;
      const y = event.clientY - bounds.top;

      if (animationFrameId.current) {
        cancelAnimationFrame(animationFrameId.current);
      }

      animationFrameId.current = requestAnimationFrame(() => {
        if (
          !lastSentPosition.current ||
          lastSentPosition.current.x !== x ||
          lastSentPosition.current.y !== y
        ) {
          provider.awareness.setLocalStateField('cursor', { x, y });
          lastSentPosition.current = { x, y };
        }
      });
    },
    [provider],
  );

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
        className={styles.canvasWrapper}
        ref={reactFlowWrapper}
        onMouseMove={handleMouseMove}
      >
        <NetworkStatsDisplay />
        <ReactFlow
          nodes={nodes}
          edges={edges}
          onNodesChange={onNodesChange}
          onEdgesChange={onEdgesChange}
          onConnect={onConnect}
          nodeTypes={nodeTypes}
          fitView
          minZoom={0.1}
          maxZoom={4}
          zoomOnScroll={true}
          zoomOnPinch={true}
          zoomOnDoubleClick={true}
          preventScrolling={true}
        >
          <Background 
            variant={BackgroundVariant.Dots}
            gap={16}
            size={1}
            color="#e5e7eb"
          />
          <Controls 
            position="top-left"
            style={{
              top: '50%',
              transform: 'translateY(-50%)',
              left: '10px'
            }}
          />
        </ReactFlow>
        {remoteCursors.map((cursor: RenderableCursor) => (
          <RemoteCursor
            key={cursor.clientID}
            x={cursor.x}
            y={cursor.y}
            name={cursor.name}
            color={cursor.color}
          />
        ))}
      </div>
    </ReactFlowProvider>
  );
}; 