/** @fileoverview This hook encapsulates all the complex logic for the main interactive canvas, including state synchronization with Y.js, React Flow callbacks, and real-time cursor tracking. */
import { useSyncedGraph } from '../../y/useSyncedGraph';
import { useYDoc } from '../../y/DocProvider';
import { useRemoteCursors } from '../../y/usePresence';
import { useCallback, useRef, useEffect } from 'react';
import type { Connection, Edge, Node, NodeChange, EdgeChange } from '@xyflow/react';
import { applyNodeChanges, applyEdgeChanges, addEdge } from '@xyflow/react';

export const useCanvas = () => {
  const [nodes, setNodes, edges, setEdges] = useSyncedGraph();
  const { provider } = useYDoc();
  const remoteCursors = useRemoteCursors();
  const reactFlowWrapper = useRef<HTMLDivElement>(null);

  const animationFrameId = useRef<number | null>(null);
  const lastSentPosition = useRef<{ x: number; y: number } | null>(null);

  const onNodesChange = useCallback(
    (changes: NodeChange[]) => {
      const updatedNodes = applyNodeChanges(changes, nodes) as Node[];
      
      // Update node dimensions based on selection state
      const nodesWithCorrectDimensions = updatedNodes.map(node => ({
        ...node,
        width: node.selected ? 280 : 180, // Match CSS: 280px when selected, 180px when not
      }));
      
      setNodes(nodesWithCorrectDimensions);
    },
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

  return {
    nodes,
    edges,
    onNodesChange,
    onEdgesChange,
    onConnect,
    reactFlowWrapper,
    handleMouseMove,
    remoteCursors,
  };
};
