/** @fileoverview Defines the main interactive canvas for VisuaML, integrating React Flow for graph visualization, Yjs for real-time collaboration, and components for displaying network statistics and remote user cursors. */
import { useEffect, useRef } from 'react';
import {
  ReactFlow,
  Background,
  BackgroundVariant,
  Controls,
  ReactFlowProvider,
  useReactFlow,
  useStore,
} from '@xyflow/react';
import type { RenderableCursor } from '../../y/usePresence';
import '@xyflow/react/dist/style.css';
import MLNode from '../nodes/MLNode/MLNode';
import { RemoteCursor } from '../RemoteCursor/RemoteCursor';
import { NetworkStatsDisplay } from '../NetworkStatsDisplay/NetworkStatsDisplay';
import { useCanvas } from './useCanvas';
import { useNetworkStore } from '../../store/networkStore';

const nodeTypes = { transformer: MLNode };

/** Fits the view to nodes at a comfortable scale when the graph first loads (e.g. after Simple CNN auto-load). */
function FitViewOnFirstLoad() {
  const { fitView } = useReactFlow();
  const nodeCount = useStore((s) => (s.nodes?.length ?? 0));
  const hasFitted = useRef(false);

  useEffect(() => {
    if (nodeCount === 0 || hasFitted.current) return;
    hasFitted.current = true;
    const t = setTimeout(() => {
      fitView({
        padding: 0.2,
        maxZoom: 1.4,
        duration: 400,
      });
    }, 150);
    return () => clearTimeout(t);
  }, [nodeCount, fitView]);

  return null;
}

export const Canvas: React.FC = () => {
  const {
    nodes,
    edges,
    onNodesChange,
    onEdgesChange,
    onConnect,
    reactFlowWrapper,
    handleMouseMove,
    remoteCursors,
  } = useCanvas();
  const isGraphInteractive = useNetworkStore((state) => state.isGraphInteractive);

  return (
    <ReactFlowProvider>
      <div className="canvas" ref={reactFlowWrapper} onMouseMove={handleMouseMove}>
        <div className="canvas__stats-panel">
        <NetworkStatsDisplay />
        </div>
        <ReactFlow
          nodes={nodes}
          edges={edges}
          onNodesChange={onNodesChange}
          onEdgesChange={onEdgesChange}
          onConnect={onConnect}
          nodeTypes={nodeTypes}
          minZoom={0.1}
          maxZoom={4}
          nodesDraggable={isGraphInteractive}
          panOnDrag={isGraphInteractive}
          zoomOnScroll={isGraphInteractive}
          zoomOnPinch={isGraphInteractive}
          zoomOnDoubleClick={isGraphInteractive}
          preventScrolling={true}
          multiSelectionKeyCode={null}
          selectNodesOnDrag={false}
          defaultViewport={{ x: 0, y: 0, zoom: 1 }}
        >
          <FitViewOnFirstLoad />
          <Background variant={BackgroundVariant.Dots} gap={16} size={1} color="#cbd5e1" />
          <Controls
            position="top-left"
            style={{
              top: '50%',
              transform: 'translateY(-50%)',
              left: '10px',
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
