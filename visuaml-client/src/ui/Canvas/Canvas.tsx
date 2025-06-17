/** @fileoverview Defines the main interactive canvas for VisuaML, integrating React Flow for graph visualization, Yjs for real-time collaboration, and components for displaying network statistics and remote user cursors. */
import {
  ReactFlow,
  Background,
  BackgroundVariant,
  Controls,
  ReactFlowProvider,
} from '@xyflow/react';
import type { RenderableCursor } from '../../y/usePresence';
import '@xyflow/react/dist/style.css';
import MLNode from '../nodes/MLNode/MLNode';
import { RemoteCursor } from '../RemoteCursor/RemoteCursor';
import { NetworkStatsDisplay } from '../NetworkStatsDisplay/NetworkStatsDisplay';
import { useCanvas } from './useCanvas';

const nodeTypes = { transformer: MLNode };

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
          fitView
          minZoom={0.1}
          maxZoom={4}
          zoomOnScroll={true}
          zoomOnPinch={true}
          zoomOnDoubleClick={true}
          preventScrolling={true}
          multiSelectionKeyCode={null}
          selectNodesOnDrag={false}
          defaultViewport={{ x: 0, y: 0, zoom: 1 }}
        >
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
