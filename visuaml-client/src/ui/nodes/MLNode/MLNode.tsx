/** @fileoverview Defines the MLNode component, the primary visual representation for a machine learning operation in the graph. It displays a header and, when selected, a details pane with more information. Styling and sub-components are managed separately. */
import React from 'react';
import { Handle, Position, type NodeProps } from '@xyflow/react';
import classnames from 'classnames';
import type { MLNodeData } from '../types'; // Path to types.ts is now one level up
import NodeHeader from './components/NodeHeader'; // Path to sub-component
import NodeDetailsPane from './components/NodeDetailsPane'; // Path to sub-component
import styles from './styles/MLNode.module.css';

const MLNode: React.FC<NodeProps> = (props) => {
  const { id, selected, type } = props;
  const data = props.data as MLNodeData;

  const nodeColor = data.color || '#dddddd';
  const layerDisplayName = data.layerType || data.op || type || 'Node';
  const subtextForHeader = data.name || data.target;

  const nodeClasses = classnames(
    styles.nodeBase,
    type ? `react-flow__node-${type}` : '',
    { 
      [styles.nodeSelected]: selected 
    }
  );

  const inlineDynamicStyle: React.CSSProperties = {
    backgroundColor: nodeColor,
  };

  return (
    <div style={inlineDynamicStyle} className={nodeClasses}>
      <Handle type="target" position={Position.Left} style={{ background: '#4b5563' }} />
      
      <NodeHeader 
        layerDisplayName={layerDisplayName} 
        subtext={subtextForHeader} 
        isSelected={selected}
      />

      {selected && (
        <NodeDetailsPane data={data} nodeId={id} />
      )}
      
      <Handle type="source" position={Position.Right} style={{ background: '#4b5563' }} />
    </div>
  );
};

export default MLNode; 