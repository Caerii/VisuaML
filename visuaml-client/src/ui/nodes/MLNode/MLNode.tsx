/** @fileoverview Defines the MLNode component, the primary visual representation for a machine learning operation in the graph. It displays a header and, when selected, a details pane with more information. Styling and sub-components are managed separately. */
import React, { useMemo } from 'react';
import { Handle, Position, type NodeProps } from '@xyflow/react';
import classnames from 'classnames';
import NodeHeader from './components/NodeHeader';
import NodeDetailsPane from './components/NodeDetailsPane';
import { useMLNode } from './useMLNode';
import styles from './styles/MLNode.module.css';

const MLNode: React.FC<NodeProps> = (props) => {
  const {
    id,
    selected,
    type,
    data,
    layerDisplayName,
    subtextForHeader,
    inlineDynamicStyle,
  } = useMLNode(props);

  const nodeClasses = useMemo(() => classnames(
    styles.nodeBase,
    type ? `react-flow__node-${type}` : '',
    { 
      [styles.nodeSelected]: selected 
    }
  ), [type, selected]);

  return (
    <div style={inlineDynamicStyle} className={nodeClasses}>
      <Handle type="target" position={Position.Left} style={{ background: '#4b5563', width: '10px', height: '10px', borderRadius: '50%' }} />
      
      <NodeHeader 
        layerDisplayName={layerDisplayName} 
        subtext={subtextForHeader} 
        isSelected={selected}
      />

      <NodeDetailsPane data={data} nodeId={id} isSelected={selected} />
      
      <Handle type="source" position={Position.Right} style={{ background: '#4b5563', width: '10px', height: '10px', borderRadius: '50%' }} />
    </div>
  );
};

export default MLNode; 