/** @fileoverview Defines the NodeDetailsPane component, which acts as a container for all detailed information displayed when an MLNode is selected. */
import React from 'react';
import NodeInfoFields from '../NodeInfoFields';
import NodeArguments from '../NodeArguments';
import OutputShapeVisualizer from '../OutputShapeVisualizer';
import type { MLNodeData } from '../../../types';
import styles from '../../styles/MLNode.module.css';

interface NodeDetailsPaneProps {
  data: MLNodeData;
  nodeId: string;
  isSelected: boolean;
}

const NodeDetailsPane: React.FC<NodeDetailsPaneProps> = ({ data, nodeId, isSelected }) => {
  return (
    <div className={`${styles.detailsPane} ${isSelected ? styles.detailsPaneOpen : ''}`}>
      <NodeInfoFields data={data} nodeId={nodeId} />
      <NodeArguments args={data.args} kwargs={data.kwargs} />
      {data.outputShape && <OutputShapeVisualizer outputShape={data.outputShape} />}
    </div>
  );
};

export default NodeDetailsPane;
