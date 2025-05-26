/** @fileoverview Defines the NodeDetailsPane component, which acts as a container for all detailed information displayed when an MLNode is selected. It typically includes NodeInfoFields and NodeArguments. */
import React from 'react';
import NodeInfoFields from './NodeInfoFields'; // Path will be relative to this new file
import NodeArguments from './NodeArguments'; // Path will be relative to this new file
import type { MLNodeData } from '../../types'; // Adjusted path
import styles from '../styles/MLNode.module.css'; // Adjusted path

interface NodeDetailsPaneProps {
  data: MLNodeData;
  nodeId: string;
}

const NodeDetailsPane: React.FC<NodeDetailsPaneProps> = ({ data, nodeId }) => {
  return (
    <div className={styles.detailsPane} > {/* Use detailsPane style */}
      <NodeInfoFields data={data} nodeId={nodeId} />
      <NodeArguments args={data.args} kwargs={data.kwargs} />
    </div>
  );
};

export default NodeDetailsPane; 