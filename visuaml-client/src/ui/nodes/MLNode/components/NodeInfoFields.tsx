/** @fileoverview Defines the NodeInfoFields component, responsible for rendering static informational fields within the MLNode's details pane, such as node name, operation type, target, and output shape. */
import React from 'react';
import type { MLNodeData } from '../../types'; // Adjusted path
import styles from '../styles/MLNode.module.css'; // Adjusted path

interface NodeInfoFieldsProps {
  data: MLNodeData;
  nodeId: string; // Fallback for name if data.name is not present
}

const NodeInfoFields: React.FC<NodeInfoFieldsProps> = ({ data, nodeId }) => {
  return (
    <div className={styles.detailsSection}> {/* Wrap fields in a section */}
      <div className={styles.detailsSectionTitle}>Info</div>
      <div className={styles.infoField}>
        <strong>Name:</strong> <span className={styles.infoFieldValue}>{data.name || nodeId}</span>
      </div>
      {data.layerType && data.op !== data.layerType && (
        <div className={styles.infoField}>
          <strong>Type:</strong> <span className={styles.infoFieldValue}>{data.layerType}</span>
        </div>
      )}
      <div className={styles.infoField}>
        <strong>Operation:</strong> <span className={styles.infoFieldValue}>{data.op || 'N/A'}</span>
      </div>
      {data.target && String(data.target) !== data.name && String(data.target) !== data.layerType && (
        <div className={styles.infoField}>
          <strong>Target:</strong> <span className={styles.infoFieldValue}>{String(data.target)}</span>
        </div>
      )}
      {data.outputShape && (
        <div className={styles.infoField}>
          <strong>Output Shape:</strong> <span className={styles.infoFieldValue}>{data.outputShape}</span>
        </div>
      )}
    </div>
  );
};

export default NodeInfoFields; 