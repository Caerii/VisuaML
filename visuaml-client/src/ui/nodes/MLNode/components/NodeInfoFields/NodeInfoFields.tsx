/** @fileoverview Defines the NodeInfoFields component, responsible for rendering static informational fields within the MLNode's details pane. */
import React from 'react';
import type { MLNodeData } from '../../../types';

interface NodeInfoFieldsProps {
  data: MLNodeData;
  nodeId: string; // Fallback for name if data.name is not present
}

const NodeInfoFields: React.FC<NodeInfoFieldsProps> = ({ data, nodeId }) => {
  return (
    <div className="ml-node__details-section">
      <div className="ml-node__details-title">Info</div>
      <div className="ml-node__info-field">
        <strong className="ml-node__info-label">Name:</strong> 
        <span className="ml-node__info-value">{data.name || nodeId}</span>
      </div>
      {data.layerType && data.op !== data.layerType && (
        <div className="ml-node__info-field">
          <strong className="ml-node__info-label">Type:</strong> 
          <span className="ml-node__info-value">{data.layerType}</span>
        </div>
      )}
      <div className="ml-node__info-field">
        <strong className="ml-node__info-label">Operation:</strong>
        <span className="ml-node__info-value">{data.op || 'N/A'}</span>
      </div>
      {data.target &&
        String(data.target) !== data.name &&
        String(data.target) !== data.layerType && (
          <div className="ml-node__info-field">
            <strong className="ml-node__info-label">Target:</strong>
            <span className="ml-node__info-value">{String(data.target)}</span>
          </div>
        )}
      {data.outputShape && (
        <div className="ml-node__info-field">
          <strong className="ml-node__info-label">Output Shape:</strong>
          <span className="ml-node__info-value">{data.outputShape}</span>
        </div>
      )}
    </div>
  );
};

export default NodeInfoFields;
