/** @fileoverview Defines the NodeHeader component, responsible for rendering the compact, always-visible part of an MLNode. */
import React from 'react';

interface NodeHeaderProps {
  layerDisplayName: string;
  subtext?: string;
  isSelected: boolean;
}

const NodeHeader: React.FC<NodeHeaderProps> = ({ layerDisplayName, subtext }) => {
  return (
    <div className="ml-node__header">
      <div className="ml-node__title">{layerDisplayName}</div>
      {subtext && <div className="ml-node__subtitle">{subtext}</div>}
    </div>
  );
};

export default NodeHeader;
