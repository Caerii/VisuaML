/** @fileoverview Defines the NodeHeader component, responsible for rendering the compact, always-visible part of an MLNode. It displays the layer's display name and a subtext (e.g., node name or target). */
import React from 'react';
import styles from '../styles/MLNode.module.css'; // Adjusted path

interface NodeHeaderProps {
  layerDisplayName: string;
  subtext?: string;
  isSelected: boolean;
}

const NodeHeader: React.FC<NodeHeaderProps> = ({ layerDisplayName, subtext, isSelected }) => {
  return (
    // Use the .nodeHeader class for the container div
    <div className={`${styles.nodeHeader} ${isSelected ? styles.nodeHeaderSelected : ''}`}>
      <div 
        className={styles.nodeHeaderTitle} 
      >
        {layerDisplayName}
      </div>
      {subtext && (
        <div className={styles.nodeHeaderSubtext}>{subtext}</div>
      )}
    </div>
  );
};

export default NodeHeader; 