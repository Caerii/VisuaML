/** @fileoverview Defines the NodeHeader component, responsible for rendering the compact, always-visible part of an MLNode. */
import React from 'react';
import styles from '../../styles/MLNode.module.css';

interface NodeHeaderProps {
  layerDisplayName: string;
  subtext?: string;
  isSelected: boolean;
}

const NodeHeader: React.FC<NodeHeaderProps> = ({ layerDisplayName, subtext, isSelected }) => {
  return (
    <div className={`${styles.nodeHeader} ${isSelected ? styles.nodeHeaderSelected : ''}`}>
      <div className={styles.nodeHeaderTitle}>{layerDisplayName}</div>
      {subtext && <div className={styles.nodeHeaderSubtext}>{subtext}</div>}
    </div>
  );
};

export default NodeHeader;
