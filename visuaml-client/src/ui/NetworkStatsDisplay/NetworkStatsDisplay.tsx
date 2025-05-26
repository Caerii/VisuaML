/** @fileoverview Defines the NetworkStatsDisplay component, which shows key information about the currently loaded network graph, such as node/edge counts, input shapes, and component types. It sources its data from a Zustand store. */
import React from 'react';
import { useNetworkStore } from '../../store/networkStore'; // Adjusted path
import styles from './styles/NetworkStatsDisplay.module.css'; // Adjusted path

export const NetworkStatsDisplay: React.FC = () => {
  const { facts } = useNetworkStore();

  if (!facts || facts.isLoadingGraph) {
    if (facts && facts.networkName && facts.isLoadingGraph) {
      return (
        <div className={styles.statsContainer}>
          <div className={styles.statItem}>
            <strong>{facts.networkName}</strong>
          </div>
          <div className={styles.loadingGraphMessage}>Loading graph data...</div>
        </div>
      );
    }
    return <div className={styles.statsContainer}>Loading network data...</div>;
  }

  return (
    <div className={styles.statsContainer}>
      {facts.networkName && (
        <div className={styles.statItem}>
          <strong>{facts.networkName}</strong>
        </div>
      )}
      <div className={styles.statItem}>Nodes: {facts.numNodes}</div>
      <div className={styles.statItem}>Edges: {facts.numEdges}</div>
      
      {facts.inputShapes && facts.inputShapes.length > 0 && (
        <div className={styles.statItem}>
          Input Shapes:
          <ul className={styles.list}>
            {facts.inputShapes.map((shape, index) => (
              <li key={index}>{shape}</li>
            ))}
          </ul>
        </div>
      )}

      <div className={styles.statItem}>
        Component Types ({facts.componentTypes?.length || 0}):
      </div>
      {facts.componentTypes && facts.componentTypes.length > 0 ? (
        <div className={styles.tagContainer}>
          {facts.componentTypes.map((type) => (
            <span key={type} className={styles.tagItem}>
              {type}
            </span>
          ))}
        </div>
      ) : (
        <div className={styles.noTypesMessage}>No component types found.</div>
      )}
    </div>
  );
}; 