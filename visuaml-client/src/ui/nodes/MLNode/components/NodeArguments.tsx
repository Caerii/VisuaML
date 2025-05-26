/** @fileoverview Defines the NodeArguments component, which displays the positional (args) and keyword (kwargs) arguments of an MLNode within its details pane. It utilizes the ArgumentRenderer for individual argument rendering. */
import React from 'react';
import ArgumentRenderer from './ArgumentRenderer'; // Path will be relative to this new file
import type { MLNodeData, ResolvedArgValue } from '../../types'; // Adjusted path
import styles from '../styles/MLNode.module.css'; // Adjusted path

interface NodeArgumentsProps {
  args?: MLNodeData['args'];
  kwargs?: MLNodeData['kwargs'];
}

const NodeArguments: React.FC<NodeArgumentsProps> = ({ args, kwargs }) => {
  const hasArgs = Array.isArray(args) && args.length > 0;
  const hasKwargs = kwargs && typeof kwargs === 'object' && Object.keys(kwargs).length > 0;

  if (!hasArgs && !hasKwargs) {
    return null; // Render nothing if there are no args or kwargs
  }

  return (
    <React.Fragment>
      {hasArgs && (
        <div className={styles.infoField}> {/* Reuse .infoField for margin */}
          <strong className={styles.infoLabel}>Args:</strong>
          <ul className={styles.argumentsList}>
            {args!.map((arg: ResolvedArgValue, index: number) => (
              <li key={`arg-${index}`}>
                <ArgumentRenderer arg={arg} keyPrefix={`arg-val-${index}`} />
              </li>
            ))}
          </ul>
        </div>
      )}
      {hasKwargs && (
        <div className={styles.infoField}> {/* Reuse .infoField for margin */}
          <strong className={styles.infoLabel}>KwArgs:</strong>
          <ul className={styles.argumentsList}>
            {Object.entries(kwargs!).map(([key, value]: [string, ResolvedArgValue], i: number) => (
              <li key={`kwarg-${key}-${i}`}>
                <span className={styles.argumentKey}>{key}:</span>{' '}
                <ArgumentRenderer arg={value} keyPrefix={`kwarg-val-${key}-${i}`} />
              </li>
            ))}
          </ul>
        </div>
      )}
    </React.Fragment>
  );
};

export default NodeArguments; 