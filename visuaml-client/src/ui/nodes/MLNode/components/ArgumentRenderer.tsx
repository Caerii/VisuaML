/** @fileoverview Defines the ArgumentRenderer component, a recursive renderer for displaying various types of arguments (primitives, arrays, objects, and source node links) within the MLNode details pane. It handles the visual representation of how arguments are connected or defined. */
import React from 'react';
import type { ResolvedArgValue, ResolvedArgSourceNode } from '../../types'; // Corrected path
import styles from '../styles/MLNode.module.css'; // Adjusted path

interface ArgumentRendererProps {
  arg: ResolvedArgValue;
  keyPrefix: string; // For generating unique React keys during recursion
}

const ArgumentRenderer: React.FC<ArgumentRendererProps> = ({ arg, keyPrefix }) => {
  if (arg === null) {
    return <span key={keyPrefix}>null</span>;
  }

  // Check for ResolvedArgSourceNode (must check specific properties)
  if (typeof arg === 'object' && ('source_node' in arg || 'source_nodes' in arg)) {
    const sourceNodeArg = arg as ResolvedArgSourceNode;
    let sourceText = 'unknown source';

    if (sourceNodeArg.source_nodes && sourceNodeArg.source_nodes.length > 0) {
      sourceText = `nodes [${sourceNodeArg.source_nodes.join(', ')}]`;
    } else if (sourceNodeArg.source_node) {
      sourceText = `node '${sourceNodeArg.source_node}'`;
    }

    return (
      <span key={keyPrefix} className={styles.sourceNodeLink}>
        from {sourceText}
        {sourceNodeArg.transitive && (
          <span className={styles.transitiveHint}> (via filtered)</span>
        )}
      </span>
    );
  }

  if (Array.isArray(arg)) {
    return (
      <span key={`${keyPrefix}-array`}>
        [
        {arg.map((item: ResolvedArgValue, index: number) => (
          <React.Fragment key={`${keyPrefix}-item-${index}`}>
            <ArgumentRenderer arg={item} keyPrefix={`${keyPrefix}-val-${index}`} />
            {index < arg.length - 1 ? ", " : ""}
          </React.Fragment>
        ))}
        ]
      </span>
    );
  }

  if (typeof arg === 'object' && arg !== null) {
    return (
      <span key={`${keyPrefix}-object`}>
        {"{"}
        {Object.entries(arg as { [key: string]: ResolvedArgValue }).map(
          ([entryKey, value]: [string, ResolvedArgValue], index: number, arr: Array<[string, ResolvedArgValue]>) => (
            <React.Fragment key={`${keyPrefix}-prop-${entryKey}`}>
              <span style={{ fontStyle: 'italic' }}>'{entryKey}'</span>:{" "}
              <ArgumentRenderer arg={value} keyPrefix={`${keyPrefix}-val-${entryKey}`} />
              {index < arr.length - 1 ? ", " : ""}
            </React.Fragment>
          )
        )}
        {"}"}
      </span>
    );
  }

  // Default to string representation for primitives (string, number, boolean)
  return <span key={keyPrefix}>{String(arg)}</span>;
};

export default ArgumentRenderer; 