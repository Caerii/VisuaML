/** @fileoverview Defines the NodeArguments component and its recursive sub-renderer for displaying MLNode arguments. */
import React from 'react';
import type { ResolvedArgValue, MLNodeData } from '../../../types';
import { isSourceNode } from './guards';
import styles from '../../styles/MLNode.module.css';

// ===========================
// ARGUMENT RENDERER
// ===========================

interface ArgumentRendererProps {
  arg: ResolvedArgValue;
}

const ArgumentRenderer: React.FC<ArgumentRendererProps> = ({ arg }) => {
  if (arg === null) {
    return <span>null</span>;
  }

  if (isSourceNode(arg)) {
    const sourceText = arg.source_nodes?.length
      ? `nodes [${arg.source_nodes.join(', ')}]`
      : `node '${arg.source_node}'`;

    return (
      <span className={styles.sourceNodeLink}>
        from {sourceText}
        {arg.transitive && <span className={styles.transitiveHint}> (via filtered)</span>}
      </span>
    );
  }

  if (Array.isArray(arg)) {
    return (
      <span>
        [
        {arg.map((item, index) => (
          <React.Fragment key={index}>
            <ArgumentRenderer arg={item} />
            {index < arg.length - 1 && ', '}
          </React.Fragment>
        ))}
        ]
      </span>
    );
  }

  if (typeof arg === 'object') {
    return (
      <span>
        {'{'}
        {Object.entries(arg).map(([key, value], index, arr) => (
          <React.Fragment key={key}>
            <span style={{ fontStyle: 'italic' }}>'{key}'</span>: <ArgumentRenderer arg={value} />
            {index < arr.length - 1 && ', '}
          </React.Fragment>
        ))}
        {'}'}
      </span>
    );
  }

  return <span>{String(arg)}</span>;
};

// ===========================
// MAIN COMPONENT
// ===========================

interface NodeArgumentsProps {
  args?: MLNodeData['args'];
  kwargs?: MLNodeData['kwargs'];
}

const NodeArguments: React.FC<NodeArgumentsProps> = ({ args, kwargs }) => {
  const hasArgs = Array.isArray(args) && args.length > 0;
  const hasKwargs = kwargs && typeof kwargs === 'object' && Object.keys(kwargs).length > 0;

  if (!hasArgs && !hasKwargs) {
    return null;
  }

  return (
    <div className={styles.detailsSection}>
      <div className={styles.detailsSectionTitle}>Arguments</div>
      {hasArgs && (
        <div className={styles.infoField}>
          <strong className={styles.infoLabel}>Args:</strong>
          <ul className={styles.argumentsList}>
            {args.map((arg, index) => (
              <li key={index}>
                <ArgumentRenderer arg={arg} />
              </li>
            ))}
          </ul>
        </div>
      )}
      {hasKwargs && (
        <div className={styles.infoField}>
          <strong className={styles.infoLabel}>KwArgs:</strong>
          <ul className={styles.argumentsList}>
            {Object.entries(kwargs).map(([key, value]) => (
              <li key={key}>
                <span className={styles.argumentKey}>{key}:</span> <ArgumentRenderer arg={value} />
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
};

export default NodeArguments;
