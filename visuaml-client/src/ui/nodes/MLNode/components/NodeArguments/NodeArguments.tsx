/** @fileoverview Defines the NodeArguments component and its recursive sub-renderer for displaying MLNode arguments. */
import React from 'react';
import type { ResolvedArgValue, MLNodeData } from '../../../types';
import { isSourceNode } from './guards';

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
      <span className="ml-node__source-link">
        from {sourceText}
        {arg.transitive && <span className="ml-node__transitive-hint"> (via filtered)</span>}
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
    <div className="ml-node__details-section">
      <div className="ml-node__details-title">Arguments</div>
      {hasArgs && (
        <div className="ml-node__info-field">
          <strong className="ml-node__info-label ml-node__info-label--bright">Args:</strong>
          <ul className="ml-node__arguments-list">
            {args.map((arg, index) => (
              <li key={index} className="ml-node__argument-item">
                <ArgumentRenderer arg={arg} />
              </li>
            ))}
          </ul>
        </div>
      )}
      {hasKwargs && (
        <div className="ml-node__info-field">
          <strong className="ml-node__info-label ml-node__info-label--bright">KwArgs:</strong>
          <ul className="ml-node__arguments-list">
            {Object.entries(kwargs).map(([key, value]) => (
              <li key={key} className="ml-node__argument-item">
                <span className="ml-node__argument-key">{key}:</span> <ArgumentRenderer arg={value} />
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
};

export default NodeArguments;
