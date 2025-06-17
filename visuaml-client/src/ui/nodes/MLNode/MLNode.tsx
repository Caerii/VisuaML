/** @fileoverview Defines the MLNode component, the primary visual representation for a machine learning operation in the graph. It displays a header and, when selected, a details pane with more information. Styling and sub-components are managed separately. */
import React, { useMemo, useEffect, useState } from 'react';
import { Handle, Position, type NodeProps } from '@xyflow/react';
import classnames from 'classnames';
import NodeHeader from './components/NodeHeader';
import NodeDetailsPane from './components/NodeDetailsPane';
import { useMLNode } from './useMLNode';

const MLNode: React.FC<NodeProps> = (props) => {
  const { id, selected, type, data, layerDisplayName, subtextForHeader, inlineDynamicStyle } =
    useMLNode(props);

  const [isExpanding, setIsExpanding] = useState(false);
  const [isCollapsing, setIsCollapsing] = useState(false);
  const [hasBeenSelected, setHasBeenSelected] = useState(false);
  const [isInitialized, setIsInitialized] = useState(false);

  // Initialize after a short delay to prevent initial cascade
  useEffect(() => {
    const timer = setTimeout(() => setIsInitialized(true), 100);
    return () => clearTimeout(timer);
  }, []);

  // Handle expansion/collapse animations
  useEffect(() => {
    if (!isInitialized) return; // Don't animate until initialized

    if (selected) {
      setHasBeenSelected(true);
      setIsCollapsing(false);
      setIsExpanding(true);
      const timer = setTimeout(() => setIsExpanding(false), 600);
      return () => clearTimeout(timer);
    } else if (hasBeenSelected) {
      // Only animate collapse if the node has been previously selected
      setIsExpanding(false);
      setIsCollapsing(true);
      const timer = setTimeout(() => setIsCollapsing(false), 600);
      return () => clearTimeout(timer);
    }
  }, [selected, hasBeenSelected, isInitialized]);

  const nodeClasses = useMemo(
    () =>
      classnames('ml-node', type ? `react-flow__node-${type}` : '', {
        'ml-node--selected': selected,
        'ml-node--expanding': isExpanding,
        'ml-node--collapsing': isCollapsing,
      }),
    [type, selected, isExpanding, isCollapsing],
  );

  return (
    <div style={inlineDynamicStyle} className={nodeClasses}>
      <Handle
        type="target"
        position={Position.Left}
        className="ml-node__handle ml-node__handle--input"
        style={{ top: '32px' }}
      />

      <NodeHeader
        layerDisplayName={layerDisplayName}
        subtext={subtextForHeader}
        isSelected={selected}
      />

      <NodeDetailsPane data={data} nodeId={id} isSelected={selected} />

      <Handle
        type="source"
        position={Position.Right}
        className="ml-node__handle ml-node__handle--output"
        style={{ top: '32px' }}
      />
    </div>
  );
};

export default MLNode;
