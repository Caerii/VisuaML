/** @fileoverview Controller hook for the MLNode component. */
import type { NodeProps } from '@xyflow/react';
import type { MLNodeData } from '../types';

export const useMLNode = (props: NodeProps) => {
  const { id, selected, type, data: rawData } = props;
  const data = rawData as MLNodeData;

  const nodeColor = data.color || '#dddddd';
  const layerDisplayName = data.layerType || data.op || type || 'Node';
  const subtextForHeader = data.name || data.target;

  const inlineDynamicStyle: React.CSSProperties = {
    backgroundColor: nodeColor,
  };

  return {
    id,
    selected,
    type,
    data,
    layerDisplayName,
    subtextForHeader,
    inlineDynamicStyle,
  };
};
