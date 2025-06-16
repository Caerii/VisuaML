/** @fileoverview Type guards for identifying specific argument types. */
import type { ResolvedArgValue, ResolvedArgSourceNode } from '../../../types';

export const isSourceNode = (arg: ResolvedArgValue): arg is ResolvedArgSourceNode => {
  return typeof arg === 'object' && arg !== null && ('source_node' in arg || 'source_nodes' in arg);
};
