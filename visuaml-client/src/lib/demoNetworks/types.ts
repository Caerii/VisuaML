/**
 * Type definitions for demo networks
 */
import type { Node, Edge } from '@xyflow/react';
import type { MLNodeData } from '../../ui/nodes/types';

export interface DemoNetwork {
  id: string;
  name: string;
  description: string;
  nodes: Node<MLNodeData>[];
  edges: Edge[];
}
