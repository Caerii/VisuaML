import dagre from 'dagre';
import { type Node, type Edge } from '@xyflow/react';

export type { Node, Edge };

interface LayoutOptions {
  rankdir?: 'TB' | 'BT' | 'LR' | 'RL'; // Top-to-bottom, Bottom-to-top, Left-to-right, Right-to-left
  align?: 'UL' | 'UR' | 'DL' | 'DR';
  nodesep?: number; // separation between nodes in the same rank
  edgesep?: number; // separation between edges
  ranksep?: number; // separation between ranks
  marginx?: number;
  marginy?: number;
}

const DEFAULT_NODE_WIDTH = 200;
const DEFAULT_NODE_HEIGHT = 60;

export const autoLayout = (
  nodes: Node[],
  edges: Edge[],
  options?: LayoutOptions,
): { nodes: Node[]; edges: Edge[] } => {
  const g = new dagre.graphlib.Graph();

  // Set layout options
  const layoutOptions = {
    rankdir: 'LR',
    nodesep: 70,
    ranksep: 90,
    ...options,
  };
  g.setGraph(layoutOptions);

  // Default to assigning a new object for edge labels, if not present
  g.setDefaultEdgeLabel(() => ({}));

  nodes.forEach((node) => {
    // Use provided dimensions or defaults
    const width = node.width ?? DEFAULT_NODE_WIDTH;
    const height = node.height ?? DEFAULT_NODE_HEIGHT;
    g.setNode(node.id, { label: node.id, width, height });
  });

  edges.forEach((edge) => {
    g.setEdge(edge.source, edge.target);
  });

  dagre.layout(g);

  const laidOutNodes = nodes.map((node) => {
    const nodeWithPosition = g.node(node.id);
    // Adjust position to be center of the node, React Flow uses top-left
    const x = nodeWithPosition.x - (node.width ?? DEFAULT_NODE_WIDTH) / 2;
    const y = nodeWithPosition.y - (node.height ?? DEFAULT_NODE_HEIGHT) / 2;

    return {
      ...node,
      position: { x, y },
      // Update width and height if they were defaulted by dagre and not initially set
      width: node.width ?? DEFAULT_NODE_WIDTH,
      height: node.height ?? DEFAULT_NODE_HEIGHT,
    };
  });

  return { nodes: laidOutNodes, edges };
};
