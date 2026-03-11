/**
 * Main demo networks file
 * Aggregates all demo networks and applies auto-layout
 * 
 * To regenerate these demos from actual models, run:
 *   pnpm generate-demos-fast
 * 
 * This will process the actual models and generate JSON files in demo-networks/
 */
import type { DemoNetwork } from './types';
import { autoLayout } from '../autoLayout';

// Import individual demo networks
import { simpleCNNDemo } from './demos/simpleCNN';
import { simpleMLPDemo } from './demos/simpleMLP';
import { resnetBlockDemo } from './demos/resnetBlock';
import { transformerBlockDemo } from './demos/transformerBlock';
import { lstmDemo } from './demos/lstm';
import { autoencoderDemo } from './demos/autoencoder';
import { deepCNNDemo } from './demos/deepCNN';

/**
 * All demo networks with auto-layout applied
 */
const demosWithLayout: DemoNetwork[] = [
  simpleCNNDemo,
  simpleMLPDemo,
  resnetBlockDemo,
  transformerBlockDemo,
  lstmDemo,
  autoencoderDemo,
  deepCNNDemo,
].map((demo) => {
  const { nodes, edges } = autoLayout(demo.nodes, demo.edges);
  return {
    ...demo,
    nodes,
    edges,
  };
});

export const DEMO_NETWORKS: DemoNetwork[] = demosWithLayout;

/**
 * Get a demo network by ID
 */
export const getDemoNetwork = (id: string): DemoNetwork | undefined => {
  return DEMO_NETWORKS.find((demo) => demo.id === id);
};

/**
 * Check if a network ID is a demo network
 */
export const isDemoNetwork = (id: string): boolean => {
  return DEMO_NETWORKS.some((demo) => demo.id === id);
};
