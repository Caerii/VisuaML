/**
 * Client-side demo networks that work without backend servers
 * 
 * These are placeholder demos. To use actual model data from the monorepo, run:
 *   pnpm generate-demos-fast
 * 
 * This will process the actual models and generate JSON files in demo-networks/
 * 
 * Demo networks are now organized in the demoNetworks/ directory.
 * This file re-exports everything for backward compatibility.
 */
export type { DemoNetwork } from './demoNetworks/types';
export {
  DEMO_NETWORKS,
  getDemoNetwork,
  isDemoNetwork,
} from './demoNetworks/index';
