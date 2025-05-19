import { describe, it, expect } from 'vitest';
import { TransformerNode } from '@/features/graph/nodes/TransformerNode';

// Define a minimal type for the node prop for this smoke test
interface TestNode {
  data: {
    layerType: string;
    name: string;
    color: string;
  };
  // Add other properties if TransformerNode expects them, e.g., id, position
}

describe('TransformerNode', () => {
  it('renders without crashing', () => {
    const node: TestNode = { data:{ layerType:'Linear', name:'fc1', color:'#fff' } };
    expect(() => TransformerNode(node)).not.toThrow();
  });
}); 