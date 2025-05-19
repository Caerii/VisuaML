import { describe, it, expect } from 'vitest';
import { TransformerNode } from '@/features/graph/nodes/TransformerNode';

describe('TransformerNode', () => {
  it('renders without crashing', () => {
    const node = { data:{ layerType:'Linear', name:'fc1', color:'#fff' } } as any;
    expect(() => TransformerNode(node)).not.toThrow();
  });
}); 