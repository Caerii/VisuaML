import { describe, it, expect } from 'vitest';

describe('Smoke Tests', () => {
  it('should pass basic assertion', () => {
    expect(1 + 1).toBe(2);
  });

  it('should be able to import basic utilities', () => {
    // Test that we can import from our lib directory
    expect(typeof import('../lib/colors')).toBe('object');
  });
}); 