# -*- coding: utf-8 -*-
"""Test script for fixed models."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from visuaml import export_model_open_hypergraph

def test_fixed_models():
    """Test all fixed models."""
    print("Testing Fixed Models with Open-Hypergraph Export")
    print("=" * 50)
    
    # Test FixedSimpleCNN
    print("\n1. Testing FixedSimpleCNN...")
    try:
        result = export_model_open_hypergraph(
            'models.FixedSimpleCNN', 
            sample_input_args=((1, 1, 28, 28),)
        )
        print(f"   ✅ Success: {len(result['nodes'])} nodes, {len(result['hyperedges'])} hyperedges")
    except Exception as e:
        print(f"   ❌ Failed: {str(e)[:100]}...")
    
    # Test FixedBasicRNN
    print("\n2. Testing FixedBasicRNN...")
    try:
        result = export_model_open_hypergraph(
            'models.FixedBasicRNN', 
            sample_input_args=((1, 10, 10),)
        )
        print(f"   ✅ Success: {len(result['nodes'])} nodes, {len(result['hyperedges'])} hyperedges")
    except Exception as e:
        print(f"   ❌ Failed: {str(e)[:100]}...")
    
    # Test FixedDemoNet
    print("\n3. Testing FixedDemoNet...")
    try:
        result = export_model_open_hypergraph(
            'models.FixedDemoNet', 
            sample_input_args=((1, 32),),
            sample_input_dtypes=['long']
        )
        print(f"   ✅ Success: {len(result['nodes'])} nodes, {len(result['hyperedges'])} hyperedges")
    except Exception as e:
        print(f"   ❌ Failed: {str(e)[:100]}...")
    
    print("\n" + "=" * 50)
    print("All fixed models tested!")

if __name__ == "__main__":
    test_fixed_models() 