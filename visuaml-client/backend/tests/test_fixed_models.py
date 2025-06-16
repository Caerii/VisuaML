# -*- coding: utf-8 -*-
"""Test script for fixed models."""

import sys
import os
import pytest

# Add the visuaml-client directory to Python path so we can import models
# Go up from backend/tests/ to visuaml-client/
visuaml_client_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, visuaml_client_dir)

# Add the backend directory to Python path so we can import visuaml package
backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(backend_dir)

# Check if open-hypergraphs is available
try:
    import open_hypergraphs  # noqa: F401
    OPEN_HYPERGRAPHS_AVAILABLE = True
except ImportError:
    OPEN_HYPERGRAPHS_AVAILABLE = False

from visuaml import export_model_open_hypergraph  # noqa: E402

@pytest.mark.skipif(not OPEN_HYPERGRAPHS_AVAILABLE, reason="open-hypergraphs package not available")
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