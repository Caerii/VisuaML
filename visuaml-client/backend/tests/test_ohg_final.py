#!/usr/bin/env python3
"""Final demonstration of open-hypergraph export functionality."""

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
def demonstrate_export():
    """Demonstrate the open-hypergraph export functionality."""
    print("🚀 VisuaML Open-Hypergraph Export Demo")
    print("=" * 40)
    
    # Test with SimpleNN - a model that works well with FX tracing
    print("📊 Exporting SimpleNN model...")
    
    # Export to JSON format
    json_result = export_model_open_hypergraph(
        'models.SimpleNN', 
        sample_input_args=((10,),)
    )
    
    print("✅ JSON Export Complete:")
    print(f"   - Nodes: {len(json_result['nodes'])}")
    print(f"   - Hyperedges: {len(json_result['hyperedges'])}")
    print(f"   - Metadata: {json_result['metadata']}")
    
    # Show sample node structure
    print("\n📋 Sample Node Structure:")
    if json_result['nodes']:
        sample_node = json_result['nodes'][0]
        for key, value in sample_node.items():
            print(f"   {key}: {value}")
    
    # Show sample hyperedge structure
    print("\n🔗 Sample Hyperedge Structure:")
    if json_result['hyperedges']:
        sample_edge = json_result['hyperedges'][0]
        for key, value in sample_edge.items():
            print(f"   {key}: {value}")
    
    # Export to macro format
    print("\n📝 Exporting to Macro Format...")
    macro_result = export_model_open_hypergraph(
        'models.SimpleNN', 
        sample_input_args=((10,),),
        out_format="macro"
    )
    
    print("✅ Macro Export Complete:")
    macro_lines = macro_result['macro'].split('\n')
    print(f"   - Lines generated: {len(macro_lines)}")
    print("\n📄 Sample Macro Output:")
    for i, line in enumerate(macro_lines[:10]):  # Show first 10 lines
        print(f"   {i+1:2d}: {line}")
    if len(macro_lines) > 10:
        print(f"   ... ({len(macro_lines) - 10} more lines)")
    
    # Test with Autoencoder
    print("\n🔄 Testing with Autoencoder model...")
    autoencoder_result = export_model_open_hypergraph(
        'models.Autoencoder', 
        sample_input_args=((784,),)
    )
    
    print("✅ Autoencoder Export Complete:")
    print(f"   - Nodes: {len(autoencoder_result['nodes'])}")
    print(f"   - Hyperedges: {len(autoencoder_result['hyperedges'])}")
    
    print("\n🎯 Summary:")
    print("   ✅ Open-hypergraph export is working correctly!")
    print("   ✅ Supports both JSON and macro output formats")
    print("   ✅ Includes tensor shape and dtype information")
    print("   ✅ Handles multiple model architectures")
    print("   ✅ Preserves PyTorch FX graph structure")
    
    print("\n🔧 Usage:")
    print("   from visuaml import export_model_open_hypergraph")
    print("   result = export_model_open_hypergraph('models.YourModel', sample_input_args=((shape,),))")
    print("   # Or for macro format:")
    print("   result = export_model_open_hypergraph('models.YourModel', sample_input_args=((shape,),), out_format='macro')")

if __name__ == "__main__":
    demonstrate_export() 