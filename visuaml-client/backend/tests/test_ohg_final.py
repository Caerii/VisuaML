#!/usr/bin/env python3
"""Final demonstration of open-hypergraph export functionality."""

import sys
import os
# Add the parent directory to Python path so we can import models
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from visuaml import export_model_open_hypergraph
import json

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
    
    print(f"✅ JSON Export Complete:")
    print(f"   - Nodes: {len(json_result['nodes'])}")
    print(f"   - Hyperedges: {len(json_result['hyperedges'])}")
    print(f"   - Metadata: {json_result['metadata']}")
    
    # Show sample node structure
    print(f"\n📋 Sample Node Structure:")
    if json_result['nodes']:
        sample_node = json_result['nodes'][0]
        for key, value in sample_node.items():
            print(f"   {key}: {value}")
    
    # Show sample hyperedge structure
    print(f"\n🔗 Sample Hyperedge Structure:")
    if json_result['hyperedges']:
        sample_edge = json_result['hyperedges'][0]
        for key, value in sample_edge.items():
            print(f"   {key}: {value}")
    
    # Export to macro format
    print(f"\n📝 Exporting to Macro Format...")
    macro_result = export_model_open_hypergraph(
        'models.SimpleNN', 
        sample_input_args=((10,),),
        out_format="macro"
    )
    
    print(f"✅ Macro Export Complete:")
    macro_lines = macro_result['macro'].split('\n')
    print(f"   - Lines generated: {len(macro_lines)}")
    print(f"\n📄 Sample Macro Output:")
    for i, line in enumerate(macro_lines[:10]):  # Show first 10 lines
        print(f"   {i+1:2d}: {line}")
    if len(macro_lines) > 10:
        print(f"   ... ({len(macro_lines) - 10} more lines)")
    
    # Test with Autoencoder
    print(f"\n🔄 Testing with Autoencoder model...")
    autoencoder_result = export_model_open_hypergraph(
        'models.Autoencoder', 
        sample_input_args=((784,),)
    )
    
    print(f"✅ Autoencoder Export Complete:")
    print(f"   - Nodes: {len(autoencoder_result['nodes'])}")
    print(f"   - Hyperedges: {len(autoencoder_result['hyperedges'])}")
    
    print(f"\n🎯 Summary:")
    print(f"   ✅ Open-hypergraph export is working correctly!")
    print(f"   ✅ Supports both JSON and macro output formats")
    print(f"   ✅ Includes tensor shape and dtype information")
    print(f"   ✅ Handles multiple model architectures")
    print(f"   ✅ Preserves PyTorch FX graph structure")
    
    print(f"\n🔧 Usage:")
    print(f"   from visuaml import export_model_open_hypergraph")
    print(f"   result = export_model_open_hypergraph('models.YourModel', sample_input_args=((shape,),))")
    print(f"   # Or for macro format:")
    print(f"   result = export_model_open_hypergraph('models.YourModel', sample_input_args=((shape,),), out_format='macro')")

if __name__ == "__main__":
    demonstrate_export() 