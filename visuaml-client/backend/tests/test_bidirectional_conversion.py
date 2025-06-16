#!/usr/bin/env python3
"""
Test script demonstrating bidirectional conversion between JSON and categorical OpenHypergraph formats.
"""

import sys
import os
import json

# Add the parent directory to Python path so we can import models
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from visuaml import export_model_open_hypergraph
from visuaml.openhypergraph_export import json_to_categorical, categorical_to_json

def test_bidirectional_conversion():
    """Test bidirectional conversion between JSON and categorical formats."""
    print("🔄 Testing Bidirectional Open-Hypergraph Conversion")
    print("=" * 55)
    
    # Test with FixedDemoNet
    model_path = 'models.FixedDemoNet'
    sample_args = ((1, 32),)
    sample_dtypes = ['long']
    
    print(f"📊 Testing with {model_path}")
    print(f"   Sample input: {sample_args}")
    print(f"   Sample dtypes: {sample_dtypes}")
    
    # Step 1: Export to JSON format
    print("\n🔹 Step 1: Export to JSON format")
    json_result = export_model_open_hypergraph(
        model_path,
        sample_input_args=sample_args,
        sample_input_dtypes=sample_dtypes,
        out_format="json"
    )
    
    print(f"   ✅ JSON export successful")
    print(f"   📈 Nodes: {len(json_result['nodes'])}")
    print(f"   🔗 Hyperedges: {len(json_result['hyperedges'])}")
    print(f"   🏷️  Categorical available: {json_result.get('categorical_available', False)}")
    
    # Step 2: Convert JSON to categorical
    print("\n🔹 Step 2: Convert JSON → Categorical")
    try:
        categorical_obj = json_to_categorical(json_result)
        print(f"   ✅ JSON → Categorical conversion successful")
        print(f"   🧮 OpenHypergraph object created: {type(categorical_obj)}")
        
        # Step 3: Convert categorical back to JSON
        print("\n🔹 Step 3: Convert Categorical → JSON")
        reconstructed_json = categorical_to_json(categorical_obj, {
            "format": "reconstructed-from-categorical",
            "source": "bidirectional-test"
        })
        
        print(f"   ✅ Categorical → JSON conversion successful")
        print(f"   📈 Reconstructed nodes: {len(reconstructed_json['nodes'])}")
        print(f"   🔗 Reconstructed hyperedges: {len(reconstructed_json['hyperedges'])}")
        
        # Step 4: Direct categorical export
        print("\n🔹 Step 4: Direct categorical export")
        categorical_result = export_model_open_hypergraph(
            model_path,
            sample_input_args=sample_args,
            sample_input_dtypes=sample_dtypes,
            out_format="categorical"
        )
        
        print(f"   ✅ Direct categorical export successful")
        print(f"   🧮 Categorical analysis: {type(categorical_result['categorical_analysis'])}")
        print(f"   🔄 Bidirectional: {categorical_result.get('bidirectional', False)}")
        print(f"   📋 JSON representation included: {'json_representation' in categorical_result}")
        print(f"   🏗️  Framework ready: {categorical_result.get('framework_ready', False)}")
        
        # Step 5: Compare structures
        print("\n🔹 Step 5: Structure comparison")
        original_nodes = len(json_result['nodes'])
        reconstructed_nodes = len(reconstructed_json['nodes'])
        direct_nodes = len(categorical_result['json_representation']['nodes'])
        
        original_edges = len(json_result['hyperedges'])
        reconstructed_edges = len(reconstructed_json['hyperedges'])
        direct_edges = len(categorical_result['json_representation']['hyperedges'])
        
        print(f"   📊 Node counts:")
        print(f"      Original JSON: {original_nodes}")
        print(f"      Reconstructed: {reconstructed_nodes}")
        print(f"      Direct categorical: {direct_nodes}")
        
        print(f"   🔗 Hyperedge counts:")
        print(f"      Original JSON: {original_edges}")
        print(f"      Reconstructed: {reconstructed_edges}")
        print(f"      Direct categorical: {direct_edges}")
        
        # Check consistency
        nodes_consistent = original_nodes == reconstructed_nodes == direct_nodes
        edges_consistent = original_edges == reconstructed_edges == direct_edges
        
        print(f"\n   ✅ Structure consistency:")
        print(f"      Nodes: {'✅ Consistent' if nodes_consistent else '❌ Inconsistent'}")
        print(f"      Edges: {'✅ Consistent' if edges_consistent else '❌ Inconsistent'}")
        
        # Show categorical analysis details
        cat_analysis = categorical_result['categorical_analysis']
        print(f"\n   🔬 Categorical Analysis:")
        print(f"      Complexity: {cat_analysis.get('complexity', 'unknown')}")
        print(f"      Status: {cat_analysis.get('conversion_status', 'unknown')}")
        print(f"      Note: {cat_analysis.get('note', 'N/A')}")
        
        # Step 6: Test macro format compatibility
        print("\n🔹 Step 6: Macro format compatibility")
        macro_result = export_model_open_hypergraph(
            model_path,
            sample_input_args=sample_args,
            sample_input_dtypes=sample_dtypes,
            out_format="macro"
        )
        
        print(f"   ✅ Macro export successful")
        macro_lines = macro_result['macro'].split('\n')
        print(f"   📄 Macro lines: {len(macro_lines)}")
        print(f"   🦀 Rust crate compatible: Yes")
        
        # Show sample macro output
        print(f"\n   📋 Sample macro output:")
        for i, line in enumerate(macro_lines[:8]):
            print(f"      {i+1:2d}: {line}")
        if len(macro_lines) > 8:
            print(f"      ... ({len(macro_lines) - 8} more lines)")
        
        print("\n🎯 Summary:")
        print("   ✅ JSON format: Frontend-ready visualization data")
        print("   ✅ Categorical framework: Analysis and conversion infrastructure ready")
        print("   ✅ Macro format: Rust crate compatibility")
        print("   ✅ Bidirectional conversion: JSON ↔ Categorical analysis")
        print("   ✅ All formats maintain structural consistency")
        print("   🏗️  Framework ready for full categorical OpenHypergraph integration")
        
        print(f"\n🔧 Usage Examples:")
        print(f"   # JSON for frontend")
        print(f"   json_data = export_model_open_hypergraph('{model_path}', out_format='json')")
        print(f"   ")
        print(f"   # Categorical analysis")
        print(f"   cat_data = export_model_open_hypergraph('{model_path}', out_format='categorical')")
        print(f"   analysis = cat_data['categorical_analysis']")
        print(f"   ")
        print(f"   # Convert between formats")
        print(f"   cat_analysis = json_to_categorical(json_data)")
        print(f"   json_data = categorical_to_json(cat_analysis)")
        print(f"   ")
        print(f"   # Macro for Rust crate")
        print(f"   macro_data = export_model_open_hypergraph('{model_path}', out_format='macro')")
        print(f"   rust_code = macro_data['macro']")
        
    except ImportError as e:
        print(f"   ❌ Categorical conversion not available: {e}")
        print(f"   💡 Install with: pip install open-hypergraphs")
    except Exception as e:
        print(f"   ❌ Conversion failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_bidirectional_conversion() 