#!/usr/bin/env python3
"""
Test script to verify export functionality works with the frontend.
"""

import requests
import json

def test_export_endpoints():
    """Test all export endpoints to ensure they work with the frontend."""
    base_url = "http://localhost:8787"
    
    # Test data
    test_model = "models.FixedSimpleCNN"
    sample_args = "((1, 1, 28, 28),)"
    
    print("🧪 Testing Export Endpoints")
    print("=" * 50)
    
    # Test 1: JSON Export
    print("\n1. Testing JSON Export...")
    try:
        response = requests.post(f"{base_url}/api/export-hypergraph", json={
            "modelPath": test_model,
            "format": "json",
            "sampleInputArgs": sample_args
        })
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ JSON Export: {len(data.get('nodes', []))} nodes, {len(data.get('hyperedges', []))} hyperedges")
        else:
            print(f"❌ JSON Export failed: {response.status_code}")
            print(response.text)
    except Exception as e:
        print(f"❌ JSON Export error: {e}")
    
    # Test 2: Macro Export
    print("\n2. Testing Macro Export...")
    try:
        response = requests.post(f"{base_url}/api/export-hypergraph", json={
            "modelPath": test_model,
            "format": "macro",
            "sampleInputArgs": sample_args
        })
        
        if response.status_code == 200:
            data = response.json()
            macro_content = data.get('macro_syntax') or data.get('macro')
            if macro_content:
                print(f"✅ Macro Export: {len(macro_content)} characters")
                print(f"   Preview: {macro_content[:100]}...")
            else:
                print("❌ Macro Export: No macro content found")
                print(f"   Available fields: {list(data.keys())}")
        else:
            print(f"❌ Macro Export failed: {response.status_code}")
            print(response.text)
    except Exception as e:
        print(f"❌ Macro Export error: {e}")
    
    # Test 3: Categorical Export
    print("\n3. Testing Categorical Export...")
    try:
        response = requests.post(f"{base_url}/api/export-hypergraph", json={
            "modelPath": test_model,
            "format": "categorical",
            "sampleInputArgs": sample_args
        })
        
        if response.status_code == 200:
            data = response.json()
            analysis = data.get('categorical_analysis', {})
            json_data = data.get('json_data', {})
            
            print(f"✅ Categorical Export:")
            print(f"   Analysis: {analysis.get('nodes', 0)} nodes, {analysis.get('hyperedges', 0)} hyperedges")
            print(f"   Complexity: {analysis.get('complexity', 'unknown')}")
            print(f"   JSON data: {len(json_data.get('nodes', []))} nodes")
            print(f"   Library available: {data.get('library_available', False)}")
        else:
            print(f"❌ Categorical Export failed: {response.status_code}")
            print(response.text)
    except Exception as e:
        print(f"❌ Categorical Export error: {e}")
    
    # Test 4: Enhanced Import with Export Format
    print("\n4. Testing Enhanced Import Endpoint...")
    try:
        response = requests.post(f"{base_url}/api/import", json={
            "modelPath": test_model,
            "exportFormat": "openhg-json",
            "sampleInputArgs": sample_args
        })
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Enhanced Import: {len(data.get('nodes', []))} nodes, {len(data.get('edges', []))} edges")
            if 'hyperedges' in data:
                print(f"   Also includes: {len(data.get('hyperedges', []))} hyperedges")
        else:
            print(f"❌ Enhanced Import failed: {response.status_code}")
            print(response.text)
    except Exception as e:
        print(f"❌ Enhanced Import error: {e}")
    
    print("\n" + "=" * 50)
    print("🎯 Test Summary:")
    print("   - JSON Export: Standard hypergraph format")
    print("   - Macro Export: Rust-compatible syntax")
    print("   - Categorical Export: Mathematical analysis")
    print("   - Enhanced Import: Unified endpoint")
    print("\n✨ Frontend should now handle all export formats correctly!")

if __name__ == "__main__":
    test_export_endpoints() 