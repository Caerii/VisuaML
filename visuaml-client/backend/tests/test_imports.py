"""Test basic imports to verify CI setup."""

import sys
import os

# Add the backend directory to Python path so we can import visuaml package
backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, backend_dir)

def test_torch_import():
    """Test that PyTorch can be imported."""
    import torch
    assert torch.__version__ is not None
    print(f"✅ PyTorch {torch.__version__} imported successfully")

def test_visuaml_imports():
    """Test that visuaml modules can be imported."""
    # These should work if the path is set up correctly
    try:
        from visuaml.graph_export import export_model_graph  # noqa: E402
        from visuaml.model_loader import load_model_class  # noqa: E402
        from visuaml.filters import GraphFilter  # noqa: E402
        
        # Actually use the imports to avoid F401 errors
        assert export_model_graph is not None
        assert load_model_class is not None
        assert GraphFilter is not None
        
        print("✅ Core visuaml modules imported successfully")
        assert True
    except ImportError as e:
        print(f"❌ visuaml import failed: {e}")
        # Don't fail the test, just report the issue
        assert False, f"visuaml imports failed: {e}"

def test_optional_imports():
    """Test optional dependencies."""
    try:
        from visuaml.openhypergraph_export import export_model_open_hypergraph  # noqa: E402
        
        # Actually use the import to avoid F401 error
        assert export_model_open_hypergraph is not None
        
        print("✅ Optional openhypergraph module imported successfully")
    except ImportError as e:
        print(f"ℹ️ Optional import not available: {e}")
        # This is expected if open-hypergraphs is not installed
        pass
    
    # This should always pass
    assert True 