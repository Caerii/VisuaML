# -*- coding: utf-8 -*-
"""
Diagnostic script for analyzing models that fail open-hypergraph export.
This script identifies issues and provides solutions to make models compatible.
"""

import sys
import os
import torch
from torch.fx import symbolic_trace
from torch.fx.passes.shape_prop import ShapeProp

# Add the parent directory to Python path so we can import models
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from visuaml import export_model_open_hypergraph

class ModelDiagnostic:
    def __init__(self):
        self.issues = []
    
    def add_issue(self, issue_type, description, solution):
        self.issues.append({
            'type': issue_type,
            'description': description,
            'solution': solution
        })
    
    def print_report(self, model_name):
        print(f"\nDiagnostic Report for {model_name}")
        print("=" * 50)
        
        if not self.issues:
            print("No issues found - model should work with open-hypergraph export!")
            return
        
        for i, issue in enumerate(self.issues, 1):
            print(f"\nIssue #{i}: {issue['type']}")
            print(f"   Description: {issue['description']}")
            print(f"   Solution: {issue['solution']}")

def analyze_simple_cnn():
    """Analyze SimpleCNN issues."""
    print("\nAnalyzing SimpleCNN...")
    
    from models.SimpleCNN import SimpleCNN
    
    diagnostic = ModelDiagnostic()
    
    # The issue: input shape (1, 28, 28) vs expected (1, 1, 28, 28)
    print("Issue identified: Input shape mismatch")
    print("   Expected: (batch, channels, height, width) = (1, 1, 28, 28)")
    print("   Provided: (1, 28, 28) - missing channel dimension")
    
    diagnostic.add_issue(
        "Input Shape Mismatch",
        "Sample input shape (1, 28, 28) doesn't match expected (1, 1, 28, 28)",
        "Use sample_input_args=((1, 1, 28, 28),) instead of ((1, 28, 28),)"
    )
    
    # Test with correct input shape
    correct_input = torch.randn(1, 1, 28, 28)
    model = SimpleCNN()
    
    try:
        # Test the linear layer size calculation
        test_conv_output = model.pool2(model.relu2(model.conv2(
            model.pool1(model.relu1(model.conv1(correct_input)))
        )))
        expected_fc_input = test_conv_output.numel() // test_conv_output.size(0)
        actual_fc_input = model.fc.in_features
        
        if expected_fc_input != actual_fc_input:
            diagnostic.add_issue(
                "Linear Layer Size Mismatch",
                f"FC layer expects {actual_fc_input} features but gets {expected_fc_input}",
                f"Change fc layer to nn.Linear({expected_fc_input}, num_classes)"
            )
        else:
            print(f"Linear layer size is correct: {actual_fc_input}")
            
        # Test FX tracing
        traced = symbolic_trace(model)
        print("FX tracing successful")
        
        # Test shape propagation
        sp = ShapeProp(traced)
        sp.propagate(correct_input)
        print("Shape propagation successful")
        
    except Exception as e:
        diagnostic.add_issue(
            "FX Tracing/Shape Propagation Error",
            f"Error during tracing or shape propagation: {str(e)}",
            "Check model structure for dynamic operations"
        )
    
    diagnostic.print_report("SimpleCNN")
    return diagnostic

def analyze_basic_rnn():
    """Analyze BasicRNN issues."""
    print("\nAnalyzing BasicRNN...")
    
    from models.BasicRNN import BasicRNN
    
    diagnostic = ModelDiagnostic()
    
    # The main issue: dynamic tensor creation with .size() calls
    diagnostic.add_issue(
        "Dynamic Tensor Creation",
        "Uses x.size(0) in torch.zeros() which FX can't trace",
        "Pre-allocate hidden states or use torch.jit.script for RNNs"
    )
    
    diagnostic.add_issue(
        "Device-dependent Operations", 
        "Uses .to(x.device) which is dynamic",
        "Remove device operations or handle them outside forward()"
    )
    
    diagnostic.add_issue(
        "Conditional Logic",
        "Has if/else statements based on RNN type",
        "Create separate model classes for each RNN type"
    )
    
    # Test with a simple input - using underscore prefix for intentionally unused variables
    try:
        model = BasicRNN()
        _traced = symbolic_trace(model)
        print("Unexpected: FX tracing succeeded (this usually fails)")
    except Exception as e:
        print(f"Expected: FX tracing failed as predicted: {str(e)[:100]}...")
    
    diagnostic.print_report("BasicRNN")
    return diagnostic

def analyze_demo_net():
    """Analyze DemoNet issues."""
    print("\nAnalyzing DemoNet...")
    
    from models.DemoNet import DemoNet
    
    diagnostic = ModelDiagnostic()
    
    # The main issue: embedding expects Long tensors but gets Float
    diagnostic.add_issue(
        "Embedding Input Type Mismatch",
        "Embedding layer expects Long tensor indices but gets Float tensor",
        "Use sample_input_dtypes=['long'] or provide integer tensor indices"
    )
    
    diagnostic.add_issue(
        "Input Interpretation",
        "Model expects token indices but test provides continuous values",
        "Use integer indices in range [0, vocab_size) for embedding input"
    )
    
    # Test with correct input type
    sample_input_float = torch.randn(3, 32, 32)  # Wrong type
    sample_input_long = torch.randint(0, 1000, (1, 32))  # Correct type
    
    model = DemoNet()
    
    try:
        # This should fail
        _output = model(sample_input_float)
        print("Unexpected: Float input worked")
    except Exception as e:
        print(f"Expected: Float input failed: {str(e)[:100]}...")
    
    try:
        # This should work
        _output = model(sample_input_long)
        print("Long input works correctly")
        
        # Test FX tracing with correct input
        traced = symbolic_trace(model)
        print("FX tracing successful with correct input")
        
        # Test shape propagation
        sp = ShapeProp(traced)
        sp.propagate(sample_input_long)
        print("Shape propagation successful")
        
    except Exception as e:
        print(f"Long input failed: {str(e)[:100]}...")
        diagnostic.add_issue(
            "Additional Issues",
            f"Even with correct input type, model fails: {str(e)}",
            "Check for other dynamic operations in the model"
        )
    
    diagnostic.print_report("DemoNet")
    return diagnostic

def test_fixes():
    """Test the proposed fixes."""
    print("\nTesting Proposed Fixes...")
    print("=" * 30)
    
    # Test SimpleCNN fix
    print("Testing SimpleCNN fix...")
    try:
        result = export_model_open_hypergraph(
            'models.SimpleCNN', 
            sample_input_args=((1, 1, 28, 28),)  # Correct shape
        )
        print(f"SimpleCNN fixed! Generated {len(result['nodes'])} nodes")
    except Exception as e:
        print(f"SimpleCNN still fails: {str(e)[:100]}...")
    
    # Test DemoNet fix
    print("\nTesting DemoNet fix...")
    try:
        result = export_model_open_hypergraph(
            'models.DemoNet', 
            sample_input_args=((1, 32),),  # Correct shape
            sample_input_dtypes=['long']   # Correct dtype
        )
        print(f"DemoNet fixed! Generated {len(result['nodes'])} nodes")
    except Exception as e:
        print(f"DemoNet still fails: {str(e)[:100]}...")

def main():
    """Main diagnostic function."""
    print("VisuaML Model Compatibility Diagnostic")
    print("=" * 50)
    print("Analyzing models that failed open-hypergraph export...")
    
    # Analyze each problematic model
    analyze_simple_cnn()
    analyze_basic_rnn() 
    analyze_demo_net()
    
    # Test the fixes
    test_fixes()
    
    print("\nSummary of Issues and Solutions:")
    print("=" * 40)
    print("1. SimpleCNN: Input shape mismatch - use (1, 1, 28, 28)")
    print("2. BasicRNN: Dynamic operations - simplify or use torch.jit.script")
    print("3. DemoNet: Wrong input dtype - use Long tensors for embeddings")
    
    print("\nGeneral Guidelines for FX-Compatible Models:")
    print("- Avoid dynamic tensor shapes (x.size(0) in tensor creation)")
    print("- Avoid conditional logic based on runtime values")
    print("- Use correct input shapes and dtypes")
    print("- Minimize device-dependent operations")
    print("- Keep models simple and static for best FX tracing support")

if __name__ == "__main__":
    main() 