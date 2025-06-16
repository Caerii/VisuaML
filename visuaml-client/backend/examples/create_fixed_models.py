# -*- coding: utf-8 -*-
"""
Script to create fixed versions of models that fail open-hypergraph export.
"""

import os

def create_fixed_simple_cnn():
    """Create a fixed version of SimpleCNN."""
    fixed_code = '''import torch
import torch.nn as nn
import torch.nn.functional as F

class FixedSimpleCNN(nn.Module):
    """
    Fixed version of SimpleCNN that works with open-hypergraph export.
    
    Key fixes:
    - Proper input shape handling (1, 1, 28, 28)
    - Correct linear layer size calculation
    """
    def __init__(self, num_classes=10):
        super(FixedSimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Correct calculation: 28x28 -> 14x14 -> 7x7, so 32*7*7=1568
        self.fc = nn.Linear(32 * 7 * 7, num_classes)

    def forward(self, x):
        # x shape: [batch_size, 1, 28, 28]
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return x

if __name__ == '__main__':
    # Test the fixed model
    try:
        model = FixedSimpleCNN()
        print("FixedSimpleCNN model created successfully.")
        
        # Test with correct input shape
        test_input = torch.randn(1, 1, 28, 28)
        output = model(test_input)
        print(f"Output shape: {output.shape}")
        
        # Test FX tracing
        from torch.fx import symbolic_trace
        traced = symbolic_trace(model)
        print("FX tracing successful!")
        
    except Exception as e:
        print(f"Error with FixedSimpleCNN: {e}")

# Usage for open-hypergraph export:
# export_model_open_hypergraph('models.FixedSimpleCNN', sample_input_args=((1, 1, 28, 28),))
'''
    
    with open('../models/FixedSimpleCNN.py', 'w', encoding='utf-8') as f:
        f.write(fixed_code)
    print("Created FixedSimpleCNN.py")

def create_fixed_basic_rnn():
    """Create a fixed version of BasicRNN."""
    fixed_code = '''import torch
import torch.nn as nn

class FixedBasicRNN(nn.Module):
    """
    Fixed version of BasicRNN that works with FX tracing.
    
    Key fixes:
    - Removed dynamic tensor creation (torch.zeros with x.size())
    - Removed device-dependent operations (.to(x.device))
    - Simplified to single RNN type
    - Let PyTorch handle hidden state initialization
    """
    def __init__(self, input_size=10, hidden_size=20, output_size=5):
        super(FixedBasicRNN, self).__init__()
        self.hidden_size = hidden_size
        
        # Use built-in RNN without manual hidden state initialization
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x shape: [batch_size, seq_len, input_size]
        # Let PyTorch handle hidden state initialization automatically
        out, _ = self.rnn(x)
        
        # Take the output from the last time step
        out = self.fc(out[:, -1, :])
        return out

class FixedBasicLSTM(nn.Module):
    """
    Fixed LSTM version that works with FX tracing.
    """
    def __init__(self, input_size=10, hidden_size=20, output_size=5):
        super(FixedBasicLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Let PyTorch handle hidden state initialization
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

class FixedBasicGRU(nn.Module):
    """
    Fixed GRU version that works with FX tracing.
    """
    def __init__(self, input_size=10, hidden_size=20, output_size=5):
        super(FixedBasicGRU, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Let PyTorch handle hidden state initialization
        out, _ = self.gru(x)
        out = self.fc(out[:, -1, :])
        return out

if __name__ == '__main__':
    # Test the fixed models
    try:
        model_rnn = FixedBasicRNN()
        print("FixedBasicRNN model created successfully.")
        
        test_input = torch.randn(1, 10, 10)
        output = model_rnn(test_input)
        print(f"RNN Output shape: {output.shape}")
        
        # Test FX tracing
        from torch.fx import symbolic_trace
        traced = symbolic_trace(model_rnn)
        print("RNN FX tracing successful!")

        model_lstm = FixedBasicLSTM()
        traced_lstm = symbolic_trace(model_lstm)
        print("LSTM FX tracing successful!")

        model_gru = FixedBasicGRU()
        traced_gru = symbolic_trace(model_gru)
        print("GRU FX tracing successful!")

    except Exception as e:
        print(f"Error with Fixed RNN models: {e}")

# Usage for open-hypergraph export:
# export_model_open_hypergraph('models.FixedBasicRNN', sample_input_args=((1, 10, 10),))
'''
    
    with open('../models/FixedBasicRNN.py', 'w', encoding='utf-8') as f:
        f.write(fixed_code)
    print("Created FixedBasicRNN.py")

def create_fixed_demo_net():
    """Create a fixed version of DemoNet."""
    fixed_code = '''import torch
import torch.nn as nn

class FixedDemoNet(nn.Module):
    """
    Fixed version of DemoNet that works with open-hypergraph export.
    
    Key fixes:
    - Simplified architecture to avoid complex attention mechanism
    - Proper handling of embedding input types
    - Removed dropout during forward pass for tracing
    """
    def __init__(self, vocab_size=1000, embed_dim=64, hidden_size=128, output_size=10):
        super(FixedDemoNet, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # Simplified LSTM without dropout for better FX compatibility
        self.lstm = nn.LSTM(embed_dim, hidden_size, batch_first=True)
        
        # Simplified output layers
        self.fc1 = nn.Linear(hidden_size, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, output_size)
        
    def forward(self, x):
        # x should be Long tensor with token indices [batch, seq_len]
        x = self.embedding(x)
        
        # LSTM processing - let PyTorch handle hidden state initialization
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Use last hidden state (simplified from attention mechanism)
        context = h_n[-1]  # Use last layer's hidden state
        
        # Final layers
        x = self.fc1(context)
        x = self.relu(x)
        x = self.fc2(x)
        
        return x

class FixedDemoNetSimple(nn.Module):
    """
    Even simpler version of DemoNet for maximum compatibility.
    """
    def __init__(self, vocab_size=1000, embed_dim=64, output_size=10):
        super(FixedDemoNetSimple, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, 128, batch_first=True)
        self.fc = nn.Linear(128, output_size)

    def forward(self, x):
        # x should be Long tensor with token indices
        x = self.embedding(x)
        out, (h_n, _) = self.lstm(x)
        # Use last hidden state
        x = self.fc(h_n[-1])
        return x

if __name__ == '__main__':
    # Test the fixed model
    try:
        model = FixedDemoNet()
        print("FixedDemoNet model created successfully.")
        
        # Test with correct input type (Long tensor)
        test_input = torch.randint(0, 1000, (1, 32))  # (batch_size, seq_len)
        output = model(test_input)
        print(f"Output shape: {output.shape}")
        
        # Test FX tracing
        from torch.fx import symbolic_trace
        traced = symbolic_trace(model)
        print("FX tracing successful!")
        
        # Test simple version
        model_simple = FixedDemoNetSimple()
        traced_simple = symbolic_trace(model_simple)
        print("Simple version FX tracing successful!")
        
    except Exception as e:
        print(f"Error with FixedDemoNet: {e}")

# Usage for open-hypergraph export:
# export_model_open_hypergraph('models.FixedDemoNet', 
#                              sample_input_args=((1, 32),),
#                              sample_input_dtypes=['long'])
'''
    
    with open('../models/FixedDemoNet.py', 'w', encoding='utf-8') as f:
        f.write(fixed_code)
    print("Created FixedDemoNet.py")

def test_fixed_models():
    """Test the fixed models with open-hypergraph export."""
    print("\nTesting fixed models with open-hypergraph export...")
    
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from visuaml import export_model_open_hypergraph
    
    # Test FixedSimpleCNN
    try:
        result = export_model_open_hypergraph(
            'models.FixedSimpleCNN', 
            sample_input_args=((1, 1, 28, 28),)
        )
        print(f"✅ FixedSimpleCNN: {len(result['nodes'])} nodes, {len(result['hyperedges'])} hyperedges")
    except Exception as e:
        print(f"❌ FixedSimpleCNN failed: {str(e)[:100]}...")
    
    # Test FixedBasicRNN
    try:
        result = export_model_open_hypergraph(
            'models.FixedBasicRNN', 
            sample_input_args=((1, 10, 10),)
        )
        print(f"✅ FixedBasicRNN: {len(result['nodes'])} nodes, {len(result['hyperedges'])} hyperedges")
    except Exception as e:
        print(f"❌ FixedBasicRNN failed: {str(e)[:100]}...")
    
    # Test FixedDemoNet
    try:
        result = export_model_open_hypergraph(
            'models.FixedDemoNet', 
            sample_input_args=((1, 32),),
            sample_input_dtypes=['long']
        )
        print(f"✅ FixedDemoNet: {len(result['nodes'])} nodes, {len(result['hyperedges'])} hyperedges")
    except Exception as e:
        print(f"❌ FixedDemoNet failed: {str(e)[:100]}...")

def main():
    """Create all fixed models."""
    print("Creating fixed versions of problematic models...")
    
    create_fixed_simple_cnn()
    create_fixed_basic_rnn()
    create_fixed_demo_net()
    
    print("\nFixed models created! Testing with open-hypergraph export...")
    test_fixed_models()
    
    print("\n" + "="*60)
    print("SUMMARY: Model Fixes for Open-Hypergraph Compatibility")
    print("="*60)
    print("1. FixedSimpleCNN.py - Correct input shapes and layer sizes")
    print("2. FixedBasicRNN.py - Removed dynamic operations, separate RNN types")
    print("3. FixedDemoNet.py - Simplified architecture, proper input types")
    print("\nAll fixed models should now work with:")
    print("- PyTorch FX tracing")
    print("- Shape propagation")
    print("- Open-hypergraph export")

if __name__ == "__main__":
    main() 