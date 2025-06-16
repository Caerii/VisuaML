import torch
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
