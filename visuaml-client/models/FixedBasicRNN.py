import torch
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
