import torch
import torch.nn as nn

class DemoNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Demo network with LSTM and attention-like mechanism
        self.embedding = nn.Embedding(1000, 64)  # vocab size 1000, embed dim 64
        
        # LSTM layers
        self.lstm = nn.LSTM(64, 128, num_layers=2, batch_first=True, dropout=0.2)
        
        # Simple attention mechanism
        self.attention_linear = nn.Linear(128, 128)
        self.attention_softmax = nn.Softmax(dim=1)
        
        # Output layers
        self.fc1 = nn.Linear(128, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, 10)
        
    def forward(self, x):
        # Assuming x is token indices [batch, seq_len]
        x = self.embedding(x)
        
        # LSTM processing
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Simple attention
        attention_weights = self.attention_linear(lstm_out)
        attention_weights = self.attention_softmax(attention_weights)
        
        # Apply attention (simplified - just using last hidden state for now)
        context = h_n[-1]  # Use last layer's hidden state
        
        # Final layers
        x = self.fc1(context)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

# Test if the model can be instantiated
if __name__ == '__main__':
    model = DemoNet()
    print("DemoNet created successfully") 