import torch
import torch.nn as nn

class BasicRNN(nn.Module):
    def __init__(self, input_size=10, hidden_size=20, num_layers=1, output_size=5, rnn_type='RNN'):
        super(BasicRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        if rnn_type == 'LSTM':
            self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        elif rnn_type == 'GRU':
            self.rnn = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        else:
            self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x shape: [batch_size, seq_len, input_size]
        # Initialize hidden state and cell state (for LSTM)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        if isinstance(self.rnn, nn.LSTM):
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
            out, _ = self.rnn(x, (h0, c0))
        else:
            out, _ = self.rnn(x, h0)
        
        # Take the output from the last time step
        out = self.fc(out[:, -1, :])
        return out

if __name__ == '__main__':
    # Example usage:
    # For fx_export, provide sample_input_args like "((1, 10, 10),)" (batch_size, seq_len, input_size)
    try:
        model_rnn = BasicRNN(rnn_type='RNN')
        print("BasicRNN (RNN type) model created successfully.")
        # test_input = torch.randn(1, 10, 10)
        # output = model_rnn(test_input)
        # print(f"RNN Output shape: {output.shape}")

        model_lstm = BasicRNN(rnn_type='LSTM')
        print("BasicRNN (LSTM type) model created successfully.")
        # output_lstm = model_lstm(test_input)
        # print(f"LSTM Output shape: {output_lstm.shape}")

        model_gru = BasicRNN(rnn_type='GRU')
        print("BasicRNN (GRU type) model created successfully.")
        # output_gru = model_gru(test_input)
        # print(f"GRU Output shape: {output_gru.shape}")

    except Exception as e:
        print(f"Error creating BasicRNN model: {e}") 