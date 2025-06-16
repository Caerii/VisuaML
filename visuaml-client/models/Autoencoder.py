import torch
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self, input_dim=784, encoding_dim=32):
        super(Autoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True),
            nn.Linear(64, encoding_dim),
            nn.ReLU(True) # Or Sigmoid/Tanh if you want bounded encoding
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 64),
            nn.ReLU(True),
            nn.Linear(64, 128),
            nn.ReLU(True),
            nn.Linear(128, input_dim),
            nn.Sigmoid() # Or Tanh, or nothing if inputs aren't normalized to [0,1] or [-1,1]
        )

    def forward(self, x):
        # x assumed to be flattened, e.g. [batch_size, input_dim]
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

if __name__ == '__main__':
    # Example usage:
    # For fx_export, provide sample_input_args like "((1, 784),)" (batch_size, flattened_input_dim)
    try:
        model = Autoencoder(input_dim=784, encoding_dim=32)
        print("Autoencoder model created successfully.")
        # Test with dummy input:
        # test_input = torch.randn(1, 784) 
        # output = model(test_input)
        # print(f"Output shape: {output.shape}")
    except Exception as e:
        print(f"Error creating Autoencoder model: {e}") 