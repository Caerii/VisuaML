import torch
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
