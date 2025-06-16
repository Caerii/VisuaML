import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Assuming input image size that results in 7x7 feature map after 2 pooling layers
        # e.g., 28x28 -> 14x14 -> 7x7. For fx_export, a sample input might be torch.randn(1, 1, 28, 28)
        self.fc = nn.Linear(32 * 7 * 7, num_classes)

    def forward(self, x):
        # x shape: [batch_size, 1, 28, 28]
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(x.size(0), -1) # Flatten
        x = self.fc(x)
        return x

if __name__ == '__main__':
    # Example usage:
    # For fx_export, provide sample_input_args like "((1, 1, 28, 28),)"
    # and ensure the input dimensions match assumptions (e.g. leading to 32*7*7 features for fc layer)
    try:
        model = SimpleCNN()
        print("SimpleCNN model created successfully.")
        # To test with a dummy input:
        # test_input = torch.randn(1, 1, 28, 28)
        # output = model(test_input)
        # print(f"Output shape: {output.shape}")
    except Exception as e:
        print(f"Error creating SimpleCNN: {e}") 