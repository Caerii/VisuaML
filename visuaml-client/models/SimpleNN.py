import torch
import torch.nn as nn

class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Simple feedforward neural network
        self.fc1 = nn.Linear(10, 32)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(32, 16)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(16, 8)
        self.output = nn.Linear(8, 3)
        
    def forward(self, x):
        # x is the input tensor
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        out = self.output(x)
        return out

# Test if the model can be instantiated and traced
if __name__ == '__main__':
    model = SimpleNN()
    print("SimpleNN model created successfully") 