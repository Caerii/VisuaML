import torch
import torch.nn as nn

class MyTinyGPT(nn.Module):
    def __init__(self):
        super().__init__()
        # Define some simple layers for demonstration
        self.embedding = nn.Embedding(100, 10) # Vocab size 100, embedding dim 10. Currently unused in forward.
        self.linear1 = nn.Linear(10, 20)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(20, 5)
        self.output_layer = nn.Linear(5,2) # Example output

    def forward(self, x_tokens):
        # x_tokens is the input placeholder from the forward method signature.
        # We assume for FX tracing that x_tokens will be a tensor compatible with self.linear1's input.
        # (e.g., shape [batch_size, 10])
        # The actual values don't matter for symbolic_trace unless concrete_args are used,
        # but the data flow path does.

        # Previously, temp_val = torch.randn(1, 10) was used here.
        # Now we use x_tokens to establish the connection from the input placeholder.
        x = self.linear1(x_tokens) 
        x = self.relu(x)
        x = self.linear2(x)
        x = self.output_layer(x)
        return x

# To test if this file is runnable and the class is defined:
# if __name__ == '__main__':
#     model = MyTinyGPT()
#     print("MyTinyGPT model class defined and instantiated.")
#     # To test tracing:
#     # try:
#     #     # For FX trace to work with this forward pass, it will create a placeholder for x_tokens.
#     #     # No concrete input is strictly needed for symbolic_trace unless model logic demands it.
#     #     gm = torch.fx.symbolic_trace(model)
#     #     print("Symbolic trace successful.")
#     #     # print(gm.graph)
#     # except Exception as e:
#     #     print(f"Error during symbolic trace test: {e}") 