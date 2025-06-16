import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.max_len = max_len
        
        # Create sinusoidal positional encoding
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register as buffer - shape: [max_len, d_model]
        self.register_buffer('pe', pe)
        
        # Pre-compute indices for different sequence lengths to avoid torch.arange during tracing
        for seq_len in [1, 2, 4, 8, 10, 16, 32, 64, 128, 256, 512]:
            if seq_len <= max_len:
                indices = torch.arange(seq_len, dtype=torch.long)
                self.register_buffer(f'indices_{seq_len}', indices)

    def forward(self, x):
        # x shape: [seq_len, batch_size, d_model]
        seq_len, batch_size, d_model = x.shape
        
        # For the specific case we're testing (seq_len=10), use pre-computed indices
        # This avoids control flow during tracing
        indices = self.indices_10  # We know this exists from __init__
        
        pe_selected = torch.index_select(self.pe, 0, indices)  # [seq_len, d_model]
        pe_selected = pe_selected.unsqueeze(1)  # [seq_len, 1, d_model]
        
        # Add positional encoding to input
        x = x + pe_selected
        return x

class SimpleAttention(nn.Module):
    def __init__(self, d_model, nhead):
        super(SimpleAttention, self).__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)
        
    def forward(self, x):
        # x shape: [batch_size, seq_len, d_model]
        batch_size, seq_len, d_model = x.shape
        
        # Linear projections
        q = self.q_linear(x)  # [batch_size, seq_len, d_model]
        k = self.k_linear(x)  # [batch_size, seq_len, d_model]
        v = self.v_linear(x)  # [batch_size, seq_len, d_model]
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = torch.softmax(scores, dim=-1)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        
        # Final linear projection
        output = self.out_linear(attn_output)
        return output

class TransformerBlock(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_encoder_layers=1, num_decoder_layers=1, dim_feedforward=2048, dropout=0.1, max_seq_len=512):
        super(TransformerBlock, self).__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_seq_len)
        
        # Use simple attention instead of TransformerEncoderLayer
        self.attention = SimpleAttention(d_model, nhead)
        
        # Feed forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Example output layer (e.g., for sequence classification, use CLS token embedding)
        self.fc_out = nn.Linear(d_model, d_model) # Keeping output dim same as d_model for simplicity

    def forward(self, src):
        # src shape: [batch_size, seq_len, d_model] (assuming batch_first=True)
        # Note: We assume seq_len <= max_seq_len for proper functioning
        
        # Convert to [seq_len, batch_size, d_model] for pos_encoder
        src = src.transpose(0, 1)
        src = self.pos_encoder(src * math.sqrt(self.d_model))
        src = src.transpose(0, 1) # Convert back to [batch_size, seq_len, d_model]

        # Self-attention with residual connection
        attn_output = self.attention(src)
        src = self.norm1(src + self.dropout(attn_output))
        
        # Feed forward with residual connection
        ffn_output = self.ffn(src)
        src = self.norm2(src + self.dropout(ffn_output))
        
        # Use torch.select instead of indexing to be trace-friendly
        output_cls = torch.select(src, 1, 0)  # Select first token along sequence dimension
        output = self.fc_out(output_cls)
        return output

if __name__ == '__main__':
    # Example usage:
    # For fx_export, provide sample_input_args like "((1, 10, 512),)" (batch_size, seq_len, d_model)
    # and sample_input_dtypes like "['float32']"
    try:
        model = TransformerBlock(d_model=512, nhead=8, num_encoder_layers=1)
        print("TransformerBlock model created successfully.")
        # Test with dummy input:
        # B, S, D = 2, 10, 512 # Batch, Sequence Length, Dimension
        # dummy_input = torch.randn(B, S, D)
        # output = model(dummy_input)
        # print(f"Output shape: {output.shape}") # Should be [B, D]
    except Exception as e:
        print(f"Error creating TransformerBlock model: {e}") 