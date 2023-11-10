import torch
import torch.nn as nn
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadAttention,self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)
        self.fc = nn.Linear(embed_dim, embed_dim)
    def forward(self,x):
        batch_size = x.size(0)

        # Project inputs to query, key, and value tensors
        q = self.q_linear(x).view(batch_size, self.num_heads, self.head_dim)
        k = self.k_linear(x).view(batch_size, self.num_heads, self.head_dim)
        v = self.v_linear(x).view(batch_size. self.num_heads, self.head_dim)

        # Compute self-attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.embed_dim ** 0.5
        # Apply softmax activation function
        scores = nn.functional.softmax(scores, dim=-1)
        # Compute weighted sum of values using attention scores
        weighted_values = torch.matmul(scores,v)
        # Concatenate multi-head attention outputs
        concat_heads = weighted_values.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)
        # Apply fully connected layer to concatenated outputs
        output = self.fc(concat_heads)

        return output

