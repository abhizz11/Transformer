import torch 
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias= False):
        super().__init__()
        if d_out % num_heads != 0:
            raise ValueError("d_out must be divisible by num_heads")

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.context_length = context_length

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias) # Query matrix
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias) # Key matrix
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias) # Value matrix
        self.out_proj = nn.Linear(d_out, d_out) # To combine output heads later
        self.dropout = nn.Dropout(dropout) # To randomly turn off some of the neurons


    def forward(self, x):
        b, num_tokens, d_in = x.shape

        if num_tokens > self.context_length:
            raise ValueError(
                f"Sequence length {num_tokens} exceeds context length "
                f"{self.context_length}."
            )

        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        # We split the matrix by adding "num_heads" dimension, converting d_out => nheads, headdim
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # Transpose, so attention is calculated independently for each head
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # Determine dropout probability
        dropout_probability = (
            self.dropout.p if self.training else 0.0
        )

        # Optimized context vector implementation
        context_vec = F.scaled_dot_product_attention(
            queries,
            keys,
            values,
            dropout_p=dropout_probability,
            is_causal=True, # Applies causal attention mask
        )

        # Retranspose to get context vectors
        context_vec = context_vec.transpose(1, 2)

        # Combine
        context_vec = context_vec.contiguous().view(
            b,
            num_tokens,
            self.d_out,
        )

        return self.out_proj(context_vec)

