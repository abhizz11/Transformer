# This file implements a Causal Multi-Head Attention layer. This is the core engine inside models like GPT. 

import torch 
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias= False):
        '''
        The parameters:
        d_in: Total size of the attention input
        d_out: Total size of the final attention output
        num_heads: The number of attention heads (heads process different relationships like grammar, S-V-A) 
        head_dim: The size of each individual head
        dropout: The rate with which we want to randomly turn off some neurons

        '''
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
        '''The function defines what happens when data passes through each layer.
        Let's suppose our sequence length is 3 (I love math).
        Embedding dimension is 4 and number of heads is 2
        Head dimension is 4 / 2 = 2
        Batch is 1

        Initially the shape is (3, 4) (tokens, emb dim) 
        Key, Query, Value matrices have the shape (4 * 4) (din, dout)
        so after multiplication we get (3, 4) * (4, 4) for all three
        the resulting shapes are (3, 4)
        
        Now, the model splts the QKV matrices across h independent heads.
        Head1 looks at grammar, Head2 looks at context, etc.

        Reshape operation -> (3, 4) -> (3, 2, 2) corresponding to (Tokens, head, head dimension)
        We transpose so the head dimensions are isolated -> (2, 3, 2) (Head, Tokens, head dimension)

        Next, we strip of the first dimension and we are left with a 2D matrix
        of shape (3,2). Key matrix goes through the same thing and we transpose it
        to become (2, 3)

        Then, we take the dot product Q * K' (3 * 2) * (2, 3) = (3, 3)

        Result A: (3, 3) is our attention map and because this is causal attention,
        we force the upper triangular part of this (3, 3) matrix to be -inf

        After masking, we divide by dk(head dimension) to stabilize gradients.

        Now, we have 2 heads each outputting (3, 2) matrix we need to glue them back together
        Getting (2, 3, 2) then we transpose again to get (3, 2, 2)

        Next, we crush the heads and head dimensions back together to get (3, 4). 
        At last, we multily by a final matrix to mix the information from 
        different heads one last time. 

        Operation (3, 4) * (4, 4) 

        Final shape = (3, 4) heavily contextualized vector


        '''
        b, num_tokens, d_in = x.shape

        if num_tokens > self.context_length:
            raise ValueError(
                f"Sequence length {num_tokens} exceeds context length "
                f"{self.context_length}."
            )

        queries = self.W_query(x) # what you are looking for, (what you type into search bar for a book) 
        keys = self.W_key(x) # What the item has to offer, (book tags like "romance", "thriller", etc)
        values = self.W_value(x) # Actual text inside the book (Core content that you want to read once)

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

