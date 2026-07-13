import torch
import torch.nn as nn


inputs = torch.tensor(
  [[0.43, 0.15, 0.89], # Your     (x^1)
   [0.55, 0.87, 0.66], # journey  (x^2)
   [0.57, 0.85, 0.64], # starts   (x^3)
   [0.22, 0.58, 0.33], # with     (x^4)
   [0.77, 0.25, 0.10], # one      (x^5)
   [0.05, 0.80, 0.55], # step     (x^6)
])
'''Practice part
torch.manual_seed(789)

x_2 = inputs[1] # A
d_in = inputs.shape[1] # B -> inputs.shape = [6,3] 6 rows/tokens, 3 columns/dimensions
d_out = 2 # C  # compressed to this dimension after the output
attention = attentionweights.SelfAttention(d_in, d_out)

queries = attention.W_query(inputs) # A
keys = attention.W_key(inputs)
attn_scores = queries @ keys.T
attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim = 1)

context_length = attn_scores.shape[0]
# Torch.ones creates a tensor full of 1's and torch.tril makes it lower traingular
mask_simple = torch.tril(torch.ones(context_length, context_length)) 

masked_simple = attn_weights * mask_simple # masking future tokens, model can only attend to tokens in the past

# Let's normalize again
row_sums = masked_simple.sum(dim=1, keepdim=True)
masked_simple_norm = masked_simple / row_sums
print(masked_simple_norm)

# The previous approach is clunky, and requires renormalization, so the following approach is better

# Instead of using a lower traingular matrix, we can use the upper triangular matrix too
mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
masked = attn_scores.masked_fill(mask.bool(), -torch.inf) # We created an upper triangular matrix and filled all the zeroes with '-inf'


# Now we can do the normalization. 
attn_weights = torch.softmax(masked / keys.shape[-1] ** 0.5, dim=1)

# Dropouts:
dropout = torch.nn.Dropout(0.1) # We turn 10 % of the neurons randomly to make the behavior more stable

print(dropout(attn_weights))
'''
batch = torch.stack((inputs, inputs), dim=0)
class CausalAttention(nn.Module):
    
    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))
    
    def forward(self, x):
        b, num_tokens, d_in = x.shape # batch dimension number of tokens, and their dimension
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        attn_scores = queries @ keys.transpose(1,2) # We just want to transpose each of the batch
        attn_scores.masked_fill_(
            self.mask.bool()[:num_tokens, :num_tokens], -torch.inf
        ) # using num_tokens to handle edge cases where num tokens < context length

        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5, dim = -1
        )
        attn_weights = self.dropout(attn_weights) # Implementing dropout

        context_vec = attn_weights @ values 
        return context_vec 
  
d_in = inputs.shape[1]
d_out = 2
torch.manual_seed(123)
context_length = batch.shape[1]
ca = CausalAttention(d_in, d_out, context_length, 0.0)
context_vecs = ca(batch)
print(ca)
print(context_vecs.shape)