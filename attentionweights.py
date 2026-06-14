# Implementing attention mechanism with trainable weights (Q,K,V) matrices
import torch 

# Attention mechanism for "Your journey starts here"

inputs = torch.tensor(
  [[0.43, 0.15, 0.89], # Your     (x^1)
   [0.55, 0.87, 0.66], # journey  (x^2)
   [0.57, 0.85, 0.64], # starts   (x^3)
   [0.22, 0.58, 0.33], # with     (x^4)
   [0.77, 0.25, 0.10], # one      (x^5)
   [0.05, 0.80, 0.55], # step     (x^6)
])

x_2 = inputs[1] # A
d_in = inputs.shape[1] # B -> inputs.shape = [6,3] 6 rows/tokens, 3 columns/dimensions
d_out = 2 # C  # compressed to this dimension after the output

torch.manual_seed(123) # Setting seed for reproducible outputs

# During training grad should be True
W_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_key = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)

query_2 = x_2 @ W_query
key_2 = x_2 @ W_key 
value_2 = x_2 @ W_value 

# Even though we are just creating context vector for 2nd token, we still need all keys and values
keys = inputs @ W_key 
values = inputs @ W_value

print("keys shape", keys.shape)
print("values shape", values.shape)


keys_2 = keys[1]
attn_score_2 = query_2 @ keys.T 
print(attn_score_2)

d_k = keys.shape[-1]
sqrt = d_k ** 0.5
attn_weights_2 = torch.softmax(attn_score_2 / sqrt, dim=-1)
print(attn_weights_2)

context_vec_2 = attn_weights_2 @ values 
print(context_vec_2)

# Now, we are creating a class to make it compact