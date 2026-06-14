import torch
import attentionweights

inputs = torch.tensor(
  [[0.43, 0.15, 0.89], # Your     (x^1)
   [0.55, 0.87, 0.66], # journey  (x^2)
   [0.57, 0.85, 0.64], # starts   (x^3)
   [0.22, 0.58, 0.33], # with     (x^4)
   [0.77, 0.25, 0.10], # one      (x^5)
   [0.05, 0.80, 0.55], # step     (x^6)
])

torch.manual_seed(789)

x_2 = inputs[1] # A
d_in = inputs.shape[1] # B -> inputs.shape = [6,3] 6 rows/tokens, 3 columns/dimensions
d_out = 2 # C  # compressed to this dimension after the output
attention = attentionweights.SelfAttention(d_in, d_out)

print(attention(inputs))