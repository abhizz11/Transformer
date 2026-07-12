# Implementing a simplified attention mechanism
import torch 

# Attention mechanism for "Your journey starts here"

inputs = torch.tensor(
  [[0.43, 0.15, 0.89], # Your     (x^1)
   [0.55, 0.87, 0.66], # journey  (x^2)
   [0.57, 0.85, 0.64], # starts   (x^3)
   [0.22, 0.58, 0.33], # with     (x^4)
   [0.77, 0.25, 0.10], # one      (x^5)
   [0.05, 0.80, 0.55], # step     (x^6)
#    [0.4419, 0.6515, 0.5683] # journey-context vector
])


import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D

words = ["Your", "journey", "starts", "here", "journey's context vector"]

x_cords = inputs[:,0].numpy()
y_cords = inputs[:, 1].numpy()
z_cords = inputs[:, 2].numpy()

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
colors = ['r','g','b','c','m','y','r']

for (x, y, z, word, color) in zip(x_cords, y_cords, z_cords, words, colors):
    ax.quiver(0, 0, 0, x, y, z, color=color, arrow_length_ratio=0.05)
    ax.text(x, y, z, word, fontsize=10, color=color)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

ax.set_xlim([0,1])
ax.set_ylim([0,1])
ax.set_zlim([0,1])

plt.title("3D Plot of Word Embeddings")
# plt.show() # generates and opens an image

# Attention score for "journey" 2nd word/token 
query = inputs[1]

attn_scores_2 = torch.empty(inputs.shape[0])
for i, x_i in enumerate(inputs):
    attn_scores_2[i] = torch.dot(x_i, query) # dot product

print(attn_scores_2)

# Normalizing attention scores
attn_weights_2_tmp = attn_scores_2 / attn_scores_2.sum()
print("Attention weights: ", attn_weights_2_tmp)
print("Sum: ", attn_weights_2_tmp.sum()) # This should be 1


# Simplified function similar to Softmax
def softmax(x):
    return torch.exp(x) / torch.exp(x).sum(dim=0) # 1d tensor, similar to sum(arr[0]) 

attn_sftmax = softmax(attn_scores_2)
print("Attention weights: ", attn_sftmax)
print("Attention sum: ", attn_sftmax.sum())


# Pytorch softmax
attn_weights_pysft = torch.softmax(attn_scores_2, dim = 0)
print("Attention weights: ", attn_weights_pysft)
print("Attention sum: ", attn_weights_pysft.sum())

context_vec2 = torch.zeros(query.shape)
for i, x_i in enumerate(inputs):
    context_vec2 += attn_weights_pysft[i] * x_i

print(context_vec2)

attn_scores = torch.empty(6, 6)

# O(n^2) approach computationally heavy
attn_scores = torch.empty(6, 6)

for i, x_i in enumerate(inputs):
    for j, x_j in enumerate(inputs):
        attn_scores[i, j] = torch.dot(x_i, x_j)

print(attn_scores)


# Same thing with matrix multiplication, pytorch implementation
attn_scores_mtx = inputs @ inputs.T
print(attn_scores_mtx)

attn_weights_mtx = torch.softmax(attn_scores_mtx, dim=-1)  # Dim = -1 means the columns are normalized
print(attn_weights_mtx)

all_context = attn_weights_mtx @ inputs 
print(all_context) # All of the context vectors