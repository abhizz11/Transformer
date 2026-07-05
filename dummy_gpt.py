import torch
import torch.nn as nn
import tiktoken
import multiheadedattention as mha

GPT_CONFIG_124M = {
    "vocab_size": 50257, # Size of Model's vocab
    "context_length": 1024, # Words it can process and remember at one time
    "emb_dim": 768, # Embedding dimension, (Different meanings of the same word)
    "n_heads": 12, # Number of Attention heads (Different interpretations of the same sequence)
    "n_layers": 12, # Layers in the transforemer
    "drop_rate": 0.1, # Percent of neurons to randomly turn off
    "qkv_bias": False # Query-key-value bias  
}

# Just a dummy class for now
class DummyGPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"]) # Create an embedding matrix of 50527 words and 768 dimensions
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"]) # At any given time, we can only process context words
        self.drop_emb = nn.Dropout(cfg["drop_rate"]) # 10% drop rate

        # Just a placeholder for Transformer Block
        self.trf_blocks = nn.Sequential(
            *[DummyTransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )
        
        # Just a placeholder for LayerNorm
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False
        )

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)

        logits = self.out_head(x)
        return logits 

# Transformer Block
class TransformerBlock(nn.Module):
    '''
    Inside a transformer block, the input tensor is normalized, and processed by a multiheaded causal 
    attention. This attention output undergoes dropout and is added back to the original un-normalized
    input via a residual, shortcut connection to preserve gradient flow. The updated tensor is 
    normalized again before it passes through FFN, which expands and contracts its dimensions to
    capture non-linear relationships. Next, we do a second round of dropout and shortcut connections to
    produce an enriched context vector.   
    '''
    def __init__(self, cfg):
        super().__init__()
        self.att = mha.MultiHeadAttention(
            d_in = cfg["emb_dim"],
            d_out = cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads = cfg["n_heads"],
            dropout = cfg["drop_rate"]
            qkv_bias=cfg["qkv_bias"]
        )
        self.ff = FeedForward(cfg) # Expand
        self.norm1 = LayerNorm(cfg["emb_dim"]) # Normalize
        self.norm2 = LayerNorm[cfg["emb_dim"]] # Normalize
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"]) # Dropout

    
    def forward(self, x):
        # Shortcut connection for attention block
        shortcut = x
        x = self.norm1(x)
        x = self.att(x) # Shape changes and comes back after multihead attention
        x = self.drop_shortcut(x)
        x = x + shortcut # Add the original input back 

        # Shortcut connection for Feed forward block
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x) 
        x = self.drop_shortcut(x)
        x = x + shortcut # Add the original input back

        return x 

# Normalization class
class LayerNorm(nn.Module):
    '''
    The goal of this class is to force input to have a mean of 0 and a variance of 1.
    Forcing the layers to do that, prevents the vanishing gradient problem.  Without normalization some gradients either shrink to zero (learning stops entirely) or explode to infinity, causing the loss to register as a NaN (not a number). LayerNormalization provides stability
    '''
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5 # To ensure we don't have division by zero errors
        self.scale = nn.Parameter(torch.ones(emb_dim)) # scale, shift give us the flexibility to "undo" the normalization 
        self.shift = nn.Parameter(torch.zeros(emb_dim)) # Helps us reshape the distribution

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False) # unbiased = False so that we divide by n instead of n - 1
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift   

# GELU Class
class GELU(nn.Module):
    '''
    GELU (Gaussian Error Linear Unit) is a mathematical smoothing of ReLU.
    ReLU turns off a neuron if it's negative, crushing it to zero. GELU is 
    a mathematical smoothing of ReLU. The smooth transition allows the model 
    to retain a tiny bit of uncertainty for negative values, making LLM tra-
    ining more stable.
    '''
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.004715 * torch.pow(x,3))
        )      

# Feed Forward class
class FeedForward(nn.Module):
    '''
    MHA figures out which tokens are related to each other, FFN retrieves stroed concepts
    learned during training and injects them into the token. FFN works in isolation, ignoring
    the sequence completely, and applies math to every single token individually.

    Expand ==> Filter (GELU) ==> Compress
    '''
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]), # Expanding the dimension by 4
            GELU(), # Smoothening the negative neurons
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]), # Compressing back to the size
        )
    def forward(self, x):
        return self.layers(x)


# Example neural network for shortcut connections
class DeepNeuralNetwork(nn.Module):
    '''
    Shortcut, Residual connections prevent vanishing gradient problem. When we backpropagate without shortcut connections, the final layer can shrink down to a microscopic value like 0.000002. If the gradient is near zero, the weights in the earlier layers barely update. This bottlenecks the entire network and we can't train properly.
    '''
    def __init__(self, layer_sizes, use_shortcut):
        super().__init__()
        self.use_shortcut = use_shortcut
        self.layers = nn.ModuleList([
            nn.Sequential(nn.Linear(layer_sizes[0], layer_sizes[1]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[1], layer_sizes[2]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[2], layer_sizes[3]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[3], layer_sizes[4]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[4], layer_sizes[5]), GELU())
        ])

    def forward(self, x):
        for layer in self.layers:
            # Compute the output for current layer
            layer_output = layer(x)
            if self.use_shortcut and x.shape == layer_output.shape:
                x = x + layer_output # h(x) = x + f(x)
            else:
                x = layer_output
        
        return x

# function to print gradients
def print_gradients(model, x):
    output = model(x)
    target = torch.tensor([[0.]])

    # calculate loss based on how close the target and output are
    loss = nn.MSELoss()
    loss = loss(output, target)

    loss.backward() # backward pass

    for name, param in model.named_parameters():
        if 'weight' in name:
            print(f"{name} has gradient mean of {param.grad.abs().mean().item()}")


tokenizer = tiktoken.get_encoding("gpt2")
batch = []
txt1 = "Every effort moves you"
txt2 = "Every day holds a"
batch.append(torch.tensor(tokenizer.encode(txt1)))
batch.append(torch.tensor(tokenizer.encode(txt2)))
batch = torch.stack(batch, dim=0)
print(batch)

# Instance of DummyGPT
torch.manual_seed(123)
model = DummyGPTModel(GPT_CONFIG_124M)
logits = model(batch)
# print(logits.shape)

# # Instance of ffn
# ffn = FeedForward(GPT_CONFIG_124M)
# x = torch.ran(2, 3, 768)
# out = ffn(x)
# print(out.shape) # should be [2,3,768] after expansion and compression

# Instance for shortcut connections in Neural Network
layer_sizes = [3, 3, 3, 3, 3, 1]
sample_input = torch.tensor([[1., 0., -1.]])
model_without_shortcut = DeepNeuralNetwork(
layer_sizes, use_shortcut=False
)
print_gradients(model_without_shortcut, sample_input)
# Outputs:
# layers.0.0.weight has gradient mean of 1.1759034350689035e-06
# layers.1.0.weight has gradient mean of 3.0806938866589917e-06
# layers.2.0.weight has gradient mean of 7.358218681474682e-06
# layers.3.0.weight has gradient mean of 0.000168326630955562
# layers.4.0.weight has gradient mean of 0.00635495176538825

model_with_shortcut = DeepNeuralNetwork(layer_sizes, use_shortcut=True)
print_gradients(model_with_shortcut, sample_input)
# Outputs:
# layers.0.0.weight has gradient mean of 0.00030286217224784195
# layers.1.0.weight has gradient mean of 0.0005237706936895847
# layers.2.0.weight has gradient mean of 0.00045960216084495187
# layers.3.0.weight has gradient mean of 0.00032706503407098353
# layers.4.0.weight has gradient mean of 0.01308580581098795