import torch
import torch.nn as nn
import tiktoken

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

# Just a dummy class for now 
class DummyTransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # A simple placeholder 
    
    def forward(self, x):
        return x 

# Normalization class
class LayerNorm(nn.Module):
    '''
    The goal of this class is to force input to have a mean of 0 and a variance of 1.
    Forcing the layers to do that, prevents the vanishing gradient problem.
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
    def __init__():
        super().__init__()

    def forward(self, x):
        return 0.5 * x * torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.004715 * torch.pow(x,3))
        )      

# Feed Forward class
class FeedForward(nn.module):
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
print(logits.shape)

# Instance of ffn
ffn = FeedForward(GPT_CONFIG_124M)
x = torch.ran(2, 3, 768)
out = ffn(x)
print(out.shape) # should be [2,3,768] after expansion and compression