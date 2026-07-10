import torch
import torch.nn as nn
import tiktoken
import multiheadedattention as mha

GPT_CONFIG_124M = {
    "vocab_size": 50257, # Size of Model's vocab
    "context_length": 256, # Words it can process and remember at one time
    "emb_dim": 768, # Embedding dimension, (Different meanings of the same word)
    "n_heads": 12, # Number of Attention heads (Different interpretations of the same sequence)
    "n_layers": 12, # Layers in the transforemer
    "drop_rate": 0.1, # Percent of neurons to randomly turn off
    "qkv_bias": False # Query-key-value bias  
}

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
            dropout = cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"]
        )
        self.ff = FeedForward(cfg) # Expand
        self.norm1 = LayerNorm(cfg["emb_dim"]) # Normalize
        self.norm2 = LayerNorm(cfg["emb_dim"]) # Normalize
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


# GPT Model class
class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"]) # Create an embedding matrix of 50527 words and 768 dimensions
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"]) # At any given time, we can only process context words
        self.drop_emb = nn.Dropout(cfg["drop_rate"]) # 10% drop rate

        # Just a placeholder for Transformer Block
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )
        
        # Layer Normalization
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

# Simple generate function 
def generate(model, idx, max_new_tokens, context_size):
    # loop runs until max_new_tokens have been generated
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:] # Only work on the context size window, trim the excess tokens
        
        # Need torch.no_grad() for training phase but not for generation
        with torch.no_grad():
            logits = model(idx_cond) # P
        
        logits = logits[:, -1, :] # Last row of logits

        probs = torch.softmax(logits, dim = -1) # Probability sampling on the last row
        idx_next = torch.argmax(probs, dim=-1, keepdim=True) # Choose the token with max probability

        idx = torch.cat((idx, idx_next), dim=1) # Append it for next token generation

    return idx # Return at the end

# Text to token id function
def text_to_token_ids(text, tokenizer):
    '''
    Model cannot read Python lists. It can only perform operations on PyTorch Tensors.
    We use this function to create a bridge between model and the tokenizer, before passing the 
    input tokens, we convert it to a tensor. The GPT Model However expects a batch instead of 
    a single tensor, so we are adding a dimension before feeding it to the model.
    '''
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0) # Add dimension
    return encoded_tensor

# Token id to text function
def token_ids_to_text(token_ids, tokenizer):
    '''
    Removing the introduced batch in text_to_token_ids function and then decoding it to return
    a python list of texts
    '''
    flat = token_ids.squeeze(0) # Remove last dimension
    return tokenizer.decode(flat.tolist())


tokenizer = tiktoken.get_encoding("gpt2")

# GPT MODEL
model = GPTModel(GPT_CONFIG_124M)
prompt = "Hey How are you"
model.eval()
encode = tokenizer.encode(prompt)
print("Encoded text: ", encode)
output = generate(
    model = model,
    idx = text_to_token_ids(prompt, tokenizer),
    max_new_tokens=10,
    context_size=GPT_CONFIG_124M["context_length"]
    )

# print("Output text: ", token_ids_to_text(output, tokenizer)) # Output text:  Hey How are you insulated SEAL spray monarchrecordedcerpt workload Morty rollsAmerican

# Measuring cross-entropy loss
'''
We are trying to maximize the probability of generated tokens to be as close to 1 as possible, 
but working with probabilities is messy, so we use logarithms. However, since probability is 
between 0 and 1. the logs come out as negative values, so we take the negative log and now the
problems shifts from maximizing to minimizing. Now, we want the logarithms to be as close to 0 as
possible. 
'''

inputs = torch.tensor([[16833, 3626, 6100],   # ["every effort moves",
                       [40,    1107, 588]])   #  "I really like"]

targets = torch.tensor([[3626, 6100, 345  ],  # [" effort moves you",
                        [1107,  588, 11311]]) #  " really like chocolate"]

with torch.no_grad():
    logits = model(inputs)

logits_flat = logits.flatten(0,1)
targets_flat = targets.flatten()

loss = torch.nn.functional.cross_entropy(logits_flat, targets_flat)
print(loss)