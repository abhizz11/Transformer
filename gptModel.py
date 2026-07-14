'''
 This file implements the GPT Model with weight tying.
 A standard LLM has two massive matrices that deal with vocab,
 input embedding and output head (logits) 
 Weight tying forces these two distinct layer to share the exact
 same matrix in memory (one just transpose of other), saving RAM usage
 and training time

'''
from __future__ import annotations
import torch
import torch.nn as nn
import multiheadedattention as mha

TINYSTORIES_CONFIG_29M = {
    "vocab_size": 8_000, # Size of Model's vocab
    "context_length": 256, # Words it can process and remember at one time
    "emb_dim": 512, # Embedding dimension, (Different meanings of the same word)
    "n_heads": 8, # Number of Attention heads (Different interpretations of the same sequence)
    "n_layers": 8, # Layers in the transformer
    "drop_rate": 0.1, # Percent of neurons to randomly turn off
    "qkv_bias": False # Query-key-value bias  
}

GPT_CONFIG_124M = TINYSTORIES_CONFIG_29M 

# Feed Forward class
class FeedForward(nn.Module):
    '''
    MHA figures out which tokens are related to each other, FFN retrieves stored concepts
    learned during training and injects them into the token. FFN works in isolation, ignoring
    the sequence completely, and applies math to every single token individually.

    Expand ==> Filter (GELU) ==> Compress
    '''
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]), # Expanding the dimension by 4
            nn.GELU(approximate="tanh"), # Smoothening the negative neurons
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
        self.norm1 = nn.LayerNorm(cfg["emb_dim"]) # Normalize
        self.norm2 = nn.LayerNorm(cfg["emb_dim"]) # Normalize
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"]) # Dropout

    
    def forward(self, x):
        x = x + self.drop_shortcut(self.att(self.norm1(x)))
        x = x + self.drop_shortcut(self.ff(self.norm2(x)))
        return x


# GPT Model class
class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = dict(cfg)
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"]) # Create an embedding matrix of 8k words and 512 dimensions
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"]) # At any given time, we can only process context words
        self.drop_emb = nn.Dropout(cfg["drop_rate"]) # 10% drop rate

        # Transformer Block
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )
        
        # Layer Normalization
        self.final_norm = nn.LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False
        )
        self.apply(self._init_weights)

        # Weight typing, input_embeddings and output logits use one matrix. Removes 8000 * 512 = 4_096_000 duplicate parameters
        self.out_head.weight = self.tok_emb.weight
    
    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        '''Function overrides PyTorch's default random weight assignments
        and manually sets the initial starting numbers for NN's matrices before training begins. If the 
        model starts with wrong weights, it will physically be unable to learn.
        The purpose of this initialization is to stabilize the math during the very first steps of training.
        '''
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, in_idx):
        _, seq_len = in_idx.shape
        context_length = self.cfg["context_length"]
        if seq_len > context_length:
            raise ValueError(
                f"Sequence length {seq_len} exceeds context length "
                f"{context_length}."
            )


        token_embeddings = self.tok_emb(in_idx)
        positions = torch.arange(seq_len, device=in_idx.device)
        position_embeddings = self.pos_emb(positions)

        x = self.drop_emb(token_embeddings + position_embeddings)
        x = self.trf_blocks(x)
        x = self.final_norm(x)

        return self.out_head(x)

# Generate function
@torch.inference_mode() # Disable gradient calculation
def generate(
        model, # model
        idx, # Input ids
        max_new_tokens, # Tokens to generate
        context_size, # how much of the text model remembers at a time
        temperature=0.8, # how random the model is
        top_k=40, # prevents the token from picking nonsensical tokens
        eos_id=None
        ):
    was_training = model.training # Check if model was training for later
    model.eval()

    try: 
        # loop runs until max_new_tokens have been generated
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -context_size:] # Only work on the context size window, trim the excess tokens
            logits = model(idx_cond)[:, -1, :] # Last row of digits
            
            # Greedy decoding. Pick the token with the highest score, this is deterministic 
            if temperature <= 0:
                idx_next = torch.argmax(logits, dim=-1, keepdim=True) # argmax gives out token with highest logits
            else:
                # A temperature < 1 sharpens the distribution, high scores higher low scores lower
                logits = logits / temperature  # temperature > 1 increases randomness

                
                if top_k is not None:
                    k = min(top_k, logits.shape[-1]) # Safety check, so that the K is not larger than model's vocab
                    top_values, _ = torch.topk(logits, k) # Isolates the k highest_scoring tokens
                    cutoff = top_values[:, -1].unsqueeze(-1) # Isolate the score of the lowest token within k-group
                    logits = logits.masked_fill(logits < cutoff, -torch.inf) # Below cutoff turn any other logit score to -inf
                
                probabilities = torch.softmax(logits, dim=-1) # Normalize
                idx_next = torch.multinomial(probabilities, num_samples=1) # Instead of picking the highest one, roll a die and pick the next token 

            idx = torch.cat((idx, idx_next), dim=1) # Append it for next token generation

            if eos_id is not None and torch.all(idx_next == eos_id):
                break
    finally:
        if was_training: # Change the state if it was training
            model.train()

    return idx # Return at the end

# Text to token id function
def text_to_token_ids(text, tokenizer):
    '''
    Model cannot read Python lists. It can only perform operations on PyTorch Tensors.
    We use this function to create a bridge between model and the tokenizer, before passing the 
    input tokens, we convert it to a tensor. The GPT Model However expects a batch instead of 
    a single tensor, so we are adding a dimension before feeding it to the model.
    '''
    encoded = tokenizer.encode(text)
    token_ids = encoded.ids if hasattr(encoded, "ids") else encoded
    return torch.tensor(token_ids, dtype=torch.long).unsqueeze(0)

# Token id to text function
def token_ids_to_text(token_ids, tokenizer, skip_special_tokens=True):
    '''
    Removing the introduced batch in text_to_token_ids function and then decoding it to return
    a python list of texts
    '''
    ids = token_ids.detach().cpu().reshape(-1).tolist()
    return tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)

# Returns the total number of parameters in the model
def count_parameters(model):
    return sum(parameter.numel() for parameter in model.parameters())

