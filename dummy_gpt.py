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
        self.final_norm = DummyLayerNorm(cfg["emb_dim"])
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

# Just a dummer layer for normalization
class DummyLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
    
    def forward(self, x):
        return x 

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
print(logits)