import os
import urllib.request
import tiktoken
import dataset as d
import gptModel
import torch

# Measuring cross-entropy loss
'''
We are trying to maximize the probability of generated tokens to be as close to 1 as possible, 
but working with probabilities is messy, so we use logarithms. However, since probability is 
between 0 and 1. the logs come out as negative values, so we take the negative log and now the
problems shifts from maximizing to minimizing. Now, we want the logarithms to be as close to 0 as
possible. 
'''

file_path = "C:/Users/owner/Desktop/Transformer/the-verdict.txt"
url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt"

GPT_CONFIG_124M = {
    "vocab_size": 50257, # Size of Model's vocab
    "context_length": 256, # Words it can process and remember at one time
    "emb_dim": 768, # Embedding dimension, (Different meanings of the same word)
    "n_heads": 12, # Number of Attention heads (Different interpretations of the same sequence)
    "n_layers": 12, # Layers in the transforemer
    "drop_rate": 0.1, # Percent of neurons to randomly turn off
    "qkv_bias": False # Query-key-value bias  
}

# Load the file, if it exists in the file_path, else download
if not os.path.exists(file_path):
    with urllib.request.urlopen(url) as response:
        text_data = response.read().decode('utf-8')
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(text_data)
else:
    with open(file_path, "r", encoding="utf-8") as file:
        text_data = file.read()


# Load tokenizer
tokenizer = tiktoken.get_encoding("gpt2")

# Sanity check
print(text_data[:99])
print(text_data[-99:])

total_characters = len(text_data)
total_tokens = len(tokenizer.encode(text_data))

print("Characters:", total_characters)
print("Tokens:", total_tokens)

# Training variables
total_tokens = len(tokenizer.encode(text_data))
train_ratio = 0.90 #  We split data into two parts and only train on the 90%
split_idx = int(train_ratio * len(text_data))
train_data = text_data[:split_idx]
val_data = text_data[split_idx:]

# Load the training batch, to teach the model 
train_loader = d.create_dataloader(
    train_data,
    batch_size=2,
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    drop_last=False,
    shuffle=False,
    num_workers=0
)

# Load the value batch, this is for honesty, to check if model is memorizing the dataset or learning patterns
val_loader = d.create_dataloader(
    val_data,
    batch_size=2,
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    drop_last=False,
    shuffle=False,
    num_workers=0
)

if total_tokens * (train_ratio) < GPT_CONFIG_124M["context_length"]:
    print("Not enough tokens for the training loader. "
          "Try to lower the `GPT_CONFIG_124M['context_length']` or "
          "increase the `training_ratio`")

if total_tokens * (1-train_ratio) < GPT_CONFIG_124M["context_length"]:
    print("Not enough tokens for the validation loader. "
          "Try to lower the `GPT_CONFIG_124M['context_length']` or "
          "decrease the `training_ratio`")

print("Train loader:")
for x, y in train_loader:
    print(x.shape, y.shape)

print("\nValidation loader:")
for x, y in val_loader:
    print(x.shape, y.shape)

train_tokens = 0
for input_batch, target_batch in train_loader:
    train_tokens += input_batch.numel()

val_tokens = 0
for input_batch, target_batch in val_loader:
    val_tokens += input_batch.numel()

print("Training tokens:", train_tokens)
print("Validation tokens:", val_tokens)
print("All tokens:", train_tokens + val_tokens)

# Calculates the loss for a single batch
def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device) # Load data into the same device
    logits = model(input_batch) # Get the logits
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten()) # reshape the tensor and calculate cross-entropy loss
    return loss # Return loss for this specific batch

# Calculates the loss for loader 
def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0 # Loss tally
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device) # Calculate loss for each batch
            total_loss += loss.item()
        else:
            break 
    
    return total_loss / num_batches # Return the average loss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


model = gptModel.GPTModel(GPT_CONFIG_124M)
model.to(device)

with torch.no_grad(): # Disable gradient tracking for efficiency because we are not training, yet
    train_loss = calc_loss_loader(train_loader, model, device)
    val_loss = calc_loss_loader(val_loader, model, device)

print("Training loss:", train_loss)
print("Validation loss:", val_loss)