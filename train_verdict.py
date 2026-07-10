import os
import urllib.request
import tiktoken
import dataset as d
import gptModel
import torch
import time 
from datasets import load_dataset

ds = load_dataset("roneneldan/TinyStories")

start_time = time.time()

# Measuring cross-entropy loss
'''
We are trying to maximize the probability of generated tokens to be as close to 1 as possible, 
but working with probabilities is messy, so we use logarithms. However, since probability is 
between 0 and 1, the logs come out as negative values, so we take the negative log and now the
problem shifts from maximizing to minimizing. Now, we want the logarithms to be as close to 0 as
possible. 
'''

file_path = "C:/Users/owner/Desktop/Transformer/the-verdict.txt"
url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt"

GPT_CONFIG_124M = {
    "vocab_size": 50257, # Size of Model's vocab
    "context_length": 256, # Words it can process and remember at one time
    "emb_dim": 768, # Embedding dimension, (Different meanings of the same word)
    "n_heads": 12, # Number of Attention heads (Different interpretations of the same sequence)
    "n_layers": 12, # Layers in the transformer
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

total_characters = len(text_data)

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

# Load the validation batch, this is for honesty, to check if model is memorizing the dataset or learning patterns
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


# Train model
def train_model(
        model, 
        train_loader,
        val_loader,
        optimizer,
        device,
        num_epochs,
        eval_freq,
        eval_iter,
        start_context,
        tokenizer
):
    # To track losses and tokens seen
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    for epoch in range(num_epochs):
        model.train() # setting model to training mode

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad() # Reset loss gradients from previous batch iteration
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward() # Calculate loss gradients
            optimizer.step() # Update model weights using loss gradients
            tokens_seen += input_batch.numel() # The total number of elements seen in the input batch
            global_step += 1

            # Evaluation step
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch + 1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")
        
        # Print a sample text after each epoch
        generate_and_print_sample(
            model, tokenizer, device, start_context
        )

    return train_losses, val_losses, track_tokens_seen 

# Evalautes the model, every eval_freq th iteration to visualize the train loss and validation loss
def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval() # change to eval mode
    with torch.no_grad(): # No grad here, we are just calculating losses
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    
    model.train() # Change it back to train
    return train_loss, val_loss

# Visualizes the model's output after each epoch
def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval() # Change to eval mode
    context_size = model.pos_emb.weight.shape[0]
    encoded = gptModel.text_to_token_ids(start_context, tokenizer).to(device) # convert text to token
    with torch.no_grad(): # No training, so no grad
        token_ids = gptModel.generate(
            model = model,
            idx= encoded,
            max_new_tokens=50,
            context_size=context_size
        ) # generate output
    decoded_text = gptModel.token_ids_to_text(token_ids, tokenizer) # decode the output
    print(decoded_text.replace("\n", " ")) # print
    model.train() # change to training mode again


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


model = gptModel.GPTModel(GPT_CONFIG_124M)
model.to(device)

# Adam stands for Adaptive Moment Estimation, instead of using a single, fixed learning rate
# for everything, Adam individually adapts the learning rate for each parameter based on how they have been changing

# lr = learning rate, if it's too high, might jump over the valley (loss) if it's too low, will reach slower
# weight_decay = regularization penalty designed to prevent model from overfitting
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0002, weight_decay=0.75)

num_epochs = 10 # Number of times the model will see the entire dataset
train_losses, val_losses, tokens_seen = train_model(
    model, train_loader, val_loader, optimizer, device, 
    num_epochs=num_epochs, eval_freq=5, eval_iter=5,
    start_context="Every day is beautiful", tokenizer=tokenizer
)
end_time = time.time()
execution_time_minutes = (end_time - start_time) / 60
print(f"Training completed in {execution_time_minutes:.2f} minutes ")

