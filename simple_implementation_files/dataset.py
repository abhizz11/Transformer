import tiktoken
from torch.utils.data import Dataset, DataLoader
import torch 

class GPTDataSet(Dataset):
    '''Defines how individual training samples are extracted'''
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []
    
        # Get encoded text
        tokens = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        # Sliding window to iterate over the tokens capturing max tokens at a time
        # Stride is the number of steps
        for i in range(0, len(tokens) - max_length, stride):
            input_chunk = tokens[i: i+max_length] # input
            target_chunk = tokens[i+1:i+max_length+1] # target, next word prediction
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))
        
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]

# Helper function to manage dataset and packages
def create_dataloader(txt, batch_size=4, max_length=256, stride=128, shuffle=True, drop_last=True, num_workers=0):
    tokenizer = tiktoken.get_encoding("gpt2")

    dataset = GPTDataSet(txt, tokenizer, max_length, stride) # Dataset objet

    # Dataloader is highly efficient PyTorch data manager
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, # Group (input, target) together so GPU can process simultaneously
        shuffle=shuffle, # Randomizes the order of batches
        drop_last=drop_last, # Setting this to True drops undersized batch to prevent tensor shape mismatches
        num_workers=num_workers
    )

    return dataloader
