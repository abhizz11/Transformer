import tiktoken
from torch.utils.data import Dataset, DataLoader
import torch 

with open("the-verdict.txt", "r", encoding="utf-8") as file:
    raw_text = file.read() # Contains the entire text


class GPTDataSet(Dataset):
    
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []
    
        tokens = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        for i in range(0, len(tokens) - max_length, stride):
            input_chunk = tokens[i: i+max_length]
            target_chunk = tokens[i+1:i+max_length+1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))
        
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]

def create_dataloader(txt, batch_size=4, max_length=256, stride=128, shuffle=True, drop_last=True, num_workers=0):
    tokenizer = tiktoken.get_encoding("gpt2")

    dataset = GPTDataSet(txt, tokenizer, max_length, stride)

    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )

    return dataloader

dataloader = create_dataloader(
    raw_text, 
    batch_size=1,
    max_length=4,
    stride=1,
    shuffle=False
)

data_iter = iter(dataloader)
first_batch = next(data_iter)
print(first_batch)