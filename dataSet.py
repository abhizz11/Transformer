from pathlib import Path 
from __future__ import annotations 

import numpy as np
import torch 
from torch.utils.data import DataLoader, Dataset 

class TokenDataset(Dataset):
    def __init__(
            self,
            token_file: str | Path,
            context_length: int,
    ) -> None:
        self.token_file = Path(token_file)
        self.context_length = context_length # Max number of tokens the model can see at a time

        if not self.token_file.exists():
            raise FileNotFoundError(f"Token file not found: {self.token_file}")
        
        # uint16 is exactly 2 bytes. if file isn't divisble by 2 it means the file is corrupted
        if self.token_file.stat().st_size % np.dtype(np.unit16).itemsize != 0:
            raise ValueError(
                f"{self.token_file} is not a valid uint16 token file."
            )

        # Instead of loading an entire dataset into the RAM, create a virtual array and read the 
        # data directly from hard drive only when explicitly asked
        self.tokens = np.memmap( 
            token_file,
            dtype=np.uint16, # Tokenizer's vocab can be upto 65,535 (our size is 8k)
            mode="r", # Opens the file in read-only mode to prevent accidental overwrites
        )

        if len(self.tokens) <= context_length:
            raise ValueError(
                "Token file must contain more tokens than context_length."
            )

    # Tells pytorch how many sequences can be extracted from the file
    def __len__(self) -> int:
        # subtracts 1 from the total number of tokens because the last token is kind of like the "answer key"
        return (len(self.tokens) - 1) // self.context_length

    # When PyTorch asks for an item. this is where the actual token chopping happens
    def __getitem__(
            self,
            index: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if index < 0 or index >= len(self):
            raise IndexError(index)

        start = index * self.context_length
        end = start + self.context_length + 1 # +1 so that if context length is 4 grab 5 tokens

        # Pull a specific chunk from the hard drive via memmap. Convert dtype=np.int64 
        chunk = np.array(
            self.tokens[start:end],
            dtype = np.int64,
            copy=True, # Ensures we aren't accidentally modifying the read-only memory map
        )

        chunk_tensor = torch.from_numpy(chunk)

        # If the prompt is [The cat sat on the mat]
        # Input ids contains [The cat sat on the]
        # Target ids contains [cat sat on the mat]
        input_ids = chunk_tensor[:-1]
        target_ids = chunk_tensor[1:]

        return input_ids, target_ids

def create_dataloader(
        token_file: str | Path,
        context_length: int,
        batch_size: int,
        shuffle: bool,
        num_workers: int = 0,
        pin_memory: bool = True,
        drop_last: bool = True, 
        seed: int = 42 
) -> DataLoader:
    dataset = TokenDataset(
        token_file=token_file,
        context_length=context_length,   
    )
    generator = torch.Generator().manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=batch_size, # How many sequences to push through the network
        shuffle=shuffle, # Randomly scramble the dataset to prevent model from memorizing dataset order
        drop_last=drop_last, #  100 / 32, remainder = 4. we leave the last 4 to prevent crash from small batch size 
        num_workers=num_workers, # Number of separate CPU cores dedicated strictly to fetching data
        pin_memory=pin_memory, # PyTorch optimization
        persistent_workers=num_workers > 0, # Keeps worker processes alive between epochs rather than killing and restarting
        generator=generator if shuffle else None # If the training restarts the dataset is shuffled in exact order
    )