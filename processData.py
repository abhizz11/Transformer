import numpy as np
from datasets import load_dataset
from tokenizers import Tokenizer

EOS_TOKEN = "<|endoftext|>"


def tokenize_split(
        split: str,
        output_file: str,
        tokenizer_file: str,
        max_tokens: int | None = None,
) -> None:
    tokenizer = Tokenizer.from_file(tokenizer_file) # Loads the tokenizer we trained
    eos_id = tokenizer.token_to_id(EOS_TOKEN) # Load eos id

    if eos_id is None: # If there's no eos id stop executing to prevent building a corrupted dataset.
        raise ValueError("EOS token is missing from the tokenizer.")
    
    # Load dataset according to the split (train, validation) and streaming=True
    dataset = load_dataset("roneneldan/TinyStories", split=split, streaming=True)

    tokens_written = 0

    # Open the target file on the hard drive and write raw computer binaries
    with open(output_file, "wb") as output:
        for example in dataset:
            token_ids = tokenizer.encode(example["text"]).ids # Convert to tokens
            token_ids.append(eos_id) # Add eos id

            if max_tokens is not None: # Progress checker, if max_new_tokens reached stop else continue
                remaining = max_tokens - tokens_written

                if remaining <= 0:
                    break 
                
                token_ids = token_ids[:remaining]
            
            # Standard python integer takes upto 28 bytes of memory, using 16 bits mean the final dataset file
            # will be 15 times smaller 
            np.asarray(token_ids, dtype=np.uint16).tofile(output)
            tokens_written += len(token_ids)

            if max_tokens is not None and tokens_written >= max_tokens:
                break 
    
    print(f"Wrote {tokens_written:,} tokens to {output_file}")
    