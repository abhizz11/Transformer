from __future__ import annotations

from pathlib import Path 
import numpy as np
from datasets import load_dataset
from tokenizers import Tokenizer
from tqdm.auto import tqdm

EOS_TOKEN = "<|endoftext|>"


def tokenize_split(
        split: str,
        output_file: str,
        tokenizer_file: str,
        max_tokens: int | None = None,
        shuffle_buffer_size: int = 10_000,
        seed: int = 42,
        overwrite: bool = False,
) -> int:
    output_path = Path(output_file)
    tokenizer_path = Path(tokenizer_file)

    if output_path.exists() and not overwrite:
        token_count = output_path.stat().st_size // np.dtype(np.uint16).itemsize
        print(f"Using existing file: {output_path} ({token_count:,} tokens)")
        return token_count
    
    if max_tokens is not None and max_tokens <= 0:
        raise ValueError("max_tokens must be positive or None")

    tokenizer = Tokenizer.from_file(str(tokenizer_path)) # Loads the tokenizer we trained
    vocab_size = tokenizer.get_vocab_size()

    if vocab_size > np.iinfo(np.uint16).max + 1:
        raise ValueError("Vocabulary is too large to store as uint16.")
    

    eos_id = tokenizer.token_to_id(EOS_TOKEN) # Load eos id
    if eos_id is None: # If there's no eos id stop executing to prevent building a corrupted dataset.
        raise ValueError("EOS token is missing from the tokenizer.")
    
    # Load dataset according to the split (train, validation) and streaming=True
    dataset = load_dataset("roneneldan/TinyStories", split=split, streaming=True)

    # Shuffle according to the seed
    if shuffle_buffer_size > 0 and split == "train":
        dataset = dataset.shuffle(
            seed=seed,
            buffer_size=shuffle_buffer_size,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    tokens_written = 0
    
    progress = tqdm(
        total=max_tokens,
        unit="tok",
        unit_scale=True,
        desc=f"Tokenizing {split}",
    )

    try:
        # Open the target file on the hard drive and write raw computer binaries
        with temporary_path.open("wb") as output:
            for example in dataset:
                token_ids = tokenizer.encode(example["text"]).ids # Convert to tokens
                token_ids.append(eos_id) # Add eos id

                if max_tokens is not None: # Progress checker, if max_new_tokens reached stop else continue
                    remaining = max_tokens - tokens_written

                    if remaining <= 0:
                        break 
                    
                    token_ids = token_ids[:remaining]
                
                # Standard python integer takes up to 28 bytes of memory, using 16 bits mean the final dataset file
                # will be 15 times smaller 
                np.asarray(token_ids, dtype=np.uint16).tofile(output)
                tokens_written += len(token_ids)
                progress.update(len(token_ids))

                if max_tokens is not None and tokens_written >= max_tokens:
                    break 
        
        temporary_path.replace(output_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
    finally:
        progress.close()
    
    print(f"Wrote {tokens_written:,} tokens to {output_file}")
    return tokens_written
    