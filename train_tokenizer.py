from collections.abc import Iterator 

from datasets import load_dataset # Loading dataset
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers # Modules required to train a tokenizer

VOCAB_SIZE = 8_000 # vocab size for transformer model
MAX_TOKENIZER_STORIES = 200_000 # Number of stories to train tokenizer for
BATCH_SIZE = 1_000 # Process 1000 stories first before moving on to the next

EOS_TOKEN = "<|endoftext|>" # End of sequence token
UNK_TOKEN = "<|unk|>" # unknown tokens 

def story_batches() -> Iterator[list[str]]:
    dataset = load_dataset(
        "roneneldan/TinyStories",
        split = "train",
        streaming = True
    )
    batch = [] # List to store stories
    stories_seen = 0 # To count the number of stories seen for iterator

    for example in dataset:
        batch.append(example["text"])
        stories_seen += 1

        if len(batch) == BATCH_SIZE: # yield pauses the function and hands it over to the Tokenizer to be processed
            yield batch 
            batch = []
        
        if stories_seen >= MAX_TOKENIZER_STORIES: # If seen more than 200k stories stop
            break 
    
    if batch:
        yield batch 

def main():
    # Create a tokenizer object, using Byte-Pair encoding model
    tokenizer = Tokenizer(
        models.BPE(unk_token=UNK_TOKEN)
    )

    # Convert text into raw computer bytes. This algo works with a strictly limited 256 basic building blocks
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(
        add_prefix_space=False # Treat the first word as it is don't assume there's a space at the beginning
    )

    tokenizer.decoder = decoders.ByteLevel() # Reverse of the pre tokenizer, converts numbers into bytes then into human readable text

    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE, # Stop learning new tokens after hitting this limit
        min_frequency=2, # Pattern must appear at least twice to be considered new token
        special_tokens=[UNK_TOKEN, EOS_TOKEN], # Special tokens
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), # Seeds the dictionary with 256 fundamental byte values before training
        show_progress=True
    )

    tokenizer.train_from_iterator(
        story_batches(), # calls the story_batches function for batching
        trainer=trainer, # feed the batches onto trainer
        length=MAX_TOKENIZER_STORIES, # go on until 200_000 stories
    )

    tokenizer.save("tinystories_tokenizer.json")

    print("Tokenizer saved.")
    print("Vocabulary size: ", tokenizer.get_vocab_size())


if __name__ == "__main__":
    main()