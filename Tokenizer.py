import re

with open("the-verdict.txt", "r", encoding="utf-8") as file:
    raw_text = file.read() # Contains the entire text


res = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text) # First split the text by punctuation and whitespace
result = [item.strip() for item in res if item.strip()] # Keep items that are not whitespace

all_tokens = sorted(set(result)) # Get unique tokens
new_tokens = ["<|EOS|>", "<|UNK|>"] # S[ecial tokens for end of sentence and unknown words 
all_tokens.extend(new_tokens) # Add special tokens to the list of all tokens


vocab = {token:idx for idx,token in enumerate(all_tokens)} # Create a mapping of token to index

class Tokenizer:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {idx:token for token, idx in vocab.items()} # Create a mapping of index to token
    
    def encode(self, text):
        tokens = re.split(r'([,.:;?_!"()\']|--|\s)', text) # Split the input text into tokens
        tokens = [token.strip() for token in tokens if token.strip()] # Remove whitespace tokens

        txt = [self.str_to_int.get(token, self.str_to_int["<|UNK|>"]) for token in tokens] # Convert tokens to indices, using <|UNK|> for unknown tokens
        txt.append(self.str_to_int["<|EOS|>"]) # Append the end of sentence token
        return txt
    
    def decode(self, token_ids):
        tokens = " ".join([self.int_to_str.get(token_id, "<|UNK|>") for token_id in token_ids]) # Convert indices back to tokens, using <|UNK|> for unknown indices
        tokens = re.sub(r'\s+([,.?!"()\'])', r'\1', tokens)
        return tokens # Join the tokens into a single string

Tokenizer = Tokenizer(vocab) # Create an instance of the Tokenizer class with the vocabulary
txt = "The day is pretty nice, abhi" # Example text to encode
en = Tokenizer.encode(txt)
dc = Tokenizer.decode(en)
