import re

with open("the-verdict.txt", "r", encoding="utf-8") as file:
    raw_text = file.read() # Contains the entire text


res = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text) # First split the text by punctuation and whitespace
result = [item.strip() for item in res if item.strip()] # Keep items that are not whitespace

all_tokens = sorted(set(result)) # Get unique tokens
vocab_size = len(all_tokens) # Get the size of the vocabulary

encoding = {token:idx for idx,token in enumerate(all_tokens)} # Create a mapping of token to index
decoding = {idx:token for idx,token in enumerate(all_tokens)} # Create a mapping of index to token

print(decoding)