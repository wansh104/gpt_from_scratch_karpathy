import torch

# 1. Load the dataset
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

print(f"Dataset size: {len(text):,} characters")

# 2. Vocabulary & Character Mapping
chars = sorted(list(set(text)))
vocab_size = len(chars)
print(f"Vocab size: {vocab_size}")

stoi = { ch: i for i, ch in enumerate(chars) }
itos = { i: ch for i, ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s]           # string -> list of ints
decode = lambda l: ''.join([itos[i] for i in l])  # list of ints -> string

# Quick Test: Encode & Decode sample text
sample_str = "hello world"
encoded_sample = encode(sample_str)
print(f"\nSample: '{sample_str}'")
print(f"Encoded: {encoded_sample}")
print(f"Decoded back: '{decode(encoded_sample)}'")

# 3. Full Dataset Tensor & Train/Val Split (90% train, 10% val)
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

print(f"\nTrain tokens: {len(train_data):,} | Validation tokens: {len(val_data):,}")