import torch
import torch.nn as nn
from torch.nn import functional as F

# Hyperparameters
batch_size = 32  # how many independent sequences will we process in parallel?
block_size = 8   # what is the maximum context length for predictions?
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200

torch.manual_seed(1337)

# 1. Loading shakespeare dataset
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

print(f"Dataset size: {len(text):,} characters")

# 2. Make vocab (65 unique characters)
chars = sorted(list(set(text)))
vocab_size = len(chars)
print(f"Vocab size: {vocab_size}")

# Embedding mapping: character <-> integer
stoi = { ch: i for i, ch in enumerate(chars) }
itos = { i: ch for i, ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s]           # string -> list of ints
decode = lambda l: ''.join([itos[i] for i in l])  # list of ints -> string

# 3. Full Dataset Tensor & Train/Val Split (90% train, 10% val)
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

print(f"Train tokens: {len(train_data):,} | Validation tokens: {len(val_data):,}")

# Data loading batch function (returns batches (4) of chunk size (8) characters of input and target tensors)
def get_batch(split):
    data_source = train_data if split == 'train' else val_data
    ix = torch.randint(len(data_source) - block_size, (batch_size,))
    x = torch.stack([data_source[i:i+block_size] for i in ix])
    y = torch.stack([data_source[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y # y is x right-shifted by 1 character (bascailly target for our input seq)

@torch.no_grad() # no need to track for eval
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

#^^ returns avergae loss across number of iterations 

# 4. Bigram Language Model Definition
class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        # Each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        # idx (input) and targets(output) are both (B, T) tensor of integers 
        logits = self.token_embedding_table(idx) # (B, T, C)
        #logits are the raw prediction scores for each character in input sequence idx fetched from that first constructor above this function

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    # basically returns prediction scores and calculates loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            logits, loss = self(idx)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            probs = F.softmax(logits, dim=-1) # (B, C)
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1) picks 1 from 65 probabilities
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1) adds it to generated sequence
        return idx

model = BigramLanguageModel(vocab_size)
m = model.to(device)

# 5. Optimization & Training Loop
optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)

for iter in range(max_iters):

    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    xb, yb = get_batch('train')

    logits, loss = m(xb, yb)# calculate loss and raw predictions 
    optimizer.zero_grad(set_to_none=True)# set gradients to 0 before each step
    loss.backward()# loss for current parameters
    optimizer.step()# update parameters

# 6. Generate output from the trained Bigram baseline
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print("\n--- Generated Text ---")
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))