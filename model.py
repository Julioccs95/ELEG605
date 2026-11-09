"""
model.py - minimal transformer for PatternBoost.

Based on makemoretokens.py (Karpathy's makemore). Kept exactly as in the original:
  ModelConfig, NewGELU, CausalSelfAttention, Block, Transformer, generate

Removed: Bigram, MLP, RNN, GRU, BoW, TensorBoard, logger, logprobs, eval dataset,
argparse, print_samples, InfiniteDataLoader, CUDA_LAUNCH_BLOCKING.

Simplified: CharDataset -> SeqDataset (takes integer ids from vocab.encode, so no
"V" strings or string-to-id mapping), and the training/sampling code from the
__main__ block -> train() and sample().

Id 0 is the START/STOP token (vocab.py shifts all real tokens by +1).
"""

#Import the classes of interest
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader


# -----------------------------------------------------------------------------
# Model (unchanged from makemoretokens.py)

#Here we check for the model parameters
@dataclass
class ModelConfig:
    block_size: int = None  # length of the input sequences (incl. START token)
    #Longest sequence the model can read in tokens counting the starting tokens. the default is max_len + 1
    #This is also the length of the attention mask and the position embeddings
    vocab_size: int = None  # ids are in range [0 .. vocab_size - 1]
    #The amount of different ids that exist from the BPE plus 1 for start/stop id
    n_layer: int = 4
    #n_layer = how many transformer blocks are stacked
    #Each attention has a feed forward layer. The more layers the more complex patterns it learns
    n_embd: int = 64
    #The length of the vector that represents each token inside the model. The larger the vector
    #The more details and add more parameters.
    n_head: int = 4
    #How many attention heads split up. each head focus on different pattern such as 
    #nearby tokens, or far back in the sequence. Thus is split n_embd / n_head or (64/4=16 per head)

#GELU is the equivalent of relu, but instead of using a hard cutoff it is more smooth
class NewGELU(nn.Module):
    """GELU activation (as in GPT-2)."""
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


class CausalSelfAttention(nn.Module):
    """Multi-head attention where each token only sees earlier tokens."""

    #Obtains the models parameter dataclass
    def __init__(self, config):
        super().__init__()
        #This is important as they share equal dimension by number of heads, splitting the vector
        assert config.n_embd % config.n_head == 0
        #Create the linear layer to output the Q,K,V
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)  # query, key, value
        #Output the projection back into original size
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)      # output projection
        # lower-triangular mask: position t can only attend to positions <= t
        #Create all 1's block matrix and then keeping the lower triangular parts
        #This is essentially the cuasal mask
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                             #Changes shape (4,4) to (1,1,4,4) and register in the model as self.bias
                                     .view(1, 1, config.block_size, config.block_size))

        #Denote in the head or embd in the class
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        #Denote the batch size, sequence length, and embedding size
        B, T, C = x.size()
        #Here we apply the linear layer first which makes the shape (B,T,3*n_emb), then split into 3 equal dimensions
        #dim = 2 means to apply it into the 3rd position
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        #eg. change (32, 20, 64) to (32, 20, 4, 16) then .transpose(1, 2) meaning dimension 1 and 2 swap
        #so we go from (B, T, heads, head_size) (32,20,4,16) -> (B, heads, T, head_size) (32,4,20,16)
        #Does the same in k,q,v
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        #k.transpose(-2, -1) swaps the last 2 dimensions to make q be able to do matrix multiplication together
        #k.size(-1) gives the size of one attention head dimension
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        #self.bias[...] gets the lower triangular mask for the sequence length
        #Create mask with ==0 which are the values we want to block
        #Wherever the mask is true(0) replace with -inf
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        #Apply the softmax into the last dimension
        #This leaves with att.shape = (B, n_heads, T, T) before the softmax
        #Turns the attention values into softmax [0.5, 1.0, 2.5]
        att = F.softmax(att, dim=-1)
        #Apply the attention weights into value vectors
        y = att @ v
        #Shape starts here (B, heads, T, head_size) then swaps transpose(1, 2) dim 1 and 2
        #.contiguous() is used to reshape safely by changing the memory
        #Then view becomes (B, 4, T, 16) to (B, T, 64) which is merged by view()
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        #Passes into the projection linear layer
        return self.c_proj(y)


class Block(nn.Module):
    """Attention + feedforward, each with LayerNorm and a residual connection."""
    #Initializes with the model configurations
    def __init__(self, config):
        #Initializes the pytorch module
        super().__init__()
        #Layer of normalization before the attention
        self.ln_1 = nn.LayerNorm(config.n_embd)
        #Applies the les attention layer
        self.attn = CausalSelfAttention(config)
        #Second layer of normalization
        self.ln_2 = nn.LayerNorm(config.n_embd)
        #This is the MLP part
        self.mlp = nn.ModuleDict(dict(
            c_fc=nn.Linear(config.n_embd, 4 * config.n_embd),
            c_proj=nn.Linear(4 * config.n_embd, config.n_embd),
            #Activation between the 2 layers
            act=NewGELU(),
        ))
        m = self.mlp
        self.mlpf = lambda x: m.c_proj(m.act(m.c_fc(x)))
        #The lambda defines the feed-forward portion
        #x -> linear(4 -> 16) -> GELU -> Linear(16 -> 4)

    def forward(self, x):
        #Takes x into normalization 1, then run self attention
        x = x + self.attn(self.ln_1(x))
        #normalize again in ln2 then feed forward network, then add
        #x again
        x = x + self.mlpf(self.ln_2(x))
        return x


class Transformer(nn.Module):
    """GPT-2 style language model."""

    #Initialize the model from model configuration
    def __init__(self, config):
        super().__init__()
        #Denotes the class block size
        self.block_size = config.block_size
        #Initializes the dictionary for the transformer components
        self.transformer = nn.ModuleDict(dict(
            #The first line is the token/word embedding
            #The arguments are the number of possible id's and the size of each vector (embedding)
            #eg (nn.Embedding(100,64)) where we essentially have 100 rows and 64 columns
            wte=nn.Embedding(config.vocab_size, config.n_embd),   # token embeddings
            wpe=nn.Embedding(config.block_size, config.n_embd),   # position embeddings
            #This creates the transformer blocks
            #This creates a list of blocks [block(config), block(config), block(config)]
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            #The final layernorm applied after all transformer blocks
            ln_f=nn.LayerNorm(config.n_embd),
        ))
        #Final prediction layer where the input is the size of embedding, and output is the vocab size
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def get_block_size(self):
        return self.block_size


#The model gets the idx (input tokens ID's) and expected next tokens
    def forward(self, idx, targets=None):
        #Get the shape of the input
        #batch size and sequence length
        b, t = idx.size()
        #Make sure the sequence isnt longer than the models maximum length
        assert t <= self.block_size, f"Sequence length {t} > block size {self.block_size}"
        #Create the positional numbers
        #.unsqueeze(0) adds a batch dimension
        #example : [0, 1, 2, 3, 4] -> (5,)
        #Then turns into [[0, 1, 2, 3, 4]] -> (1, 5)
        #Makes it suitable to look up the positional embeddings
        pos = torch.arange(0, t, dtype=torch.long, device=idx.device).unsqueeze(0)

        #look up the token embeddings wte(idx) and position embeddigngs wpe(pos)
        #Add them to make the input representation
        x = self.transformer.wte(idx) + self.transformer.wpe(pos)
        #Go through each transformer block
        for block in self.transformer.h:
            #Pass the embeddings into the transformer blocks
            #Each block has attention + MLP
            x = block(x)
        #Logits is obtained by applying the final layer norm and 
        #Final prediciton layer
        logits = self.lm_head(self.transformer.ln_f(x))

        loss = None
        if targets is not None:
            # ignore_index=-1 skips the padding after STOP
            #Computes the cross entropy loss logits.view(-1, logits.size(-1))
            #What this means is that say that we have the shape of logits like (32,20,100) then view(-1) would reshape it to (340(32 * 20), 100) and logit.size(-1)
            #to be 100. All together it is 340,100
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            #returns the logit and loss
        return logits, loss

#Do no track the gradients
@torch.no_grad()
def generate(model, idx, max_new_tokens, temperature=1.0, top_k=None):
    #Idx - The prompt or starting tokens
    #top_k0 - optional if to choose between the top k options
    #max_new_tokens - how much new tokens to generate
    """Extend idx one sampled token at a time, feeding each prediction back in."""
    block_size = model.get_block_size()
    for _ in range(max_new_tokens):
        #generate one token at a time
        idx_cond = idx if idx.size(1) <= block_size else idx[:, -block_size:]
        #This makes sure that the generated output is within the block size, if not then we trim it
        logits, _ = model(idx_cond)
        #gets the logits from the model class or transformer, basically gets the prediciton
        logits = logits[:, -1, :] / temperature
        #Keep only the prediction for the next token (the last position)
        if top_k is not None:
            v, _ = torch.topk(logits, top_k)
            #Gets the top k largest logits
            logits[logits < v[:, [-1]]] = -float('Inf')
            #Set the rest of the tokens as -inf so they can never be chosen
            #dim = -1 then it goes into (32,100) the 100 samples
        probs = F.softmax(logits, dim=-1)
        #Convert the scores(logits) into probabilities
        idx_next = torch.multinomial(probs, num_samples=1)
        #randomly samples the the next token accoriding to the probabilities
        idx = torch.cat((idx, idx_next), dim=1)
        #Append the new token to the sequence
        #dim = 1 is because we have batch and tokens as (B,T) and we want to apply into dim=1
    return idx


# -----------------------------------------------------------------------------
# Dataset (simplified CharDataset)

class SeqDataset(Dataset):
    """
    Each item is a list of token ids from vocab.encode.
      x = [0, t1, t2, ..., tn, 0, 0, ...]     (START, tokens, padding)
      y = [t1, t2, ..., tn, 0, -1, -1, ...]   (next token, STOP, ignored)
    """

    def __init__(self, seqs, max_len):
        self.seqs = seqs
        #list of tokenized sequences
        self.max_len = max_len
        #max length of the sequences

    def __len__(self):
        #When you call len(dataset) then you get the length of the sequences in the list
        return len(self.seqs)

    def __getitem__(self, i):
        ix = torch.tensor(self.seqs[i][:self.max_len], dtype=torch.long)
        #get sequence i and truncate to the max length
        x = torch.zeros(self.max_len + 1, dtype=torch.long)
        y = torch.zeros(self.max_len + 1, dtype=torch.long)
        #create input and output as all 0 sequence
        x[1:1 + len(ix)] = ix
        y[:len(ix)] = ix
        y[len(ix) + 1:] = -1
        return x, y


''''
Get item example:
ix = [5, 8, 10]

x = [0, 0, 0, 0, 0, 0]
y = [0, 0, 0, 0, 0, 0]

x[1:1 + len(ix)] = ix -> x = [0, 5, 8, 10, 0, 0]

y[:len(ix)] = ix -> y = [5, 8, 10, 0, 0, 0]
#First 0 is the start token but the immediate after 10 is the stop one

y[len(ix) + 1:] = -1 -> y = [5, 8, 10, 0, -1, -1]
#Eveyrhting after stop is -1


final result 
x = [0, 5, 8, 10, 0, 0]
y = [5, 8, 10, 0, -1, -1]
'''

# -----------------------------------------------------------------------------
# Training and sampling (from the __main__ block of makemoretokens.py)

def build_model(vocab_size, max_len, n_layer=4, n_embd=64, n_head=4, device="cpu"):
    """max_len = longest encoded sequence; +1 makes room for the START token."""
    #Initializes the model with the default model specs we denote at the top
    config = ModelConfig(vocab_size=vocab_size, block_size=max_len + 1,
                         n_layer=n_layer, n_embd=n_embd, n_head=n_head)
    return Transformer(config).to(device)


def train(model, seqs, max_len, steps=2000, batch_size=32, lr=5e-4,
          weight_decay=0.01, device="cpu", print_every=200):
    """Train on a list of token-id lists. Returns the final loss."""
    #DataLoader(...) provides the dataset to be provided in batches
    loader = DataLoader(SeqDataset(seqs, max_len), batch_size=batch_size, shuffle=True)
    #Shuffle = True to prevent the model to seeing the same sequence of training points
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay,
                                  betas=(0.9, 0.99), eps=1e-8)
    model.train()
    step, loss = 0, None
    while step < steps:
        for X, Y in loader:
            X, Y = X.to(device), Y.to(device)
            _, loss = model(X, Y)
            optimizer.zero_grad(set_to_none=True)
            #Delete the old gradients before computing the new ones in this case instead of 0 it's NONE
            loss.backward()
            optimizer.step()
            step += 1
            if step % print_every == 0:
                print(f"step {step} | loss {loss.item():.4f}")
            if step >= steps:
                break
    return loss.item()


def sample(model, num, max_len, temperature=1.0, top_k=None, batch_size=1000, device="cpu"):
    """Return `num` sampled sequences as lists of ids (ready for vocab.decode)."""
    model.eval()
    out = []
    #num is the number of sequence
    while len(out) < num:
        n = min(batch_size, num - len(out))
        #n= decides how many more sequences to generate
        #Create the all 0 vector
        start = torch.zeros(n, 1, dtype=torch.long, device=device)  # START token
        #Generate the encodede tokens
        rows = generate(model, start, max_len, temperature, top_k).to("cpu")
        #Drop the start token and put the generated sequence to out
        out.extend(rows[:, 1:].tolist())  # drop START; vocab.decode stops at STOP
    return out


if __name__ == "__main__":
    # Quick self-test: the loss should fall well below its starting value
    import random
    import vocab

    M = 105  # N = 15
    data = ["".join(random.choice("01") for _ in range(M)) for _ in range(500)]
    vocab.train_tokenizer(data, n_tokens=100)
    seqs = [vocab.encode(s) for s in data]
    max_len = max(len(s) for s in seqs)

    model = build_model(vocab.vocab_size(), max_len)
    train(model, seqs, max_len, steps=500)

    decoded = [vocab.decode(r) for r in sample(model, 20, max_len)]
    ok = sum(len(s) == M for s in decoded)
    print(f"{ok}/20 samples have the right length ({M} bits)")
