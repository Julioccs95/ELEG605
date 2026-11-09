"""
vocab.py - minimal encoder/decoder for PatternBoost.

Based on tokenizer.py and decoder.py from the original repo, merged into one file:
  - train_tokenizer(): learns BPE tokens from good constructions (tokenizer.py)
  - encode():          bitstring -> token ids                      (tokenizer.py)
  - decode():          token ids -> bitstring                      (decoder.py)

Removed compared with the originals: argparse, temp.txt, the "V" prefix and
comma-separated files. Everything stays in memory as Python lists.

Id 0 is reserved for the transformer's START/STOP token, so every BPE id is
shifted by +1 (the original did the same thing inside CharDataset).
"""

import os
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

#The path where the tokenizer will be read/loaded from
TOKENIZER_FILE = "tokenizer_data/tokenizer.json"

#Define the empty variable as global variable
_tokenizer = None


def train_tokenizer(lines: list[str], n_tokens: int = 100, max_lines: int = 100_000):
    """Train BPE on bitstrings (one per list item) and save it. Do this ONCE per run."""
    #This tells python to let _tokenizer to be changed inside the tokenizer trainer
    global _tokenizer
    #Initializes the tokenizer, then specifies to use BPE
    tok = Tokenizer(BPE())
    #Before the BPE runs, split the text on whitespace
    tok.pre_tokenizer = Whitespace()
    # initial_alphabet makes sure "0" and "1" always exist as tokens expecially as starting tokens
    #here we create the rules for bpe
    #vocab_size = max number of uniqe tokens it can learn
    trainer = BpeTrainer(vocab_size=n_tokens, initial_alphabet=["0", "1"])
    #Actually performs the traing based on the data
    tok.train_from_iterator(lines[:max_lines], trainer)

    #Creates the folder if it does not currently exists
    #exist_ok = True -> If the file exist, then do not raise an error
    os.makedirs(os.path.dirname(TOKENIZER_FILE), exist_ok=True)
    #Save the trained tokenizer to disk
    tok.save(TOKENIZER_FILE)
    #Store the tokenizer in the global variable so the current file program can use it immediately 
    #Without reloading the file
    _tokenizer = tok


def load_tokenizer():
    """Load the saved tokenizer (reuse it every round so ids never change)."""
    global _tokenizer
    if not os.path.exists(TOKENIZER_FILE):
        raise FileNotFoundError(f"No tokenizer at {TOKENIZER_FILE}. Train one first.")
    _tokenizer = Tokenizer.from_file(TOKENIZER_FILE)


def vocab_size() -> int:
    """Size the transformer needs: all BPE tokens + 1 for the START/STOP id 0."""
    return _tokenizer.get_vocab_size() + 1


def encode(s: str) -> list[int]:
    """Bitstring -> token ids (shifted +1 so 0 stays free)."""
    #Encode means to actually encode a string like "0111" into respective id's like [2, 3]
    return [i + 1 for i in _tokenizer.encode(s.strip()).ids]


def decode(ids: list[int]) -> str:
    """Token ids -> bitstring. Stops at the first 0 (STOP) and skips unknown ids."""
    out = []
    for i in ids:
        if i == 0:
            break
        out.append(i - 1)
    # decode() puts spaces between tokens, so remove them (as in decoder.py)
    #Example input ids = [3, 4, 0], out = [2, 3]
    #Then for our example, it may return full string "01 11" and why we just replace the string into no whitespace
    return _tokenizer.decode(out).replace(" ", "")


if __name__ == "__main__":
    # Round-trip test: decode(encode(s)) must give back s exactly
    import random
    M = 105  # N = 15
    data = ["".join(random.choice("01") for _ in range(M)) for _ in range(500)]
    train_tokenizer(data, n_tokens=100)
    load_tokenizer()
    for s in data:
        assert decode(encode(s)) == s
    print(f"Round trip OK on {len(data)} strings, vocab size {vocab_size()}")
    print("Example:", len(data[0]), "bits ->", len(encode(data[0])), "tokens")
