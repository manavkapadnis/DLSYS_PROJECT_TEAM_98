"""
Dataset utilities for loading and preprocessing text datasets
"""
import numpy as np


class TextDataset:
    """
    Simple text dataset wrapper
    """
    def __init__(self, data, vocab):
        self.data = data
        self.vocab = vocab
        self.vocab_size = len(vocab)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


def build_vocab(text_tokens, max_vocab_size=50000):
    """
    Build vocabulary from text tokens
    
    Args:
        text_tokens: list of tokens
        max_vocab_size: maximum vocabulary size
        
    Returns:
        vocab: dictionary mapping tokens to indices
        idx_to_token: list mapping indices to tokens
    """
    # Count token frequencies
    token_counts = {}
    for token in text_tokens:
        token_counts[token] = token_counts.get(token, 0) + 1
    
    # Sort by frequency
    sorted_tokens = sorted(token_counts.items(), key=lambda x: x[1], reverse=True)
    
    # Build vocabulary
    vocab = {"<pad>": 0, "<unk>": 1, "<sos>": 2, "<eos>": 3}
    idx_to_token = ["<pad>", "<unk>", "<sos>", "<eos>"]
    
    for token, _ in sorted_tokens[:max_vocab_size - 4]:
        if token not in vocab:
            vocab[token] = len(vocab)
            idx_to_token.append(token)
    
    return vocab, idx_to_token


def tokenize_simple(text):
    """
    Simple word-level tokenization
    
    Args:
        text: input text string
        
    Returns:
        tokens: list of token strings
    """
    # Simple whitespace tokenization
    tokens = text.lower().split()
    return tokens


def encode_text(text, vocab):
    """
    Encode text to token indices
    
    Args:
        text: input text string
        vocab: vocabulary dictionary
        
    Returns:
        indices: numpy array of token indices
    """
    tokens = tokenize_simple(text)
    indices = [vocab.get(token, vocab["<unk>"]) for token in tokens]
    return np.array(indices)


def decode_text(indices, idx_to_token):
    """
    Decode token indices to text
    
    Args:
        indices: numpy array of token indices
        idx_to_token: list mapping indices to tokens
        
    Returns:
        text: decoded text string
    """
    tokens = [idx_to_token[idx] if idx < len(idx_to_token) else "<unk>" 
              for idx in indices]
    return " ".join(tokens)
