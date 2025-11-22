"""
Training script for Pythia-70M with HuggingFace Datasets Support
Supports WikiText-2 and TinyStories datasets
"""
import sys
sys.path.append('./python')
import time
import numpy as np
import needle as ndl
import needle.nn as nn
from pythia_model import create_pythia_70m, PythiaConfig
import argparse
import os
import pickle


def load_dataset_huggingface(dataset_name, max_tokens=None):
    """
    Load dataset from HuggingFace
    
    Args:
        dataset_name: "wikitext-2" or "tinystories"
        max_tokens: maximum number of tokens to use
        
    Returns:
        train_data, val_data, vocab_size
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: HuggingFace datasets library not installed!")
        print("Install with: pip install datasets")
        print("Falling back to synthetic data...")
        return load_synthetic_data(max_tokens or 100000)
    
    print(f"Loading {dataset_name} from HuggingFace...")
    
    if dataset_name == "wikitext-2":
        # Load WikiText-2
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
        train_text = " ".join(dataset["train"]["text"])
        val_text = " ".join(dataset["validation"]["text"])
        
    elif dataset_name == "tinystories":
        # Load TinyStories
        dataset = load_dataset("roneneldan/TinyStories")
        train_text = " ".join([item["text"] for item in dataset["train"][:10000]])  # Limit for memory
        val_text = " ".join([item["text"] for item in dataset["validation"][:1000]])
        
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Simple tokenization (word-level)
    print("Tokenizing...")
    train_tokens = train_text.lower().split()
    val_tokens = val_text.lower().split()
    
    # Build vocabulary
    vocab = {"<pad>": 0, "<unk>": 1, "<sos>": 2, "<eos>": 3}
    for token in train_tokens:
        if token not in vocab:
            vocab[token] = len(vocab)
    
    vocab_size = len(vocab)
    print(f"Vocabulary size: {vocab_size}")
    
    # Convert to indices
    train_data = np.array([vocab.get(token, vocab["<unk>"]) for token in train_tokens])
    val_data = np.array([vocab.get(token, vocab["<unk>"]) for token in val_tokens])
    
    # Limit tokens if specified
    if max_tokens:
        train_data = train_data[:max_tokens]
        val_data = val_data[:int(max_tokens * 0.1)]
    
    print(f"Train tokens: {len(train_data)}")
    print(f"Validation tokens: {len(val_data)}")
    
    return train_data, val_data, vocab_size


def load_synthetic_data(max_tokens=100000):
    """
    Fallback: Create synthetic data
    """
    print("Using synthetic data...")
    vocab_size = 10000
    data = np.random.randint(0, vocab_size, size=max_tokens)
    
    split_idx = int(0.9 * len(data))
    train_data = data[:split_idx]
    val_data = data[split_idx:]
    
    return train_data, val_data, vocab_size


def batchify(data, batch_size, seq_len, device):
    """
    Arrange data into batches
    
    Returns:
        batches: array of shape (n_batches, batch_size, seq_len)
    """
    # Calculate number of complete sequences
    n_sequences = len(data) // (seq_len + 1)  # +1 for targets
    n_batches = n_sequences // batch_size
    
    # Trim data
    total_len = n_batches * batch_size * (seq_len + 1)
    data = data[:total_len]
    
    # Reshape
    data = data.reshape((n_batches, batch_size, seq_len + 1))
    
    return data


def get_batch(batches, idx, device):
    """
    Get a single batch
    
    Returns:
        inputs: (batch_size, seq_len)
        targets: (batch_size, seq_len)
    """
    batch = batches[idx]
    inputs = batch[:, :-1]
    targets = batch[:, 1:]
    
    inputs_tensor = ndl.Tensor(inputs, device=device, dtype="float32")
    targets_tensor = ndl.Tensor(targets, device=device, dtype="float32")
    
    return inputs_tensor, targets_tensor


def train_epoch(model, batches, optimizer, device, clip_grad=1.0):
    """
    Train for one epoch
    
    Returns:
        avg_loss: average loss over epoch
        tokens_per_sec: throughput
    """
    model.train()
    total_loss = 0.0
    total_tokens = 0
    start_time = time.time()
    
    n_batches = len(batches)
    
    for i in range(n_batches):
        # Get batch
        inputs, targets = get_batch(batches, i, device)
        batch_size, seq_len = inputs.shape
        
        # Forward pass
        optimizer.reset_grad()
        logits, loss = model(inputs, targets)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping (optional)
        if clip_grad > 0:
            for param in model.parameters():
                if param.grad is not None:
                    grad_data = param.grad.numpy()
                    grad_norm = np.linalg.norm(grad_data)
                    if grad_norm > clip_grad:
                        param.grad = ndl.Tensor(
                            grad_data * (clip_grad / grad_norm),
                            device=param.grad.device
                        )
        
        # Update
        optimizer.step()
        
        # Accumulate stats
        loss_val = loss.numpy()
        if isinstance(loss_val, np.ndarray):
            loss_val = loss_val.item()
        total_loss += loss_val * batch_size * seq_len
        total_tokens += batch_size * seq_len
        
        # Print progress
        if (i + 1) % 10 == 0:
            elapsed = time.time() - start_time
            tokens_per_sec = total_tokens / elapsed
            print(f"  Batch {i+1}/{n_batches} | "
                  f"Loss: {loss_val:.4f} | "
                  f"Tokens/sec: {tokens_per_sec:.0f}")
    
    elapsed = time.time() - start_time
    avg_loss = total_loss / total_tokens
    tokens_per_sec = total_tokens / elapsed
    
    return avg_loss, tokens_per_sec


def evaluate(model, batches, device):
    """
    Evaluate model
    
    Returns:
        avg_loss: average loss
        perplexity: perplexity
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    
    n_batches = min(len(batches), 50)  # Evaluate on subset
    
    for i in range(n_batches):
        inputs, targets = get_batch(batches, i, device)
        batch_size, seq_len = inputs.shape
        
        # Forward pass (no gradients)
        logits, loss = model(inputs, targets)
        
        # Accumulate
        loss_val = loss.numpy()
        if isinstance(loss_val, np.ndarray):
            loss_val = loss_val.item()
        total_loss += loss_val * batch_size * seq_len
        total_tokens += batch_size * seq_len
    
    avg_loss = total_loss / total_tokens
    perplexity = np.exp(avg_loss)
    
    return avg_loss, perplexity


def save_checkpoint(model, optimizer, epoch, loss, filepath):
    """
    Save model checkpoint
    
    Args:
        model: PythiaLM model
        optimizer: optimizer
        epoch: current epoch
        loss: current loss
        filepath: path to save checkpoint
    """
    print(f"Saving checkpoint to {filepath}...")
    
    checkpoint = {
        'epoch': epoch,
        'loss': loss,
        'config': {
            'vocab_size': model.config.vocab_size,
            'd_model': model.config.d_model,
            'num_heads': model.config.num_heads,
            'num_layers': model.config.num_layers,
            'd_ff': model.config.d_ff,
            'max_seq_len': model.config.max_seq_len,
            'dropout': model.config.dropout,
            'use_sparse_attention': model.config.use_sparse_attention,
            'sparse_block_size': model.config.sparse_block_size,
            'sparse_pattern': model.config.sparse_pattern,
        },
        'model_state': {},
        'optimizer_state': {
            't': getattr(optimizer, 't', 0),
        }
    }
    
    # Save model parameters
    for i, param in enumerate(model.parameters()):
        checkpoint['model_state'][f'param_{i}'] = param.numpy()
    
    # Save optimizer state
    if hasattr(optimizer, 'm'):
        checkpoint['optimizer_state']['m'] = {
            i: optimizer.m[param].numpy() 
            for i, param in enumerate(model.parameters()) 
            if param in optimizer.m
        }
    if hasattr(optimizer, 'v'):
        checkpoint['optimizer_state']['v'] = {
            i: optimizer.v[param].numpy() 
            for i, param in enumerate(model.parameters()) 
            if param in optimizer.v
        }
    
    # Save to file
    with open(filepath, 'wb') as f:
        pickle.dump(checkpoint, f)
    
    print(f"Checkpoint saved successfully!")


def load_checkpoint(filepath, device):
    """
    Load model from checkpoint
    
    Args:
        filepath: path to checkpoint
        device: device to load model on
        
    Returns:
        model, optimizer, epoch, loss
    """
    print(f"Loading checkpoint from {filepath}...")
    
    with open(filepath, 'rb') as f:
        checkpoint = pickle.load(f)
    
    # Recreate model
    config = PythiaConfig(**checkpoint['config'], device=device)
    from pythia_model import PythiaLM
    model = PythiaLM(config)
    
    # Load parameters
    model_params = list(model.parameters())
    for i, param in enumerate(model_params):
        if f'param_{i}' in checkpoint['model_state']:
            param.data = ndl.Tensor(
                checkpoint['model_state'][f'param_{i}'],
                device=device
            ).data
    
    # Recreate optimizer
    optimizer = ndl.optim.Adam(model.parameters(), lr=3e-4, weight_decay=0.01)
    
    # Load optimizer state
    if 't' in checkpoint['optimizer_state']:
        optimizer.t = checkpoint['optimizer_state']['t']
    
    if 'm' in checkpoint['optimizer_state']:
        for i, param in enumerate(model_params):
            if i in checkpoint['optimizer_state']['m']:
                optimizer.m[param] = ndl.Tensor(
                    checkpoint['optimizer_state']['m'][i],
                    device=device
                ).data
    
    if 'v' in checkpoint['optimizer_state']:
        for i, param in enumerate(model_params):
            if i in checkpoint['optimizer_state']['v']:
                optimizer.v[param] = ndl.Tensor(
                    checkpoint['optimizer_state']['v'][i],
                    device=device
                ).data
    
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    
    print(f"Checkpoint loaded: epoch {epoch}, loss {loss:.4f}")
    
    return model, optimizer, epoch, loss


def train(
    model,
    train_data,
    val_data,
    config,
    n_epochs=10,
    batch_size=32,
    seq_len=128,
    lr=3e-4,
    device=None,
    checkpoint_dir=None,
    eval_only=False
):
    """
    Main training loop
    """
    print("=" * 80)
    print("Training Configuration")
    print("=" * 80)
    print(f"Model: Pythia-70M")
    print(f"Sparse attention: {config.use_sparse_attention}")
    print(f"Epochs: {n_epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Sequence length: {seq_len}")
    print(f"Learning rate: {lr}")
    print(f"Device: {device}")
    print(f"Eval only: {eval_only}")
    print("=" * 80)
    
    # Prepare batches
    print("Preparing data...")
    train_batches = batchify(train_data, batch_size, seq_len, device)
    val_batches = batchify(val_data, batch_size, seq_len, device)
    print(f"Train batches: {len(train_batches)}")
    print(f"Val batches: {len(val_batches)}")
    
    if eval_only:
        # Evaluation only
        print("\nRunning evaluation...")
        val_loss, val_ppl = evaluate(model, val_batches, device)
        print(f"Validation Loss: {val_loss:.4f}")
        print(f"Validation Perplexity: {val_ppl:.2f}")
        return {'val_loss': val_loss, 'val_ppl': val_ppl}
    
    # Optimizer
    optimizer = ndl.optim.Adam(model.parameters(), lr=lr, weight_decay=0.01)
    
    # Training loop
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(n_epochs):
        print(f"\n{'='*80}")
        print(f"Epoch {epoch + 1}/{n_epochs}")
        print(f"{'='*80}")
        
        # Train
        train_loss, tokens_per_sec = train_epoch(model, train_batches, optimizer, device)
        train_losses.append(train_loss)
        
        # Evaluate
        val_loss, val_ppl = evaluate(model, val_batches, device)
        val_losses.append(val_loss)
        
        print(f"\n{'='*80}")
        print(f"Epoch {epoch + 1} Summary:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  Val Perplexity: {val_ppl:.2f}")
        print(f"  Throughput: {tokens_per_sec:.0f} tokens/sec")
        print(f"{'='*80}")
        
        # Save checkpoint
        if checkpoint_dir and val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pkl')
            save_checkpoint(model, optimizer, epoch + 1, val_loss, checkpoint_path)
            print(f"  New best validation loss: {best_val_loss:.4f}")
    
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_val_loss': best_val_loss
    }


def main():
    parser = argparse.ArgumentParser(description='Train Pythia-70M with HuggingFace Datasets')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--seq_len', type=int, default=128, help='Sequence length')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    
    # Model parameters
    parser.add_argument('--sparse', action='store_true', help='Use sparse attention')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'], help='Device')
    
    # Dataset parameters
    parser.add_argument('--dataset', type=str, default='wikitext-2', 
                       choices=['wikitext-2', 'tinystories', 'synthetic'],
                       help='Dataset to use')
    parser.add_argument('--max_tokens', type=int, default=1000000, help='Max tokens to use')
    
    # Checkpoint parameters
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', 
                       help='Directory to save checkpoints')
    parser.add_argument('--load_checkpoint', type=str, default=None, 
                       help='Path to checkpoint to load')
    parser.add_argument('--eval_only', action='store_true', 
                       help='Only run evaluation (requires --load_checkpoint)')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'cuda':
        try:
            device = ndl.cuda()
        except:
            print("CUDA not available, falling back to CPU")
            device = ndl.cpu()
    else:
        device = ndl.cpu()
    
    # Load data
    print(f"Loading {args.dataset} dataset...")
    if args.dataset == 'synthetic':
        train_data, val_data, vocab_size = load_synthetic_data(args.max_tokens)
    else:
        train_data, val_data, vocab_size = load_dataset_huggingface(args.dataset, args.max_tokens)
    
    # Load checkpoint or create new model
    if args.load_checkpoint:
        model, optimizer, start_epoch, _ = load_checkpoint(args.load_checkpoint, device)
        config = model.config
    else:
        # Create model
        print("Creating model...")
        model, config = create_pythia_70m(
            vocab_size=vocab_size,
            max_seq_len=args.seq_len,
            use_sparse_attention=args.sparse,
            device=device
        )
    
    # Train or evaluate
    results = train(
        model=model,
        train_data=train_data,
        val_data=val_data,
        config=config,
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        lr=args.lr,
        device=device,
        checkpoint_dir=args.checkpoint_dir if not args.eval_only else None,
        eval_only=args.eval_only
    )
    
    print("\n" + "=" * 80)
    print("Training Complete!" if not args.eval_only else "Evaluation Complete!")
    if not args.eval_only:
        print(f"Best Validation Loss: {results['best_val_loss']:.4f}")
    print("=" * 80)
    
    return results


if __name__ == "__main__":
    results = main()