"""
Training script for Pythia-70M on WikiText-2
"""
import sys
sys.path.append('./python')
import time
import numpy as np
import needle as ndl
import needle.nn as nn
from pythia_model import create_pythia_70m, PythiaConfig
import argparse


def load_wikitext2_simple(max_tokens=100000):
    """
    Simple WikiText-2 loader (for demonstration)
    In practice, use the HuggingFace datasets library
    """
    # For demo purposes, create synthetic data
    # In real implementation, load from WikiText-2 dataset
    vocab_size = 10000
    data = np.random.randint(0, vocab_size, size=max_tokens)
    return data, vocab_size


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


def train(
    model,
    train_data,
    val_data,
    config,
    n_epochs=10,
    batch_size=32,
    seq_len=128,
    lr=3e-4,
    device=None
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
    print("=" * 80)
    
    # Prepare batches
    print("Preparing data...")
    train_batches = batchify(train_data, batch_size, seq_len, device)
    val_batches = batchify(val_data, batch_size, seq_len, device)
    print(f"Train batches: {len(train_batches)}")
    print(f"Val batches: {len(val_batches)}")
    
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
        
        # Save best model (in practice, save to disk)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f"  New best validation loss: {best_val_loss:.4f}")
    
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_val_loss': best_val_loss
    }


def main():
    parser = argparse.ArgumentParser(description='Train Pythia-70M')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--seq_len', type=int, default=128, help='Sequence length')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--sparse', action='store_true', help='Use sparse attention')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'], help='Device')
    parser.add_argument('--max_tokens', type=int, default=1000000, help='Max tokens to use')
    
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
    print("Loading WikiText-2 data...")
    data, vocab_size = load_wikitext2_simple(max_tokens=args.max_tokens)
    
    # Split into train/val
    split_idx = int(0.9 * len(data))
    train_data = data[:split_idx]
    val_data = data[split_idx:]
    
    # Create model
    print("Creating model...")
    model, config = create_pythia_70m(
        vocab_size=vocab_size,
        max_seq_len=args.seq_len,
        use_sparse_attention=args.sparse,
        device=device
    )
    
    # Train
    results = train(
        model=model,
        train_data=train_data,
        val_data=val_data,
        config=config,
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        lr=args.lr,
        device=device
    )
    
    print("\n" + "=" * 80)
    print("Training Complete!")
    print(f"Best Validation Loss: {results['best_val_loss']:.4f}")
    print("=" * 80)
    
    return results


if __name__ == "__main__":
    results = main()
