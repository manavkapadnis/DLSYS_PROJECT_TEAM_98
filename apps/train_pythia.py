"""
Training script for Pythia-70M with Memory Optimizations
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
from collections import Counter
import gc


def load_dataset_huggingface(dataset_name, max_tokens=None, vocab_size=10000):
    """
    Load dataset from HuggingFace with FIXED vocabulary size
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: HuggingFace datasets library not installed!")
        print("Install with: pip install datasets")
        print("Falling back to synthetic data...")
        return load_synthetic_data(max_tokens or 100000, vocab_size)
    
    print(f"Loading {dataset_name} from HuggingFace...")
    
    if dataset_name == "wikitext-2":
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
        train_text = " ".join(dataset["train"]["text"])
        val_text = " ".join(dataset["validation"]["text"])
        
    elif dataset_name == "tinystories":
        dataset = load_dataset("roneneldan/TinyStories")
        train_text = " ".join([item["text"] for item in dataset["train"][:10000]])
        val_text = " ".join([item["text"] for item in dataset["validation"][:1000]])
        
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    print("Tokenizing...")
    train_tokens = train_text.lower().split()
    val_tokens = val_text.lower().split()
    
    # Build vocabulary with size limit
    print(f"Building vocabulary (max size: {vocab_size})...")
    
    # Count token frequencies
    token_counts = Counter(train_tokens)
    
    # Reserve space for special tokens
    special_tokens = ["<pad>", "<unk>", "<sos>", "<eos>"]
    max_vocab_tokens = vocab_size - len(special_tokens)
    
    # Get most frequent tokens
    most_common = token_counts.most_common(max_vocab_tokens)
    
    # Build vocabulary
    vocab = {token: idx for idx, token in enumerate(special_tokens)}
    for token, _ in most_common:
        if token not in vocab:
            vocab[token] = len(vocab)
    
    actual_vocab_size = len(vocab)
    print(f"Actual vocabulary size: {actual_vocab_size}")
    print(f"Coverage: {sum(count for token, count in most_common) / len(train_tokens) * 100:.2f}%")
    
    # Convert to indices with UNK for out-of-vocab tokens
    unk_idx = vocab["<unk>"]
    train_data = np.array([vocab.get(token, unk_idx) for token in train_tokens])
    val_data = np.array([vocab.get(token, unk_idx) for token in val_tokens])
    
    # Limit tokens if specified
    if max_tokens:
        train_data = train_data[:max_tokens]
        val_data = val_data[:int(max_tokens * 0.1)]
    
    print(f"Train tokens: {len(train_data)}")
    print(f"Validation tokens: {len(val_data)}")
    
    return train_data, val_data, actual_vocab_size


def load_synthetic_data(max_tokens=100000, vocab_size=10000):
    """Create synthetic data"""
    print("Using synthetic data...")
    data = np.random.randint(0, vocab_size, size=max_tokens)
    
    split_idx = int(0.9 * len(data))
    train_data = data[:split_idx]
    val_data = data[split_idx:]
    
    return train_data, val_data, vocab_size


def batchify_streaming(data, batch_size, seq_len):
    """Create batches on-the-fly to save memory"""
    n_sequences = len(data) // (seq_len + 1)
    n_batches = n_sequences // batch_size
    
    for batch_idx in range(n_batches):
        batch_data = []
        for seq_idx in range(batch_size):
            idx = (batch_idx * batch_size + seq_idx) * (seq_len + 1)
            if idx + seq_len + 1 <= len(data):
                batch_data.append(data[idx:idx + seq_len + 1])
        
        if len(batch_data) == batch_size:
            yield np.array(batch_data)


def get_batch(batch_data, device):
    """Get a single batch from pre-fetched data"""
    inputs = batch_data[:, :-1]
    targets = batch_data[:, 1:]
    
    inputs_tensor = ndl.Tensor(inputs, device=device, dtype="float32")
    targets_tensor = ndl.Tensor(targets, device=device, dtype="float32")
    
    return inputs_tensor, targets_tensor


def train_epoch(model, train_data, batch_size, seq_len, optimizer, device, clip_grad=1.0):
    """
    Train for one epoch with memory optimization
    """
    model.train()
    total_loss = 0.0
    total_tokens = 0
    start_time = time.time()
    batch_count = 0
    
    for batch_data in batchify_streaming(train_data, batch_size, seq_len):
        # Get batch
        inputs, targets = get_batch(batch_data, device)
        
        # Forward pass
        optimizer.reset_grad()
        logits, loss = model(inputs, targets)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
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
        
        batch_tokens = batch_size * seq_len
        total_loss += loss_val * batch_tokens
        total_tokens += batch_tokens
        batch_count += 1
        
        # MEMORY OPTIMIZATION: Clear computation graph periodically
        if batch_count % 10 == 0:
            # Force garbage collection
            gc.collect()
            
            elapsed = time.time() - start_time
            tokens_per_sec = total_tokens / elapsed
            print(f"  Batch {batch_count} | "
                  f"Loss: {loss_val:.4f} | "
                  f"Tokens/sec: {tokens_per_sec:.0f}")
        
        # Clear intermediate tensors
        del inputs, targets, logits, loss
    
    elapsed = time.time() - start_time
    avg_loss = total_loss / total_tokens if total_tokens > 0 else 0
    tokens_per_sec = total_tokens / elapsed if elapsed > 0 else 0
    
    return avg_loss, tokens_per_sec


def evaluate(model, val_data, batch_size, seq_len, device, max_batches=50):
    """Evaluate model"""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    batch_count = 0
    
    for batch_data in batchify_streaming(val_data, batch_size, seq_len):
        if batch_count >= max_batches:
            break
        
        inputs, targets = get_batch(batch_data, device)
        
        # Forward pass (no gradients)
        logits, loss = model(inputs, targets)
        
        # Accumulate
        loss_val = loss.numpy()
        if isinstance(loss_val, np.ndarray):
            loss_val = loss_val.item()
        
        batch_tokens = batch_size * seq_len
        total_loss += loss_val * batch_tokens
        total_tokens += batch_tokens
        batch_count += 1
        
        # Clean up
        del inputs, targets, logits, loss
    
    avg_loss = total_loss / total_tokens if total_tokens > 0 else 0
    perplexity = np.exp(avg_loss)
    
    return avg_loss, perplexity


def save_checkpoint(model, optimizer, epoch, loss, filepath):
    """Save model checkpoint"""
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
    
    with open(filepath, 'wb') as f:
        pickle.dump(checkpoint, f)
    
    print(f"Checkpoint saved successfully!")


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
    """Main training loop with memory optimizations"""
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
    
    if eval_only:
        print("\nRunning evaluation...")
        val_loss, val_ppl = evaluate(model, val_data, batch_size, seq_len, device)
        print(f"Validation Loss: {val_loss:.4f}")
        print(f"Validation Perplexity: {val_ppl:.2f}")
        return {'val_loss': val_loss, 'val_ppl': val_ppl}
    
    # Optimizer with lower weight decay for memory efficiency
    optimizer = ndl.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0)
    
    # Training loop
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(n_epochs):
        print(f"\n{'='*80}")
        print(f"Epoch {epoch + 1}/{n_epochs}")
        print(f"{'='*80}")
        
        # Train
        train_loss, tokens_per_sec = train_epoch(
            model, train_data, batch_size, seq_len, optimizer, device
        )
        train_losses.append(train_loss)
        
        # Evaluate
        val_loss, val_ppl = evaluate(model, val_data, batch_size, seq_len, device)
        val_losses.append(val_loss)
        
        print(f"\n{'='*80}")
        print(f"Epoch {epoch + 1} Summary:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  Val Perplexity: {val_ppl:.2f}")
        print(f"  Throughput: {tokens_per_sec:.0f} tokens/sec")
        
        # Memory usage info
        if hasattr(device, '__repr__') and 'cuda' in str(device):
            try:
                import subprocess
                result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,nounits,noheader'], 
                                      capture_output=True, text=True)
                memory_used = result.stdout.strip()
                print(f"  GPU Memory: {memory_used} MB")
            except:
                pass
        
        print(f"{'='*80}")
        
        # Save checkpoint
        if checkpoint_dir and val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pkl')
            save_checkpoint(model, optimizer, epoch + 1, val_loss, checkpoint_path)
            print(f"  New best validation loss: {best_val_loss:.4f}")
        
        # Force garbage collection between epochs
        gc.collect()
    
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_val_loss': best_val_loss
    }


def main():
    parser = argparse.ArgumentParser(description='Train Pythia-70M with Memory Optimizations')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--seq_len', type=int, default=128)
    parser.add_argument('--lr', type=float, default=3e-4)
    
    # Model parameters
    parser.add_argument('--sparse', action='store_true')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'])
    
    # Dataset parameters
    parser.add_argument('--dataset', type=str, default='wikitext-2', 
                       choices=['wikitext-2', 'tinystories', 'synthetic'])
    parser.add_argument('--max_tokens', type=int, default=1000000)
    parser.add_argument('--vocab_size', type=int, default=10000)
    
    # Checkpoint parameters
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')
    parser.add_argument('--load_checkpoint', type=str, default=None)
    parser.add_argument('--eval_only', action='store_true')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'cuda':
        try:
            device = ndl.cuda()
            print("Using CUDA")
        except:
            print("CUDA not available, falling back to CPU")
            device = ndl.cpu()
    else:
        device = ndl.cpu()
    
    # Load data
    print(f"Loading {args.dataset} dataset...")
    if args.dataset == 'synthetic':
        train_data, val_data, vocab_size = load_synthetic_data(
            args.max_tokens, args.vocab_size
        )
    else:
        train_data, val_data, vocab_size = load_dataset_huggingface(
            args.dataset, args.max_tokens, args.vocab_size
        )
    
    # Create model
    print("Creating model...")
    model, config = create_pythia_70m(
        vocab_size=vocab_size,
        max_seq_len=args.seq_len,
        use_sparse_attention=args.sparse,
        device=device
    )
    
    # Print memory estimate
    total_params = config.get_total_params()
    memory_est = total_params * 4 / 1e9  # GB for fp32
    print(f"\nModel Parameters: {total_params / 1e6:.1f}M")
    print(f"Model Memory (fp32): {memory_est:.2f} GB")
    print(f"Estimated Training Memory: {memory_est * 3:.2f} GB (model + gradients + optimizer)")
    
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