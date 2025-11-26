"""
Memory-Optimized Training Script for Pythia-70M
Supports gradient accumulation and efficient memory management
"""
import sys
sys.path.append('./python')
import time
import numpy as np
import needle as ndl
import needle.nn as nn
from pythia_model import create_pythia_70m
import argparse
import os
import pickle
from collections import Counter
import gc


def load_dataset_huggingface(dataset_name, max_tokens=None, vocab_size=10000):
    """Load dataset with fixed vocabulary size"""
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: HuggingFace datasets not installed!")
        return load_synthetic_data(max_tokens or 100000, vocab_size)
    
    print(f"Loading {dataset_name}...")
    
    if dataset_name == "wikitext-2":
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
        train_text = " ".join(dataset["train"]["text"])
        val_text = " ".join(dataset["validation"]["text"])
    elif dataset_name == "tinystories":
        dataset = load_dataset("roneneldan/TinyStories")
        train_text = " ".join([print(item) for item in dataset["train"][:10000]])
        val_text = " ".join([item["text"] for item in dataset["validation"][:1000]])
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Tokenize
    train_tokens = train_text.lower().split()
    val_tokens = val_text.lower().split()
    
    # Build vocabulary with size limit
    token_counts = Counter(train_tokens)
    special_tokens = ["<pad>", "<unk>", "<sos>", "<eos>"]
    max_vocab_tokens = vocab_size - len(special_tokens)
    most_common = token_counts.most_common(max_vocab_tokens)
    
    vocab = {token: idx for idx, token in enumerate(special_tokens)}
    for token, _ in most_common:
        if token not in vocab:
            vocab[token] = len(vocab)
    
    print(f"Vocabulary size: {len(vocab)}")
    
    # Convert to indices
    unk_idx = vocab["<unk>"]
    train_data = np.array([vocab.get(token, unk_idx) for token in train_tokens])
    val_data = np.array([vocab.get(token, unk_idx) for token in val_tokens])
    
    if max_tokens:
        train_data = train_data[:max_tokens]
        val_data = val_data[:int(max_tokens * 0.1)]
    
    print(f"Train tokens: {len(train_data)}, Val tokens: {len(val_data)}")
    return train_data, val_data, len(vocab)


def load_synthetic_data(max_tokens=100000, vocab_size=10000):
    """Create synthetic data"""
    data = np.random.randint(0, vocab_size, size=max_tokens)
    split_idx = int(0.9 * len(data))
    return data[:split_idx], data[split_idx:], vocab_size


def batchify_streaming(data, batch_size, seq_len):
    """Create batches on-the-fly"""
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
    """Get a single batch"""
    inputs = batch_data[:, :-1]
    targets = batch_data[:, 1:]
    
    inputs_tensor = ndl.Tensor(inputs, device=device, dtype="float32")
    targets_tensor = ndl.Tensor(targets, device=device, dtype="float32")
    
    return inputs_tensor, targets_tensor


def train_epoch_with_accumulation(model, train_data, batch_size, seq_len, optimizer, 
                                   device, accumulation_steps=4, clip_grad=1.0):
    """
    Train with gradient accumulation for memory efficiency
    """
    model.train()
    total_loss = 0.0
    total_tokens = 0
    batch_count = 0
    accumulated_loss = 0.0
    
    print(f"  Using gradient accumulation: {accumulation_steps} steps")
    start_time = time.time()
    
    for batch_data in batchify_streaming(train_data, batch_size, seq_len):
        inputs, targets = get_batch(batch_data, device)
        
        # Forward pass
        logits, loss = model(inputs, targets)
        
        # Scale loss for accumulation
        loss_scaled = loss / accumulation_steps
        
        # Backward pass
        loss_scaled.backward()
        
        # Accumulate
        loss_val = loss.numpy()
        if isinstance(loss_val, np.ndarray):
            loss_val = loss_val.item()
        
        accumulated_loss += loss_val
        batch_count += 1
        
        # Update every accumulation_steps
        if batch_count % accumulation_steps == 0:
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
            
            # Update weights
            optimizer.step()
            optimizer.reset_grad()
            
            # Track stats
            batch_tokens = batch_size * seq_len * accumulation_steps
            total_loss += accumulated_loss * batch_tokens
            total_tokens += batch_tokens
            accumulated_loss = 0.0
            
            # Memory cleanup every 10 updates
            if (batch_count // accumulation_steps) % 10 == 0:
                gc.collect()
                elapsed = time.time() - start_time
                tokens_per_sec = total_tokens / elapsed if elapsed > 0 else 0
                avg_loss = total_loss / total_tokens if total_tokens > 0 else 0
                print(f"    Step {batch_count//accumulation_steps} | "
                      f"Loss: {avg_loss:.4f} | "
                      f"Tokens/sec: {tokens_per_sec:.0f}")
        
        # Clear intermediate tensors
        del inputs, targets, logits, loss
    
    # Final update if needed
    if batch_count % accumulation_steps != 0:
        optimizer.step()
        optimizer.reset_grad()
    
    elapsed = time.time() - start_time
    avg_loss = total_loss / total_tokens if total_tokens > 0 else 0
    tokens_per_sec = total_tokens / elapsed if elapsed > 0 else 0
    
    return avg_loss, tokens_per_sec


def evaluate(model, val_data, batch_size, seq_len, device, max_batches=50):
    """Evaluate model with perplexity"""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    batch_count = 0
    
    for batch_data in batchify_streaming(val_data, batch_size, seq_len):
        if batch_count >= max_batches:
            break
        
        inputs, targets = get_batch(batch_data, device)
        logits, loss = model(inputs, targets)
        
        loss_val = loss.numpy()
        if isinstance(loss_val, np.ndarray):
            loss_val = loss_val.item()
        
        batch_tokens = batch_size * seq_len
        total_loss += loss_val * batch_tokens
        total_tokens += batch_tokens
        batch_count += 1
        
        del inputs, targets, logits, loss
    
    avg_loss = total_loss / total_tokens if total_tokens > 0 else 0
    perplexity = np.exp(avg_loss)
    
    return avg_loss, perplexity


def save_checkpoint(model, optimizer, epoch, loss, filepath):
    """Save model checkpoint"""
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
        'optimizer_state': {'t': getattr(optimizer, 't', 0)},
    }
    
    for i, param in enumerate(model.parameters()):
        checkpoint['model_state'][f'param_{i}'] = param.numpy()
    
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
    
    print(f"  Checkpoint saved: {filepath}")


def train(model, train_data, val_data, config, n_epochs=10, batch_size=8, 
          seq_len=512, lr=3e-4, device=None, checkpoint_dir=None, 
          accumulation_steps=4):
    """Main training loop with gradient accumulation"""
    print("=" * 80)
    print("TRAINING CONFIGURATION")
    print("=" * 80)
    print(f"Model: Pythia-70M")
    print(f"Sparse attention: {config.use_sparse_attention}")
    print(f"Pattern: {config.sparse_pattern if config.use_sparse_attention else 'N/A'}")
    print(f"Epochs: {n_epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Sequence length: {seq_len}")
    print(f"Gradient accumulation: {accumulation_steps}")
    print(f"Effective batch size: {batch_size * accumulation_steps}")
    print(f"Learning rate: {lr}")
    print("=" * 80)
    
    # Optimizer
    optimizer = ndl.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0)
    
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    perplexities = []
    
    for epoch in range(n_epochs):
        print(f"\n{'='*80}")
        print(f"Epoch {epoch + 1}/{n_epochs}")
        print(f"{'='*80}")
        
        # Train
        train_loss, tokens_per_sec = train_epoch_with_accumulation(
            model, train_data, batch_size, seq_len, optimizer, device, 
            accumulation_steps
        )
        train_losses.append(train_loss)
        
        # Evaluate
        val_loss, val_ppl = evaluate(model, val_data, batch_size, seq_len, device)
        val_losses.append(val_loss)
        perplexities.append(val_ppl)
        
        print(f"\n{'='*80}")
        print(f"Epoch {epoch + 1} Summary:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  Perplexity: {val_ppl:.2f}")
        print(f"  Throughput: {tokens_per_sec:.0f} tokens/sec")
        print(f"{'='*80}")
        
        # Save checkpoint
        if checkpoint_dir and val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pkl')
            save_checkpoint(model, optimizer, epoch + 1, val_loss, checkpoint_path)
        
        gc.collect()
    
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'perplexities': perplexities,
        'best_val_loss': best_val_loss
    }


def main():
    parser = argparse.ArgumentParser()
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--seq_len', type=int, default=512)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--accumulation_steps', type=int, default=4)
    
    # Model parameters
    parser.add_argument('--sparse', action='store_true')
    parser.add_argument('--pattern', type=str, default='local', 
                       choices=['local', 'global', 'mixed'])
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'])
    
    # Dataset parameters
    parser.add_argument('--dataset', type=str, default='wikitext-2')
    parser.add_argument('--max_tokens', type=int, default=500000)
    parser.add_argument('--vocab_size', type=int, default=10000)
    
    # Checkpoint
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')
    
    args = parser.parse_args()
    
    # Device
    device = ndl.cuda() if args.device == 'cuda' else ndl.cpu()
    print(f"Using device: {device}")
    
    # Load data
    print(f"\nLoading {args.dataset}...")
    if args.dataset == 'synthetic':
        train_data, val_data, vocab_size = load_synthetic_data(
            args.max_tokens, args.vocab_size
        )
    else:
        train_data, val_data, vocab_size = load_dataset_huggingface(
            args.dataset, args.max_tokens, args.vocab_size
        )
    
    # Create model
    print("\nCreating model...")
    from pythia_model import PythiaConfig, PythiaLM
    
    config = PythiaConfig(
        vocab_size=vocab_size,
        d_model=512,
        num_heads=8,
        num_layers=6,
        d_ff=2048,
        max_seq_len=args.seq_len,
        dropout=0.1,
        device=device,
        dtype="float32",
        use_sparse_attention=args.sparse,
        sparse_block_size=64,
        sparse_pattern=args.pattern
    )
    
    model = PythiaLM(config)
    
    total_params = config.get_total_params()
    print(f"\nModel: {total_params / 1e6:.1f}M parameters")
    print(f"Sparse: {args.sparse}")
    if args.sparse:
        print(f"Pattern: {args.pattern}")
    
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
        device=device,
        checkpoint_dir=args.checkpoint_dir,
        accumulation_steps=args.accumulation_steps
    )
    
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE!")
    print(f"Best Val Loss: {results['best_val_loss']:.4f}")
    print(f"Best Perplexity: {min(results['perplexities']):.2f}")
    print("=" * 80)
    
    return results


if __name__ == "__main__":
    results = main()