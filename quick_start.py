"""
Quick Start Example - Pythia-70M with Sparse Attention

This script demonstrates the basic usage of Pythia-70M model
with both dense and sparse attention.
"""
import sys
sys.path.append('./python')
import numpy as np
import needle as ndl
from pythia_model import create_pythia_70m


def demo_basic_usage():
    """Demonstrate basic model creation and forward pass"""
    print("="*80)
    print("PYTHIA-70M QUICK START DEMO")
    print("="*80)
    
    # Setup
    device = ndl.cpu()
    vocab_size = 10000
    max_seq_len = 128
    batch_size = 4
    seq_len = 64
    
    print("\n1. Creating Models...")
    print("-"*80)
    
    # Dense model
    print("Creating dense attention model...")
    model_dense, config_dense = create_pythia_70m(
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        use_sparse_attention=False,
        device=device
    )
    
    # Sparse model
    print("Creating sparse attention model...")
    model_sparse, config_sparse = create_pythia_70m(
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        use_sparse_attention=True,
        device=device
    )
    
    print("\n2. Model Configuration")
    print("-"*80)
    print(f"Architecture: Pythia-70M")
    print(f"  Parameters: ~{config_dense.get_total_params() / 1e6:.1f}M")
    print(f"  Layers: {config_dense.num_layers}")
    print(f"  Hidden dim: {config_dense.d_model}")
    print(f"  Attention heads: {config_dense.num_heads}")
    print(f"  FFN dim: {config_dense.d_ff}")
    
    print("\n3. Forward Pass Test")
    print("-"*80)
    
    # Create sample input
    input_ids = ndl.Tensor(
        np.random.randint(0, vocab_size, (batch_size, seq_len)),
        device=device
    )
    print(f"Input shape: {input_ids.shape}")
    
    # Forward pass - Dense
    print("\nDense attention forward pass...")
    logits_dense, loss_dense = model_dense(input_ids)
    print(f"  Output shape: {logits_dense.shape}")
    print(f"  Expected: ({batch_size}, {seq_len}, {vocab_size})")
    
    # Forward pass - Sparse
    print("\nSparse attention forward pass...")
    logits_sparse, loss_sparse = model_sparse(input_ids)
    print(f"  Output shape: {logits_sparse.shape}")
    print(f"  Expected: ({batch_size}, {seq_len}, {vocab_size})")
    
    print("\n4. Output Statistics")
    print("-"*80)
    logits_dense_np = logits_dense.numpy()
    logits_sparse_np = logits_sparse.numpy()
    
    print(f"Dense output:")
    print(f"  Mean: {logits_dense_np.mean():.4f}")
    print(f"  Std: {logits_dense_np.std():.4f}")
    print(f"  Min: {logits_dense_np.min():.4f}")
    print(f"  Max: {logits_dense_np.max():.4f}")
    
    print(f"\nSparse output:")
    print(f"  Mean: {logits_sparse_np.mean():.4f}")
    print(f"  Std: {logits_sparse_np.std():.4f}")
    print(f"  Min: {logits_sparse_np.min():.4f}")
    print(f"  Max: {logits_sparse_np.max():.4f}")
    
    print("\n5. Generation Test")
    print("-"*80)
    
    # Simple generation test
    prompt = ndl.Tensor(
        np.array([[1, 2, 3, 4, 5]]),
        device=device
    )
    
    print("Generating 10 new tokens...")
    print(f"Prompt: {prompt.numpy()[0]}")
    
    model_dense.eval()
    generated = model_dense.generate(prompt, max_new_tokens=10, temperature=1.0)
    print(f"Generated (dense): {generated.numpy()[0]}")
    
    model_sparse.eval()
    generated = model_sparse.generate(prompt, max_new_tokens=10, temperature=1.0)
    print(f"Generated (sparse): {generated.numpy()[0]}")
    
    print("\n" + "="*80)
    print("✓ DEMO COMPLETE")
    print("="*80)
    print("\nNext steps:")
    print("1. Run training: python apps/train_pythia.py --epochs 5")
    print("2. Run benchmarks: python apps/benchmark.py")
    print("3. Explore notebook: jupyter notebook demo_notebook.ipynb")


def demo_sparse_patterns():
    """Demonstrate sparse attention patterns"""
    print("\n" + "="*80)
    print("SPARSE ATTENTION PATTERNS DEMO")
    print("="*80)
    
    from nn_sparse_attention import BlockSparsePattern
    
    seq_len = 256
    block_size = 64
    
    print(f"\nSequence length: {seq_len}")
    print(f"Block size: {block_size}")
    print(f"Number of blocks: {seq_len // block_size}")
    
    # Generate patterns
    print("\n1. Local Pattern (Sliding Window)")
    print("-"*80)
    local = BlockSparsePattern.local_pattern(seq_len, block_size, window_size=1)
    sparsity_local = (1 - local.sum() / local.size) * 100
    print(f"  Sparsity: {sparsity_local:.1f}%")
    print(f"  Attention blocks: {local.sum()} / {local.size}")
    
    print("\n2. Global Pattern (Strided)")
    print("-"*80)
    global_pattern = BlockSparsePattern.global_pattern(seq_len, block_size, stride=2)
    sparsity_global = (1 - global_pattern.sum() / global_pattern.size) * 100
    print(f"  Sparsity: {sparsity_global:.1f}%")
    print(f"  Attention blocks: {global_pattern.sum()} / {global_pattern.size}")
    
    print("\n3. Mixed Pattern (Local + Global)")
    print("-"*80)
    mixed = BlockSparsePattern.mixed_pattern(seq_len, block_size, window_size=1, stride=4)
    sparsity_mixed = (1 - mixed.sum() / mixed.size) * 100
    print(f"  Sparsity: {sparsity_mixed:.1f}%")
    print(f"  Attention blocks: {mixed.sum()} / {mixed.size}")
    
    print("\n4. Computational Savings")
    print("-"*80)
    full_attention_ops = seq_len * seq_len
    local_ops = local.sum() * block_size * block_size
    global_ops = global_pattern.sum() * block_size * block_size
    mixed_ops = mixed.sum() * block_size * block_size
    
    print(f"Dense attention operations: {full_attention_ops:,}")
    print(f"Local sparse operations: {local_ops:,} ({local_ops/full_attention_ops*100:.1f}%)")
    print(f"Global sparse operations: {global_ops:,} ({global_ops/full_attention_ops*100:.1f}%)")
    print(f"Mixed sparse operations: {mixed_ops:,} ({mixed_ops/full_attention_ops*100:.1f}%)")
    
    print(f"\nTheoretical speedup:")
    print(f"  Local: {full_attention_ops/local_ops:.2f}x")
    print(f"  Global: {full_attention_ops/global_ops:.2f}x")
    print(f"  Mixed: {full_attention_ops/mixed_ops:.2f}x")


def demo_complexity_comparison():
    """Compare complexity of dense vs sparse attention"""
    print("\n" + "="*80)
    print("COMPLEXITY COMPARISON")
    print("="*80)
    
    d_model = 512
    n_heads = 8
    n_layers = 6
    block_size = 64
    sparsity = 0.75
    
    seq_lengths = [128, 256, 512, 1024]
    
    print(f"\nModel: Pythia-70M")
    print(f"  Hidden dim: {d_model}")
    print(f"  Heads: {n_heads}")
    print(f"  Layers: {n_layers}")
    print(f"  Sparsity: {sparsity*100:.0f}%")
    
    print(f"\n{'Seq Len':<12} {'Dense FLOPs':<15} {'Sparse FLOPs':<15} {'Speedup':<10}")
    print("-"*60)
    
    for seq_len in seq_lengths:
        # Dense: O(n²d)
        dense_flops = n_layers * seq_len * seq_len * d_model
        
        # Sparse: O(n * block * d)
        sparse_flops = n_layers * seq_len * block_size * d_model * (1 - sparsity)
        
        speedup = dense_flops / sparse_flops
        
        print(f"{seq_len:<12} {dense_flops/1e9:<15.2f} {sparse_flops/1e9:<15.2f} {speedup:<10.2f}x")


if __name__ == "__main__":
    print("\n")
    print("#" * 80)
    print("#" + " " * 78 + "#")
    print("#" + " " * 20 + "PYTHIA-70M QUICK START" + " " * 37 + "#")
    print("#" + " " * 78 + "#")
    print("#" * 80)
    
    # Run demos
    demo_basic_usage()
    demo_sparse_patterns()
    demo_complexity_comparison()
    
    print("\n" + "="*80)
    print("ALL DEMOS COMPLETE!")
    print("="*80)
    print("\nFor more details, see:")
    print("  - README.md for full documentation")
    print("  - demo_notebook.ipynb for interactive exploration")
    print("  - train_pythia.py for training examples")
    print("  - benchmark.py for comprehensive benchmarks")
    print("="*80)
