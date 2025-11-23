"""
Analysis: Why Sparse Attention Has No Speedup in Current Implementation
"""
import sys
sys.path.append('./python')
import time
import numpy as np
import needle as ndl
from needle.nn.nn_sparse_attention import BlockSparseMultiHeadAttention
from needle.nn.nn_transformer import MultiHeadAttention


def analyze_sparse_vs_dense():
    """Compare operations in sparse vs dense attention"""
    print("="*80)
    print("SPARSE ATTENTION PERFORMANCE ANALYSIS")
    print("="*80)
    
    # Setup
    device = ndl.cpu()  # or ndl.cuda()
    batch_size = 4
    num_heads = 8
    seq_len = 256
    dim_head = 64
    
    # Create random inputs
    shape = (batch_size, num_heads, seq_len, dim_head)
    q = ndl.Tensor(np.random.randn(*shape), device=device)
    k = ndl.Tensor(np.random.randn(*shape), device=device)
    v = ndl.Tensor(np.random.randn(*shape), device=device)
    
    # Dense attention
    print("\n1. Dense Attention Operations:")
    print("-"*80)
    dense_attn = MultiHeadAttention(device=device)
    
    start = time.time()
    # Key operations in dense attention:
    scores = q @ k.transpose((2, 3))  # O(n²d) operations
    print(f"  QK^T matmul: {scores.shape} = {seq_len}² × {dim_head} = {seq_len*seq_len*dim_head:,} ops")
    
    # Actual computation
    dense_out, _ = dense_attn(q, k, v)
    dense_time = time.time() - start
    print(f"  Time: {dense_time*1000:.2f} ms")
    
    # Sparse attention
    print("\n2. Sparse Attention Operations:")
    print("-"*80)
    sparse_attn = BlockSparseMultiHeadAttention(
        device=device,
        block_size=64,
        sparse_pattern="local"
    )
    
    start = time.time()
    
    # Problem 1: Mask creation on CPU
    mask = sparse_attn.create_block_mask(seq_len, device)
    mask_time = time.time() - start
    print(f"  Mask creation (CPU): {mask_time*1000:.2f} ms")
    
    # Problem 2: Still doing full matmul!
    scores = sparse_attn.matmul(q, k)  # This is still O(n²d)!
    print(f"  QK^T matmul: SAME as dense! {seq_len}² × {dim_head} = {seq_len*seq_len*dim_head:,} ops")
    
    # Problem 3: Masking is just element-wise multiply
    mask_tensor = ndl.Tensor(mask, device=device)
    masked_scores = scores + mask_tensor  # Still processes all elements
    print(f"  Masking: Element-wise add on {seq_len}² elements")
    
    # Actual computation
    start = time.time()
    sparse_out, _ = sparse_attn(q, k, v)
    sparse_time = time.time() - start
    print(f"  Time: {sparse_time*1000:.2f} ms")
    
    print("\n3. Performance Comparison:")
    print("-"*80)
    print(f"Dense time: {dense_time*1000:.2f} ms")
    print(f"Sparse time: {sparse_time*1000:.2f} ms") 
    print(f"Speedup: {dense_time/sparse_time:.2f}x")
    
    print("\n4. Why No Speedup?")
    print("-"*80)
    print("❌ Sparse attention still computes FULL attention matrix")
    print("❌ Mask is applied AFTER computation (no savings)")
    print("❌ Mask creation happens on CPU (overhead)")
    print("❌ No specialized sparse matrix operations")
    
    print("\n5. What Real Sparse Attention Needs:")
    print("-"*80)
    print("✓ Skip computation for masked blocks entirely")
    print("✓ Sparse matrix storage format")
    print("✓ Custom CUDA kernels for sparse ops")
    print("✓ Block-level parallelization")
    
    # Show theoretical savings
    print("\n6. Theoretical Savings with 75% Sparsity:")
    print("-"*80)
    full_ops = seq_len * seq_len * dim_head * batch_size * num_heads
    sparse_ops = full_ops * 0.25  # Only 25% of blocks computed
    print(f"Dense operations: {full_ops/1e9:.2f} GFLOPs")
    print(f"Sparse operations (ideal): {sparse_ops/1e9:.2f} GFLOPs")
    print(f"Theoretical speedup: {full_ops/sparse_ops:.1f}x")
    print(f"Actual speedup: ~1.0x (no optimization)")


def profile_memory_usage():
    """Profile memory usage of sparse attention"""
    print("\n\n" + "="*80)
    print("MEMORY USAGE ANALYSIS")
    print("="*80)
    
    batch_size = 8
    seq_len = 128
    d_model = 512
    num_heads = 8
    
    print(f"\nConfiguration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Model dimension: {d_model}")
    print(f"  Heads: {num_heads}")
    
    # Attention memory
    print("\nAttention Memory (per layer):")
    print("-"*80)
    
    # Dense attention
    qkv_memory = 3 * batch_size * seq_len * d_model * 4 / 1e6  # MB
    scores_memory = batch_size * num_heads * seq_len * seq_len * 4 / 1e6
    total_dense = qkv_memory + scores_memory
    
    print(f"Dense attention:")
    print(f"  Q,K,V tensors: {qkv_memory:.2f} MB")
    print(f"  Attention scores: {scores_memory:.2f} MB")
    print(f"  Total: {total_dense:.2f} MB")
    
    # Current "sparse" attention (not really sparse!)
    mask_memory = seq_len * seq_len * 4 / 1e6
    total_sparse_current = total_dense + mask_memory
    
    print(f"\nCurrent 'sparse' attention:")
    print(f"  Same as dense PLUS mask: {mask_memory:.2f} MB")
    print(f"  Total: {total_sparse_current:.2f} MB")
    print(f"  ❌ Actually uses MORE memory!")
    
    # True sparse attention
    sparsity = 0.75
    sparse_scores = scores_memory * (1 - sparsity)
    total_sparse_ideal = qkv_memory + sparse_scores
    
    print(f"\nIdeal sparse attention (75% sparse):")
    print(f"  Q,K,V tensors: {qkv_memory:.2f} MB")
    print(f"  Sparse attention scores: {sparse_scores:.2f} MB")
    print(f"  Total: {total_sparse_ideal:.2f} MB")
    print(f"  ✓ Memory savings: {(1 - total_sparse_ideal/total_dense)*100:.1f}%")


if __name__ == "__main__":
    print("\n🔍 ANALYZING SPARSE ATTENTION IMPLEMENTATION...\n")
    
    # Run analysis
    analyze_sparse_vs_dense()
    profile_memory_usage()
    
    print("\n\n" + "="*80)
    print("CONCLUSION")
    print("="*80)
    print("\nThe current 'sparse' attention implementation:")
    print("1. Still computes the FULL attention matrix")
    print("2. Only masks values AFTER computation") 
    print("3. Actually uses MORE memory than dense")
    print("4. Has CPU overhead for mask creation")
    print("\nTo get real speedup, you need:")
    print("- Custom CUDA kernels that skip masked blocks")
    print("- Sparse matrix storage formats")
    print("- Block-level computation (not element-wise)")
    print("="*80)
