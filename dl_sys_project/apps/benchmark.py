"""
Benchmark script to compare Dense vs Sparse attention performance
"""
import sys
sys.path.append('./python')
import time
import numpy as np
import matplotlib.pyplot as plt
import needle as ndl
from pythia_model import create_pythia_70m


def benchmark_forward_pass(model, batch_size, seq_len, vocab_size, device, n_runs=10):
    """
    Benchmark forward pass performance
    
    Returns:
        times: list of execution times
        memory: estimated memory usage
    """
    # Create input
    input_ids = ndl.Tensor(
        np.random.randint(0, vocab_size, (batch_size, seq_len)),
        device=device
    )
    
    model.eval()
    times = []
    
    # Warmup
    for _ in range(2):
        _, _ = model(input_ids)
    
    # Benchmark
    for _ in range(n_runs):
        start = time.time()
        logits, _ = model(input_ids)
        elapsed = time.time() - start
        times.append(elapsed)
    
    # Estimate memory (simplified)
    memory_estimate = seq_len * seq_len * model.config.num_heads * model.config.num_layers * 4 / 1e6  # MB
    
    return times, memory_estimate


def run_benchmark_suite():
    """
    Run comprehensive benchmark comparing dense vs sparse attention
    """
    print("="*80)
    print("PYTHIA-70M BENCHMARK SUITE")
    print("Dense vs Sparse Attention Comparison")
    print("="*80)
    
    device = ndl.cpu()
    vocab_size = 10000
    
    # Test configurations
    configs = [
        {'batch_size': 4, 'seq_len': 64},
        {'batch_size': 4, 'seq_len': 128},
        {'batch_size': 4, 'seq_len': 256},
        {'batch_size': 2, 'seq_len': 512},
    ]
    
    results = {
        'dense': {'times': [], 'memory': []},
        'sparse': {'times': [], 'memory': []},
        'config_labels': []
    }
    
    for config in configs:
        batch_size = config['batch_size']
        seq_len = config['seq_len']
        label = f"B{batch_size}_S{seq_len}"
        results['config_labels'].append(label)
        
        print(f"\n{'='*80}")
        print(f"Configuration: Batch={batch_size}, SeqLen={seq_len}")
        print(f"{'='*80}")
        
        # Create models
        print("\nCreating dense model...")
        model_dense, _ = create_pythia_70m(
            vocab_size=vocab_size,
            max_seq_len=seq_len,
            use_sparse_attention=False,
            device=device
        )
        
        print("Creating sparse model...")
        model_sparse, _ = create_pythia_70m(
            vocab_size=vocab_size,
            max_seq_len=seq_len,
            use_sparse_attention=True,
            device=device
        )
        
        # Benchmark dense
        print("\nBenchmarking dense attention...")
        times_dense, mem_dense = benchmark_forward_pass(
            model_dense, batch_size, seq_len, vocab_size, device
        )
        avg_time_dense = np.mean(times_dense)
        std_time_dense = np.std(times_dense)
        results['dense']['times'].append(avg_time_dense)
        results['dense']['memory'].append(mem_dense)
        
        print(f"  Average time: {avg_time_dense*1000:.2f} ± {std_time_dense*1000:.2f} ms")
        print(f"  Memory estimate: {mem_dense:.2f} MB")
        
        # Benchmark sparse
        print("\nBenchmarking sparse attention...")
        times_sparse, mem_sparse = benchmark_forward_pass(
            model_sparse, batch_size, seq_len, vocab_size, device
        )
        avg_time_sparse = np.mean(times_sparse)
        std_time_sparse = np.std(times_sparse)
        results['sparse']['times'].append(avg_time_sparse)
        results['sparse']['memory'].append(mem_sparse)
        
        print(f"  Average time: {avg_time_sparse*1000:.2f} ± {std_time_sparse*1000:.2f} ms")
        print(f"  Memory estimate: {mem_sparse:.2f} MB")
        
        # Speedup
        speedup_time = avg_time_dense / avg_time_sparse
        speedup_mem = mem_dense / mem_sparse
        
        print(f"\n{'='*80}")
        print(f"RESULTS:")
        print(f"  Time speedup: {speedup_time:.2f}x")
        print(f"  Memory reduction: {speedup_mem:.2f}x")
        print(f"{'='*80}")
    
    # Generate plots
    plot_benchmark_results(results)
    
    # Summary table
    print("\n" + "="*80)
    print("SUMMARY TABLE")
    print("="*80)
    print(f"{'Config':<12} {'Dense (ms)':<12} {'Sparse (ms)':<12} {'Speedup':<10}")
    print("-"*80)
    for i, label in enumerate(results['config_labels']):
        dense_ms = results['dense']['times'][i] * 1000
        sparse_ms = results['sparse']['times'][i] * 1000
        speedup = dense_ms / sparse_ms
        print(f"{label:<12} {dense_ms:<12.2f} {sparse_ms:<12.2f} {speedup:<10.2f}x")
    print("="*80)
    
    return results


def plot_benchmark_results(results):
    """
    Create visualization of benchmark results
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    labels = results['config_labels']
    x = np.arange(len(labels))
    width = 0.35
    
    # Time comparison
    times_dense = [t * 1000 for t in results['dense']['times']]
    times_sparse = [t * 1000 for t in results['sparse']['times']]
    
    ax1.bar(x - width/2, times_dense, width, label='Dense', color='steelblue', alpha=0.8)
    ax1.bar(x + width/2, times_sparse, width, label='Sparse', color='coral', alpha=0.8)
    ax1.set_xlabel('Configuration', fontsize=12)
    ax1.set_ylabel('Forward Pass Time (ms)', fontsize=12)
    ax1.set_title('Forward Pass Performance Comparison', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Speedup
    speedups = [dense / sparse for dense, sparse in zip(times_dense, times_sparse)]
    ax2.bar(x, speedups, color='green', alpha=0.7)
    ax2.axhline(y=1.0, color='r', linestyle='--', linewidth=2, label='Baseline')
    ax2.set_xlabel('Configuration', fontsize=12)
    ax2.set_ylabel('Speedup (×)', fontsize=12)
    ax2.set_title('Sparse Attention Speedup', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=45)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, v in enumerate(speedups):
        ax2.text(i, v + 0.1, f'{v:.2f}×', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/benchmark_results.png', dpi=150, bbox_inches='tight')
    print("\nBenchmark plot saved to: /mnt/user-data/outputs/benchmark_results.png")
    plt.close()


def theoretical_complexity_analysis():
    """
    Analyze theoretical complexity benefits
    """
    print("\n" + "="*80)
    print("THEORETICAL COMPLEXITY ANALYSIS")
    print("="*80)
    
    seq_lengths = [64, 128, 256, 512, 1024, 2048]
    d_model = 512
    n_heads = 8
    n_layers = 6
    block_size = 64
    sparsity = 0.75
    
    print(f"\nModel Configuration:")
    print(f"  Hidden dim: {d_model}")
    print(f"  Heads: {n_heads}")
    print(f"  Layers: {n_layers}")
    print(f"  Block size: {block_size}")
    print(f"  Sparsity: {sparsity*100:.0f}%")
    
    print(f"\n{'Seq Len':<12} {'Dense GFLOPs':<15} {'Sparse GFLOPs':<15} {'Memory Dense (MB)':<18} {'Memory Sparse (MB)':<18} {'Speedup':<10}")
    print("-"*110)
    
    for seq_len in seq_lengths:
        # Dense: O(n²d)
        dense_flops = n_layers * seq_len * seq_len * d_model / 1e9
        dense_memory = seq_len * seq_len * n_layers * n_heads * 4 / 1e6
        
        # Sparse: O(n * block * (1 - sparsity) * d)
        sparse_flops = n_layers * seq_len * block_size * d_model * (1 - sparsity) / 1e9
        sparse_memory = seq_len * block_size * n_layers * n_heads * (1 - sparsity) * 4 / 1e6
        
        speedup = dense_flops / sparse_flops
        
        print(f"{seq_len:<12} {dense_flops:<15.2f} {sparse_flops:<15.2f} {dense_memory:<18.2f} {sparse_memory:<18.2f} {speedup:<10.2f}x")
    
    print("="*110)


if __name__ == "__main__":
    # Run benchmarks
    results = run_benchmark_suite()
    
    # Theoretical analysis
    theoretical_complexity_analysis()
    
    print("\n" + "="*80)
    print("BENCHMARK COMPLETE")
    print("="*80)
