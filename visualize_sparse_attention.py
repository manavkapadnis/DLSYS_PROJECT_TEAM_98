"""
Visualize Sparse Attention Patterns and Performance
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def visualize_sparse_patterns():
    """Visualize different sparse attention patterns"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Sparse Attention Patterns', fontsize=16, fontweight='bold')
    
    seq_len = 16  # Small for visualization
    block_size = 4
    
    # 1. Dense Attention
    ax = axes[0, 0]
    dense = np.ones((seq_len, seq_len))
    # Apply causal mask
    for i in range(seq_len):
        for j in range(i+1, seq_len):
            dense[i, j] = 0
    
    im = ax.imshow(dense, cmap='Blues', vmin=0, vmax=1)
    ax.set_title('Dense Attention (Causal)', fontweight='bold')
    ax.set_xlabel('Keys/Values')
    ax.set_ylabel('Queries')
    
    # Add block boundaries
    for i in range(0, seq_len+1, block_size):
        ax.axhline(i-0.5, color='red', linewidth=0.5, alpha=0.5)
        ax.axvline(i-0.5, color='red', linewidth=0.5, alpha=0.5)
    
    # 2. Local (Sliding Window)
    ax = axes[0, 1]
    local = np.zeros((seq_len, seq_len))
    for i in range(seq_len):
        # Each position attends to positions within its block and neighboring blocks
        block_i = i // block_size
        for j in range(seq_len):
            block_j = j // block_size
            if abs(block_i - block_j) <= 1 and j <= i:  # Causal
                local[i, j] = 1
    
    ax.imshow(local, cmap='Greens', vmin=0, vmax=1)
    ax.set_title('Local/Sliding Window', fontweight='bold')
    ax.set_xlabel('Keys/Values')
    ax.set_ylabel('Queries')
    
    # Add block boundaries
    for i in range(0, seq_len+1, block_size):
        ax.axhline(i-0.5, color='red', linewidth=0.5, alpha=0.5)
        ax.axvline(i-0.5, color='red', linewidth=0.5, alpha=0.5)
    
    # 3. Global (Strided)
    ax = axes[0, 2]
    strided = np.zeros((seq_len, seq_len))
    stride = 2
    for i in range(seq_len):
        block_i = i // block_size
        for j in range(seq_len):
            block_j = j // block_size
            # Attend to every stride-th block + same block
            if (block_j % stride == 0 or block_i == block_j) and j <= i:
                strided[i, j] = 1
    
    ax.imshow(strided, cmap='Oranges', vmin=0, vmax=1)
    ax.set_title('Global/Strided', fontweight='bold')
    ax.set_xlabel('Keys/Values')
    ax.set_ylabel('Queries')
    
    # Add block boundaries
    for i in range(0, seq_len+1, block_size):
        ax.axhline(i-0.5, color='red', linewidth=0.5, alpha=0.5)
        ax.axvline(i-0.5, color='red', linewidth=0.5, alpha=0.5)
    
    # 4. Current Implementation
    ax = axes[1, 0]
    ax.text(0.5, 0.7, 'Current "Sparse" Implementation:', 
            ha='center', va='center', transform=ax.transAxes, 
            fontsize=14, fontweight='bold')
    ax.text(0.5, 0.5, '1. Compute FULL attention matrix\n2. Apply mask (multiply)\n3. No computation savings!', 
            ha='center', va='center', transform=ax.transAxes, 
            fontsize=12, color='red')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    # 5. Ideal Implementation
    ax = axes[1, 1]
    ax.text(0.5, 0.7, 'Ideal Sparse Implementation:', 
            ha='center', va='center', transform=ax.transAxes, 
            fontsize=14, fontweight='bold')
    ax.text(0.5, 0.5, '1. Only compute active blocks\n2. Skip masked regions entirely\n3. ~4x speedup!', 
            ha='center', va='center', transform=ax.transAxes, 
            fontsize=12, color='green')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    # 6. Sparsity Statistics
    ax = axes[1, 2]
    patterns = ['Dense', 'Local', 'Strided']
    active = [np.sum(dense), np.sum(local), np.sum(strided)]
    total = seq_len * seq_len
    sparsity = [(total - a) / total * 100 for a in active]
    
    bars = ax.bar(patterns, sparsity, color=['blue', 'green', 'orange'], alpha=0.7)
    ax.set_ylabel('Sparsity (%)', fontweight='bold')
    ax.set_title('Sparsity Levels', fontweight='bold')
    ax.set_ylim(0, 100)
    
    # Add value labels
    for bar, sp in zip(bars, sparsity):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{sp:.1f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/sparse_attention_patterns.png', dpi=150, bbox_inches='tight')
    plt.close()


def visualize_computation_flow():
    """Visualize how current vs ideal sparse attention works"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
    fig.suptitle('Sparse Attention Computation Flow', fontsize=16, fontweight='bold')
    
    # Current Implementation
    ax1.set_title('Current Implementation (No Speedup)', fontsize=14, color='red')
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.axis('off')
    
    # Flow diagram
    steps = [
        (5, 9, "Q, K, V\n(n×d)", 'lightblue'),
        (5, 7, "Compute Q×K^T\n(n×n matrix)", 'lightcoral'),
        (5, 5, "Create Mask\n(n×n matrix)", 'lightyellow'),
        (5, 3, "Apply Mask\n(element-wise)", 'lightcoral'),
        (5, 1, "Softmax & V\n(n×n × n×d)", 'lightcoral'),
    ]
    
    for x, y, text, color in steps:
        rect = patches.FancyBboxPatch((x-1.5, y-0.4), 3, 0.8, 
                                     boxstyle="round,pad=0.1", 
                                     facecolor=color, edgecolor='black')
        ax1.add_patch(rect)
        ax1.text(x, y, text, ha='center', va='center', fontsize=10)
    
    # Arrows
    for i in range(len(steps)-1):
        ax1.arrow(5, steps[i][1]-0.4, 0, -1.2, head_width=0.3, head_length=0.2, fc='black', ec='black')
    
    # Annotations
    ax1.text(8.5, 7, "O(n²d)\nFULL computation!", ha='center', va='center', 
             fontsize=11, color='red', fontweight='bold',
             bbox=dict(boxstyle="round,pad=0.3", facecolor='white', edgecolor='red'))
    
    # Ideal Implementation
    ax2.set_title('Ideal Sparse Implementation (4x Faster)', fontsize=14, color='green')
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.axis('off')
    
    # Flow diagram
    steps = [
        (5, 9, "Q, K, V\n(n×d)", 'lightblue'),
        (5, 7, "Block Pattern\n(precomputed)", 'lightgreen'),
        (5, 5, "Compute ONLY\nactive blocks", 'lightgreen'),
        (5, 3, "Block Softmax\n(sparse)", 'lightgreen'),
        (5, 1, "Sparse V mult\n(~0.25n²d)", 'lightgreen'),
    ]
    
    for x, y, text, color in steps:
        rect = patches.FancyBboxPatch((x-1.5, y-0.4), 3, 0.8, 
                                     boxstyle="round,pad=0.1", 
                                     facecolor=color, edgecolor='black')
        ax2.add_patch(rect)
        ax2.text(x, y, text, ha='center', va='center', fontsize=10)
    
    # Arrows
    for i in range(len(steps)-1):
        ax2.arrow(5, steps[i][1]-0.4, 0, -1.2, head_width=0.3, head_length=0.2, fc='black', ec='black')
    
    # Annotations
    ax2.text(8.5, 5, "O(n²d × sparsity)\nOnly ~25% ops!", ha='center', va='center', 
             fontsize=11, color='green', fontweight='bold',
             bbox=dict(boxstyle="round,pad=0.3", facecolor='white', edgecolor='green'))
    
    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/sparse_attention_flow.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_performance_comparison():
    """Plot theoretical vs actual performance"""
    seq_lengths = [128, 256, 512, 1024]
    
    # Theoretical speedups
    dense_flops = [n**2 for n in seq_lengths]
    sparse_flops_ideal = [n**2 * 0.25 for n in seq_lengths]  # 75% sparse
    sparse_flops_actual = dense_flops  # Current implementation
    
    # Normalize to dense=1.0
    speedup_ideal = [d/s for d, s in zip(dense_flops, sparse_flops_ideal)]
    speedup_actual = [1.0] * len(seq_lengths)  # No speedup
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # FLOPs comparison
    x = np.arange(len(seq_lengths))
    width = 0.25
    
    bars1 = ax1.bar(x - width, dense_flops, width, label='Dense', color='blue', alpha=0.7)
    bars2 = ax1.bar(x, sparse_flops_actual, width, label='Current Sparse', color='red', alpha=0.7)
    bars3 = ax1.bar(x + width, sparse_flops_ideal, width, label='Ideal Sparse', color='green', alpha=0.7)
    
    ax1.set_xlabel('Sequence Length', fontsize=12)
    ax1.set_ylabel('Relative FLOPs', fontsize=12)
    ax1.set_title('Computational Cost Comparison', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(seq_lengths)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Speedup comparison
    bars1 = ax2.bar(x - width/2, speedup_actual, width, label='Current (Actual)', color='red', alpha=0.7)
    bars2 = ax2.bar(x + width/2, speedup_ideal, width, label='Ideal (Theoretical)', color='green', alpha=0.7)
    
    ax2.set_xlabel('Sequence Length', fontsize=12)
    ax2.set_ylabel('Speedup vs Dense', fontsize=12)
    ax2.set_title('Sparse Attention Speedup', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(seq_lengths)
    ax2.axhline(y=1.0, color='blue', linestyle='--', label='Dense baseline')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bars in [bars2]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}×', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/sparse_attention_performance.png', dpi=150, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    print("Generating sparse attention visualizations...")
    
    visualize_sparse_patterns()
    print("✓ Saved: sparse_attention_patterns.png")
    
    visualize_computation_flow()
    print("✓ Saved: sparse_attention_flow.png")
    
    plot_performance_comparison()
    print("✓ Saved: sparse_attention_performance.png")
    
    print("\nAll visualizations saved to /mnt/user-data/outputs/")
