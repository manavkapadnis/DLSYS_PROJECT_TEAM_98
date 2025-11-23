"""
Memory Profiling Script for Pythia-70M
Helps identify memory bottlenecks
"""
import sys
sys.path.append('./python')
import numpy as np
import needle as ndl
from pythia_model import create_pythia_70m
import gc
import psutil
import os


def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def profile_model_memory(vocab_size=10000, batch_size=8, seq_len=128, device=None):
    """Profile memory usage of model components"""
    print("="*80)
    print("PYTHIA-70M MEMORY PROFILING")
    print("="*80)
    
    if device is None:
        device = ndl.cpu()
    
    print(f"\nConfiguration:")
    print(f"  Vocab size: {vocab_size}")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Device: {device}")
    
    # Initial memory
    gc.collect()
    mem_start = get_memory_usage()
    print(f"\nInitial memory: {mem_start:.2f} MB")
    
    # Create model
    print("\n1. Creating model...")
    model, config = create_pythia_70m(
        vocab_size=vocab_size,
        max_seq_len=seq_len,
        use_sparse_attention=False,
        device=device
    )
    gc.collect()
    mem_after_model = get_memory_usage()
    model_memory = mem_after_model - mem_start
    print(f"   Model memory: {model_memory:.2f} MB")
    
    # Count parameters
    total_params = 0
    param_details = []
    
    print("\n2. Parameter breakdown:")
    print("-"*80)
    
    # Token embedding
    token_emb_params = vocab_size * config.d_model
    total_params += token_emb_params
    param_details.append(("Token Embedding", token_emb_params, token_emb_params * 4 / 1e6))
    
    # Position embedding
    pos_emb_params = config.max_seq_len * config.d_model
    total_params += pos_emb_params
    param_details.append(("Position Embedding", pos_emb_params, pos_emb_params * 4 / 1e6))
    
    # Transformer layers
    per_layer = 4 * config.d_model * config.d_model + 2 * config.d_model * config.d_ff
    total_params += config.num_layers * per_layer
    param_details.append(("Transformer Layers", config.num_layers * per_layer, 
                         config.num_layers * per_layer * 4 / 1e6))
    
    # LM head
    lm_head_params = config.d_model * vocab_size
    total_params += lm_head_params
    param_details.append(("LM Head", lm_head_params, lm_head_params * 4 / 1e6))
    
    # Print details
    for name, params, memory_mb in param_details:
        print(f"   {name:<20}: {params/1e6:>6.1f}M params = {memory_mb:>6.1f} MB")
    
    print(f"   {'='*50}")
    print(f"   {'Total':<20}: {total_params/1e6:>6.1f}M params = {total_params*4/1e6:>6.1f} MB")
    
    # Test forward pass
    print("\n3. Testing forward pass...")
    input_ids = ndl.Tensor(
        np.random.randint(0, vocab_size, (batch_size, seq_len)),
        device=device
    )
    
    gc.collect()
    mem_before_forward = get_memory_usage()
    
    logits, _ = model(input_ids)
    
    gc.collect()
    mem_after_forward = get_memory_usage()
    forward_memory = mem_after_forward - mem_before_forward
    print(f"   Forward pass memory: {forward_memory:.2f} MB")
    
    # Test with targets (loss computation)
    print("\n4. Testing with loss computation...")
    targets = ndl.Tensor(
        np.random.randint(0, vocab_size, (batch_size, seq_len)),
        device=device
    )
    
    gc.collect()
    mem_before_loss = get_memory_usage()
    
    _, loss = model(input_ids, targets)
    
    gc.collect()
    mem_after_loss = get_memory_usage()
    loss_memory = mem_after_loss - mem_before_loss
    print(f"   Loss computation memory: {loss_memory:.2f} MB")
    
    # Test backward pass
    print("\n5. Testing backward pass...")
    gc.collect()
    mem_before_backward = get_memory_usage()
    
    loss.backward()
    
    gc.collect()
    mem_after_backward = get_memory_usage()
    backward_memory = mem_after_backward - mem_before_backward
    print(f"   Backward pass memory: {backward_memory:.2f} MB")
    
    # Summary
    print("\n" + "="*80)
    print("MEMORY SUMMARY")
    print("="*80)
    print(f"Model parameters: {total_params/1e6:.1f}M = {total_params*4/1e6:.1f} MB")
    print(f"Model in memory: {model_memory:.1f} MB")
    print(f"Forward pass: +{forward_memory:.1f} MB")
    print(f"Loss computation: +{loss_memory:.1f} MB")
    print(f"Backward pass: +{backward_memory:.1f} MB")
    print(f"Total peak memory: {mem_after_backward:.1f} MB")
    
    # GPU memory if available
    if 'cuda' in str(device):
        try:
            import subprocess
            result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total', 
                                   '--format=csv,nounits,noheader'], 
                                  capture_output=True, text=True)
            used, total = map(int, result.stdout.strip().split(','))
            print(f"\nGPU Memory: {used} MB / {total} MB ({used/total*100:.1f}%)")
        except:
            pass
    
    return {
        'total_params': total_params,
        'model_memory': model_memory,
        'forward_memory': forward_memory,
        'backward_memory': backward_memory,
        'peak_memory': mem_after_backward
    }


def test_embedding_memory():
    """Test memory usage of embedding layer specifically"""
    print("\n\n" + "="*80)
    print("EMBEDDING LAYER MEMORY TEST")
    print("="*80)
    
    device = ndl.cpu()
    vocab_size = 10000
    embedding_dim = 512
    seq_len = 128
    batch_size = 8
    
    print(f"\nConfiguration:")
    print(f"  Vocab size: {vocab_size}")
    print(f"  Embedding dim: {embedding_dim}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Batch size: {batch_size}")
    
    # Test old implementation (if available)
    print("\n1. Testing embedding implementations...")
    
    from needle.nn.nn_sequence import Embedding
    
    # Create embedding layer
    gc.collect()
    mem_start = get_memory_usage()
    
    embedding = Embedding(vocab_size, embedding_dim, device=device)
    
    gc.collect()
    mem_after_create = get_memory_usage()
    print(f"   Embedding layer created: {mem_after_create - mem_start:.2f} MB")
    
    # Create input
    input_ids = ndl.Tensor(
        np.random.randint(0, vocab_size, (seq_len, batch_size)),
        device=device
    )
    
    # Forward pass
    gc.collect()
    mem_before_forward = get_memory_usage()
    
    output = embedding(input_ids)
    
    gc.collect()
    mem_after_forward = get_memory_usage()
    forward_memory = mem_after_forward - mem_before_forward
    
    print(f"   Forward pass memory: {forward_memory:.2f} MB")
    
    # Expected vs actual
    expected_one_hot = seq_len * batch_size * vocab_size * 4 / 1e6
    print(f"\n   Expected one-hot memory (if created): {expected_one_hot:.2f} MB")
    print(f"   Actual forward memory: {forward_memory:.2f} MB")
    
    if forward_memory > expected_one_hot * 0.8:
        print("   ❌ WARNING: Embedding is likely creating full one-hot matrix!")
    else:
        print("   ✓ Embedding is memory efficient")


if __name__ == "__main__":
    print("\n🔍 PROFILING PYTHIA-70M MEMORY USAGE...\n")
    
    # Profile different configurations
    configs = [
        {"batch_size": 8, "seq_len": 128},
        {"batch_size": 16, "seq_len": 128},
        {"batch_size": 32, "seq_len": 128},
    ]
    
    results = []
    for config in configs:
        print(f"\n\n{'#'*80}")
        print(f"Testing batch_size={config['batch_size']}, seq_len={config['seq_len']}")
        print('#'*80)
        
        result = profile_model_memory(
            batch_size=config['batch_size'],
            seq_len=config['seq_len']
        )
        result.update(config)
        results.append(result)
        
        # Clean up
        gc.collect()
    
    # Test embedding specifically
    test_embedding_memory()
    
    # Summary table
    print("\n\n" + "="*80)
    print("SCALING SUMMARY")
    print("="*80)
    print(f"{'Batch':<8} {'Seq':<8} {'Peak Mem (MB)':<15} {'Per Sample':<15}")
    print("-"*80)
    
    for r in results:
        per_sample = r['peak_memory'] / r['batch_size']
        print(f"{r['batch_size']:<8} {r['seq_len']:<8} {r['peak_memory']:<15.1f} {per_sample:<15.2f}")
