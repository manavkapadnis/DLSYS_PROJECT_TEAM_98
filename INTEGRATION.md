# Integration Guide - Pythia-70M with Block-Sparse Attention

This guide provides step-by-step instructions for integrating the Pythia-70M sparse attention implementation into your Needle repository.

## Prerequisites

- Needle framework installed and working
- Python 3.7+
- CUDA 10.0+ (optional, for GPU acceleration)
- HuggingFace datasets library (optional, for real datasets)

## Step 1: File Placement

Copy the following files to your Needle repository:

### Core Implementation Files

```bash
# Sparse attention module
cp python/needle/nn/nn_sparse_attention.py <needle_repo>/python/needle/nn/

# Update nn __init__
cp python/needle/nn/__init__.py <needle_repo>/python/needle/nn/

# Model implementation
cp apps/pythia_model.py <needle_repo>/apps/

# Training script
cp apps/train_pythia.py <needle_repo>/apps/

# Benchmark script
cp apps/benchmark.py <needle_repo>/apps/

# Quick start demo
cp apps/quick_start.py <needle_repo>/apps/

# Dataset utilities
cp python/needle/data/datasets/text_dataset.py <needle_repo>/python/needle/data/datasets/
```

### CUDA Backend (Optional)

```bash
# CUDA kernels with sparse attention support
cp src/ndarray_backend_cuda.cu <needle_repo>/src/
```

### Documentation

```bash
# Main documentation
cp README.md <needle_repo>/

# This integration guide
cp INTEGRATION.md <needle_repo>/

# Demo notebook
cp demo_notebook.ipynb <needle_repo>/
```

## Step 2: Build the Project

### Rebuild CUDA Backend (if using CUDA)

```bash
cd <needle_repo>
make clean
make
```

This will compile the new CUDA kernels including the sparse attention operations.

### Verify Build

```bash
python3 -c "import needle; print('Needle loaded successfully')"
python3 -c "from needle.nn import nn_sparse_attention; print('Sparse attention module loaded')"
```

## Step 3: Install Dependencies

### Required

```bash
pip install numpy
```

### Optional (for HuggingFace datasets)

```bash
pip install datasets transformers
```

### Optional (for visualization)

```bash
pip install matplotlib jupyter
```

## Step 4: Verification

### Test 1: Import Check

```python
import sys
sys.path.append('./python')
import needle as ndl
from pythia_model import create_pythia_70m
from needle.nn.nn_sparse_attention import BlockSparsePattern

print("All imports successful!")
```

### Test 2: Model Creation

```python
import needle as ndl
from pythia_model import create_pythia_70m

device = ndl.cpu()

# Create dense model
model_dense, config = create_pythia_70m(
    vocab_size=10000,
    max_seq_len=128,
    use_sparse_attention=False,
    device=device
)
print(f"Dense model created: {config.get_total_params() / 1e6:.1f}M parameters")

# Create sparse model
model_sparse, config = create_pythia_70m(
    vocab_size=10000,
    max_seq_len=128,
    use_sparse_attention=True,
    device=device
)
print(f"Sparse model created: {config.get_total_params() / 1e6:.1f}M parameters")
```

### Test 3: Forward Pass

```python
import numpy as np
import needle as ndl
from pythia_model import create_pythia_70m

device = ndl.cpu()
model, config = create_pythia_70m(use_sparse_attention=True, device=device)

# Create sample input
input_ids = ndl.Tensor(
    np.random.randint(0, 10000, (4, 64)),
    device=device
)

# Forward pass
logits, _ = model(input_ids)
print(f"Forward pass successful!")
print(f"Output shape: {logits.shape}")
```

### Test 4: Training Script

```bash
# Quick test with synthetic data
python apps/train_pythia.py \
    --dataset synthetic \
    --epochs 1 \
    --batch_size 4 \
    --seq_len 64 \
    --max_tokens 10000
```

## Step 5: Running Experiments

### Experiment 1: Dense vs Sparse Comparison

```bash
# Train dense model
python apps/train_pythia.py \
    --dataset wikitext-2 \
    --epochs 10 \
    --batch_size 32 \
    --seq_len 128 \
    --checkpoint_dir ./checkpoints/dense

# Train sparse model
python apps/train_pythia.py \
    --dataset wikitext-2 \
    --epochs 10 \
    --batch_size 32 \
    --seq_len 128 \
    --sparse \
    --checkpoint_dir ./checkpoints/sparse
```

### Experiment 2: Performance Benchmarking

```bash
python apps/benchmark.py
```

### Experiment 3: Interactive Exploration

```bash
jupyter notebook demo_notebook.ipynb
```

## Common Issues and Solutions

### Issue 1: CUDA Build Fails

**Solution**: Ensure CUDA toolkit is installed and cmake can find it:

```bash
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
make clean
make
```

### Issue 2: HuggingFace Datasets Not Found

**Solution**: Install datasets or use synthetic data:

```bash
pip install datasets
# Or use synthetic data:
python apps/train_pythia.py --dataset synthetic
```

### Issue 3: Out of Memory

**Solution**: Reduce batch size or sequence length:

```bash
python apps/train_pythia.py \
    --batch_size 16 \
    --seq_len 64
```

### Issue 4: Import Errors

**Solution**: Ensure paths are correct:

```python
import sys
sys.path.append('./python')
sys.path.append('./apps')
```

## Configuration Options

### Model Configuration

Edit `pythia_model.py` to adjust:

- `vocab_size`: Size of vocabulary
- `d_model`: Hidden dimension (512 for Pythia-70M)
- `num_heads`: Number of attention heads (8 for Pythia-70M)
- `num_layers`: Number of transformer layers (6 for Pythia-70M)
- `d_ff`: FFN dimension (2048 for Pythia-70M)
- `max_seq_len`: Maximum sequence length
- `dropout`: Dropout rate
- `sparse_block_size`: Block size for sparse attention (64)
- `sparse_pattern`: Pattern type ("local", "global", "mixed")

### Training Configuration

Pass via command line:

```bash
python apps/train_pythia.py \
    --epochs 20 \
    --batch_size 64 \
    --seq_len 256 \
    --lr 5e-4 \
    --sparse \
    --device cuda
```

## Performance Tuning

### CPU Optimization

- Use larger batch sizes (32-64)
- Reduce sequence length for memory efficiency
- Enable sparse attention for speedup

### CUDA Optimization

- Increase batch size (64-128)
- Use longer sequences (256-512)
- Sparse attention provides best speedup on GPU

### Memory Optimization

- Reduce `d_model` or `num_layers` for smaller model
- Use gradient accumulation for large batch sizes
- Enable sparse attention to reduce memory footprint

## Validation Checklist

- [ ] All files copied to correct locations
- [ ] Project builds successfully (`make`)
- [ ] Imports work without errors
- [ ] Dense model forward pass succeeds
- [ ] Sparse model forward pass succeeds
- [ ] Training script runs (even on synthetic data)
- [ ] Benchmark script produces results
- [ ] Checkpoint save/load works
- [ ] (Optional) HuggingFace datasets load correctly
- [ ] (Optional) CUDA backend works

## Next Steps

1. **Explore the demo notebook**: `jupyter notebook demo_notebook.ipynb`
2. **Run benchmarks**: `python apps/benchmark.py`
3. **Train your first model**: `python apps/train_pythia.py`
4. **Experiment with sparse patterns**: Modify `sparse_pattern` in config
5. **Scale up**: Try larger models or longer sequences

## Support

For issues or questions:

1. Check the README.md for usage examples
2. Review code comments in source files
3. Examine demo_notebook.ipynb for interactive examples
4. Verify your environment meets prerequisites

## Advanced: Custom Sparse Patterns

To implement custom sparse patterns:

1. Edit `python/needle/nn/nn_sparse_attention.py`
2. Add new pattern method to `BlockSparsePattern` class
3. Update `create_block_mask()` to support new pattern
4. Example:

```python
@staticmethod
def custom_pattern(seq_len: int, block_size: int, param: int):
    n_blocks = (seq_len + block_size - 1) // block_size
    mask = np.zeros((n_blocks, n_blocks), dtype=bool)
    
    # Your custom logic here
    for i in range(n_blocks):
        mask[i, i] = True  # Always attend to self
        # Add your pattern logic
    
    return mask
```

## Debugging Tips

### Enable Verbose Logging

Add to training script:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Check Tensor Shapes

Add print statements:

```python
print(f"Input shape: {input_ids.shape}")
print(f"Output shape: {logits.shape}")
```

### Verify Gradient Flow

```python
for param in model.parameters():
    if param.grad is not None:
        print(f"Grad norm: {np.linalg.norm(param.grad.numpy())}")
```

## Conclusion

You should now have a fully integrated Pythia-70M implementation with block-sparse attention in your Needle repository. Start with the quick_start.py demo, then explore training and benchmarking!