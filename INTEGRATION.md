# Integration Instructions

## How to Integrate Pythia-70M into Needle Repository

### 1. File Placement

Copy the provided files to the following locations in your Needle repository:

```bash
# Model and training files
cp pythia_model.py <needle_root>/apps/
cp train_pythia.py <needle_root>/apps/
cp benchmark.py <needle_root>/apps/

# Sparse attention module
cp nn_sparse_attention.py <needle_root>/python/needle/nn/

# Demo notebook
cp demo_notebook.ipynb <needle_root>/

# Documentation
cp README.md <needle_root>/PYTHIA_README.md
```

### 2. Update Needle Module Imports

Add the sparse attention module to the nn package:

**Edit: `<needle_root>/python/needle/nn/__init__.py`**

```python
# Add this line:
from .nn_sparse_attention import *
```

### 3. Build Needle Backend

```bash
cd <needle_root>
make clean
make
```

### 4. Verify Installation

Run this test to verify everything is working:

```python
import sys
sys.path.append('./python')
import needle as ndl
from apps.pythia_model import create_pythia_70m

# Create model
device = ndl.cpu()
model, config = create_pythia_70m(
    use_sparse_attention=False,
    device=device
)

print("✓ Model created successfully!")
print(f"✓ Total parameters: ~{config.get_total_params() / 1e6:.1f}M")
```

### 5. Run Examples

#### Train Dense Model
```bash
cd <needle_root>
python apps/train_pythia.py --epochs 5 --batch_size 16 --seq_len 128
```

#### Train Sparse Model
```bash
python apps/train_pythia.py --epochs 5 --batch_size 16 --seq_len 128 --sparse
```

#### Run Benchmarks
```bash
python apps/benchmark.py
```

#### Run Demo Notebook
```bash
jupyter notebook demo_notebook.ipynb
```

## File Structure After Integration

```
needle/
├── apps/
│   ├── pythia_model.py          # NEW: Pythia-70M implementation
│   ├── train_pythia.py          # NEW: Training script
│   ├── benchmark.py             # NEW: Benchmark script
│   ├── models.py                # Existing
│   └── simple_ml.py            # Existing
├── python/
│   └── needle/
│       ├── nn/
│       │   ├── nn_basic.py           # Existing
│       │   ├── nn_conv.py            # Existing
│       │   ├── nn_sequence.py        # Existing
│       │   ├── nn_transformer.py     # Existing
│       │   └── nn_sparse_attention.py # NEW: Sparse attention
│       └── ...
├── demo_notebook.ipynb          # NEW: Demo notebook
├── PYTHIA_README.md            # NEW: Project documentation
└── README.md                    # Existing
```

## Dependencies

The implementation uses only standard Needle components:
- `needle.nn` - Neural network modules
- `needle.ops` - Operations
- `needle.init` - Initialization functions
- `needle.optim` - Optimizers

No additional external dependencies required beyond what Needle already uses.

## Expected Outputs

After running the demo notebook, you should see:

1. **Model Creation**: Successfully creates Pythia-70M with ~70M parameters
2. **Sparse Patterns**: Visualization of local, global, and mixed attention patterns
3. **Performance**: 2-4x speedup on forward pass (CPU), higher on GPU
4. **Training**: Similar convergence for dense and sparse models
5. **Benchmarks**: Detailed performance comparison across sequence lengths

## Troubleshooting

### Import Errors

If you see `ModuleNotFoundError: No module named 'needle'`:
```bash
export PYTHONPATH=$PYTHONPATH:<needle_root>/python
```

### Build Errors

If `make` fails:
```bash
# Install build dependencies
sudo apt-get install build-essential python3-dev

# Rebuild
make clean
make
```

### CUDA Errors

If CUDA is not available:
- The code will automatically fall back to CPU
- Use `--device cpu` flag explicitly in training scripts

### Memory Issues

If you run out of memory:
- Reduce batch size: `--batch_size 8`
- Reduce sequence length: `--seq_len 64`
- Use sparse attention: `--sparse`

## Performance Tips

### For Faster Training

1. Use GPU if available:
   ```bash
   python apps/train_pythia.py --device cuda --sparse
   ```

2. Increase batch size (if memory allows):
   ```bash
   python apps/train_pythia.py --batch_size 64
   ```

3. Use sparse attention:
   ```bash
   python apps/train_pythia.py --sparse
   ```

### For Better Quality

1. Train longer:
   ```bash
   python apps/train_pythia.py --epochs 20
   ```

2. Use learning rate scheduling (can be added to train_pythia.py)

3. Increase model size (modify config in pythia_model.py)

## Next Steps

### Extend the Implementation

1. **Add More Sparse Patterns**:
   - Edit `nn_sparse_attention.py`
   - Implement BigBird pattern, Reformer pattern, etc.

2. **CUDA Optimization**:
   - Create `src/ndarray_backend_cuda_sparse.cu`
   - Implement efficient sparse matrix operations

3. **Larger Models**:
   - Modify `pythia_model.py` PythiaConfig
   - Create Pythia-410M, Pythia-1B configurations

4. **Better Datasets**:
   - Integrate HuggingFace datasets
   - Add WikiText-2, TinyStories loaders

## Contact & Support

For questions or issues:
1. Check the main README.md
2. Review demo_notebook.ipynb for examples
3. Examine the code comments for implementation details

## Validation Checklist

- [ ] Files copied to correct locations
- [ ] Needle backend built successfully (`make`)
- [ ] Import test passes
- [ ] Dense model trains without errors
- [ ] Sparse model trains without errors
- [ ] Benchmark script runs successfully
- [ ] Demo notebook executes all cells
- [ ] Speedup observed (sparse vs dense)

If all items are checked, your integration is successful! ✓
