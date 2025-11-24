# Fixes Applied - Sparse Attention & Checkpoint Loading

## Issues Resolved

### 1. CUDA Memory Error ("illegal memory access was encountered")

**Root Cause:**
- CUDA device not properly initialized before memory allocation
- Missing error checking and synchronization
- Potential memory leaks in sparse attention kernel

**Fixes Applied:**

**File: `src/ndarray_backend_cuda.cu`**
- Added explicit CUDA device initialization in `CudaArray` constructor
- Added `cudaSetDevice(0)` and `cudaDeviceSynchronize()` calls
- Improved error messages with detailed CUDA error strings
- Added `cuda_init()` function for explicit initialization
- Added `cuda_reset()` function for debugging memory issues
- Fixed memory leak risks in `BlockSparseAttention` and `ConvertToBlockMask`
- Added comprehensive error checking for all `cudaMalloc`, `cudaMemcpy` operations
- Added proper cleanup on errors to prevent memory leaks

**File: `python/needle/backend_ndarray/ndarray.py`**
- Added automatic CUDA initialization on first device access
- Added initialization status tracking to avoid redundant init calls

### 2. Missing `load_checkpoint` Function

**Root Cause:**
- `load_checkpoint` function was referenced but not implemented

**Fixes Applied:**

**File: `apps/train_pythia.py`**
- Added `load_checkpoint(filepath, device=None)` function (lines 278-337)
- Loads model architecture from saved config
- Restores model parameters from checkpoint
- Returns model, optimizer state, epoch, and loss
- Compatible with existing `save_checkpoint` format

## Usage

### Loading a Checkpoint

```python
from train_pythia import load_checkpoint

# Load checkpoint
model, optimizer, epoch, loss = load_checkpoint(
    "checkpoints/model_epoch_10.pkl",
    device=ndl.cuda()
)

print(f"Resumed from epoch {epoch} with loss {loss:.4f}")
```

### Rebuilding CUDA Backend

After applying fixes, rebuild the CUDA backend:

```bash
make clean
make lib
```

**Note:** Requires CUDA toolkit to be installed. The build will automatically detect CUDA and compile the backend if available.

## Key Improvements

1. **Robust CUDA Initialization**: Device is now properly initialized before any memory operations
2. **Better Error Messages**: CUDA errors now include detailed error strings for easier debugging
3. **Memory Leak Prevention**: All error paths properly clean up allocated memory
4. **Checkpoint Loading**: Full checkpoint loading functionality matching the save format
5. **Synchronization**: Proper kernel synchronization to catch runtime errors immediately

## Files Modified

1. `src/ndarray_backend_cuda.cu` - CUDA backend improvements
2. `python/needle/backend_ndarray/ndarray.py` - CUDA initialization
3. `apps/train_pythia.py` - Added load_checkpoint function

## Testing

To verify the fixes work:

```python
import needle as ndl

# Test CUDA initialization
device = ndl.cuda()
print("CUDA initialized successfully")

# Test model creation with sparse attention
from pythia_model import create_pythia_70m
model, config = create_pythia_70m(
    vocab_size=10000,
    max_seq_len=128,
    use_sparse_attention=True,
    device=device
)
print("Sparse model created successfully")
```
