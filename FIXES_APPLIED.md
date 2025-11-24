# Fixes Applied - Sparse Attention & Checkpoint Loading

## Issues Resolved

### 1. CUDA Memory Error ("illegal memory access was encountered")

**Root Causes:**
1. CUDA device not properly initialized before memory allocation
2. **Critical Bug:** Incorrect metadata parsing in `ConvertToBlockMask` - was reading `num_active` from index 2 instead of index 1
3. Missing bounds checking in BlockSparseAttention kernel causing out-of-bounds memory access
4. Missing error checking and synchronization
5. No CUDA context recovery after kernel failures

**Fixes Applied:**

**File: `src/ndarray_backend_cuda.cu`**
- **CRITICAL FIX:** Fixed metadata parsing bug in `ConvertToBlockMask`:
  - Changed `num_active = sparse_blocks[2]` to `num_active = sparse_blocks[1]`
  - This was causing incorrect memory allocation and out-of-bounds access
- Added comprehensive bounds checking in `BlockSparseAttentionKernel`:
  - Query block index validation
  - Bounds checks for Q, K, V tile loading with zero-padding
  - Bounds check for output writing
  - Division by zero protection (check `l_i > 0.0f`)
- Added validation in `ConvertToBlockMask`:
  - Size validation before parsing metadata
  - Clear error messages with expected vs actual sizes
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

**File: `python/needle/nn/nn_sparse_attention.py`**
- Added CUDA context recovery on kernel failure:
  - Automatically resets and reinitializes CUDA device when kernel fails
  - Prevents cascading failures from corrupted CUDA context
  - Falls back to slow implementation after recovery attempt

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

1. **Fixed Critical Metadata Bug**: Corrected sparse block metadata parsing that was causing illegal memory access
2. **Comprehensive Bounds Checking**: All kernel memory accesses now validated to prevent out-of-bounds errors
3. **Robust CUDA Initialization**: Device is now properly initialized before any memory operations
4. **CUDA Context Recovery**: Automatic device reset on kernel failure prevents cascading errors
5. **Better Error Messages**: CUDA errors now include detailed error strings for easier debugging
6. **Memory Leak Prevention**: All error paths properly clean up allocated memory
7. **Checkpoint Loading**: Full checkpoint loading functionality matching the save format
8. **Synchronization**: Proper kernel synchronization to catch runtime errors immediately

## Files Modified

1. `src/ndarray_backend_cuda.cu` - **Critical metadata bug fix + bounds checking**
2. `python/needle/nn/nn_sparse_attention.py` - CUDA context recovery
3. `python/needle/backend_ndarray/ndarray.py` - CUDA initialization
4. `apps/train_pythia.py` - Added load_checkpoint function

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
