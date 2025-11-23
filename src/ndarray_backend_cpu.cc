#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <limits>
#include <vector>

namespace needle {
namespace cpu {

#define ALIGNMENT 256
#define TILE 8
typedef float scalar_t;
const size_t ELEM_SIZE = sizeof(scalar_t);


/**
 * This is a utility structure for maintaining an array aligned to ALIGNMENT boundaries in
 * memory.  This alignment should be at least TILE * ELEM_SIZE, though we make it even larger
 * here by default.
 */
struct AlignedArray {
  AlignedArray(const size_t size) {
    int ret = posix_memalign((void**)&ptr, ALIGNMENT, size * ELEM_SIZE);
    if (ret != 0) throw std::bad_alloc();
    this->size = size;
  }
  ~AlignedArray() { free(ptr); }
  size_t ptr_as_int() {return (size_t)ptr; }
  scalar_t* ptr;
  size_t size;
};



void Fill(AlignedArray* out, scalar_t val) {
  /**
   * Fill the values of an aligned array with val
   */
  for (int i = 0; i < out->size; i++) {
    out->ptr[i] = val;
  }
}

// Helper function to convert flat index to strided offset
size_t GetOffset(size_t idx, const std::vector<int32_t>& shape, 
                 const std::vector<int32_t>& strides) {
  size_t offset = 0;
  size_t remaining = idx;
  
  for (int i = shape.size() - 1; i >= 0; i--) {
    size_t coord = remaining % shape[i];
    remaining /= shape[i];
    offset += coord * strides[i];
  }
  
  return offset;
}


void Compact(const AlignedArray& a, AlignedArray* out, std::vector<int32_t> shape,
             std::vector<int32_t> strides, size_t offset) {
  size_t size = 1;
  for (int32_t dim : shape) {
    size *= dim;
  }

  for (size_t i = 0; i < size; i++) {
    out->ptr[i] = a.ptr[offset + GetOffset(i, shape, strides)];
  }
}

void EwiseSetitem(const AlignedArray& a, AlignedArray* out, std::vector<int32_t> shape,
                  std::vector<int32_t> strides, size_t offset) {
  size_t size = 1;
  for (int32_t dim : shape) {
    size *= dim;
  }

  for (size_t i = 0; i < size; i++) {
    out->ptr[offset + GetOffset(i, shape, strides)] = a.ptr[i];
  }
}

void ScalarSetitem(const size_t size, scalar_t val, AlignedArray* out, std::vector<int32_t> shape,
                   std::vector<int32_t> strides, size_t offset) {
  for (size_t i = 0; i < size; i++) {
    out->ptr[offset + GetOffset(i, shape, strides)] = val;
  }
}

void EwiseAdd(const AlignedArray& a, const AlignedArray& b, AlignedArray* out) {
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = a.ptr[i] + b.ptr[i];
  }
}

void ScalarAdd(const AlignedArray& a, scalar_t val, AlignedArray* out) {
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = a.ptr[i] + val;
  }
}

// ============================================================================
// Multiplication
// ============================================================================

void EwiseMul(const AlignedArray& a, const AlignedArray& b, AlignedArray* out) {
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = a.ptr[i] * b.ptr[i];
  }
}

void ScalarMul(const AlignedArray& a, scalar_t val, AlignedArray* out) {
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = a.ptr[i] * val;
  }
}

// ============================================================================
// Division
// ============================================================================

void EwiseDiv(const AlignedArray& a, const AlignedArray& b, AlignedArray* out) {
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = a.ptr[i] / b.ptr[i];
  }
}

void ScalarDiv(const AlignedArray& a, scalar_t val, AlignedArray* out) {
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = a.ptr[i] / val;
  }
}

// ============================================================================
// Power (scalar only)
// ============================================================================

void ScalarPower(const AlignedArray& a, scalar_t val, AlignedArray* out) {
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = std::pow(a.ptr[i], val);
  }
}

// ============================================================================
// Maximum
// ============================================================================

void EwiseMaximum(const AlignedArray& a, const AlignedArray& b, AlignedArray* out) {
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = (a.ptr[i] > b.ptr[i]) ? a.ptr[i] : b.ptr[i];
  }
}

void ScalarMaximum(const AlignedArray& a, scalar_t val, AlignedArray* out) {
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = (a.ptr[i] > val) ? a.ptr[i] : val;
  }
}

// ============================================================================
// Equality (returns 0.0 or 1.0)
// ============================================================================

void EwiseEq(const AlignedArray& a, const AlignedArray& b, AlignedArray* out) {
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = (a.ptr[i] == b.ptr[i]) ? 1.0f : 0.0f;
  }
}

void ScalarEq(const AlignedArray& a, scalar_t val, AlignedArray* out) {
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = (a.ptr[i] == val) ? 1.0f : 0.0f;
  }
}

// ============================================================================
// Greater or Equal (returns 0.0 or 1.0)
// ============================================================================

void EwiseGe(const AlignedArray& a, const AlignedArray& b, AlignedArray* out) {
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = (a.ptr[i] >= b.ptr[i]) ? 1.0f : 0.0f;
  }
}

void ScalarGe(const AlignedArray& a, scalar_t val, AlignedArray* out) {
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = (a.ptr[i] >= val) ? 1.0f : 0.0f;
  }
}

// ============================================================================
// Logarithm (elementwise only)
// ============================================================================

void EwiseLog(const AlignedArray& a, AlignedArray* out) {
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = std::log(a.ptr[i]);
  }
}

// ============================================================================
// Exponential (elementwise only)
// ============================================================================

void EwiseExp(const AlignedArray& a, AlignedArray* out) {
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = std::exp(a.ptr[i]);
  }
}

// ============================================================================
// Hyperbolic Tangent (elementwise only)
// ============================================================================

void EwiseTanh(const AlignedArray& a, AlignedArray* out) {
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = std::tanh(a.ptr[i]);
  }
}


void Matmul(const AlignedArray& a, const AlignedArray& b, AlignedArray* out, uint32_t m, uint32_t n,
            uint32_t p) {
  // Initialize output to zero
  for (uint32_t i = 0; i < m * p; i++) {
    out->ptr[i] = 0;
  }

  // Naive matrix multiplication
  for (uint32_t i = 0; i < m; i++) {
    for (uint32_t j = 0; j < p; j++) {
      for (uint32_t k = 0; k < n; k++) {
        out->ptr[i * p + j] += a.ptr[i * n + k] * b.ptr[k * p + j];
      }
    }
  }
}

inline void AlignedDot(const float* __restrict__ a,
                       const float* __restrict__ b,
                       float* __restrict__ out) {
  a = (const float*)__builtin_assume_aligned(a, TILE * ELEM_SIZE);
  b = (const float*)__builtin_assume_aligned(b, TILE * ELEM_SIZE);
  out = (float*)__builtin_assume_aligned(out, TILE * ELEM_SIZE);

  for (uint32_t i = 0; i < TILE; i++) {
    for (uint32_t j = 0; j < TILE; j++) {
      for (uint32_t k = 0; k < TILE; k++) {
        out[i * TILE + j] += a[i * TILE + k] * b[k * TILE + j];
      }
    }
  }
}

void MatmulTiled(const AlignedArray& a, const AlignedArray& b, AlignedArray* out, uint32_t m,
                 uint32_t n, uint32_t p) {
  uint32_t m_tiles = m / TILE;
  uint32_t n_tiles = n / TILE;
  uint32_t p_tiles = p / TILE;

  // Initialize output to zero
  for (uint32_t i = 0; i < m_tiles * p_tiles * TILE * TILE; i++) {
    out->ptr[i] = 0;
  }

  // Tile-based matrix multiplication
  for (uint32_t i = 0; i < m_tiles; i++) {
    for (uint32_t j = 0; j < p_tiles; j++) {
      for (uint32_t k = 0; k < n_tiles; k++) {
        // Get pointers to the tiles
        const float* a_tile = a.ptr + (i * n_tiles + k) * TILE * TILE;
        const float* b_tile = b.ptr + (k * p_tiles + j) * TILE * TILE;
        float* out_tile = out->ptr + (i * p_tiles + j) * TILE * TILE;
        
        // Multiply and accumulate
        AlignedDot(a_tile, b_tile, out_tile);
      }
    }
  }
}

void ReduceMax(const AlignedArray& a, AlignedArray* out, size_t reduce_size) {
  size_t out_size = a.size / reduce_size;
  
  for (size_t i = 0; i < out_size; i++) {
    scalar_t max_val = a.ptr[i * reduce_size];
    for (size_t j = 1; j < reduce_size; j++) {
      scalar_t val = a.ptr[i * reduce_size + j];
      if (val > max_val) {
        max_val = val;
      }
    }
    out->ptr[i] = max_val;
  }
}

void ReduceSum(const AlignedArray& a, AlignedArray* out, size_t reduce_size) {
  size_t out_size = a.size / reduce_size;
  
  for (size_t i = 0; i < out_size; i++) {
    scalar_t sum = 0.0f;
    for (size_t j = 0; j < reduce_size; j++) {
      sum += a.ptr[i * reduce_size + j];
    }
    out->ptr[i] = sum;
  }
}

// ============================================================================
// Block Sparse Attention (CPU)
// ============================================================================

void BlockSparseAttention(
    const AlignedArray& q,
    const AlignedArray& k, 
    const AlignedArray& v,
    AlignedArray* out,
    const std::vector<int>& sparse_blocks, // [num_rows, num_active, off0...offN, idx0...idxM]
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim
) {
    // Unpack Mask data (CSR Format)
    int num_rows = sparse_blocks[0];
    // int num_active = sparse_blocks[1]; // Not strictly needed for loop traversal
    
    // Offsets start at index 3
    // Indices start at index 3 + (num_rows + 1)
    const int* offsets = &sparse_blocks[3];
    const int* indices = &sparse_blocks[3 + num_rows + 1];
    
    // Block size (TILE) assumed to be passed or fixed. 
    // Usually passed as arg, but here we can infer or use TILE constant.
    // In CPU implementation, we can iterate element-wise or block-wise.
    int block_size = TILE; 
    
    float scale = 1.0f / std::sqrt((float)head_dim);

    // Iterate Batch
    for (int b = 0; b < batch_size; b++) {
        // Iterate Heads
        for (int h = 0; h < num_heads; h++) {
            
            int batch_head_offset = (b * num_heads + h) * (seq_len * head_dim);

            // Iterate Query Blocks
            for (int i_block = 0; i_block < num_rows; i_block++) {
                
                // Get range of Key blocks for this Query block
                int start = offsets[i_block];
                int end = offsets[i_block + 1];
                
                if (start == end) continue; // Skip empty blocks

                // Iterate over rows within the Query Block
                for (int i_in_block = 0; i_in_block < block_size; i_in_block++) {
                    int i_global = i_block * block_size + i_in_block;
                    if (i_global >= seq_len) break;

                    // Row-wise Online Softmax variables
                    float m_i = -std::numeric_limits<float>::infinity();
                    float l_i = 0.0f;
                    
                    // Temp accumulator for output row (size head_dim)
                    // We assume head_dim is small enough for stack, or alloc
                    std::vector<float> o_i(head_dim, 0.0f);

                    // Iterate Active Key Blocks
                    for (int kb = start; kb < end; kb++) {
                        int j_block = indices[kb];
                        
                        // Iterate rows in Key block (columns in Attn matrix)
                        for (int j_in_block = 0; j_in_block < block_size; j_in_block++) {
                            int j_global = j_block * block_size + j_in_block;
                            // Bounds check not strictly needed if padded, but good for safety
                            if (j_global >= seq_len) continue; 
                            
                            // Compute Score S_ij = Q[i] . K[j]
                            float score = 0.0f;
                            int q_offset = batch_head_offset + i_global * head_dim;
                            int k_offset = batch_head_offset + j_global * head_dim;
                            
                            for (int d = 0; d < head_dim; d++) {
                                score += q.ptr[q_offset + d] * k.ptr[k_offset + d];
                            }
                            score *= scale;

                            // Update Online Softmax stats
                            float m_prev = m_i;
                            m_i = std::max(m_prev, score);
                            float exp_score = std::exp(score - m_i);
                            float correction = std::exp(m_prev - m_i);
                            
                            l_i = l_i * correction + exp_score;
                            
                            // Load V[j] and accumulate
                            int v_offset = batch_head_offset + j_global * head_dim;
                            for (int d = 0; d < head_dim; d++) {
                                o_i[d] = o_i[d] * correction + exp_score * v.ptr[v_offset + d];
                            }
                        }
                    }
                    
                    // Final Write Back: Out = o_i / l_i
                    int out_row_offset = batch_head_offset + i_global * head_dim;
                    for (int d = 0; d < head_dim; d++) {
                        out->ptr[out_row_offset + d] = o_i[d] / l_i;
                    }
                }
            }
        }
    }
}


}  // namespace cpu
}  // namespace needle

PYBIND11_MODULE(ndarray_backend_cpu, m) {
  namespace py = pybind11;
  using namespace needle;
  using namespace cpu;

  m.attr("__device_name__") = "cpu";
  m.attr("__tile_size__") = TILE;

  py::class_<AlignedArray>(m, "Array")
      .def(py::init<size_t>(), py::return_value_policy::take_ownership)
      .def("ptr", &AlignedArray::ptr_as_int)
      .def_readonly("size", &AlignedArray::size);

  // return numpy array (with copying for simplicity, otherwise garbage
  // collection is a pain)
  m.def("to_numpy", [](const AlignedArray& a, std::vector<size_t> shape,
                       std::vector<size_t> strides, size_t offset) {
    std::vector<size_t> numpy_strides = strides;
    std::transform(numpy_strides.begin(), numpy_strides.end(), numpy_strides.begin(),
                   [](size_t& c) { return c * ELEM_SIZE; });
    return py::array_t<scalar_t>(shape, numpy_strides, a.ptr + offset);
  });

  // convert from numpy (with copying)
  m.def("from_numpy", [](py::array_t<scalar_t> a, AlignedArray* out) {
    std::memcpy(out->ptr, a.request().ptr, out->size * ELEM_SIZE);
  });

  m.def("fill", Fill);
  m.def("compact", Compact);
  m.def("ewise_setitem", EwiseSetitem);
  m.def("scalar_setitem", ScalarSetitem);
  m.def("ewise_add", EwiseAdd);
  m.def("scalar_add", ScalarAdd);

  m.def("ewise_mul", EwiseMul);
  m.def("scalar_mul", ScalarMul);
  m.def("ewise_div", EwiseDiv);
  m.def("scalar_div", ScalarDiv);
  m.def("scalar_power", ScalarPower);

  m.def("ewise_maximum", EwiseMaximum);
  m.def("scalar_maximum", ScalarMaximum);
  m.def("ewise_eq", EwiseEq);
  m.def("scalar_eq", ScalarEq);
  m.def("ewise_ge", EwiseGe);
  m.def("scalar_ge", ScalarGe);

  m.def("ewise_log", EwiseLog);
  m.def("ewise_exp", EwiseExp);
  m.def("ewise_tanh", EwiseTanh);

  m.def("matmul", Matmul);
  m.def("matmul_tiled", MatmulTiled);

  m.def("reduce_max", ReduceMax);
  m.def("reduce_sum", ReduceSum);
  
  // Register CPU Block Sparse Attention
  m.def("block_sparse_attention", BlockSparseAttention);
}

}