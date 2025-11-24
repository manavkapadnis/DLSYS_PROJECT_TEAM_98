#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <algorithm>

namespace needle {
namespace cpu {

#define ALIGNMENT 256
#define TILE 8
typedef float scalar_t;
const size_t ELEM_SIZE = sizeof(scalar_t);

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
  for (int i = 0; i < out->size; i++) {
    out->ptr[i] = val;
  }
}

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

void ScalarPower(const AlignedArray& a, scalar_t val, AlignedArray* out) {
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = std::pow(a.ptr[i], val);
  }
}

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

void EwiseLog(const AlignedArray& a, AlignedArray* out) {
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = std::log(a.ptr[i]);
  }
}

void EwiseExp(const AlignedArray& a, AlignedArray* out) {
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = std::exp(a.ptr[i]);
  }
}

void EwiseTanh(const AlignedArray& a, AlignedArray* out) {
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = std::tanh(a.ptr[i]);
  }
}

void Matmul(const AlignedArray& a, const AlignedArray& b, AlignedArray* out, uint32_t m, uint32_t n,
            uint32_t p) {
  for (uint32_t i = 0; i < m * p; i++) {
    out->ptr[i] = 0;
  }

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

  for (uint32_t i = 0; i < m_tiles * p_tiles * TILE * TILE; i++) {
    out->ptr[i] = 0;
  }

  for (uint32_t i = 0; i < m_tiles; i++) {
    for (uint32_t j = 0; j < p_tiles; j++) {
      for (uint32_t k = 0; k < n_tiles; k++) {
        const float* a_tile = a.ptr + (i * n_tiles + k) * TILE * TILE;
        const float* b_tile = b.ptr + (k * p_tiles + j) * TILE * TILE;
        float* out_tile = out->ptr + (i * p_tiles + j) * TILE * TILE;
        
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

void BlockSparseAttention(const AlignedArray& q, const AlignedArray& k, const AlignedArray& v,
                         AlignedArray* out, const std::vector<int32_t>& metadata,
                         uint32_t batch_size, uint32_t num_heads, uint32_t seq_len, 
                         uint32_t head_dim) {
  /**
   * Block-sparse attention kernel
   * 
   * Args:
   *   q, k, v: (batch, heads, seq_len, head_dim) compact arrays
   *   out: output array same shape as q
   *   metadata: CSR format [n_blocks, num_active, ...offsets, ...indices]
   *   batch_size, num_heads, seq_len, head_dim: dimensions
   */
  
  // Parse metadata
  int32_t n_blocks = metadata[0];
  int32_t num_active = metadata[1];
  std::vector<int32_t> offsets(metadata.begin() + 2, metadata.begin() + 2 + n_blocks + 1);
  std::vector<int32_t> indices(metadata.begin() + 2 + n_blocks + 1, metadata.end());
  
  int32_t block_size = seq_len / n_blocks;
  scalar_t scale = 1.0f / std::sqrt((scalar_t)head_dim);
  
  // Initialize output to zero
  for (size_t i = 0; i < out->size; i++) {
    out->ptr[i] = 0.0f;
  }
  
  // Process each batch and head
  for (uint32_t b = 0; b < batch_size; b++) {
    for (uint32_t h = 0; h < num_heads; h++) {
      size_t base_offset = (b * num_heads + h) * seq_len * head_dim;
      
      // Process each query block
      for (int32_t q_block = 0; q_block < n_blocks; q_block++) {
        int32_t q_start = q_block * block_size;
        int32_t q_end = std::min((int32_t)seq_len, (q_block + 1) * block_size);
        
        // Temporary storage for attention scores
        std::vector<scalar_t> scores(block_size * seq_len, -1e10f);
        
        // Get active key blocks for this query block
        for (int32_t idx = offsets[q_block]; idx < offsets[q_block + 1]; idx++) {
          int32_t k_block = indices[idx];
          int32_t k_start = k_block * block_size;
          int32_t k_end = std::min((int32_t)seq_len, (k_block + 1) * block_size);
          
          // Compute scores: Q @ K^T for this block pair
          for (int32_t qi = q_start; qi < q_end; qi++) {
            for (int32_t ki = k_start; ki < k_end; ki++) {
              scalar_t dot = 0.0f;
              for (uint32_t d = 0; d < head_dim; d++) {
                dot += q.ptr[base_offset + qi * head_dim + d] * 
                       k.ptr[base_offset + ki * head_dim + d];
              }
              scores[(qi - q_start) * seq_len + ki] = dot * scale;
            }
          }
        }
        
        // Softmax over each query position
        for (int32_t qi = q_start; qi < q_end; qi++) {
          int local_qi = qi - q_start;
          
          // Find max for numerical stability
          scalar_t max_score = -1e10f;
          for (uint32_t ki = 0; ki < seq_len; ki++) {
            if (scores[local_qi * seq_len + ki] > max_score) {
              max_score = scores[local_qi * seq_len + ki];
            }
          }
          
          // Exp and sum
          scalar_t sum_exp = 0.0f;
          for (uint32_t ki = 0; ki < seq_len; ki++) {
            scalar_t val = scores[local_qi * seq_len + ki];
            if (val > -1e9f) {  // Only valid positions
              scores[local_qi * seq_len + ki] = std::exp(val - max_score);
              sum_exp += scores[local_qi * seq_len + ki];
            } else {
              scores[local_qi * seq_len + ki] = 0.0f;
            }
          }
          
          // Normalize
          if (sum_exp > 0) {
            for (uint32_t ki = 0; ki < seq_len; ki++) {
              scores[local_qi * seq_len + ki] /= sum_exp;
            }
          }
          
          // Weighted sum of values
          for (uint32_t d = 0; d < head_dim; d++) {
            scalar_t sum = 0.0f;
            for (uint32_t ki = 0; ki < seq_len; ki++) {
              sum += scores[local_qi * seq_len + ki] * v.ptr[base_offset + ki * head_dim + d];
            }
            out->ptr[base_offset + qi * head_dim + d] = sum;
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

  m.def("to_numpy", [](const AlignedArray& a, std::vector<size_t> shape,
                       std::vector<size_t> strides, size_t offset) {
    std::vector<size_t> numpy_strides = strides;
    std::transform(numpy_strides.begin(), numpy_strides.end(), numpy_strides.begin(),
                   [](size_t& c) { return c * ELEM_SIZE; });
    return py::array_t<scalar_t>(shape, numpy_strides, a.ptr + offset);
  });

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
  m.def("block_sparse_attention", BlockSparseAttention);
}