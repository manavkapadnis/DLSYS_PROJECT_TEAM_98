#include <cuda_runtime.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <iostream>
#include <sstream>

namespace needle {
namespace cuda {

#define BASE_THREAD_NUM 256
#define TILE 4
typedef float scalar_t;
const size_t ELEM_SIZE = sizeof(scalar_t);

struct CudaArray {
  CudaArray(const size_t size) {
    cudaError_t err = cudaMalloc(&ptr, size * ELEM_SIZE);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
    this->size = size;
  }
  ~CudaArray() { cudaFree(ptr); }
  size_t ptr_as_int() { return (size_t)ptr; }
  
  scalar_t* ptr;
  size_t size;
};

struct CudaDims {
  dim3 block, grid;
};

CudaDims CudaOneDim(size_t size) {
  CudaDims dim;
  size_t num_blocks = (size + BASE_THREAD_NUM - 1) / BASE_THREAD_NUM;
  dim.block = dim3(BASE_THREAD_NUM, 1, 1);
  dim.grid = dim3(num_blocks, 1, 1);
  return dim;
}

#define MAX_VEC_SIZE 8
struct CudaVec {
  uint32_t size;
  int32_t data[MAX_VEC_SIZE];
};

CudaVec VecToCuda(const std::vector<int32_t>& x) {
  CudaVec shape;
  if (x.size() > MAX_VEC_SIZE) throw std::runtime_error("Exceeded CUDA supported max dimesions");
  shape.size = x.size();
  for (size_t i = 0; i < x.size(); i++) {
    shape.data[i] = x[i];
  }
  return shape;
}

__global__ void FillKernel(scalar_t* out, scalar_t val, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = val;
}

void Fill(CudaArray* out, scalar_t val) {
  CudaDims dim = CudaOneDim(out->size);
  FillKernel<<<dim.grid, dim.block>>>(out->ptr, val, out->size);
}

__global__ void CompactKernel(const scalar_t* a, scalar_t* out, size_t size, CudaVec shape,
                              CudaVec strides, size_t offset) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  
  if (gid < size) {
    size_t idx_offset = 0;
    size_t remaining = gid;
    
    for (int i = shape.size - 1; i >= 0; i--) {
      size_t coord = remaining % shape.data[i];
      remaining /= shape.data[i];
      idx_offset += coord * strides.data[i];
    }
    
    out[gid] = a[offset + idx_offset];
  }
}

void Compact(const CudaArray& a, CudaArray* out, std::vector<int32_t> shape,
             std::vector<int32_t> strides, size_t offset) {
  CudaDims dim = CudaOneDim(out->size);
  CompactKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size, VecToCuda(shape),
                                         VecToCuda(strides), offset);
}

__global__ void EwiseSetitemKernel(const scalar_t* a, scalar_t* out, size_t size,
                                   CudaVec shape, CudaVec strides, size_t offset) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  
  if (gid < size) {
    size_t idx_offset = 0;
    size_t remaining = gid;
    
    for (int i = shape.size - 1; i >= 0; i--) {
      size_t coord = remaining % shape.data[i];
      remaining /= shape.data[i];
      idx_offset += coord * strides.data[i];
    }
    
    out[offset + idx_offset] = a[gid];
  }
}

void EwiseSetitem(const CudaArray& a, CudaArray* out, std::vector<int32_t> shape,
                  std::vector<int32_t> strides, size_t offset) {
  CudaDims dim = CudaOneDim(a.size);
  EwiseSetitemKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, a.size,
                                               VecToCuda(shape), VecToCuda(strides), offset);
}

__global__ void ScalarSetitemKernel(scalar_t val, scalar_t* out, size_t size,
                                    CudaVec shape, CudaVec strides, size_t offset) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  
  if (gid < size) {
    size_t idx_offset = 0;
    size_t remaining = gid;
    
    for (int i = shape.size - 1; i >= 0; i--) {
      size_t coord = remaining % shape.data[i];
      remaining /= shape.data[i];
      idx_offset += coord * strides.data[i];
    }
    
    out[offset + idx_offset] = val;
  }
}

void ScalarSetitem(size_t size, scalar_t val, CudaArray* out, std::vector<int32_t> shape,
                   std::vector<int32_t> strides, size_t offset) {
  CudaDims dim = CudaOneDim(size);
  ScalarSetitemKernel<<<dim.grid, dim.block>>>(val, out->ptr, size,
                                                VecToCuda(shape), VecToCuda(strides), offset);
}

__global__ void EwiseAddKernel(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = a[gid] + b[gid];
}

void EwiseAdd(const CudaArray& a, const CudaArray& b, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);
  EwiseAddKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size);
}

__global__ void ScalarAddKernel(const scalar_t* a, scalar_t val, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = a[gid] + val;
}

void ScalarAdd(const CudaArray& a, scalar_t val, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);
  ScalarAddKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);
}

__global__ void EwiseMulKernel(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = a[gid] * b[gid];
}

void EwiseMul(const CudaArray& a, const CudaArray& b, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);
  EwiseMulKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size);
}

__global__ void ScalarMulKernel(const scalar_t* a, scalar_t val, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = a[gid] * val;
}

void ScalarMul(const CudaArray& a, scalar_t val, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);
  ScalarMulKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);
}

__global__ void EwiseDivKernel(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = a[gid] / b[gid];
}

void EwiseDiv(const CudaArray& a, const CudaArray& b, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);
  EwiseDivKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size);
}

__global__ void ScalarDivKernel(const scalar_t* a, scalar_t val, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = a[gid] / val;
}

void ScalarDiv(const CudaArray& a, scalar_t val, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);
  ScalarDivKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);
}

__global__ void ScalarPowerKernel(const scalar_t* a, scalar_t val, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = pow(a[gid], val);
}

void ScalarPower(const CudaArray& a, scalar_t val, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);
  ScalarPowerKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);
}

__global__ void EwiseMaximumKernel(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = (a[gid] > b[gid]) ? a[gid] : b[gid];
}

void EwiseMaximum(const CudaArray& a, const CudaArray& b, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);
  EwiseMaximumKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size);
}

__global__ void ScalarMaximumKernel(const scalar_t* a, scalar_t val, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = (a[gid] > val) ? a[gid] : val;
}

void ScalarMaximum(const CudaArray& a, scalar_t val, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);
  ScalarMaximumKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);
}

__global__ void EwiseEqKernel(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = (a[gid] == b[gid]) ? 1.0f : 0.0f;
}

void EwiseEq(const CudaArray& a, const CudaArray& b, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);
  EwiseEqKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size);
}

__global__ void ScalarEqKernel(const scalar_t* a, scalar_t val, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = (a[gid] == val) ? 1.0f : 0.0f;
}

void ScalarEq(const CudaArray& a, scalar_t val, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);
  ScalarEqKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);
}

__global__ void EwiseGeKernel(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = (a[gid] >= b[gid]) ? 1.0f : 0.0f;
}

void EwiseGe(const CudaArray& a, const CudaArray& b, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);
  EwiseGeKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size);
}

__global__ void ScalarGeKernel(const scalar_t* a, scalar_t val, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = (a[gid] >= val) ? 1.0f : 0.0f;
}

void ScalarGe(const CudaArray& a, scalar_t val, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);
  ScalarGeKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);
}

__global__ void EwiseLogKernel(const scalar_t* a, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = log(a[gid]);
}

void EwiseLog(const CudaArray& a, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);
  EwiseLogKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size);
}

__global__ void EwiseExpKernel(const scalar_t* a, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = exp(a[gid]);
}

void EwiseExp(const CudaArray& a, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);
  EwiseExpKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size);
}

__global__ void EwiseTanhKernel(const scalar_t* a, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = tanh(a[gid]);
}

void EwiseTanh(const CudaArray& a, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);
  EwiseTanhKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size);
}

__global__ void MatmulKernel(const scalar_t* a, const scalar_t* b, scalar_t* out,
                             uint32_t M, uint32_t N, uint32_t P) {
  size_t row = blockIdx.y * blockDim.y + threadIdx.y;
  size_t col = blockIdx.x * blockDim.x + threadIdx.x;
  
  if (row < M && col < P) {
    scalar_t sum = 0.0f;
    for (uint32_t k = 0; k < N; k++) {
      sum += a[row * N + k] * b[k * P + col];
    }
    out[row * P + col] = sum;
  }
}

void Matmul(const CudaArray& a, const CudaArray& b, CudaArray* out, uint32_t M, uint32_t N,
            uint32_t P) {
  dim3 block(16, 16);
  dim3 grid((P + block.x - 1) / block.x, (M + block.y - 1) / block.y);
  MatmulKernel<<<grid, block>>>(a.ptr, b.ptr, out->ptr, M, N, P);
}

__global__ void ReduceMaxKernel(const scalar_t* a, scalar_t* out, size_t reduce_size, size_t out_size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  
  if (gid < out_size) {
    scalar_t max_val = a[gid * reduce_size];
    for (size_t i = 1; i < reduce_size; i++) {
      scalar_t val = a[gid * reduce_size + i];
      if (val > max_val) {
        max_val = val;
      }
    }
    out[gid] = max_val;
  }
}

void ReduceMax(const CudaArray& a, CudaArray* out, size_t reduce_size) {
  size_t out_size = a.size / reduce_size;
  CudaDims dim = CudaOneDim(out_size);
  ReduceMaxKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, reduce_size, out_size);
}

__global__ void ReduceSumKernel(const scalar_t* a, scalar_t* out, size_t reduce_size, size_t out_size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  
  if (gid < out_size) {
    scalar_t sum = 0.0f;
    for (size_t i = 0; i < reduce_size; i++) {
      sum += a[gid * reduce_size + i];
    }
    out[gid] = sum;
  }
}

void ReduceSum(const CudaArray& a, CudaArray* out, size_t reduce_size) {
  size_t out_size = a.size / reduce_size;
  CudaDims dim = CudaOneDim(out_size);
  ReduceSumKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, reduce_size, out_size);
}

__global__ void BlockSparseAttentionKernel(
    const scalar_t* q, const scalar_t* k, const scalar_t* v, scalar_t* out,
    const int32_t* offsets, const int32_t* indices,
    int32_t n_blocks, int32_t block_size,
    uint32_t batch_size, uint32_t num_heads, uint32_t seq_len, uint32_t head_dim) {
  
  uint32_t batch_head_idx = blockIdx.x;
  if (batch_head_idx >= batch_size * num_heads) return;
  
  uint32_t b = batch_head_idx / num_heads;
  uint32_t h = batch_head_idx % num_heads;
  
  uint32_t qi = blockIdx.y * blockDim.y + threadIdx.y;
  if (qi >= seq_len) return;
  
  size_t base_offset = (b * num_heads + h) * seq_len * head_dim;
  scalar_t scale = 1.0f / sqrtf((scalar_t)head_dim);
  
  int32_t q_block = qi / block_size;
  
  // Compute attention scores for this query position
  extern __shared__ scalar_t shared_mem[];
  scalar_t* scores = shared_mem + threadIdx.y * seq_len;
  
  // Initialize scores to -inf
  for (uint32_t i = threadIdx.x; i < seq_len; i += blockDim.x) {
    scores[i] = -1e10f;
  }
  __syncthreads();
  
  // Compute scores for active blocks only
  for (int32_t idx = offsets[q_block]; idx < offsets[q_block + 1]; idx++) {
    int32_t k_block = indices[idx];
    int32_t k_start = k_block * block_size;
    int32_t k_end = min((int32_t)seq_len, (k_block + 1) * block_size);
    
    for (int32_t ki = k_start + threadIdx.x; ki < k_end; ki += blockDim.x) {
      scalar_t dot = 0.0f;
      for (uint32_t d = 0; d < head_dim; d++) {
        dot += q[base_offset + qi * head_dim + d] * 
               k[base_offset + ki * head_dim + d];
      }
      scores[ki] = dot * scale;
    }
  }
  __syncthreads();
  
  // Softmax: find max
  __shared__ scalar_t max_score[32];  // Assume blockDim.y <= 32
  scalar_t local_max = -1e10f;
  for (uint32_t i = 0; i < seq_len; i++) {
    if (scores[i] > local_max) {
      local_max = scores[i];
    }
  }
  if (threadIdx.x == 0) {
    max_score[threadIdx.y] = local_max;
  }
  __syncthreads();
  
  // Softmax: exp and sum
  scalar_t sum_exp = 0.0f;
  for (uint32_t i = threadIdx.x; i < seq_len; i += blockDim.x) {
    if (scores[i] > -1e9f) {
      scores[i] = expf(scores[i] - max_score[threadIdx.y]);
      sum_exp += scores[i];
    } else {
      scores[i] = 0.0f;
    }
  }
  
  // Reduce sum across threads
  __shared__ scalar_t sum_buffer[32];
  sum_buffer[threadIdx.x] = sum_exp;
  __syncthreads();
  
  if (threadIdx.x == 0) {
    scalar_t total = 0.0f;
    for (int i = 0; i < blockDim.x && i < 32; i++) {
      total += sum_buffer[i];
    }
    sum_buffer[0] = total;
  }
  __syncthreads();
  
  // Normalize
  scalar_t sum_total = sum_buffer[0];
  if (sum_total > 0) {
    for (uint32_t i = threadIdx.x; i < seq_len; i += blockDim.x) {
      scores[i] /= sum_total;
    }
  }
  __syncthreads();
  
  // Weighted sum of values
  if (threadIdx.x == 0) {
    for (uint32_t d = 0; d < head_dim; d++) {
      scalar_t sum = 0.0f;
      for (uint32_t ki = 0; ki < seq_len; ki++) {
        sum += scores[ki] * v[base_offset + ki * head_dim + d];
      }
      out[base_offset + qi * head_dim + d] = sum;
    }
  }
}

void BlockSparseAttention(const CudaArray& q, const CudaArray& k, const CudaArray& v,
                         CudaArray* out, const std::vector<int32_t>& metadata,
                         uint32_t batch_size, uint32_t num_heads, uint32_t seq_len, 
                         uint32_t head_dim) {
  
  // Parse metadata
  int32_t n_blocks = metadata[0];
  int32_t num_active = metadata[1];
  
  // Copy metadata to device
  int32_t* d_offsets;
  int32_t* d_indices;
  
  cudaMalloc(&d_offsets, (n_blocks + 1) * sizeof(int32_t));
  cudaMalloc(&d_indices, num_active * sizeof(int32_t));
  
  cudaMemcpy(d_offsets, metadata.data() + 2, (n_blocks + 1) * sizeof(int32_t), cudaMemcpyHostToDevice);
  cudaMemcpy(d_indices, metadata.data() + 2 + n_blocks + 1, num_active * sizeof(int32_t), cudaMemcpyHostToDevice);
  
  int32_t block_size = seq_len / n_blocks;
  
  // Launch kernel
  dim3 block(32, 8);  // 32 threads for reductions, 8 for queries
  dim3 grid(batch_size * num_heads, (seq_len + block.y - 1) / block.y);
  
  size_t shared_mem_size = block.y * seq_len * sizeof(scalar_t);
  
  BlockSparseAttentionKernel<<<grid, block, shared_mem_size>>>(
      q.ptr, k.ptr, v.ptr, out->ptr,
      d_offsets, d_indices,
      n_blocks, block_size,
      batch_size, num_heads, seq_len, head_dim
  );
  
  cudaFree(d_offsets);
  cudaFree(d_indices);
}

}  // namespace cuda
}  // namespace needle

PYBIND11_MODULE(ndarray_backend_cuda, m) {
  namespace py = pybind11;
  using namespace needle;
  using namespace cuda;

  m.attr("__device_name__") = "cuda";
  m.attr("__tile_size__") = TILE;

  py::class_<CudaArray>(m, "Array")
      .def(py::init<size_t>(), py::return_value_policy::take_ownership)
      .def_readonly("size", &CudaArray::size)
      .def("ptr", &CudaArray::ptr_as_int);

  m.def("to_numpy", [](const CudaArray& a, std::vector<size_t> shape, std::vector<size_t> strides,
                       size_t offset) {
    std::vector<size_t> numpy_strides = strides;
    std::transform(numpy_strides.begin(), numpy_strides.end(), numpy_strides.begin(),
                   [](size_t& c) { return c * ELEM_SIZE; });

    scalar_t* host_ptr = (scalar_t*)std::malloc(a.size * ELEM_SIZE);
    if (host_ptr == 0) throw std::bad_alloc();
    cudaError_t err = cudaMemcpy(host_ptr, a.ptr, a.size * ELEM_SIZE, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));

    py::capsule deallocate_buffer(host_ptr, [](void* p) { free(p); });
    return py::array_t<scalar_t>(shape, numpy_strides, host_ptr + offset, deallocate_buffer);
  });

  m.def("from_numpy", [](py::array_t<scalar_t> a, CudaArray* out) {
    cudaError_t err =
        cudaMemcpy(out->ptr, a.request().ptr, out->size * ELEM_SIZE, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
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
  m.def("reduce_max", ReduceMax);
  m.def("reduce_sum", ReduceSum);
  m.def("block_sparse_attention", BlockSparseAttention);
}
