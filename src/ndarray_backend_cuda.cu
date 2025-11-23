#include <cuda_runtime.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <iostream>
#include <sstream>

namespace needle {
namespace cuda {

#define BASE_THREAD_NUM 256
#define TILE 16  // Increased tile size for better performance
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

////////////////////////////////////////////////////////////////////////////////
// Fill call
////////////////////////////////////////////////////////////////////////////////

__global__ void FillKernel(scalar_t* out, scalar_t val, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = val;
}

void Fill(CudaArray* out, scalar_t val) {
  CudaDims dim = CudaOneDim(out->size);
  FillKernel<<<dim.grid, dim.block>>>(out->ptr, val, out->size);
}

////////////////////////////////////////////////////////////////////////////////
// Compact and setitem cals
////////////////////////////////////////////////////////////////////////////////

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

////////////////////////////////////////////////////////////////////////////////
// Elementwise and scalar operations
////////////////////////////////////////////////////////////////////////////////

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

////////////////////////////////////////////////////////////////////////////////
// OPTIMIZED Matrix Multiplication with Shared Memory Tiling
////////////////////////////////////////////////////////////////////////////////

__global__ void MatmulTiledKernel(const scalar_t* a, const scalar_t* b, scalar_t* out,
                                   uint32_t M, uint32_t N, uint32_t P) {
  __shared__ scalar_t tile_a[TILE][TILE];
  __shared__ scalar_t tile_b[TILE][TILE];
  
  int row = blockIdx.y * TILE + threadIdx.y;
  int col = blockIdx.x * TILE + threadIdx.x;
  
  scalar_t sum = 0.0f;
  
  // Loop over tiles
  for (int t = 0; t < (N + TILE - 1) / TILE; t++) {
    // Load tile from A
    int a_col = t * TILE + threadIdx.x;
    if (row < M && a_col < N) {
      tile_a[threadIdx.y][threadIdx.x] = a[row * N + a_col];
    } else {
      tile_a[threadIdx.y][threadIdx.x] = 0.0f;
    }
    
    // Load tile from B
    int b_row = t * TILE + threadIdx.y;
    if (b_row < N && col < P) {
      tile_b[threadIdx.y][threadIdx.x] = b[b_row * P + col];
    } else {
      tile_b[threadIdx.y][threadIdx.x] = 0.0f;
    }
    
    __syncthreads();
    
    // Compute partial dot product
    #pragma unroll
    for (int k = 0; k < TILE; k++) {
      sum += tile_a[threadIdx.y][k] * tile_b[k][threadIdx.x];
    }
    
    __syncthreads();
  }
  
  // Write result
  if (row < M && col < P) {
    out[row * P + col] = sum;
  }
}

void Matmul(const CudaArray& a, const CudaArray& b, CudaArray* out, uint32_t M, uint32_t N,
            uint32_t P) {
  // Use tiled kernel for better performance
  dim3 block(TILE, TILE);
  dim3 grid((P + TILE - 1) / TILE, (M + TILE - 1) / TILE);
  MatmulTiledKernel<<<grid, block>>>(a.ptr, b.ptr, out->ptr, M, N, P);
}

////////////////////////////////////////////////////////////////////////////////
// Max and sum reductions (optimized with reduction patterns)
////////////////////////////////////////////////////////////////////////////////

__global__ void ReduceMaxKernel(const scalar_t* a, scalar_t* out, size_t reduce_size, size_t out_size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  
  if (gid < out_size) {
    scalar_t max_val = a[gid * reduce_size];
    for (size_t i = 1; i < reduce_size; i++) {
      scalar_t val = a[gid * reduce_size + i];
      max_val = fmaxf(max_val, val);
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
}
