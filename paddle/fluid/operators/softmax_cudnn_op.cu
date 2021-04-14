/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/math/math_cuda_utils.h"
#include "paddle/fluid/operators/softmax_op.h"
#include "paddle/fluid/platform/cuda_device_function.h"
#ifdef PADDLE_WITH_HIP
#include "paddle/fluid/platform/miopen_helper.h"
#else
#include "paddle/fluid/platform/cudnn_helper.h"
#endif
#include "paddle/fluid/platform/gpu_launch_config.h"

namespace paddle {
namespace platform {
struct CUDAPlace;
struct float16;
}  // namespace platform
}  // namespace paddle

namespace paddle {
namespace operators {

using ScopedTensorDescriptor = platform::ScopedTensorDescriptor;
using DataLayout = platform::DataLayout;
using Tensor = framework::Tensor;

// AccType: the compute type
// to improve result accuracy
template <typename T>
struct GetAccType {
  using type = T;
};

template <>
struct GetAccType<platform::float16> {
  using type = float;
};

// VecType: Vectorized data type
// to achieve maximum memory throughput
template <typename T, int N>
struct GetVecType;

template <typename T>
struct GetVecType<T, 1> {
  using type = T;
};

template <>
struct GetVecType<platform::float16, 2> {
  using type = half2;
};

template <>
struct GetVecType<platform::float16, 4> {
  using type = float2;
};

template <>
struct GetVecType<float, 2> {
  using type = float2;
};

template <>
struct GetVecType<float, 4> {
  using type = float4;
};

template <>
struct GetVecType<double, 2> {
  using type = double2;
};

template <>
struct GetVecType<double, 4> {
  using type = double4;
};

constexpr int MAX_BLOCK_DIM = 1024;

#define LAUNCH_SOFTMAX_WARP_FORWARD(Log2Elements)                  \
  case Log2Elements:                                               \
    WarpSoftmaxForward<T, float, Log2Elements><<<                  \
        blocks, threads, 0, ctx.cuda_device_context().stream()>>>( \
        out_data, x->data<T>(), N, dim, dim);                      \
    break;

#define LAUNCH_SOFTMAX_WARP_BACKWARD(Log2Elements)                 \
  case Log2Elements:                                               \
    softmax_warp_backward<T, float, Log2Elements><<<               \
        blocks, threads, 0, ctx.cuda_device_context().stream()>>>( \
        dx_data, mul_grad.data<T>(), out->data<T>(), N, dim, dim); \
    break;

static inline int SizeOutAxis(const int axis, DDim dims) {
  int size = 1;
  for (int i = axis + 1; i < dims.size(); i++) {
    size *= dims[i];
  }
  return size;
}

int log2_ceil(int value) {
  int log2_value = 0;
  while ((1 << log2_value) < value) ++log2_value;
  return log2_value;
}

template <typename T, int VLEN>
union vec_t {
  static_assert(sizeof(T) == -1, "vec_t is only available by specialization.");
};

template <>
union vec_t<float, 4> {
  float4 s;
  float v[4];
};

template <>
union vec_t<platform::float16, 4> {
  int2 s;
  platform::float16 v[4];
};

template <typename T, typename VECT, int VPT, int WARP_PER_BLOCK>
__global__ void VecSoftmaxForward(T* dst, const T* src, const int batch_size,
                                  const int softmax_ele) {
  int offset = blockIdx.x * softmax_ele * WARP_PER_BLOCK;
  int idx = threadIdx.x * VPT;

  VECT buf = reinterpret_cast<const VECT*>(&src[offset + idx])[0];
  T* bufp = reinterpret_cast<T*>(&buf);
  float4 val4;
  float* val4p = reinterpret_cast<float*>(&val4);
  for (int i = 0; i < VPT; ++i) {
    val4p[i] = static_cast<float>(bufp[i]);
  }
  float val = val4.x + val4.y + val4.z + val4.w;
  float max_val = math::warpReduceMax<float>(
      max(max(val4.x, val4.y), max(val4.z, val4.w)), 0xffffffff);
  float4 tmp4 = make_float4(__expf(val4.x - max_val), __expf(val4.y - max_val),
                            __expf(val4.z - max_val), __expf(val4.w - max_val));
  float* tmp4p = reinterpret_cast<float*>(&tmp4);
  float invsum = 1.f / (math::warpReduceSum<float>(
                            tmp4.x + tmp4.y + tmp4.z + tmp4.w, 0xffffffff) +
                        1e-6f);
  for (int i = 0; i < VPT; ++i) {
    bufp[i] = static_cast<T>(tmp4p[i] * invsum);
  }
  reinterpret_cast<VECT*>(&dst[offset + idx])[0] = buf;
}

template <typename T, int WARP_BATCH, int WARP_SIZE_SOFTMAX>
__device__ __forceinline__ void warp_reduce_sum(T* sum) {
#pragma unroll
  for (int offset = WARP_SIZE_SOFTMAX / 2; offset > 0; offset /= 2) {
#pragma unroll
    for (int i = 0; i < WARP_BATCH; ++i) {
      T sum_val = platform::CudaShuffleXorSync(0xFFFFFFFF, sum[i], offset);
      sum[i] = sum[i] + sum_val;
    }
  }
}

template <typename T, int WARP_BATCH, int WARP_SIZE_SOFTMAX>
__device__ __forceinline__ void warp_reduce_max(T* sum) {
#pragma unroll
  for (int offset = WARP_SIZE_SOFTMAX / 2; offset > 0; offset /= 2) {
#pragma unroll
    for (int i = 0; i < WARP_BATCH; ++i) {
      T max_val = platform::CudaShuffleXorSync(0xFFFFFFFF, sum[i], offset);
      sum[i] = max(sum[i], max_val);
    }
  }
}

template <typename T, typename AccT, int Log2Elements>
__global__ void WarpSoftmaxForward(T* dst, const T* src, const int batch_size,
                                   const int stride, const int element_count) {
  constexpr int next_power_of_two = 1 << Log2Elements;
  constexpr int warp_size_softmax =
      (next_power_of_two < 32) ? next_power_of_two : 32;
  constexpr int WARP_ITERATIONS = next_power_of_two / warp_size_softmax;
  constexpr int WARP_BATCH = (next_power_of_two <= 128) ? 2 : 1;

  int first_batch = (blockDim.y * blockIdx.x + threadIdx.y) * WARP_BATCH;

  int local_batches = batch_size - first_batch;
  if (local_batches > WARP_BATCH) {
    local_batches = WARP_BATCH;
  }

  int local_idx = threadIdx.x;

  src += first_batch * stride + local_idx;
  dst += first_batch * stride + local_idx;

  // load data from global memory
  AccT elements[WARP_BATCH][WARP_ITERATIONS];
  for (int i = 0; i < WARP_BATCH; ++i) {
    int batch_element_count = (i >= local_batches) ? 0 : element_count;
    for (int it = 0; it < WARP_ITERATIONS; ++it) {
      int element_index = local_idx + it * warp_size_softmax;
      if (element_index < batch_element_count) {
        elements[i][it] =
            static_cast<float>(src[i * element_count + it * warp_size_softmax]);
      } else {
        elements[i][it] = -std::numeric_limits<AccT>::infinity();
      }
    }
  }

  // compute max_value
  AccT max_value[WARP_BATCH];
#pragma unroll
  for (int i = 0; i < WARP_BATCH; ++i) {
    max_value[i] = elements[i][0];
#pragma unroll
    for (int it = 1; it < WARP_ITERATIONS; ++it) {
      max_value[i] =
          (max_value[i] > elements[i][it]) ? max_value[i] : elements[i][it];
    }
  }
  warp_reduce_max<AccT, WARP_BATCH, warp_size_softmax>(max_value);

  AccT sum[WARP_BATCH]{0.0f};
#pragma unroll
  for (int i = 0; i < WARP_BATCH; ++i) {
#pragma unroll
    for (int it = 0; it < WARP_ITERATIONS; ++it) {
      elements[i][it] = (std::exp((elements[i][it] - max_value[i])));
      sum[i] += elements[i][it];
    }
  }
  warp_reduce_sum<AccT, WARP_BATCH, warp_size_softmax>(sum);

// store result
#pragma unroll
  for (int i = 0; i < WARP_BATCH; ++i) {
    if (i >= local_batches) break;
#pragma unroll
    for (int it = 0; it < WARP_ITERATIONS; ++it) {
      int element_index = local_idx + it * warp_size_softmax;
      if (element_index < element_count) {
        dst[i * element_count + it * warp_size_softmax] =
            elements[i][it] / sum[i];
      } else {
        break;
      }
    }
  }
}

template <typename T, typename AccT, int Log2Elements>
__global__ void softmax_warp_backward(T* gradInput, const T* grad,
                                      const T* output, int batch_size,
                                      int stride, int element_count) {
  constexpr int next_power_of_two = 1 << Log2Elements;
  constexpr int warp_size_softmax =
      (next_power_of_two < 32) ? next_power_of_two : 32;
  constexpr int WARP_ITERATIONS = next_power_of_two / warp_size_softmax;
  constexpr int WARP_BATCH = (next_power_of_two <= 128) ? 2 : 1;

  int first_batch = (blockDim.y * blockIdx.x + threadIdx.y) * WARP_BATCH;

  int local_batches = batch_size - first_batch;
  if (local_batches > WARP_BATCH) {
    local_batches = WARP_BATCH;
  }

  int local_idx = threadIdx.x % warp_size_softmax;

  int thread_offset = first_batch * stride + local_idx;
  grad += thread_offset;
  output += thread_offset;
  gradInput += thread_offset;

  // load data from global memory
  AccT grad_reg[WARP_BATCH][WARP_ITERATIONS];
  AccT output_reg[WARP_BATCH][WARP_ITERATIONS];
  for (int i = 0; i < WARP_BATCH; ++i) {
    int batch_element_count = (i >= local_batches) ? 0 : element_count;
    for (int it = 0; it < WARP_ITERATIONS; ++it) {
      int element_index = local_idx + it * warp_size_softmax;
      if (element_index < batch_element_count) {
        grad_reg[i][it] =
            static_cast<AccT>(grad[i * element_count + it * warp_size_softmax]);
        output_reg[i][it] = static_cast<AccT>(
            output[i * element_count + it * warp_size_softmax]);
      } else {
        grad_reg[i][it] = AccT(0);
        output_reg[i][it] = AccT(0);
      }
    }
  }

  AccT sum[WARP_BATCH];
#pragma unroll
  for (int i = 0; i < WARP_BATCH; ++i) {
    sum[i] = grad_reg[i][0];
#pragma unroll
    for (int it = 1; it < WARP_ITERATIONS; ++it) {
      sum[i] += grad_reg[i][it];
    }
  }
  warp_reduce_sum<AccT, WARP_BATCH, warp_size_softmax>(sum);

// store result
#pragma unroll
  for (int i = 0; i < WARP_BATCH; ++i) {
    if (i >= local_batches) break;
#pragma unroll
    for (int it = 0; it < WARP_ITERATIONS; ++it) {
      int element_index = local_idx + it * warp_size_softmax;
      if (element_index < element_count) {
        // compute gradients
        gradInput[i * element_count + it * warp_size_softmax] =
            (grad_reg[i][it] - output_reg[i][it] * sum[i]);
      }
    }
  }
}

template <typename T>
__global__ void MultiplyCUDAKernel(T* C, const T* A, const T* B, int N) {
  CUDA_KERNEL_LOOP(i, N) {
    C[i] = static_cast<T>(static_cast<float>(A[i]) * static_cast<float>(B[i]));
  }
}

template <typename T, int VPT, int WARP_PER_BLOCK>
__global__ void VecSoftmaxBackward(T* dst, const T* grad, const T* src,
                                   const int batch_size,
                                   const int softmax_ele) {
  const int offset =
      blockIdx.x * softmax_ele * WARP_PER_BLOCK + threadIdx.x * VPT;

  float local_sum_gy = 0.f;
  vec_t<T, VPT> local_grad;
  vec_t<T, VPT> local_src;

  local_grad.s =
      reinterpret_cast<const decltype(local_grad.s)*>(&grad[offset])[0];
  local_src.s = reinterpret_cast<const decltype(local_src.s)*>(&src[offset])[0];

  for (int i = 0; i < VPT; ++i) {
    local_sum_gy += static_cast<float>(local_grad.v[i]) *
                    static_cast<float>(local_src.v[i]);
  }
  float sum_gy = math::warpReduceSum<float>(local_sum_gy, 0xffffffff);

  vec_t<T, VPT> local_dst;
  for (int i = 0; i < VPT; ++i) {
    local_dst.v[i] =
        static_cast<T>(static_cast<float>(local_src.v[i]) *
                       (static_cast<float>(local_grad.v[i]) - sum_gy));
  }
  reinterpret_cast<decltype(local_dst.s)*>(&dst[offset])[0] = local_dst.s;
}

// When D is larg and dim is small:
// Each block arranged by N，each thread arranged by D
// each thread compute dim number's softmax
template <typename T, typename AccT, int kVecSize>
__global__ void KeLoopDimSoftmaxBackward(T* __restrict__ dx,
                                         const T* __restrict__ out,
                                         const T* __restrict__ dout,
                                         const int N, const int dim,
                                         const int D) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int vec_id = tid * kVecSize;
  const int out_id = vec_id / D;
  if (out_id >= N) return;
  const int in_id = vec_id - out_id * D;

  // vectorization for global memory coalescing
  using VecT = typename GetVecType<T, kVecSize>::type;
  VecT vec_out, vec_dout, vec_dx;
  T* buf_out = reinterpret_cast<T*>(&vec_out);
  T* buf_dout = reinterpret_cast<T*>(&vec_dout);
  T* buf_dx = reinterpret_cast<T*>(&vec_dx);

  const int offset = out_id * dim * D + in_id;
  const T* __restrict__ out_row = out + offset;
  const T* __restrict__ dout_row = dout + offset;
  T* __restrict__ dx_row = dx + offset;

  // compute multiply then sum value
  AccT sum_val[kVecSize]{0};
  for (int dim_id = 0; dim_id < dim; dim_id++) {
    vec_out = reinterpret_cast<const VecT*>(&out_row[dim_id * D])[0];
    vec_dout = reinterpret_cast<const VecT*>(&dout_row[dim_id * D])[0];
#pragma unroll
    for (int i = 0; i < kVecSize; i++) {
      sum_val[i] +=
          static_cast<AccT>(buf_out[i]) * static_cast<AccT>(buf_dout[i]);
    }
  }

  // compute softmax gradient value
  for (int dim_id = 0; dim_id < dim; dim_id++) {
    vec_out = reinterpret_cast<const VecT*>(&out_row[dim_id * D])[0];
    vec_dout = reinterpret_cast<const VecT*>(&dout_row[dim_id * D])[0];
#pragma unroll
    for (int i = 0; i < kVecSize; i++) {
      buf_dx[i] = static_cast<T>(static_cast<AccT>(buf_out[i]) *
                                 (static_cast<AccT>(buf_dout[i]) - sum_val[i]));
    }
    reinterpret_cast<VecT*>(&dx_row[dim_id * D])[0] = vec_dx;
  }
}

template <typename T, int kVecSize>
inline void LaunchLoopDimSoftmaxBackwardKernel(const gpuStream_t& stream,
                                               T* dx_data, const T* out_data,
                                               const T* dout_data, const int N,
                                               const int dim, const int D) {
  int loop_num = N * D / kVecSize;
  int threads = std::min(loop_num, MAX_BLOCK_DIM);
  int grids = (loop_num + threads - 1) / threads;
  using AccT = typename GetAccType<T>::type;

  KeLoopDimSoftmaxBackward<T, AccT, kVecSize><<<grids, threads, 0, stream>>>(
      dx_data, out_data, dout_data, N, dim, D);
}

template <typename T>
inline void LaunchLoopDimSoftmaxBackward(const gpuStream_t& stream, T* dx_data,
                                         const T* out_data, const T* dout_data,
                                         const int N, const int dim,
                                         const int D, const int sm_count) {
  // ensure occupancy
  const int num = N * D / (MAX_BLOCK_DIM * 4);
  if (D % 4 == 0 && num / 4 >= sm_count) {
    LaunchLoopDimSoftmaxBackwardKernel<T, 4>(stream, dx_data, out_data,
                                             dout_data, N, dim, D);
  } else if (D % 2 == 0 && num / 2 >= sm_count) {
    LaunchLoopDimSoftmaxBackwardKernel<T, 2>(stream, dx_data, out_data,
                                             dout_data, N, dim, D);
  } else {
    LaunchLoopDimSoftmaxBackwardKernel<T, 1>(stream, dx_data, out_data,
                                             dout_data, N, dim, D);
  }
}

// When D is small and (dim * D) is larger
// Each block arranged by N，each thread arranged by dim * D
// each block compute (dim * D) number's softmax
template <typename T, typename AccT, int kVecSize>
__global__ void KeSpandDimDSoftmaxForward(T* __restrict__ dst,
                                          const T* __restrict__ src,
                                          const int N, const int dim,
                                          const int D) {
  extern __shared__ __align__(sizeof(AccT)) unsigned char s_mem[];
  AccT* s_data = reinterpret_cast<AccT*>(s_mem);

  // vectorization for global memory coalescing
  using VecT = typename GetVecType<T, kVecSize>::type;
  VecT vec_src, vec_dst;
  T* buf_src = reinterpret_cast<T*>(&vec_src);
  T* buf_dst = reinterpret_cast<T*>(&vec_dst);

  const int vec_id = threadIdx.x * kVecSize;
  const int vec_num = blockDim.x * kVecSize;
  for (int out_id = blockIdx.x; out_id < N; out_id += gridDim.x) {
    const T* __restrict__ src_row = src + out_id * dim * D;
    T* __restrict__ dst_row = dst + out_id * dim * D;

    // Compute each thread's max value
    AccT max_val[kVecSize];
#pragma unroll
    for (int i = 0; i < kVecSize; i++) {
      max_val[i] = -std::numeric_limits<AccT>::infinity();
    }

    for (int id = vec_id; id < dim * D; id += vec_num) {
      vec_src = reinterpret_cast<const VecT*>(&src_row[id])[0];
#pragma unroll
      for (int i = 0; i < kVecSize; i++) {
        max_val[i] = max(static_cast<AccT>(buf_src[i]), max_val[i]);
      }
    }

// write to shared memory
#pragma unroll
    for (int i = 0; i < kVecSize; i++) s_data[vec_id + i] = max_val[i];
    __syncthreads();

// compute total max value
#pragma unroll
    for (int i = 0; i < kVecSize; i++) {
      for (int k = (vec_id + i) % D; k < vec_num; k += D) {
        max_val[i] = max(s_data[k], max_val[i]);
      }
    }

    // Compute each thread's sum value
    AccT sum_val[kVecSize]{0};
    for (int id = vec_id; id < dim * D; id += vec_num) {
      vec_src = reinterpret_cast<const VecT*>(&src_row[id])[0];
#pragma unroll
      for (int i = 0; i < kVecSize; i++) {
        sum_val[i] +=
            math::exp_func(static_cast<AccT>(buf_src[i]) - max_val[i]);
      }
    }

// write to shared memory
#pragma unroll
    for (int i = 0; i < kVecSize; i++) s_data[vec_id + i] = sum_val[i];
    __syncthreads();

// compute total sum value
#pragma unroll
    for (int i = 0; i < kVecSize; i++) {
      sum_val[i] = 0;
      for (int k = (vec_id + i) % D; k < vec_num; k += D) {
        sum_val[i] += s_data[k];
      }
    }

    // Compute finally softmax result
    // TODO(jiangcheng): how to eliminate twice exp_func
    for (int id = vec_id; id < dim * D; id += vec_num) {
      vec_src = reinterpret_cast<const VecT*>(&src_row[id])[0];
#pragma unroll
      for (int i = 0; i < kVecSize; i++) {
        buf_dst[i] = static_cast<T>(
            math::exp_func(static_cast<AccT>(buf_src[i]) - max_val[i]) /
            (sum_val[i] + std::numeric_limits<AccT>::min()));
      }
      reinterpret_cast<VecT*>(&dst_row[id])[0] = vec_dst;
    }
  }
}
template <typename T, int kVecSize>
inline void LaunchSpandDimDSoftmaxForwardKernel(const gpuStream_t& stream,
                                                const T* in_data, T* out_data,
                                                const int N, const int dim,
                                                const int D) {
  // ensure occupancy
  int D_size = 128;
  if (D <= 256)
    D_size = 256;
  else if (D <= 512)
    D_size = 512;
  else if (D <= MAX_BLOCK_DIM)
    D_size = MAX_BLOCK_DIM;
  else
    assert(false && "not support");

  const int grids = N;
  const int threads = D * (D_size / D);
  using AccT = typename GetAccType<T>::type;
  KeSpandDimDSoftmaxForward<
      T, AccT,
      kVecSize><<<grids, threads, threads * kVecSize * sizeof(AccT), stream>>>(
      out_data, in_data, N, dim, D);
}
template <typename T>
inline void LaunchSpandDimDSoftmaxForward(const gpuStream_t& stream,
                                          const T* in_data, T* out_data,
                                          const int N, const int dim,
                                          const int D) {
  if (D % 4 == 0) {
    LaunchSpandDimDSoftmaxForwardKernel<T, 4>(stream, in_data, out_data, N, dim,
                                              D);
  } else if (D % 2 == 0) {
    LaunchSpandDimDSoftmaxForwardKernel<T, 2>(stream, in_data, out_data, N, dim,
                                              D);
  } else {
    LaunchSpandDimDSoftmaxForwardKernel<T, 1>(stream, in_data, out_data, N, dim,
                                              D);
  }
}

// When D is small and (dim * D) is larger
// Each block arranged by N，each thread arranged by dim * D
// each block compute (dim * D) number's softmax
template <typename T, typename AccT, int kVecSize>
__global__ void KeSpandDimDSoftmaxBackward(T* __restrict__ dx,
                                           const T* __restrict__ out,
                                           const T* __restrict__ dout,
                                           const int N, const int dim,
                                           const int D) {
  extern __shared__ __align__(sizeof(AccT)) unsigned char s_mem[];
  AccT* s_data = reinterpret_cast<AccT*>(s_mem);

  // vectorization for global memory coalescing
  using VecT = typename GetVecType<T, kVecSize>::type;
  VecT vec_out, vec_dout, vec_dx;
  T* buf_out = reinterpret_cast<T*>(&vec_out);
  T* buf_dout = reinterpret_cast<T*>(&vec_dout);
  T* buf_dx = reinterpret_cast<T*>(&vec_dx);

  const int vec_id = threadIdx.x * kVecSize;
  const int vec_num = blockDim.x * kVecSize;
  for (int out_id = blockIdx.x; out_id < N; out_id += gridDim.x) {
    const int offset = out_id * dim * D;
    const T* __restrict__ out_row = out + offset;
    const T* __restrict__ dout_row = dout + offset;
    T* __restrict__ dx_row = dx + offset;

    // Compute each thread's sum value
    AccT sum_val[kVecSize]{0};
    for (int id = vec_id; id < dim * D; id += vec_num) {
      vec_out = reinterpret_cast<const VecT*>(&out_row[id])[0];
      vec_dout = reinterpret_cast<const VecT*>(&dout_row[id])[0];
#pragma unroll
      for (int i = 0; i < kVecSize; i++) {
        sum_val[i] +=
            static_cast<AccT>(buf_out[i]) * static_cast<AccT>(buf_dout[i]);
      }
    }

// write to shared memory
#pragma unroll
    for (int i = 0; i < kVecSize; i++) {
      s_data[vec_id + i] = sum_val[i];
    }
    __syncthreads();

// compute total sum value
#pragma unroll
    for (int i = 0; i < kVecSize; i++) {
      sum_val[i] = 0;
      for (int k = (vec_id + i) % D; k < vec_num; k += D) {
        sum_val[i] += s_data[k];
      }
    }

    // Compute finally softmax result
    for (int id = vec_id; id < dim * D; id += vec_num) {
      vec_out = reinterpret_cast<const VecT*>(&out_row[id])[0];
      vec_dout = reinterpret_cast<const VecT*>(&dout_row[id])[0];
#pragma unroll
      for (int i = 0; i < kVecSize; i++) {
        buf_dx[i] =
            static_cast<T>(static_cast<AccT>(buf_out[i]) *
                           (static_cast<AccT>(buf_dout[i]) - sum_val[i]));
      }
      reinterpret_cast<VecT*>(&dx_row[id])[0] = vec_dx;
    }
  }
}

template <typename T, int kVecSize>
inline void LaunchSpandDimDSoftmaxBackwardKernel(const gpuStream_t& stream,
                                                 T* dx_data, const T* out_data,
                                                 const T* dout_data,
                                                 const int N, const int dim,
                                                 const int D) {
  // ensure occupancy
  int D_size = 128;
  if (D <= 256)
    D_size = 256;
  else if (D <= 512)
    D_size = 512;
  else if (D <= MAX_BLOCK_DIM)
    D_size = MAX_BLOCK_DIM;
  else
    assert(false && "not support");

  const int grids = N;
  const int threads = D * (D_size / D);
  using AccT = typename GetAccType<T>::type;

  KeSpandDimDSoftmaxBackward<
      T, AccT,
      kVecSize><<<grids, threads, threads * kVecSize * sizeof(AccT), stream>>>(
      dx_data, out_data, dout_data, N, dim, D);
}

template <typename T>
inline void LaunchSpandDimDSoftmaxBackward(const gpuStream_t& stream,
                                           T* dx_data, const T* out_data,
                                           const T* dout_data, const int N,
                                           const int dim, const int D) {
  if (D % 4 == 0) {
    LaunchSpandDimDSoftmaxBackwardKernel<T, 4>(stream, dx_data, out_data,
                                               dout_data, N, dim, D);
  } else if (D % 2 == 0) {
    LaunchSpandDimDSoftmaxBackwardKernel<T, 2>(stream, dx_data, out_data,
                                               dout_data, N, dim, D);
  } else {
    LaunchSpandDimDSoftmaxBackwardKernel<T, 1>(stream, dx_data, out_data,
                                               dout_data, N, dim, D);
  }
}

template <typename T>
class SoftmaxCUDNNKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* x = ctx.Input<Tensor>("X");
    auto* out = ctx.Output<Tensor>("Out");
    out->mutable_data<T>(ctx.GetPlace());
    auto* out_data = out->data<T>();

    auto dims = x->dims();
    const int rank = dims.size();
    const int axis = CanonicalAxis(ctx.Attr<int>("axis"), rank);
    const int dim = dims[axis];
    const int N = SizeToAxis(axis, dims);
    const int D = SizeOutAxis(axis, dims);

    constexpr int max_dim = 320;
    bool optimize = false;
    constexpr int warps_per_block = 4;
    if (D == 1 && dim <= max_dim && sizeof(T) <= 4) {
      if (dim == 128 && N % warps_per_block == 0) {
        optimize = true;
        // a warp for a batch, 4 elements for a thread, only support the softmax
        // dim size = 128 currently
        if (sizeof(T) == 2) {
          VecSoftmaxForward<T, int2, 4, warps_per_block><<<
              N / warps_per_block, warps_per_block * WARP_SIZE, 0,
              ctx.cuda_device_context().stream()>>>(out_data, x->data<T>(), N,
                                                    dim);
        } else if (sizeof(T) == 4) {
          VecSoftmaxForward<T, int4, 4, warps_per_block><<<
              N / warps_per_block, warps_per_block * WARP_SIZE, 0,
              ctx.cuda_device_context().stream()>>>(out_data, x->data<T>(), N,
                                                    dim);
        } else {
          assert(false && "not support");
        }
      } else if (dim < max_dim) {
        optimize = true;
        int log2_elements = static_cast<int>(log2_ceil(dim));
        const int next_power_of_two = 1 << log2_elements;

        int warp_size = (next_power_of_two < 32) ? next_power_of_two : 32;

        int batches_per_warp = (next_power_of_two <= 128) ? 2 : 1;

        // use 128 threads per block to maximimize gpu utilization
        constexpr int threads_per_block = 128;

        int warps_per_block = (threads_per_block / warp_size);
        int batches_per_block = warps_per_block * batches_per_warp;
        int blocks = (N + batches_per_block - 1) / batches_per_block;
        dim3 threads(warp_size, warps_per_block, 1);

        switch (log2_elements) {
          LAUNCH_SOFTMAX_WARP_FORWARD(0);  // 1
          LAUNCH_SOFTMAX_WARP_FORWARD(1);  // 2
          LAUNCH_SOFTMAX_WARP_FORWARD(2);  // 4
          LAUNCH_SOFTMAX_WARP_FORWARD(3);  // 8
          LAUNCH_SOFTMAX_WARP_FORWARD(4);  // 16
          LAUNCH_SOFTMAX_WARP_FORWARD(5);  // 32
          LAUNCH_SOFTMAX_WARP_FORWARD(6);  // 64
          LAUNCH_SOFTMAX_WARP_FORWARD(7);  // 128
          LAUNCH_SOFTMAX_WARP_FORWARD(8);  // 256
          LAUNCH_SOFTMAX_WARP_FORWARD(9);  // 512
          default:
            break;
        }
      }
    } else if (D > 1) {
      auto stream = ctx.cuda_device_context().stream();
      if (D <= MAX_BLOCK_DIM) {
        optimize = true;
        LaunchSpandDimDSoftmaxForward(stream, x->data<T>(), out_data, N, dim,
                                      D);
      }
    }
    if (!optimize) {
      ScopedTensorDescriptor desc;
      std::vector<int> tensor_dims = {N, dim, D, 1};
      DataLayout layout = DataLayout::kNCHW;
#ifdef PADDLE_WITH_HIP
      miopenTensorDescriptor_t desc_ = desc.descriptor<T>(layout, tensor_dims);
#else
      cudnnTensorDescriptor_t desc_ = desc.descriptor<T>(layout, tensor_dims);
#endif

      auto& dev_ctx =
          ctx.template device_context<platform::CUDADeviceContext>();
      auto handle = dev_ctx.cudnn_handle();

#ifdef PADDLE_WITH_HIP
      auto mode = axis == rank - 1 ? MIOPEN_SOFTMAX_MODE_INSTANCE
                                   : MIOPEN_SOFTMAX_MODE_CHANNEL;
      PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::miopenSoftmaxForward(
          handle, platform::CudnnDataType<T>::kOne(), desc_, x->data<T>(),
          platform::CudnnDataType<T>::kZero(), desc_, out_data));
#else
      auto mode = axis == rank - 1 ? CUDNN_SOFTMAX_MODE_INSTANCE
                                   : CUDNN_SOFTMAX_MODE_CHANNEL;
      PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::cudnnSoftmaxForward(
          handle, CUDNN_SOFTMAX_ACCURATE, mode,
          platform::CudnnDataType<T>::kOne(), desc_, x->data<T>(),
          platform::CudnnDataType<T>::kZero(), desc_, out_data));
#endif
    }
  }
};

template <typename T>
class SoftmaxGradCUDNNKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* out = ctx.Input<Tensor>("Out");
    auto* dout = ctx.Input<Tensor>(framework::GradVarName("Out"));
    auto* dx = ctx.Output<Tensor>(framework::GradVarName("X"));
    dx->mutable_data<T>(ctx.GetPlace());
    auto* dx_data = dx->data<T>();

    auto dims = out->dims();
    const int rank = dims.size();
    const int axis = CanonicalAxis(ctx.Attr<int>("axis"), rank);
    const int dim = dims[axis];
    const int N = SizeToAxis(axis, dims);
    const int D = SizeOutAxis(axis, dims);

    constexpr int warps_per_block = 4;
    constexpr bool warp_softmax_available =
        std::is_same<T, float>::value ||
        std::is_same<T, platform::float16>::value;
    bool optimize = false;
    if (D == 1 && warp_softmax_available) {
      if (dim == 128 && N % warps_per_block == 0) {
        optimize = true;
        if (std::is_same<T, float>::value) {
          VecSoftmaxBackward<float, 4, warps_per_block><<<
              N / warps_per_block, warps_per_block * WARP_SIZE, 0,
              ctx.cuda_device_context().stream()>>>(dx->data<float>(),
                                                    dout->data<float>(),
                                                    out->data<float>(), N, dim);
        } else if (std::is_same<T, platform::float16>::value) {
          VecSoftmaxBackward<platform::float16, 4, warps_per_block><<<
              N / warps_per_block, warps_per_block * WARP_SIZE, 0,
              ctx.cuda_device_context().stream()>>>(
              dx->data<platform::float16>(), dout->data<platform::float16>(),
              out->data<platform::float16>(), N, dim);
        } else {
          PADDLE_ENFORCE_EQ(
              warp_softmax_available, true,
              platform::errors::Unimplemented(
                  "Warp softmax backward is only available for fp32 and fp16"));
        }
      } else if (dim < 40 && dim % 32 != 0) {
        optimize = true;
        Tensor mul_grad;
        int numel = N * dim;
        mul_grad.mutable_data<T>({numel}, ctx.GetPlace());

        auto stream = ctx.cuda_device_context().stream();
        auto& dev_ctx =
            ctx.template device_context<platform::CUDADeviceContext>();
        auto config = GetGpuLaunchConfig1D(dev_ctx, numel);

        MultiplyCUDAKernel<T><<<config.block_per_grid.x,
                                config.thread_per_block.x, 0, stream>>>(
            mul_grad.data<T>(), dout->data<T>(), out->data<T>(), numel);

        int log2_elements = log2_ceil(dim);
        const int next_power_of_two = 1 << log2_elements;

        int warp_size = (next_power_of_two < 32) ? next_power_of_two : 32;

        int batches_per_warp = (next_power_of_two <= 128) ? 2 : 1;

        constexpr int threads_per_block = 128;

        int warps_per_block = (threads_per_block / warp_size);
        int batches_per_block = warps_per_block * batches_per_warp;
        int blocks = (N + batches_per_block - 1) / batches_per_block;
        dim3 threads(warp_size, warps_per_block, 1);

        switch (log2_elements) {
          LAUNCH_SOFTMAX_WARP_BACKWARD(0);  // 1
          LAUNCH_SOFTMAX_WARP_BACKWARD(1);  // 2
          LAUNCH_SOFTMAX_WARP_BACKWARD(2);  // 4
          LAUNCH_SOFTMAX_WARP_BACKWARD(3);  // 8
          LAUNCH_SOFTMAX_WARP_BACKWARD(4);  // 16
          LAUNCH_SOFTMAX_WARP_BACKWARD(5);  // 32
          LAUNCH_SOFTMAX_WARP_BACKWARD(6);  // 64
          LAUNCH_SOFTMAX_WARP_BACKWARD(7);  // 128
          LAUNCH_SOFTMAX_WARP_BACKWARD(8);  // 256
          LAUNCH_SOFTMAX_WARP_BACKWARD(9);  // 512
          default:
            break;
        }
      }
    } else if (D > 1) {
      auto stream = ctx.cuda_device_context().stream();
      if (D <= MAX_BLOCK_DIM) {
        optimize = true;
        LaunchSpandDimDSoftmaxBackward(stream, dx_data, out->data<T>(),
                                       dout->data<T>(), N, dim, D);
      } else if (dim <= 512) {
        optimize = true;
        // sm_count ensure occupancy
        const int sm_count = ctx.cuda_device_context().GetSMCount();
        LaunchLoopDimSoftmaxBackward(stream, dx_data, out->data<T>(),
                                     dout->data<T>(), N, dim, D, sm_count);
      }
    }
    if (!optimize) {
      ScopedTensorDescriptor desc;
      std::vector<int> tensor_dims = {N, dim, D, 1};
      DataLayout layout = DataLayout::kNCHW;
#ifdef PADDLE_WITH_HIP
      miopenTensorDescriptor_t desc_ = desc.descriptor<T>(layout, tensor_dims);
#else
      cudnnTensorDescriptor_t desc_ = desc.descriptor<T>(layout, tensor_dims);
#endif

      auto& dev_ctx =
          ctx.template device_context<platform::CUDADeviceContext>();
      auto handle = dev_ctx.cudnn_handle();

#ifdef PADDLE_WITH_HIP
      auto mode = axis == rank - 1 ? MIOPEN_SOFTMAX_MODE_INSTANCE
                                   : MIOPEN_SOFTMAX_MODE_CHANNEL;
      PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::miopenSoftmaxBackward(
          handle, platform::CudnnDataType<T>::kOne(), desc_, out->data<T>(),
          desc_, dout->data<T>(), platform::CudnnDataType<T>::kZero(), desc_,
          dx_data));
#else
      auto mode = axis == rank - 1 ? CUDNN_SOFTMAX_MODE_INSTANCE
                                   : CUDNN_SOFTMAX_MODE_CHANNEL;
      PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::cudnnSoftmaxBackward(
          handle, CUDNN_SOFTMAX_ACCURATE, mode,
          platform::CudnnDataType<T>::kOne(), desc_, out->data<T>(), desc_,
          dout->data<T>(), platform::CudnnDataType<T>::kZero(), desc_,
          dx_data));
#endif
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;
#ifdef PADDLE_WITH_HIP
// MIOPEN do not support double
REGISTER_OP_KERNEL(softmax, CUDNN, plat::CUDAPlace,
                   ops::SoftmaxCUDNNKernel<float>,
                   ops::SoftmaxCUDNNKernel<plat::float16>);
REGISTER_OP_KERNEL(softmax_grad, CUDNN, plat::CUDAPlace,
                   ops::SoftmaxGradCUDNNKernel<float>,
                   ops::SoftmaxGradCUDNNKernel<plat::float16>);
#else
REGISTER_OP_KERNEL(softmax, CUDNN, plat::CUDAPlace,
                   ops::SoftmaxCUDNNKernel<float>,
                   ops::SoftmaxCUDNNKernel<double>,
                   ops::SoftmaxCUDNNKernel<plat::float16>);
REGISTER_OP_KERNEL(softmax_grad, CUDNN, plat::CUDAPlace,
                   ops::SoftmaxGradCUDNNKernel<float>,
                   ops::SoftmaxGradCUDNNKernel<double>,
                   ops::SoftmaxGradCUDNNKernel<plat::float16>);
#endif
