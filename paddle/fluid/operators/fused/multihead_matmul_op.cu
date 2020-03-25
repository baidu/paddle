// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <cuda_runtime.h>
#include <paddle/fluid/platform/device_context.h>
#include <algorithm>
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/fluid/memory/malloc.h"
#include "paddle/fluid/operators/detail/safe_ref.h"
#include "paddle/fluid/operators/math/blas.h"
#include "paddle/fluid/operators/math/quant.h"
namespace paddle {
namespace operators {

#define FINAL_MASK 0xffffffff
#define HALF_WARP 16
#define WARP_SIZE 32

template <typename T>
__inline__ __device__ T warpReduceSum(T val, unsigned lane_mask) {
  for (int mask = HALF_WARP; mask > 0; mask >>= 1)
#if __CUDA_ARCH__ >= 350 && CUDA_VERSION >= 9000
    val += __shfl_xor_sync(lane_mask, val, mask, warpSize);
#else
    val += __shfl_xor(val, mask, warpSize);
#endif
  return val;
}

/* Calculate the sum of all elements in a block */
template <typename T>
__inline__ __device__ T blockReduceSum(T val, unsigned mask) {
  static __shared__ T shared[WARP_SIZE];
  int lane = threadIdx.x & 0x1f;
  int wid = threadIdx.x >> 5;

  val = warpReduceSum<T>(val, mask);

  if (lane == 0) shared[wid] = val;

  __syncthreads();

  // align block_span to warpSize
  int block_span = (blockDim.x + warpSize - 1) >> 5;
  val = (threadIdx.x < block_span) ? shared[lane] : static_cast<T>(0.0f);
  val = warpReduceSum<T>(val, mask);

  return val;
}

template <typename T>
__inline__ __device__ T warpReduceMax(T val, unsigned lane_mask) {
  for (int mask = HALF_WARP; mask > 0; mask >>= 1)
#if __CUDA_ARCH__ >= 350 && CUDA_VERSION >= 9000
    val = max(val, __shfl_xor_sync(lane_mask, val, mask, warpSize));
#else
    val = max(val, __shfl_xor(val, mask, warpSize));
#endif
  return val;
}

/* Calculate the maximum of all elements in a block */
template <typename T>
__inline__ __device__ T blockReduceMax(T val, unsigned mask) {
  static __shared__ T shared[WARP_SIZE];
  int lane = threadIdx.x & 0x1f;
  int wid = threadIdx.x >> 5;

  val = warpReduceMax(val, mask);

  if (lane == 0) shared[wid] = val;

  __syncthreads();

  // align block_span to warpSize
  int block_span = (blockDim.x + warpSize - 1) >> 5;
  val = (threadIdx.x < block_span) ? shared[lane] : -1e10f;
  val = warpReduceMax(val, mask);

  return val;
}

template <typename T>
__global__ void add_QKV(const T *Q, const T *K, const T *V, T *q_buf_,
                        T *k_buf_, T *v_buf_, const T *bias_q, const T *bias_k,
                        const T *bias_v, int batch_size, int seq_len,
                        int head_num, int size_per_head) {
  const T *data_ptr_q, *data_ptr_k, *data_ptr_v;
  const T *bias_ptr_q, *bias_ptr_k, *bias_ptr_v;

  int m = batch_size * seq_len;
  int n = head_num * size_per_head;

  int row_offset = (blockIdx.x % m) * n;

  data_ptr_q = Q + row_offset;
  data_ptr_k = K + row_offset;
  data_ptr_v = V + row_offset;
  // bias ptr
  bias_ptr_q = bias_q;
  bias_ptr_k = bias_k;
  bias_ptr_v = bias_v;

  int batch_id = (blockIdx.x % m) / seq_len;
  int head_id = threadIdx.x / size_per_head;
  int id_in_head = threadIdx.x % size_per_head;
  int word_start_id = (blockIdx.x) % seq_len;

#if __CUDA_ARCH__ >= 350
  T tmp_q = __ldg(&data_ptr_q[threadIdx.x]) + __ldg(&bias_ptr_q[threadIdx.x]);
  T tmp_k = __ldg(&data_ptr_k[threadIdx.x]) + __ldg(&bias_ptr_k[threadIdx.x]);
  T tmp_v = __ldg(&data_ptr_v[threadIdx.x]) + __ldg(&bias_ptr_v[threadIdx.x]);
#else
  T tmp_q = data_ptr_q[threadIdx.x] + bias_ptr_q[threadIdx.x];
  T tmp_k = data_ptr_k[threadIdx.x] + bias_ptr_k[threadIdx.x];
  T tmp_v = data_ptr_v[threadIdx.x] + bias_ptr_v[threadIdx.x];
#endif

  int target_id = batch_id * (seq_len * head_num * size_per_head) +
                  head_id * seq_len * size_per_head +
                  word_start_id * size_per_head + id_in_head;

  q_buf_[target_id] = tmp_q;
  k_buf_[target_id] = tmp_k;
  v_buf_[target_id] = tmp_v;
}

// Keep to compare performance
template <typename T>
__global__ void add_QKV_V2(const T *Q, const T *K, const T *V, T *q_buf_,
                           T *k_buf_, T *v_buf_, const T *bias_Q,
                           const T *bias_K, const T *bias_V, int batch_size,
                           int seq_len, int head_num, int size_per_head,
                           const int word_per_block) {
  const T *data_ptr;
  T *buf_ptr;
  const T *bias_ptr;

  int m = batch_size * seq_len;
  int n = head_num * size_per_head;

  int qkv_id = blockIdx.x * word_per_block / m;
  int row_offset = (blockIdx.x * word_per_block % m) * n;

  if (qkv_id == 0) {
    data_ptr = Q + row_offset;
    buf_ptr = q_buf_;
    bias_ptr = bias_Q;
  } else if (qkv_id == 1) {
    data_ptr = K + row_offset;
    buf_ptr = k_buf_;
    bias_ptr = bias_K;
  } else {
    data_ptr = V + row_offset;
    buf_ptr = v_buf_;
    bias_ptr = bias_V;
  }

  int batch_id = (blockIdx.x * word_per_block % m) / seq_len;
  int head_id = threadIdx.x / size_per_head;
  int id_in_head = threadIdx.x % size_per_head;
  int word_start_id = (blockIdx.x * word_per_block) % seq_len;

#if __CUDA_ARCH__ >= 350
  T bias = __ldg(&bias_ptr[threadIdx.x]);
#else
  T bias = bias_ptr[threadIdx.x];
#endif

  for (int i = word_start_id; i < word_start_id + word_per_block; ++i) {
    T tmp = data_ptr[threadIdx.x] + bias;

    int target_id = batch_id * (seq_len * head_num * size_per_head) +
                    head_id * seq_len * size_per_head + i * size_per_head +
                    id_in_head;

    buf_ptr[target_id] = tmp;
    data_ptr += n;
  }
}

template <typename T>
__global__ void softmax_kernel_with_eltadd(T *qk_buf_, const T *bias_qk_,
                                           const int batch_size,
                                           const int head_num,
                                           const int seq_len,
                                           const unsigned mask) {
  int qk_offset = blockIdx.x * seq_len;
  assert(blockDim.x % 32 == 0);

  __shared__ float s_sum, s_max;

  float qk = threadIdx.x < seq_len
                 ? static_cast<float>((qk_buf_[threadIdx.x + qk_offset] +
                                       bias_qk_[threadIdx.x + qk_offset]))
                 : 0.0f;
  float tmp = threadIdx.x < seq_len ? static_cast<float>(qk) : -1e20f;

  float max_val = blockReduceMax<float>(tmp, mask);

  if (threadIdx.x == 0) s_max = max_val;
  __syncthreads();

  float qk_tmp =
      threadIdx.x < seq_len ? __expf(static_cast<float>(tmp - s_max)) : 0.0f;
  float sum_val = blockReduceSum<float>(qk_tmp, mask);

  if (threadIdx.x == 0) {
    s_sum = sum_val + 1e-6f;
  }
  __syncthreads();

  if (threadIdx.x < seq_len)
    qk_buf_[threadIdx.x + qk_offset] = (T)(qk_tmp / s_sum);
}

// For verify result
template <typename T>
__global__ void elt_qk_add(const T *bias_qk, T *qk_buf, int head_num,
                           int seq_len, int size_per_head, int batch_size) {
  int m = batch_size * head_num * seq_len;
  int row_id = blockIdx.x % m;
  int dst_id = row_id * seq_len + threadIdx.x;
  const T *bias_ptr = bias_qk;
#if __CUDA_ARCH__ >= 350
  int tmp_bias = __ldg(&bias_ptr[dst_id]);
#else
  int tmp_bias = bias_ptr[dst_id];
#endif

  qk_buf[dst_id] += tmp_bias;
}

__global__ void INT32ToFP32Kernel(int count, const int32_t *in_data,
                                  float *out_data, int all_head_size,
                                  float scale0, float scale1, float scale2) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < count) {
    float scale = 1.;
    if ((tid / all_head_size) % 3 == 0) {
      scale = scale0;
    } else if ((tid / all_head_size) % 3 == 1) {
      scale = scale1;
    } else if ((tid / all_head_size) % 3 == 2) {
      scale = scale2;
    }
    out_data[tid] = scale * __int2float_rn(in_data[tid]);
  }
}

__global__ void FP32DequantKernel(int count, const float *in_data,
                                  float *out_data, int all_head_size,
                                  float scale0, float scale1, float scale2) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < count) {
    float scale = 1.;
    if ((tid / all_head_size) % 3 == 0) {
      scale = scale0;
    } else if ((tid / all_head_size) % 3 == 1) {
      scale = scale1;
    } else if ((tid / all_head_size) % 3 == 2) {
      scale = scale2;
    }
    out_data[tid] = scale * in_data[tid];
  }
}

// Compute Q*K->softmax->eltadd
template <typename T>
void MatMulWithHeadQK(const platform::CUDADeviceContext &context, int head_num,
                      int seq_len, int size_per_head, int batch_size,
                      bool q_trans, bool k_trans, T *q_buf_, T *k_buf_,
                      T *qk_buf_, const T *bias_qk, T alpha, T beta) {
  CBLAS_TRANSPOSE transA = !q_trans ? CblasNoTrans : CblasTrans;
  CBLAS_TRANSPOSE transB = !k_trans ? CblasNoTrans : CblasTrans;

  auto blas = math::GetBlas<platform::CUDADeviceContext, T>(context);
  auto stream = context.stream();

  blas.BatchedGEMM(transA, transB, seq_len, seq_len, size_per_head, alpha,
                   q_buf_, k_buf_, beta, qk_buf_, batch_size * head_num,
                   seq_len * size_per_head, seq_len * size_per_head);

  int grid = batch_size * head_num * seq_len;
  int block = seq_len;

  // Align block to 32, also limit seq_len to max block size.
  PADDLE_ENFORCE_LE(seq_len, 1024, platform::errors::InvalidArgument(
                                       "seq_len should <= 1024, "
                                       "but received seq_len is:%d",
                                       seq_len));
  if (seq_len <= 32)
    block = 32;
  else if (seq_len > 32 && seq_len <= 64)
    block = 64;
  else if (seq_len > 64 && seq_len <= 128)
    block = 128;
  else if (seq_len > 128 && seq_len <= 256)
    block = 256;
  else if (seq_len > 256 && seq_len <= 512)
    block = 512;
  else
    block = 1024;

  softmax_kernel_with_eltadd<T><<<grid, block, 0, stream>>>(
      qk_buf_, bias_qk, batch_size, head_num, seq_len, FINAL_MASK);
}

template <typename T>
__global__ void transpose(T *src, T *dst, const int batch_size,
                          const int seq_len, const int head_num,
                          const int size_per_head) {
  int batch_id = blockIdx.x / (head_num * seq_len);
  int seq_id = blockIdx.x % seq_len;
  int head_id = (blockIdx.x % (head_num * seq_len)) / seq_len;
  dst[batch_id * (head_num * seq_len * size_per_head) +
      seq_id * head_num * size_per_head + head_id * size_per_head +
      threadIdx.x] = src[blockIdx.x * size_per_head + threadIdx.x];
}

// Compute QK*V->transpose
template <typename T>
void MatMulWithHeadQKV(const platform::CUDADeviceContext &context, int head_num,
                       int seq_len, int size_per_head, int batch_size,
                       bool qk_trans, bool v_trans, T *v_buf_, const T *qk_buf_,
                       T *dst, T alpha, T beta) {
  int m = batch_size * seq_len;
  int k = head_num * size_per_head;

  auto blas = math::GetBlas<platform::CUDADeviceContext, T>(context);
  auto stream = context.stream();
  CBLAS_TRANSPOSE transA = !qk_trans ? CblasNoTrans : CblasTrans;
  CBLAS_TRANSPOSE transB = !v_trans ? CblasNoTrans : CblasTrans;

  blas.BatchedGEMM(transA, transB, seq_len, size_per_head, seq_len, alpha,
                   qk_buf_, v_buf_, beta, dst, batch_size * head_num,
                   seq_len * seq_len, seq_len * size_per_head);
}

template <typename T>
inline __device__ T add_func(T a, T b);

template <>
__device__ float add_func<float>(float a, float b) {
  return a + b;
}

template <>
__device__ float2 add_func<float2>(float2 a, float2 b) {
  float2 c;
  c.x = a.x + b.x;
  c.y = a.y + b.y;
  return c;
}

template <>
__device__ float4 add_func<float4>(float4 a, float4 b) {
  float4 c;
  c.x = a.x + b.x;
  c.y = a.y + b.y;
  c.z = a.z + b.z;
  c.w = a.w + b.w;
  return c;
}

template <typename T>
__global__ void transpose_qkv_kernel(const int H, const T *input, const T *bias,
                                     T *output) {
  // Input: BxSx3xNxH
  // Bias: 3xSxB
  // Output: 3xBxNxSxH
  int n = threadIdx.y;
  int s = blockIdx.x;
  int b = blockIdx.y;
  int m = blockIdx.z;

  const int N = blockDim.y;
  const int S = gridDim.x;
  const int B = gridDim.y;

  const int NH = N * H;
  const int NHS = NH * S;
  const int in_offset = n * H + m * NH + s * 3 * NH + b * NHS * 3;
  const int bias_offset = m * NH + n * H;
  const int out_offset = s * H + n * S * H + b * NHS + m * NHS * B;

  const int i = threadIdx.x;
  output[out_offset + i] =
      add_func(input[in_offset + i], bias[bias_offset + i]);
}

void TransQKVWithBias(const int batch, const int seq_len, const int head_size,
                      const int head_num, const float *input, const float *bias,
                      float *output, cudaStream_t stream) {
  // BxSx3xNxH + 3xNxH -> 3xBxNxSxH
  int scratch_size = batch * head_num * seq_len * seq_len;
  const dim3 grid(seq_len, batch, 3);
  // scratch % 4 == 0 to ensure the alignment
  if (head_size % 4 == 0 && scratch_size % 4 == 0) {
    const int h = head_size / 4;
    const float4 *input4 = reinterpret_cast<const float4 *>(input);
    const float4 *bias4 = reinterpret_cast<const float4 *>(bias);
    float4 *output4 = reinterpret_cast<float4 *>(output);
    const dim3 block(h, head_num, 1);

    // limit h * head_num to max block size(1024).
    PADDLE_ENFORCE_LE(h * head_num, 1024,
                      platform::errors::InvalidArgument(
                          "head_num (%d) * head_size (%d) should <= %d",
                          head_num, head_size, 1024 * 4));
    transpose_qkv_kernel<float4><<<grid, block, 0, stream>>>(h, input4, bias4,
                                                             output4);
  } else if (head_size % 2 == 0 && scratch_size % 2 == 0) {
    const int h = head_size / 2;
    const float2 *input2 = reinterpret_cast<const float2 *>(input);
    const float2 *bias2 = reinterpret_cast<const float2 *>(bias);
    float2 *output2 = reinterpret_cast<float2 *>(output);
    const dim3 block(h, head_num, 1);
    // limit h * head_num to max block size(1024).
    PADDLE_ENFORCE_LE(h * head_num, 1024,
                      platform::errors::InvalidArgument(
                          "head_num (%d) * head_size (%d) should <= %d",
                          head_num, head_size, 1024 * 2));
    transpose_qkv_kernel<float2><<<grid, block, 0, stream>>>(h, input2, bias2,
                                                             output2);
  } else {
    const dim3 block(head_size, head_num, 1);
    // limit head_size * head_num to max block size(1024).
    PADDLE_ENFORCE_LE(head_size * head_num, 1024,
                      platform::errors::InvalidArgument(
                          "head_num (%d) * head_size (%d) should <= %d",
                          head_num, head_size, 1024));
    transpose_qkv_kernel<float><<<grid, block, 0, stream>>>(head_size, input,
                                                            bias, output);
  }
}

template <typename T>
void MultiHeadGPUComputeV2(const platform::CUDADeviceContext &dev_ctx,
                           int batch, int seq_len, int head_num, int head_size,
                           T *qkptr, const T *bias_qk_ptr, T *tptr, T alpha,
                           T beta) {
  auto stream = dev_ctx.stream();
  const int tsize = batch * head_num * seq_len * head_size;

  T *qptr = tptr;
  T *kptr = qptr + tsize;
  T *vptr = kptr + tsize;
  // batch gemm stride, softmaxwithscale.
  MatMulWithHeadQK<T>(dev_ctx, head_num, seq_len, head_size, batch, false, true,
                      qptr, kptr, qkptr, bias_qk_ptr, alpha, beta);
  // batch gemm stride, transpose.
  MatMulWithHeadQKV<T>(dev_ctx, head_num, seq_len, head_size, batch, false,
                       false, vptr, qkptr, tptr, T(1.0), beta);
}

template <typename DeviceContext, typename T>
class MultiHeadMatMulV2Kernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &context) const override {
    using Tensor = framework::Tensor;
    auto *input = context.Input<framework::Tensor>("Input");
    auto *w = context.Input<framework::Tensor>("W");
    auto *bias = context.Input<framework::Tensor>("Bias");

    auto &bias_qk = detail::Ref(context.Input<framework::Tensor>("BiasQK"),
                                "Cannot find QK");

    auto *input_d = input->data<T>();
    auto *bias_d = bias->data<T>();
    auto *bias_qk_d = bias_qk.data<T>();
    T scale = static_cast<T>(context.Attr<float>("alpha"));

    int head_number = context.Attr<int>("head_number");
    // compute q*k with eltadd
    auto &device_ctx = context.template device_context<DeviceContext>();
    // should be (B * S * hidden)
    auto input_dims = input->dims();
    // shouble be (hidden * 3 * all_head_size)
    auto w_dims = w->dims();
    int batch = input_dims[0];
    int seq_len = input_dims[1];
    int hidden = input_dims[2];

    int all_head_size = w_dims[2];
    int head_size = all_head_size / head_number;

    auto *out = context.Output<framework::Tensor>("Out");
    out->Resize({batch, seq_len, all_head_size});
    auto *output_d = out->mutable_data<T>(context.GetPlace());

    // (B*S, hidden)
    const Tensor input_matrix =
        framework::ReshapeToMatrix(*input, 2 /*x_num_col_dims */);
    // (hidden, 3 * all_head_size)
    Tensor w_matrix = framework::ReshapeToMatrix(*w, 1 /*y_num_col_dims*/);

    Tensor temp_out_tensor;
    auto temp_out_dims =
        framework::make_ddim({batch, seq_len, 3, head_number, head_size});
    temp_out_tensor.Resize({batch * seq_len, framework::product(temp_out_dims) /
                                                 (batch * seq_len)});
    auto *temp_out_data = temp_out_tensor.mutable_data<T>(context.GetPlace());
    if (context.HasAttr("enable_int8") && context.Attr<bool>("enable_int8")) {
      PADDLE_ENFORCE_EQ(context.HasAttr("X_scale") &&
                            context.HasAttr("weight0_scale") &&
                            context.HasAttr("weight1_scale") &&
                            context.HasAttr("weight2_scale"),
                        true, platform::errors::InvalidArgument(
                                  "mul in multihead_matmul is set to use int8, "
                                  "but no input scale or weight scale found."));
      float in_scale = context.Attr<float>("X_scale");
      std::vector<float> weight0_scale =
          context.Attr<std::vector<float>>("weight0_scale");
      std::vector<float> weight1_scale =
          context.Attr<std::vector<float>>("weight1_scale");
      std::vector<float> weight2_scale =
          context.Attr<std::vector<float>>("weight2_scale");

      framework::Tensor input_int8;
      input_int8.Resize(input_matrix.dims());
      input_int8.mutable_data<int8_t>(context.GetPlace());
      w_matrix.mutable_data<int8_t>(context.GetPlace());
      int N = framework::vectorize(w_matrix.dims())[1];
      int K = framework::vectorize(w_matrix.dims())[0];
      int M = framework::product(input_matrix.dims()) / K;
      float scale0 =
          static_cast<float>(in_scale * weight0_scale[0] / 127. / 127.);
      float scale1 =
          static_cast<float>(in_scale * weight1_scale[0] / 127. / 127.);
      float scale2 =
          static_cast<float>(in_scale * weight2_scale[0] / 127. / 127.);

      math::QuantFp32ToInt8Functor<platform::CUDADeviceContext> quant_func;
      quant_func(device_ctx, input_matrix, in_scale / 127., &input_int8);

      const int8_t *input_int8_data = input_int8.data<int8_t>();
      const int8_t *w_int8_data = w_matrix.data<int8_t>();
      if (N % 4 == 0) {
        framework::Tensor x_int8, out_int8;
        out_int8.Resize(temp_out_tensor.dims());
        int32_t *out_int8_data =
            out_int8.mutable_data<int32_t>(context.GetPlace());
        int32_t alpha = 1;
        int32_t beta = 0;

        math::GEMMINT8Functor<platform::CUDADeviceContext> gemm_int8_func;
        gemm_int8_func(device_ctx, false, false, M, N, K, alpha,
                       input_int8_data, K, w_int8_data, N, beta, out_int8_data,
                       N);
        const int32_t *out_int32_data = out_int8.data<int32_t>();
        int numel = out_int8.numel();
        int threads = 1024;
        int blocks = (numel + threads - 1) / threads;
        float *output =
            temp_out_tensor.mutable_data<float>(device_ctx.GetPlace());

        INT32ToFP32Kernel<<<blocks, threads, 0, device_ctx.stream()>>>(
            numel, out_int32_data, output, all_head_size, scale0, scale1,
            scale2);
      } else {
        framework::Tensor out_f;
        out_f.Resize(temp_out_tensor.dims());
        float *out_data_f = out_f.mutable_data<float>(context.GetPlace());
        float *out_data =
            temp_out_tensor.mutable_data<float>(context.GetPlace());
        float alpha = 1.0f;
        float beta = 0.0f;
        math::GEMMINT8Functor<platform::CUDADeviceContext> gemm_int8_func;
        gemm_int8_func(device_ctx, false, false, M, N, K, alpha,
                       input_int8_data, K, w_int8_data, N, beta, out_data_f, N);
        int numel = out_f.numel();
        int threads = 1024;
        int blocks = (numel + threads - 1) / threads;
        FP32DequantKernel<<<blocks, threads, 0, device_ctx.stream()>>>(
            numel, out_data_f, out_data, all_head_size, scale0, scale1, scale2);
      }
    } else {
      // (B * S, hidden) * (hidden, 3 * N * H) -> (B * S * 3 * N * H)
      auto blas = math::GetBlas<platform::CUDADeviceContext, T>(device_ctx);
      blas.MatMul(input_matrix, w_matrix, &temp_out_tensor);
    }

    // temp_out_tensor.Resize(temp_out_dims);
    Tensor multihead_temp_tensor;
    // B * head_number * S * S * 1 + B * S * 3 * N * H
    int scratch_size = batch * head_number * seq_len * seq_len * 1;
    multihead_temp_tensor.Resize({scratch_size + temp_out_tensor.numel()});
    auto *multihead_temp_data =
        multihead_temp_tensor.mutable_data<T>(context.GetPlace());
    auto *qkptr = multihead_temp_data;
    auto *tptr = multihead_temp_data + scratch_size;
    auto stream = device_ctx.stream();
    // Do the transpose with bias.
    // BxSx3xNxH => tptr: 3xBxNxSxH.
    TransQKVWithBias(batch, seq_len, head_size, head_number, temp_out_data,
                     bias_d, tptr, stream);
    MultiHeadGPUComputeV2<T>(device_ctx, batch, seq_len, head_number, head_size,
                             qkptr, bias_qk_d, tptr, scale, T(0.0));

    int grid = batch * head_number * seq_len;
    int block = head_size;
    transpose<T><<<grid, block, 0, stream>>>(tptr, output_d, batch, seq_len,
                                             head_number, head_size);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(
    multihead_matmul,
    ops::MultiHeadMatMulV2Kernel<paddle::platform::CUDADeviceContext, float>);
