/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <math.h>
#include <cub/cub.cuh>
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/memory/malloc.h"
#include "paddle/fluid/operators/math/blas.h"
#include "paddle/fluid/platform/cuda_device_function.h"

namespace paddle {
namespace operators {

template <typename T>
static __device__ __forceinline__ T Relu(T x) {
  return (x > 0) ? x : 0;
}

static __device__ __forceinline__ float RealSqrt(float x) { return sqrtf(x); }
static __device__ __forceinline__ double RealSqrt(double x) { return sqrt(x); }

template <typename T>
struct PairForLayerNorm {
  __device__ __forceinline__ PairForLayerNorm() {}
  __device__ __forceinline__ PairForLayerNorm(const T& first, const T& second)
      : first_(first), second_(second) {}

  T first_;
  T second_;
};

template <typename T>
struct PairForLayerNormAddFunctor {
  __device__ __forceinline__ PairForLayerNorm<T> operator()(
      const PairForLayerNorm<T>& p1, const PairForLayerNorm<T>& p2) {
    return PairForLayerNorm<T>(p1.first_ + p2.first_, p1.second_ + p2.second_);
  }
};

template <typename T, bool DoRelu, int BlockDim>
__global__ void InplaceAddReluAddLayerNormKernel(
    const T* y, const T* bias_0, const T* bias_1, const T* scale, T* out,
    T* mean, T* variance, int M, int N, float epsilon, int bias_length) {
  using BlockReduce = cub::BlockReduce<PairForLayerNorm<T>, BlockDim>;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  __shared__ T shared_mem[BlockDim + 2];

  for (int i = blockIdx.x; i * blockDim.x + threadIdx.x < N * M;
       i += gridDim.x) {
    int id = i * blockDim.x + threadIdx.x;
    out[id] += bias_0[id % bias_length];
  }

  for (int i = blockIdx.x; i < M; i += gridDim.x) {
    int index = i * N + threadIdx.x;

    // The fisrt BlockDim elements will be saved to shared memory.
    int save_index = threadIdx.x;
    T* save_ptr = shared_mem;

    T sum_i = 0;
    T square_sum_i = 0;
    for (int j = threadIdx.x; j < N; j += blockDim.x) {
      T tmp_0 = out[index];
      // Relu
      T tmp_2 = DoRelu ? Relu(tmp_0) : tmp_0;
      // elementwise_add
      T tmp_3 = tmp_2 + y[index];

      // Save
      save_ptr[save_index] = tmp_3;
      save_ptr = out;

      index += blockDim.x;
      save_index = index;

      // For layer_norm, reduce to calculate mean and std
      sum_i += tmp_3;
      square_sum_i += (tmp_3 * tmp_3);
    }

    auto pair = BlockReduce(temp_storage)
                    .Reduce(PairForLayerNorm<T>(sum_i, square_sum_i),
                            PairForLayerNormAddFunctor<T>());

    if (threadIdx.x == 0) {
      T mean_i = static_cast<T>(pair.first_ / N);
      T variance_i = static_cast<T>(pair.second_ / N - mean_i * mean_i);
      shared_mem[BlockDim] = mean_i;
      shared_mem[BlockDim + 1] = variance_i;
      if (mean) {
        mean[blockIdx.x] = mean_i;
      }
      if (variance) {
        variance[blockIdx.x] = variance_i;
      }
    }
    __syncthreads();
    T mean_i = shared_mem[BlockDim];
    T std_i = static_cast<T>(RealSqrt(shared_mem[BlockDim + 1] + epsilon));

    index = i * N + threadIdx.x;
    // First BlockDim elements loading from shared memory.
    save_index = threadIdx.x;
    save_ptr = shared_mem;

    // For layer_norm, calculate out
    for (int j = threadIdx.x; j < N; j += blockDim.x) {
      T tmp_0 = (save_ptr[save_index] - mean_i) / std_i;
      T tmp_1 = scale ? scale[j] * tmp_0 : tmp_0;
      out[index] = bias_1 ? tmp_1 + bias_1[j] : tmp_1;

      save_ptr = out;
      index += blockDim.x;
      save_index = index;
    }
  }
}

template <typename T, int BlockDim>
__global__ void InplaceMatrixVectorAddKernel(const int N, const T* bias,
                                             T* data) {
  int offset = blockIdx.x * N;
  for (int i = threadIdx.x; i < N; i += BlockDim) {
    T temp;
#if __CUDA_ARCH__ >= 350
    temp = __ldg(data + offset + i) + __ldg(bias + i);
#else
    temp = data[offset + i] + bias[i];
#endif
    data[offset + i] = temp;
  }
}

template <typename T>
__global__ void GeluBiasKernel(const T* input, const T* bias, T* output,
                               int n_elements, int N) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const float a = 0.5;
  const float b = 0.7978845608028654;
  const float c = 0.0356774081363001;
  if (idx < n_elements) {
    const T x = input[idx];
    const T in = (bias == nullptr) ? x : (x + bias[idx % N]);
    output[idx] = (a + a * tanh(in * (c * in * in + b))) * in;
  }
}

template <typename T>
class FusedFCReshapeElementwiseLayerNormOpKernel
    : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* x = ctx.Input<framework::Tensor>("X");
    const T* x_data = x->data<T>();
    auto* out = ctx.Output<framework::Tensor>("Out");
    T* out_data = out->mutable_data<T>(ctx.GetPlace());
    auto& dev_ctx = ctx.template device_context<platform::CUDADeviceContext>();

    auto* w_1 = ctx.Input<framework::Tensor>("W_1");
    const T* w_1_data = w_1->data<T>();
    auto w_dims = w_1->dims();
    int N = w_dims[1];
    int K = w_dims[0];
    int M = framework::product(x->dims()) / K;

    int element_num = N * M;
    framework::Tensor temp_out;
    auto temp_out_dims = framework::make_ddim({element_num});
    temp_out.Resize({framework::product(temp_out_dims)});
    auto* temp_data = temp_out.mutable_data<T>(ctx.GetPlace());

    auto blas = math::GetBlas<platform::CUDADeviceContext, T>(dev_ctx);
    blas.GEMM(false, false, M, N, K, static_cast<T>(1.0), x_data, K, w_1_data,
              N, static_cast<T>(0.0), temp_data, N);

    const int gridSize = (element_num + 256 - 1) / 256;
    auto* bias0_1 = ctx.Input<framework::Tensor>("Bias0_1");
    const T* bias0_1_data = bias0_1->data<T>();
    const int threads_per_block = 256;
    GeluBiasKernel<T><<<gridSize, threads_per_block, 0, dev_ctx.stream()>>>(
        temp_data, bias0_1_data, temp_data, element_num, N);

    auto* w_2 = ctx.Input<framework::Tensor>("W_2");
    const T* w_2_data = w_2->data<T>();
    w_dims = w_2->dims();
    N = w_dims[1];
    K = w_dims[0];
    M = element_num / K;

    blas.GEMM(false, false, M, N, K, static_cast<T>(1.0), temp_data, K,
              w_2_data, N, static_cast<T>(0.0), out_data, N);
    auto* y = ctx.Input<framework::Tensor>("Y");
    auto* bias0_2 = ctx.Input<framework::Tensor>("Bias0_2");
    auto* bias_1 = ctx.Input<framework::Tensor>("Bias1");
    auto* scale = ctx.Input<framework::Tensor>("Scale");
    /*
    if (bias0_2) {
      const T* bias0_2_data = bias0_2->data<T>();
      const int threads = 256;
      const int blocks = M;
      InplaceMatrixVectorAddKernel<
          T, threads><<<blocks, threads, 0, dev_ctx.stream()>>>(N, bias0_2_data,
                                                                out_data);
    }
    */
    const T* bias0_2_data = bias0_2->data<T>();
    int bias_length = N;
    auto bias_dims = bias_1->dims();
    M = N * M / bias_dims[0];
    N = bias_dims[0];

    const T* y_data = y->data<T>();
    const T* bias_1_data = bias_1 ? bias_1->data<T>() : nullptr;
    const T* scale_data = scale ? scale->data<T>() : nullptr;

    auto* mean = ctx.Output<framework::Tensor>("Mean");
    auto* variance = ctx.Output<framework::Tensor>("Variance");

    T* mean_data = mean ? mean->mutable_data<T>(ctx.GetPlace()) : nullptr;
    T* variance_data =
        variance ? variance->mutable_data<T>(ctx.GetPlace()) : nullptr;

    bool with_relu =
        (ctx.Attr<std::string>("activation_type_2") == "relu") ? true : false;
    float epsilon = ctx.Attr<float>("epsilon");

    int max_threads = dev_ctx.GetMaxPhysicalThreadCount();
    if (with_relu) {
      switch (platform::RoundToPowerOfTwo(N)) {
        CUDA_LAUNCH_KERNEL_HELPER(
            InplaceAddReluAddLayerNormKernel<
                T, true,
                kPowerOfTwoDim><<<std::max(max_threads / kPowerOfTwoDim, 1),
                                  kPowerOfTwoDim, 0, dev_ctx.stream()>>>(
                y_data, bias0_2_data, bias_1_data, scale_data, out_data,
                mean_data, variance_data, M, N, epsilon, bias_length));
      }
    } else {
      switch (platform::RoundToPowerOfTwo(N)) {
        CUDA_LAUNCH_KERNEL_HELPER(
            InplaceAddReluAddLayerNormKernel<
                T, false,
                kPowerOfTwoDim><<<std::max(max_threads / kPowerOfTwoDim, 1),
                                  kPowerOfTwoDim, 0, dev_ctx.stream()>>>(
                y_data, bias0_2_data, bias_1_data, scale_data, out_data,
                mean_data, variance_data, M, N, epsilon, bias_length));
      }
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(fused_fc_reshape_elementwise_layernorm,
                        ops::FusedFCReshapeElementwiseLayerNormOpKernel<float>);
