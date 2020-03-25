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

#include <cub/cub.cuh>
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/math/blas.h"
#include "paddle/fluid/operators/math/quant.h"
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
__global__ void InplaceAddReluAddLayerNormKernel(const T* y, const T* bias_0,
                                                 const T* bias_1,
                                                 const T* scale, T* out,
                                                 T* mean, T* variance, int M,
                                                 int N, float epsilon) {
  using BlockReduce = cub::BlockReduce<PairForLayerNorm<T>, BlockDim>;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  __shared__ T shared_mem[BlockDim + 2];

  for (int i = blockIdx.x; i < M; i += gridDim.x) {
    int index = i * N + threadIdx.x;

    // The fisrt BlockDim elements will be saved to shared memory.
    int save_index = threadIdx.x;
    T* save_ptr = shared_mem;

    T sum_i = 0;
    T square_sum_i = 0;
    for (int j = threadIdx.x; j < N; j += blockDim.x) {
      T tmp_0 = out[index];
      // Add bias
      T tmp_1 = bias_0 ? tmp_0 + bias_0[j] : tmp_0;
      // Relu
      T tmp_2 = DoRelu ? Relu(tmp_1) : tmp_1;
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

template <typename T>
class FusedFCElementwiseLayerNormOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* x = ctx.Input<framework::Tensor>("X");
    auto* w = ctx.Input<framework::Tensor>("W");
    auto* out = ctx.Output<framework::Tensor>("Out");

    auto w_dims = w->dims();
    int N = w_dims[1];
    int K = w_dims[0];
    int M = framework::product(x->dims()) / K;
    T* out_data = out->mutable_data<T>(ctx.GetPlace());

    auto& dev_ctx = ctx.template device_context<platform::CUDADeviceContext>();
    if (ctx.HasAttr("enable_int8") && ctx.Attr<bool>("enable_int8")) {
      float in_scale = ctx.Attr<float>("X_scale");
      std::vector<float> weight_scale =
          ctx.Attr<std::vector<float>>("weight_scale");

      PADDLE_ENFORCE_EQ(weight_scale.size(), 1,
                        "weight scale size shoud be equal to 1");

      framework::Tensor x_int8;
      x_int8.Resize(x->dims());
      x_int8.mutable_data<int8_t>(ctx.GetPlace());

      math::QuantFp32ToInt8Functor<platform::CUDADeviceContext> quant_func;
      quant_func(dev_ctx, *x, in_scale / 127., &x_int8);
      // the float here represents the output type
      const int8_t* x_int8_data = x_int8.data<int8_t>();
      const int8_t* w_int8_data = w->data<int8_t>();

      if (N % 4 == 0) {
        framework::Tensor x_int8, out_int8;
        out_int8.Resize(out->dims());
        int32_t* out_int8_data = out_int8.mutable_data<int32_t>(ctx.GetPlace());
        int32_t alpha = 1;
        int32_t beta = 0;
        float scale =
            static_cast<float>(in_scale * weight_scale[0] / 127. / 127.);
        math::GEMMINT8Functor<platform::CUDADeviceContext> gemm_int8_func;
        gemm_int8_func(dev_ctx, false, false, M, N, K, alpha, x_int8_data, K,
                       w_int8_data, N, beta, out_int8_data, N);
        math::INT32ToFP32Functor<platform::CUDADeviceContext>
            int32_to_fp32_func;
        int32_to_fp32_func(dev_ctx, out_int8, out, scale);
      } else {
        float* out_data_f = out->mutable_data<float>(ctx.GetPlace());
        float alpha =
            static_cast<float>(in_scale * weight_scale[0] / 127. / 127.);
        float beta = 0.0f;
        math::GEMMINT8Functor<platform::CUDADeviceContext> gemm_int8_func;
        gemm_int8_func(dev_ctx, false, false, M, N, K, alpha, x_int8_data, K,
                       w_int8_data, N, beta, out_data_f, N);
      }
    } else {
      const T* x_data = x->data<T>();
      const T* w_data = w->data<T>();

      auto blas = math::GetBlas<platform::CUDADeviceContext, T>(dev_ctx);
      blas.GEMM(false, false, M, N, K, static_cast<T>(1.0), x_data, K, w_data,
                N, static_cast<T>(0.0), out_data, N);
    }

    auto* y = ctx.Input<framework::Tensor>("Y");
    auto* bias_0 = ctx.Input<framework::Tensor>("Bias0");
    auto* bias_1 = ctx.Input<framework::Tensor>("Bias1");
    auto* scale = ctx.Input<framework::Tensor>("Scale");

    const T* y_data = y->data<T>();
    const T* bias_0_data = bias_0 ? bias_0->data<T>() : nullptr;
    const T* bias_1_data = bias_1 ? bias_1->data<T>() : nullptr;
    const T* scale_data = scale ? scale->data<T>() : nullptr;

    auto* mean = ctx.Output<framework::Tensor>("Mean");
    auto* variance = ctx.Output<framework::Tensor>("Variance");

    T* mean_data = mean ? mean->mutable_data<T>(ctx.GetPlace()) : nullptr;
    T* variance_data =
        variance ? variance->mutable_data<T>(ctx.GetPlace()) : nullptr;

    bool with_relu =
        (ctx.Attr<std::string>("activation_type") == "relu") ? true : false;
    float epsilon = ctx.Attr<float>("epsilon");

    int max_threads = dev_ctx.GetMaxPhysicalThreadCount();
    if (with_relu) {
      switch (platform::RoundToPowerOfTwo(N)) {
        CUDA_LAUNCH_KERNEL_HELPER(
            InplaceAddReluAddLayerNormKernel<
                T, true,
                kPowerOfTwoDim><<<std::max(max_threads / kPowerOfTwoDim, 1),
                                  kPowerOfTwoDim, 0, dev_ctx.stream()>>>(
                y_data, bias_0_data, bias_1_data, scale_data, out_data,
                mean_data, variance_data, M, N, epsilon));
      }
    } else {
      switch (platform::RoundToPowerOfTwo(N)) {
        CUDA_LAUNCH_KERNEL_HELPER(
            InplaceAddReluAddLayerNormKernel<
                T, false,
                kPowerOfTwoDim><<<std::max(max_threads / kPowerOfTwoDim, 1),
                                  kPowerOfTwoDim, 0, dev_ctx.stream()>>>(
                y_data, bias_0_data, bias_1_data, scale_data, out_data,
                mean_data, variance_data, M, N, epsilon));
      }
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(fused_fc_elementwise_layernorm,
                        ops::FusedFCElementwiseLayerNormOpKernel<float>);
