/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <algorithm>
#include <string>
#include "paddle/fluid/framework/op_registry.h"

#ifdef __HIPCC__
#define __syncwarp() __all(1)
#endif

namespace paddle {
namespace operators {

#ifdef __HIPCC__
#define THREADS_PER_BLOCK 64
#else
#define THREADS_PER_BLOCK 32
#endif
#define FULL_MASK 0xffffffff

using framework::Tensor;

template <typename T>
__forceinline__ __device__ T warpReduceSum(T val) {
  for (int offset = 16; offset > 0; offset /= 2) {
#ifdef __HIPCC__
    val += __shfl_down(val, offset);
#else
    val += __shfl_down_sync(FULL_MASK, val, offset);
#endif
  }
  return val;
}

template <typename T>
__forceinline__ __device__ T blockReduceSum(T val) {
#ifdef __HIPCC__
  static __shared__ T shared[64];
#else
  static __shared__ T shared[32];
#endif
  int lane = threadIdx.x % warpSize;
  int wid = threadIdx.x / warpSize;

  val = warpReduceSum(val);
  __syncthreads();
  if (lane == 0) shared[wid] = val;

  __syncthreads();
  val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0;

  if (wid == 0) val = warpReduceSum(val);

  return val;
}

template <typename T>
__global__ void set_zero(T *x, int num) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num;
       i += blockDim.x * gridDim.x)
    x[i] = static_cast<T>(0);
}

template <typename T>
__global__ void channel_first(const T *input, T *rinput, const int channel,
                              const int height, const int width,
                              const int pad_size) {
  int n = blockIdx.x;
  int h = blockIdx.y;
  int w = blockIdx.z;

  int ch_off = threadIdx.x;
  T value;
  int dimchw = channel * height * width;
  int dimhw = height * width;

  int p_dimw = (width + 2 * pad_size);
  int p_dimh = (height + 2 * pad_size);
  int p_dimchw = channel * p_dimw * p_dimh;
  int p_dimcw = channel * p_dimw;

  for (int c = ch_off; c < channel; c += THREADS_PER_BLOCK) {
    value = input[n * dimchw + c * dimhw + h * width + w];
    rinput[n * p_dimchw + (h + pad_size) * p_dimcw + (w + pad_size) * channel +
           c] = value;
  }
}

template <typename T>
__global__ void correlation_forward(
    T *output, const int output_channel, const int output_height,
    const int output_width, const T *rinput1, const int input_channel,
    const int input_height, const int input_width, const T *rinput2,
    const int pad_size, const int kernel_size, const int max_displacement,
    const int stride1, const int stride2) {
  int p_input_width = input_width + 2 * pad_size;
  int p_input_height = input_height + 2 * pad_size;

  int kernel_rad = (kernel_size - 1) / 2;
  int displacement_rad = max_displacement / stride2;

  int displacement_size = 2 * displacement_rad + 1;

  int n = blockIdx.x;
  int h1 = blockIdx.y * stride1 + max_displacement;
  int w1 = blockIdx.z * stride1 + max_displacement;
  int c = threadIdx.x;

  int p_dimchw = p_input_height * p_input_width * input_channel;
  int p_dimcw = p_input_width * input_channel;
  int p_dimc = input_channel;

  int t_dimchw = output_channel * output_height * output_width;
  int t_dimhw = output_height * output_width;
  int t_dimw = output_width;

  int nelems = kernel_size * kernel_size * p_dimc;

  for (int tj = -displacement_rad; tj <= displacement_rad; ++tj) {
    for (int ti = -displacement_rad; ti <= displacement_rad; ++ti) {
      int w2 = w1 + ti * stride2;
      int h2 = h1 + tj * stride2;

      T acc0 = 0;
      for (int j = -kernel_rad; j <= kernel_rad; ++j) {
        for (int i = -kernel_rad; i <= kernel_rad; ++i) {
          for (int ch = c; ch < p_dimc; ch += blockDim.x) {
            int index1 =
                n * p_dimchw + (h1 + j) * p_dimcw + (w1 + i) * p_dimc + ch;
            int index2 =
                n * p_dimchw + (h2 + j) * p_dimcw + (w2 + i) * p_dimc + ch;
            acc0 += static_cast<T>(rinput1[index1] * rinput2[index2]);
          }
        }
      }
      if (blockDim.x == warpSize) {
        __syncwarp();
        acc0 = warpReduceSum(acc0);
      } else {
        __syncthreads();
        acc0 = blockReduceSum(acc0);
      }

      if (threadIdx.x == 0) {
        int tc = (tj + displacement_rad) * displacement_size +
                 (ti + displacement_rad);
        const int t_index =
            n * t_dimchw + tc * t_dimhw + blockIdx.y * t_dimw + blockIdx.z;
        output[t_index] = static_cast<T>(acc0 / nelems);
      }
    }
  }
}

// class CorrelationKernel<platform::CUDADeviceContext, T>
template <typename T>
class CorrelationCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    PADDLE_ENFORCE_EQ(platform::is_gpu_place(ctx.GetPlace()), true,
                      platform::errors::InvalidArgument(
                          "Correlation only supports GPU now."));

    auto *input1 = ctx.Input<Tensor>("Input1");
    auto *input2 = ctx.Input<Tensor>("Input2");
    int pad_size = ctx.Attr<int>("pad_size");
    int kernel_size = ctx.Attr<int>("kernel_size");
    int stride1 = ctx.Attr<int>("stride1");
    int stride2 = ctx.Attr<int>("stride2");
    int max_displacement = ctx.Attr<int>("max_displacement");
    int corr_type_multiply = ctx.Attr<int>("corr_type_multiply");

    auto *output = ctx.Output<Tensor>("Output");
    output->mutable_data<T>(ctx.GetPlace());
    auto &dev_ctx = ctx.template device_context<platform::CUDADeviceContext>();

    // base on input1, NCHW
    auto in_dims = input1->dims();
    int N = in_dims[0];
    int C = in_dims[1];
    int H = in_dims[2];
    int W = in_dims[3];

    int padded_input_height = H + 2 * pad_size;
    int padded_input_width = W + 2 * pad_size;

    Tensor rinput1 = ctx.AllocateTmpTensor<T, platform::CUDADeviceContext>(
        {N, padded_input_height, padded_input_width, C}, dev_ctx);
    rinput1.mutable_data<T>(ctx.GetPlace());

    Tensor rinput2 = ctx.AllocateTmpTensor<T, platform::CUDADeviceContext>(
        {N, padded_input_height, padded_input_width, C}, dev_ctx);
    rinput2.mutable_data<T>(ctx.GetPlace());

    set_zero<<<(rinput1.numel() + 512 - 1) / 512, 512, 0, dev_ctx.stream()>>>(
        rinput1.data<T>(), rinput1.numel());
    set_zero<<<(rinput2.numel() + 512 - 1) / 512, 512, 0, dev_ctx.stream()>>>(
        rinput2.data<T>(), rinput2.numel());
    set_zero<<<(output->numel() + 512 - 1) / 512, 512, 0, dev_ctx.stream()>>>(
        output->data<T>(), output->numel());

    auto out_dims = output->dims();
    int OC = out_dims[1];
    int OH = out_dims[2];
    int OW = out_dims[3];

    dim3 blocks_grid(N, H, W);
    dim3 threads_block(THREADS_PER_BLOCK);

    channel_first<T><<<blocks_grid, threads_block, 0, dev_ctx.stream()>>>(
        input1->data<T>(), rinput1.data<T>(), C, H, W, pad_size);
    channel_first<T><<<blocks_grid, threads_block, 0, dev_ctx.stream()>>>(
        input2->data<T>(), rinput2.data<T>(), C, H, W, pad_size);

    dim3 threadsPerBlock(THREADS_PER_BLOCK);
    dim3 totalBlocksCorr(N, OH, OW);

    correlation_forward<
        T><<<totalBlocksCorr, threadsPerBlock, 0, dev_ctx.stream()>>>(
        output->data<T>(), OC, OH, OW, rinput1.data<T>(), C, H, W,
        rinput2.data<T>(), pad_size, kernel_size, max_displacement, stride1,
        stride2);
  }
};

template <typename T>
__global__ void correlation_backward_input1(
    int item, T *grad_input1, const int input_channel, const int input_height,
    const int input_width, const T *grad_output, const int output_channel,
    const int output_height, const int output_width, const T *rinput2,
    const int pad_size, const int kernel_size, const int max_displacement,
    const int stride1, const int stride2) {
  int n = item;
  int h = blockIdx.x * stride1 + pad_size;
  int w = blockIdx.y * stride1 + pad_size;
  int c = blockIdx.z;
  int tch_off = threadIdx.x;

  int kernel_rad = (kernel_size - 1) / 2;
  int displacement_rad = max_displacement / stride2;
  int displacement_size = 2 * displacement_rad + 1;

  int xmin = (w - kernel_rad - max_displacement) / stride1;
  int ymin = (h - kernel_rad - max_displacement) / stride1;

  int xmax = (w + kernel_rad - max_displacement) / stride1;
  int ymax = (h + kernel_rad - max_displacement) / stride1;

  if (xmax < 0 || ymax < 0 || xmin >= output_width || ymin >= output_height) {
    return;
  }

  if (xmin > xmax || ymin > ymax) {
    return;
  }

  xmin = max(0, xmin);
  xmax = min(output_width - 1, xmax);

  ymin = max(0, ymin);
  ymax = min(output_height - 1, ymax);

  int p_input_width = input_width + 2 * pad_size;
  int p_input_height = input_height + 2 * pad_size;
  int p_dimchw = input_channel * p_input_height * p_input_width;
  int p_dimcw = input_channel * p_input_width;
  int p_dimc = input_channel;

  int t_dimchw = output_channel * output_height * output_width;
  int t_dimhw = output_height * output_width;
  int t_dimw = output_width;

  int o_dimchw = input_channel * input_height * input_width;
  int o_dimhw = input_height * input_width;
  int o_dimw = input_width;

  int nelems = kernel_size * kernel_size * input_channel;

  __shared__ T prod_sum[THREADS_PER_BLOCK];
  prod_sum[tch_off] = 0;

  for (int tc = tch_off; tc < output_channel; tc += THREADS_PER_BLOCK) {
    int i2 = (tc % displacement_size - displacement_rad) * stride2;
    int j2 = (tc / displacement_size - displacement_rad) * stride2;

    int index2 = n * p_dimchw + (h + j2) * p_dimcw + (w + i2) * p_dimc + c;

    T val2 = rinput2[index2];
    for (int j = ymin; j <= ymax; ++j) {
      for (int i = xmin; i <= xmax; ++i) {
        int t_index = n * t_dimchw + tc * t_dimhw + j * t_dimw + i;
        prod_sum[tch_off] += grad_output[t_index] * val2;
      }
    }
  }

  __syncthreads();

  if (tch_off == 0) {
    T reduce_sum = 0;
    for (int index = 0; index < THREADS_PER_BLOCK; index++) {
      reduce_sum += prod_sum[index];
    }
    const int index1 =
        n * o_dimchw + c * o_dimhw + (h - pad_size) * o_dimw + (w - pad_size);
    grad_input1[index1] = static_cast<T>(reduce_sum / nelems);
  }
}

template <typename T>
__global__ void correlation_backward_input2(
    int item, T *grad_input2, const int input_channel, const int input_height,
    const int input_width, const T *grad_output, const int output_channel,
    const int output_height, const int output_width, const T *rinput1,
    const int pad_size, const int kernel_size, const int max_displacement,
    const int stride1, const int stride2) {
  int n = item;
  int h = blockIdx.x * stride1 + pad_size;
  int w = blockIdx.y * stride1 + pad_size;
  int c = blockIdx.z;

  int tch_off = threadIdx.x;

  int kernel_rad = (kernel_size - 1) / 2;
  int displacement_rad = max_displacement / stride2;
  int displacement_size = 2 * displacement_rad + 1;

  int p_input_width = input_width + 2 * pad_size;
  int p_input_height = input_height + 2 * pad_size;
  int p_dimchw = input_channel * p_input_height * p_input_width;
  int p_dimcw = input_channel * p_input_width;
  int p_dimc = input_channel;

  int t_dimchw = output_channel * output_height * output_width;
  int t_dimhw = output_height * output_width;
  int t_dimw = output_width;

  int o_dimchw = input_channel * input_height * input_width;
  int o_dimhw = input_height * input_width;
  int o_dimw = input_width;

  int nelems = kernel_size * kernel_size * input_channel;

  __shared__ T prod_sum[THREADS_PER_BLOCK];
  prod_sum[tch_off] = 0;

  for (int tc = tch_off; tc < output_channel; tc += THREADS_PER_BLOCK) {
    int i2 = (tc % displacement_size - displacement_rad) * stride2;
    int j2 = (tc / displacement_size - displacement_rad) * stride2;

    int xmin = (w - kernel_rad - max_displacement - i2) / stride1;
    int ymin = (h - kernel_rad - max_displacement - j2) / stride1;

    int xmax = (w + kernel_rad - max_displacement - i2) / stride1;
    int ymax = (h + kernel_rad - max_displacement - j2) / stride1;

    if (xmax < 0 || ymax < 0 || xmin >= output_width || ymin >= output_height) {
      continue;
    }

    if (xmin > xmax || ymin > ymax) {
      continue;
    }

    xmin = max(0, xmin);
    xmax = min(output_width - 1, xmax);

    ymin = max(0, ymin);
    ymax = min(output_height - 1, ymax);

    int index1 = n * p_dimchw + (h - j2) * p_dimcw + (w - i2) * p_dimc + c;
    T val1 = rinput1[index1];
    for (int j = ymin; j <= ymax; ++j) {
      for (int i = xmin; i <= xmax; ++i) {
        int t_index = n * t_dimchw + tc * t_dimhw + j * t_dimw + i;
        prod_sum[tch_off] += grad_output[t_index] * val1;
      }
    }
  }

  __syncthreads();

  if (tch_off == 0) {
    T reduce_sum = 0;
    for (int index = 0; index < THREADS_PER_BLOCK; index++) {
      reduce_sum += prod_sum[index];
    }
    const int index2 =
        n * o_dimchw + c * o_dimhw + (h - pad_size) * o_dimw + (w - pad_size);
    grad_input2[index2] = static_cast<T>(reduce_sum / nelems);
  }
}

template <typename T>
class CorrelationCUDAGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    PADDLE_ENFORCE_EQ(platform::is_gpu_place(ctx.GetPlace()), true,
                      platform::errors::InvalidArgument(
                          "Correlation only supports GPU now."));
    const auto *input1 = ctx.Input<Tensor>("Input1");
    const auto *input2 = ctx.Input<Tensor>("Input2");
    const auto *grad_output =
        ctx.Input<Tensor>(framework::GradVarName("Output"));
    const int pad_size = ctx.Attr<int>("pad_size");
    const int kernel_size = ctx.Attr<int>("kernel_size");
    const int stride1 = ctx.Attr<int>("stride1");
    const int stride2 = ctx.Attr<int>("stride2");
    const int max_displacement = ctx.Attr<int>("max_displacement");
    const int corr_type_multiply = ctx.Attr<int>("corr_type_multiply");

    auto *grad_input1 = ctx.Output<Tensor>(framework::GradVarName("Input1"));
    grad_input1->mutable_data<T>(ctx.GetPlace());
    auto *grad_input2 = ctx.Output<Tensor>(framework::GradVarName("Input2"));
    grad_input2->mutable_data<T>(ctx.GetPlace());
    auto &dev_ctx = ctx.template device_context<platform::CUDADeviceContext>();

    auto in_dims = input1->dims();
    int N = in_dims[0];
    int C = in_dims[1];
    int H = in_dims[2];
    int W = in_dims[3];

    int padded_input_height = H + 2 * pad_size;
    int padded_input_width = W + 2 * pad_size;

    Tensor rinput1 = ctx.AllocateTmpTensor<T, platform::CUDADeviceContext>(
        {N, padded_input_height, padded_input_width, C}, dev_ctx);
    rinput1.mutable_data<T>(ctx.GetPlace());

    Tensor rinput2 = ctx.AllocateTmpTensor<T, platform::CUDADeviceContext>(
        {N, padded_input_height, padded_input_width, C}, dev_ctx);
    rinput2.mutable_data<T>(ctx.GetPlace());

    set_zero<<<(rinput1.numel() + 512 - 1) / 512, 512, 0, dev_ctx.stream()>>>(
        rinput1.data<T>(), rinput1.numel());
    set_zero<<<(rinput2.numel() + 512 - 1) / 512, 512, 0, dev_ctx.stream()>>>(
        rinput2.data<T>(), rinput2.numel());
    set_zero<<<(grad_input1->numel() + 512 - 1) / 512, 512, 0,
               dev_ctx.stream()>>>(grad_input1->data<T>(),
                                   grad_input1->numel());
    set_zero<<<(grad_input2->numel() + 512 - 1) / 512, 512, 0,
               dev_ctx.stream()>>>(grad_input2->data<T>(),
                                   grad_input2->numel());

    auto grad_out_dims = grad_output->dims();
    int GOC = grad_out_dims[1];
    int GOH = grad_out_dims[2];
    int GOW = grad_out_dims[3];

    dim3 blocks_grid(N, H, W);
    dim3 threads_block(THREADS_PER_BLOCK);

    channel_first<T><<<blocks_grid, threads_block, 0, dev_ctx.stream()>>>(
        input1->data<T>(), rinput1.data<T>(), C, H, W, pad_size);
    channel_first<T><<<blocks_grid, threads_block, 0, dev_ctx.stream()>>>(
        input2->data<T>(), rinput2.data<T>(), C, H, W, pad_size);

    dim3 threadsPerBlock(THREADS_PER_BLOCK);
    dim3 totalBlocksCorr(H, W, C);

    for (int n = 0; n < N; n++) {
      correlation_backward_input1<
          T><<<totalBlocksCorr, threadsPerBlock, 0, dev_ctx.stream()>>>(
          n, grad_input1->data<T>(), C, H, W, grad_output->data<T>(), GOC, GOH,
          GOW, rinput2.data<T>(), pad_size, kernel_size, max_displacement,
          stride1, stride2);
    }

    for (int n = 0; n < N; n++) {
      correlation_backward_input2<
          T><<<totalBlocksCorr, threadsPerBlock, 0, dev_ctx.stream()>>>(
          n, grad_input2->data<T>(), C, H, W, grad_output->data<T>(), GOC, GOH,
          GOW, rinput1.data<T>(), pad_size, kernel_size, max_displacement,
          stride1, stride2);
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(correlation, ops::CorrelationCUDAKernel<float>,
                        ops::CorrelationCUDAKernel<double>);
REGISTER_OP_CUDA_KERNEL(correlation_grad, ops::CorrelationCUDAGradKernel<float>,
                        ops::CorrelationCUDAGradKernel<double>);
