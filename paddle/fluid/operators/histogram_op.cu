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

#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/operators/histogram_op.h"
#include "paddle/fluid/platform/cuda_primitives.h"
#include "paddle/fluid/platform/gpu_launch_config.h"
#include "paddle/fluid/platform/hostdevice.h"

namespace paddle {
namespace operators {

using IndexType = int64_t;
using Tensor = framework::Tensor;
using platform::PADDLE_CUDA_NUM_THREADS;

inline int GET_BLOCKS(const int N) {
  return (N + PADDLE_CUDA_NUM_THREADS - 1) / PADDLE_CUDA_NUM_THREADS;
}

template <typename T, typename IndexType>
__device__ static IndexType GetBin(T input_value, T min_value, T max_value,
                                   int64_t nbins) {
  IndexType bin = static_cast<int>((input_value - min_value) * nbins /
                                   (max_value - min_value));
  IndexType output_index = bin < nbins - 1 ? bin : nbins - 1;
  return output_index;
}

template <typename T, typename IndexType>
__global__ void KernelHistogram(const T* input, const int total_elements,
                                const int64_t nbins, const T min_value,
                                const T max_value, int64_t* output) {
  extern __shared__ int64_t buf_hist[];
  for (int i = threadIdx.x; i < nbins; i += blockDim.x) {
    buf_hist[i] = 0;
  }
  __syncthreads();

  CUDA_KERNEL_LOOP(input_index, total_elements) {
    // const IndexType input_index = threadIdx.x + blockIdx.x * blockDim.x;
    const auto input_value = input[input_index];
    if (input_value >= min_value && input_value <= max_value) {
      const IndexType output_index =
          GetBin<T, IndexType>(input_value, min_value, max_value, nbins);
      paddle::platform::CudaAtomicAdd(&buf_hist[output_index], 1);
    }
  }
  __syncthreads();

  for (int i = threadIdx.x; i < nbins; i += blockDim.x) {
    paddle::platform::CudaAtomicAdd(&output[i], buf_hist[i]);
  }
}

template <typename DeviceContext, typename T>
class HistogramCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    PADDLE_ENFORCE_EQ(
        platform::is_gpu_place(context.GetPlace()), true,
        platform::errors::InvalidArgument("It must use CUDAPlace."));

    const Tensor* input = context.Input<framework::Tensor>("X");
    Tensor* output = context.Output<framework::Tensor>("Out");
    auto& nbins = context.Attr<int64_t>("bins");
    auto& minval = context.Attr<int>("min");
    auto& maxval = context.Attr<int>("max");

    const T* input_data = input->data<T>();
    const int input_numel = input->numel();

    T output_min = static_cast<T>(minval);
    T output_max = static_cast<T>(maxval);

    if (output_min == output_max) {
      auto input_x = framework::EigenVector<T>::Flatten(*input);

      framework::Tensor input_min_t, input_max_t;
      auto* input_min_data =
          input_min_t.mutable_data<T>({1}, context.GetPlace());
      auto* input_max_data =
          input_max_t.mutable_data<T>({1}, context.GetPlace());
      auto input_min_scala = framework::EigenScalar<T>::From(input_min_t);
      auto input_max_scala = framework::EigenScalar<T>::From(input_max_t);

      auto* place =
          context.template device_context<DeviceContext>().eigen_device();
      input_min_scala.device(*place) = input_x.minimum();
      input_max_scala.device(*place) = input_x.maximum();

      Tensor input_min_cpu, input_max_cpu;
      TensorCopySync(input_min_t, platform::CPUPlace(), &input_min_cpu);
      TensorCopySync(input_max_t, platform::CPUPlace(), &input_max_cpu);

      output_min = input_min_cpu.data<T>()[0];
      output_max = input_max_cpu.data<T>()[0];
    }
    if (output_min == output_max) {
      output_min = output_min - 1;
      output_max = output_max + 1;
    }

    PADDLE_ENFORCE_EQ(
        (std::isinf(static_cast<float>(output_min)) ||
         std::isnan(static_cast<float>(output_max)) ||
         std::isinf(static_cast<float>(output_min)) ||
         std::isnan(static_cast<float>(output_max))),
        false, platform::errors::OutOfRange("range of min, max is not finite"));
    PADDLE_ENFORCE_GE(
        output_max, output_min,
        platform::errors::InvalidArgument(
            "max must be larger or equal to min. If min and max are both zero, "
            "the minimum and maximum values of the data are used. "
            "But received max is %d, min is %d",
            maxval, minval));

    int64_t* out_data = output->mutable_data<int64_t>(context.GetPlace());
    math::SetConstant<platform::CUDADeviceContext, int64_t>()(
        context.template device_context<platform::CUDADeviceContext>(), output,
        static_cast<int64_t>(0));

    auto stream =
        context.template device_context<platform::CUDADeviceContext>().stream();
    KernelHistogram<
        T, IndexType><<<GET_BLOCKS(input_numel), PADDLE_CUDA_NUM_THREADS,
                        nbins * sizeof(int64_t), stream>>>(
        input_data, input_numel, nbins, output_min, output_max, out_data);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(
    histogram,
    ops::HistogramCUDAKernel<paddle::platform::CUDADeviceContext, int>,
    ops::HistogramCUDAKernel<paddle::platform::CUDADeviceContext, int64_t>,
    ops::HistogramCUDAKernel<paddle::platform::CUDADeviceContext, float>,
    ops::HistogramCUDAKernel<paddle::platform::CUDADeviceContext, double>);
