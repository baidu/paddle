/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */
#include "paddle/fluid/operators/elementwise/elementwise_div_op.h"
#include "paddle/fluid/operators/elementwise/elementwise_op_broadcast.cu.h"
#include "paddle/fluid/platform/complex.h"
#include "paddle/fluid/platform/float16.h"

namespace ops = paddle::operators;
namespace plat = paddle::platform;

namespace paddle {
namespace operators {

template <typename T, typename Enable = void>
struct CudaDivFunctor {
  inline HOSTDEVICE T operator()(const T* args) const {
    return args[0] / args[1];
  }
};

template <typename T>
struct CudaDivFunctor<T,
                      typename std::enable_if_t<std::is_integral<T>::value>> {
  inline HOSTDEVICE T operator()(const T* args) const {
    PADDLE_ENFORCE(args[1] != 0,
                   "Invalid Argument Error: Integer division by zero "
                   "encountered in divide. Please check the input value.");
    return args[0] / args[1];
  }
};

template <typename T>
class ElementwiseDivKernel<platform::CUDADeviceContext, T>
    : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    std::vector<const framework::Tensor*> ins;
    std::vector<framework::Tensor*> outs;
    const auto& cuda_ctx =
        ctx.template device_context<platform::CUDADeviceContext>();

    int axis = PackTensorsIntoVector<T>(ctx, &ins, &outs);
    LaunchElementwiseCudaKernel<ElementwiseType::kBinary, T, T>(
        cuda_ctx, ins, &outs, axis, CudaDivFunctor<T>());
  }
};

template <typename T>
static __global__ void SimpleElemwiseDivGradCUDAKernel(const T* x, const T* y,
                                                       const T* out,
                                                       const T* dout,
                                                       int64_t size, T* dx,
                                                       T* dy) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  while (col < size) {
    T o = dout[col];
    dx[col] = o / y[col];
    dy[col] = -o * out[col] / y[col];
    col += blockDim.x * gridDim.x;
  }
}

template <>
__global__ void
SimpleElemwiseDivGradCUDAKernel<paddle::platform::complex<float>>(
    const paddle::platform::complex<float>* x,
    const paddle::platform::complex<float>* y,
    const paddle::platform::complex<float>* out,
    const paddle::platform::complex<float>* dout, int64_t size,
    paddle::platform::complex<float>* dx,
    paddle::platform::complex<float>* dy) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  while (col < size) {
    paddle::platform::complex<float> o = dout[col];
    paddle::platform::complex<float> y_conj(y[col].real, -y[col].imag);
    paddle::platform::complex<float> out_div_y_conj((out[col] / y[col]).real,
                                                    -(out[col] / y[col]).imag);
    dx[col] = o / y_conj;
    dy[col] = -o * out_div_y_conj;
    col += blockDim.x * gridDim.x;
  }
}

template <>
__global__ void
SimpleElemwiseDivGradCUDAKernel<paddle::platform::complex<double>>(
    const paddle::platform::complex<double>* x,
    const paddle::platform::complex<double>* y,
    const paddle::platform::complex<double>* out,
    const paddle::platform::complex<double>* dout, int64_t size,
    paddle::platform::complex<double>* dx,
    paddle::platform::complex<double>* dy) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  while (col < size) {
    paddle::platform::complex<double> o = dout[col];
    paddle::platform::complex<double> y_conj(y[col].real, -y[col].imag);
    paddle::platform::complex<double> out_div_y_conj((out[col] / y[col]).real,
                                                     -(out[col] / y[col]).imag);
    dx[col] = o / y_conj;
    dy[col] = -o * out_div_y_conj;
    col += blockDim.x * gridDim.x;
  }
}

template <typename DeviceContext, typename T>
typename std::enable_if<
    std::is_same<DeviceContext, plat::CUDADeviceContext>::value>::type
elementwise_div_grad(const framework::ExecutionContext& ctx,
                     const framework::Tensor* x, const framework::Tensor* y,
                     const framework::Tensor* out,
                     const framework::Tensor* dout, framework::Tensor* dx,
                     framework::Tensor* dy) {
  dim3 block_size = dim3(PADDLE_CUDA_THREAD_SIZE, 1);
  auto size = x->numel();
  dim3 grid_size =
      dim3((size + PADDLE_CUDA_THREAD_SIZE - 1) / PADDLE_CUDA_THREAD_SIZE, 1);
  SimpleElemwiseDivGradCUDAKernel<
      T><<<grid_size, block_size, 0,
           ctx.template device_context<plat::CUDADeviceContext>().stream()>>>(
      x->data<T>(), y->data<T>(), out->data<T>(), dout->data<T>(), size,
      dx->mutable_data<T>(ctx.GetPlace()), dy->mutable_data<T>(ctx.GetPlace()));
}

}  // namespace operators
}  // namespace paddle

REGISTER_OP_CUDA_KERNEL(
    elementwise_div,
    ops::ElementwiseDivKernel<paddle::platform::CUDADeviceContext, float>,
    ops::ElementwiseDivKernel<paddle::platform::CUDADeviceContext,
                              paddle::platform::float16>,
    ops::ElementwiseDivKernel<paddle::platform::CUDADeviceContext, double>,
    ops::ElementwiseDivKernel<paddle::platform::CUDADeviceContext, int>,
    ops::ElementwiseDivKernel<paddle::platform::CUDADeviceContext, int64_t>,
    ops::ElementwiseDivKernel<paddle::platform::CUDADeviceContext,
                              paddle::platform::complex<float>>,
    ops::ElementwiseDivKernel<paddle::platform::CUDADeviceContext,
                              paddle::platform::complex<double>>);
REGISTER_OP_CUDA_KERNEL(
    elementwise_div_grad,
    ops::ElementwiseDivGradKernel<paddle::platform::CUDADeviceContext, float>,
    ops::ElementwiseDivGradKernel<paddle::platform::CUDADeviceContext,
                                  paddle::platform::float16>,
    ops::ElementwiseDivGradKernel<paddle::platform::CUDADeviceContext, double>,
    ops::ElementwiseDivGradKernel<paddle::platform::CUDADeviceContext, int>,
    ops::ElementwiseDivGradKernel<paddle::platform::CUDADeviceContext, int64_t>,
    ops::ElementwiseDivGradKernel<paddle::platform::CUDADeviceContext,
                                  paddle::platform::complex<float>>,
    ops::ElementwiseDivGradKernel<paddle::platform::CUDADeviceContext,
                                  paddle::platform::complex<double>>);
REGISTER_OP_CUDA_KERNEL(
    elementwise_div_grad_grad,
    ops::ElementwiseDivDoubleGradKernel<paddle::platform::CUDADeviceContext,
                                        float>,
    ops::ElementwiseDivDoubleGradKernel<paddle::platform::CUDADeviceContext,
                                        paddle::platform::float16>,
    ops::ElementwiseDivDoubleGradKernel<paddle::platform::CUDADeviceContext,
                                        double>,
    ops::ElementwiseDivDoubleGradKernel<paddle::platform::CUDADeviceContext,
                                        int>,
    ops::ElementwiseDivDoubleGradKernel<paddle::platform::CUDADeviceContext,
                                        int64_t>,
    ops::ElementwiseDivDoubleGradKernel<paddle::platform::CUDADeviceContext,
                                        paddle::platform::complex<float>>,
    ops::ElementwiseDivDoubleGradKernel<paddle::platform::CUDADeviceContext,
                                        paddle::platform::complex<double>>);
