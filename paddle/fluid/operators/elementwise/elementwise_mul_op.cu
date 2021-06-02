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

#include "paddle/fluid/operators/elementwise/elementwise_mul_op.h"
#include "paddle/fluid/operators/elementwise/elementwise_op_broadcast.cu.h"
#include "paddle/fluid/operators/elementwise/elementwise_op_function.cu.h"
#include "paddle/fluid/platform/complex.h"
#include "paddle/fluid/platform/float16.h"

namespace ops = paddle::operators;
namespace plat = paddle::platform;

namespace paddle {
namespace operators {

template <typename T>
struct CudaMulFunctor {
  inline HOSTDEVICE T operator()(const T* args) const {
    return args[0] * args[1];
  }
};

template <typename T>
class ElementwiseMulKernel<platform::CUDADeviceContext, T>
    : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    // int axis = -1;
    // auto x_var = ctx.InputVar("X");
    // PADDLE_ENFORCE_NOT_NULL(
    //     x_var, platform::errors::InvalidArgument(
    //                "Cannot get input Variable X, Variable name = %s.",
    //                ctx.InputName("X")));
    // auto* y = ctx.Input<framework::LoDTensor>("Y");

    // framework::Tensor x, *z;
    // std::vector<const framework::Tensor*> ins;
    // std::vector<framework::Tensor*> outs;
    // const auto& cuda_ctx =
    //     ctx.template device_context<platform::CUDADeviceContext>();

    // if (x_var->IsType<framework::LoDTensor>()) {
    //   x = x_var->Get<framework::LoDTensor>();
    //   z = ctx.Output<framework::LoDTensor>("Out");
    //   axis = PackTensorsIntoVector<T>(ctx, &ins, &outs);
    // } else if (x_var->IsType<framework::SelectedRows>()) {
    //   PADDLE_ENFORCE_EQ(y->dims().size() == 1 && y->dims()[0] == 1, true,
    //                     platform::errors::InvalidArgument(
    //                         "For elementwise_op, if X is Sparse, Y must be "
    //                         "scalar. But reveived the size of Y = %s.",
    //                         y->dims().size()));
    //   auto& x_sele = x_var->Get<framework::SelectedRows>();
    //   auto out_sele = ctx.Output<framework::SelectedRows>("Out");
    //   x = x_sele.value();
    //   out_sele->set_rows(x_sele.rows());
    //   out_sele->set_height(x_sele.height());
    //   out_sele->mutable_value()->Resize(x_sele.value().dims());
    //   out_sele->mutable_value()->mutable_data(ctx.GetPlace(), x.type());
    //   z = ctx.Output<framework::SelectedRows>("Out")->mutable_value();
    //   z->mutable_data<T>(ctx.GetPlace());
    //   outs.emplace_back(z);
    //   ins.emplace_back(&x);
    //   ins.emplace_back(y);

    //   axis = ctx.HasAttr("axis") ? ctx.Attr<int>("axis") : -1;
    //   axis = axis == -1 ? std::abs(y->dims().size() - x.dims().size()) :
    //   axis;
    // } else {
    //   PADDLE_THROW(platform::errors::InvalidArgument(
    //       "X's type[%s] is not supported by elementwise_op. X's type should
    //       be "
    //       "LoDTensor or SelectedRows.",
    //       framework::ToTypeName(x_var->Type())));
    // }
    std::vector<const framework::Tensor*> ins;
    std::vector<framework::Tensor*> outs;
    const auto& cuda_ctx =
        ctx.template device_context<platform::CUDADeviceContext>();

    int axis = PackTensorsIntoVector<T>(ctx, &ins, &outs);
    LaunchElementwiseCudaKernel<ElementwiseType::kBinary, T, T>(
        cuda_ctx, ins, &outs, axis, CudaMulFunctor<T>());
  }
};

template <typename T>
static __global__ void SimpleElemwiseMulGradCUDAKernel(const T* x, const T* y,
                                                       const T* out,
                                                       const T* dout,
                                                       int64_t size, T* dx,
                                                       T* dy) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  while (col < size) {
    T o = dout[col];
    dx[col] = y[col] * o;
    dy[col] = x[col] * o;
    col += blockDim.x * gridDim.x;
  }
}

template <>
__global__ void SimpleElemwiseMulGradCUDAKernel<plat::complex<float>>(
    const plat::complex<float>* x, const plat::complex<float>* y,
    const plat::complex<float>* out, const plat::complex<float>* dout,
    int64_t size, plat::complex<float>* dx, plat::complex<float>* dy) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  while (col < size) {
    plat::complex<float> o = dout[col];
    dx[col] = plat::complex<float>(y[col].real, -y[col].imag) * o;
    dy[col] = plat::complex<float>(x[col].real, -x[col].imag) * o;
    col += blockDim.x * gridDim.x;
  }
}

template <>
__global__ void SimpleElemwiseMulGradCUDAKernel<plat::complex<double>>(
    const plat::complex<double>* x, const plat::complex<double>* y,
    const plat::complex<double>* out, const plat::complex<double>* dout,
    int64_t size, plat::complex<double>* dx, plat::complex<double>* dy) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  while (col < size) {
    plat::complex<double> o = dout[col];
    dx[col] = plat::complex<double>(y[col].real, -y[col].imag) * o;
    dy[col] = plat::complex<double>(x[col].real, -x[col].imag) * o;
    col += blockDim.x * gridDim.x;
  }
}

template <typename DeviceContext, typename T>
typename std::enable_if<
    std::is_same<DeviceContext, plat::CUDADeviceContext>::value>::type
elementwise_mul_grad(const framework::ExecutionContext& ctx,
                     const framework::Tensor* x, const framework::Tensor* y,
                     const framework::Tensor* out,
                     const framework::Tensor* dout, framework::Tensor* dx,
                     framework::Tensor* dy) {
  dim3 block_size = dim3(PADDLE_CUDA_THREAD_SIZE, 1);
  auto size = x->numel();
  dim3 grid_size =
      dim3((size + PADDLE_CUDA_THREAD_SIZE - 1) / PADDLE_CUDA_THREAD_SIZE, 1);
  SimpleElemwiseMulGradCUDAKernel<
      T><<<grid_size, block_size, 0,
           ctx.template device_context<plat::CUDADeviceContext>().stream()>>>(
      x->data<T>(), y->data<T>(), out->data<T>(), dout->data<T>(), size,
      dx->mutable_data<T>(ctx.GetPlace()), dy->mutable_data<T>(ctx.GetPlace()));
}

}  // namespace operators
}  // namespace paddle

REGISTER_OP_CUDA_KERNEL(
    elementwise_mul, ops::ElementwiseMulKernel<plat::CUDADeviceContext, float>,
    ops::ElementwiseMulKernel<plat::CUDADeviceContext, double>,
    ops::ElementwiseMulKernel<plat::CUDADeviceContext, int>,
    ops::ElementwiseMulKernel<plat::CUDADeviceContext, int64_t>,
    ops::ElementwiseMulKernel<plat::CUDADeviceContext, plat::float16>,
    ops::ElementwiseMulKernel<plat::CUDADeviceContext, plat::complex<float>>,
    ops::ElementwiseMulKernel<plat::CUDADeviceContext, plat::complex<double>>);
REGISTER_OP_CUDA_KERNEL(
    elementwise_mul_grad,
    ops::ElementwiseMulGradKernel<plat::CUDADeviceContext, float>,
    ops::ElementwiseMulGradKernel<plat::CUDADeviceContext, double>,
    ops::ElementwiseMulGradKernel<plat::CUDADeviceContext, int>,
    ops::ElementwiseMulGradKernel<plat::CUDADeviceContext, int64_t>,
    ops::ElementwiseMulGradKernel<plat::CUDADeviceContext, plat::float16>,
    ops::ElementwiseMulGradKernel<plat::CUDADeviceContext,
                                  plat::complex<float>>,
    ops::ElementwiseMulGradKernel<plat::CUDADeviceContext,
                                  plat::complex<double>>);
REGISTER_OP_CUDA_KERNEL(
    elementwise_mul_grad_grad,
    ops::ElementwiseMulDoubleGradKernel<plat::CUDADeviceContext, float>,
    ops::ElementwiseMulDoubleGradKernel<plat::CUDADeviceContext, double>,
    ops::ElementwiseMulDoubleGradKernel<plat::CUDADeviceContext, int>,
    ops::ElementwiseMulDoubleGradKernel<plat::CUDADeviceContext, int64_t>,
    ops::ElementwiseMulDoubleGradKernel<plat::CUDADeviceContext, plat::float16>,
    ops::ElementwiseMulDoubleGradKernel<plat::CUDADeviceContext,
                                        plat::complex<float>>,
    ops::ElementwiseMulDoubleGradKernel<plat::CUDADeviceContext,
                                        plat::complex<double>>);
