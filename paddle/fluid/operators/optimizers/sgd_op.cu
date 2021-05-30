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

#include <algorithm>
#include "paddle/fluid/operators/amp/fp16_type_traits.h"
#include "paddle/fluid/operators/optimizers/sgd_op.h"
#include "paddle/fluid/platform/cuda_primitives.h"
#include "paddle/fluid/platform/float16.h"

namespace paddle {
namespace operators {

namespace {

template <typename T, typename MT>
__global__ void SGDKernel(const T* g, const T* p, const MT* learning_rate,
                          const int num, T* p_out, const MT* master_param,
                          MT* master_param_out) {
  MT lr = learning_rate[0];
  CUDA_KERNEL_LOOP(i, num) {
    MT g_data = static_cast<MT>(g[id]);
    MT p_data = master_param ? master_param[i] : static_cast<MT>(p[i]);
    p_out[i] = static_cast<T>(p_data - lr * g_data);
    if (master_param_out) {
      master_param_out[id] = p;
    }
  }
}

template <typename T, typename MT>
__global__ void SparseSGDFunctorKernel(const T* selected_rows,
                                       const int64_t* rows,
                                       const MT* learning_rate, T* tensor_out,
                                       int64_t row_numel, int64_t limit,
                                       const MT* master_param,
                                       MT* master_param_out) {
  for (int64_t i = blockIdx.x; i < limit; i += gridDim.x) {
    const T* selected_rows_ptr = selected_rows + i * row_numel;
    T* tensor_out_ptr = tensor_out + rows[i] * row_numel;
    for (int64_t index = threadIdx.x; index < row_numel; index += blockDim.x) {
      // Since index in rows of SelectedRows can be duplicate, we have to use
      // Atomic Operation to avoid concurrent write error.
      paddle::platform::CudaAtomicAdd(
          tensor_out_ptr + index,
          -static_cast<T>(1.0) * learning_rate[0] * selected_rows_ptr[index]);
    }
  }
}
}  // namespace

template <typename T>
class SGDOpKernel<platform::CUDADeviceContext, T>
    : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    const auto* param_var = ctx.InputVar("Param");
    PADDLE_ENFORCE_EQ(param_var->IsType<framework::LoDTensor>(), true,
                      platform::errors::InvalidArgument(
                          "The Var(%s)'s type should be LoDTensor, "
                          "but the received is %s",
                          ctx.InputNames("Param").front(),
                          paddle::framework::ToTypeName(param_var->Type())));

    using MPDType = typename details::MPTypeTrait<T>::Type;

    auto* param = ctx.Input<framework::Tensor>("Param");
    auto* param_out = ctx.Output<framework::Tensor>("ParamOut");
    auto* learning_rate = ctx.Input<framework::Tensor>("LearningRate");

    auto* grad_var = ctx.InputVar("Grad");

    const bool multi_precision = ctx.Attr<bool>("multi_precision");
    const LoDTensor* master_param = nullptr;
    LoDTensor* master_param_out = nullptr;

    if (multi_precision) {
      bool has_master =
          ctx.HasInput("MasterParam") && ctx.HasOutput("MasterParamOut");
      PADDLE_ENFORCE_EQ(has_master, true,
                        platform::errors::InvalidArgument(
                            "The Input(MasterParam) and Output(MasterParamOut) "
                            "should not be null when "
                            "the attr `multi_precision` is true"));
      master_param = ctx.Input<LoDTensor>("MasterParam");
      master_param_out = ctx.Output<LoDTensor>("MasterParamOut");
    }

    const MPDType* master_in_data =
        multi_precision ? master_param->data<MPDType>() : nullptr;
    MPDType* master_out_data =
        multi_precision
            ? master_param_out->mutable_data<MPDType>(ctx.GetPlace())
            : nullptr;

    // Actually, all tensors are LoDTensor except SelectedRows.
    if (grad_var->IsType<framework::LoDTensor>()) {
      param_out->mutable_data<T>(ctx.GetPlace());
      auto* grad = ctx.Input<framework::Tensor>("Grad");
      // LOG(ERROR) << "grad";
      // LOG(ERROR) << ctx.op().Input("Grad");
      auto* grad_data = grad->data<T>();
      // LOG(ERROR) << "param";
      auto* param_data = param->data<T>();
      // LOG(ERROR) << "fin";
      auto* param_out_data = param_out->data<T>();

      int block = 512;
      int grid = (param->numel() + block - 1) / block;

      SGDKernel<
          T, MPDType><<<grid, block, 0, ctx.cuda_device_context().stream()>>>(
          grad_data, param_data, learning_rate->data<MPDType>(), param->numel(),
          param_out_data, master_in_data, master_out_data);

    } else if (grad_var->IsType<framework::SelectedRows>()) {
      // TODO(qijun): In Sparse SGD operator, in-place update is enforced.
      // This manual optimization brings difficulty to track data dependency.
      // It's better to find a more elegant solution.
      PADDLE_ENFORCE_EQ(
          param, param_out,
          platform::errors::InvalidArgument(
              "The input tensor Param of SgdOp should be equal with ParamOut "
              "if variable's type is SelectedRows."));
      auto* grad = ctx.Input<framework::SelectedRows>("Grad");
      if (grad->rows().size() == 0) {
        VLOG(3) << "grad row size is 0!!";
        return;
      }

      auto in_height = grad->height();
      auto out_dims = param_out->dims();
      PADDLE_ENFORCE_EQ(in_height, out_dims[0],
                        platform::errors::InvalidArgument(
                            "The input tensor Grad's height of SgdOp should be "
                            "equal with ParamOut's dims. But received Grad's "
                            "height [%s] and ParamOut's dims [%s]",
                            in_height, out_dims[0]));

      auto& in_value = grad->value();
      auto& in_rows = grad->rows();

      int64_t in_row_numel = in_value.numel() / in_rows.size();
      PADDLE_ENFORCE_EQ(in_row_numel, param_out->numel() / in_height,
                        platform::errors::InvalidArgument(
                            "The in_row_numel of SgdOp should be equal with "
                            "param_out's numel / in_height."));

      auto* in_data = in_value.data<T>();
      auto* out_data = param_out->data<T>();

      const int kThreadsPerBlock = 256;
      int thread_x = kThreadsPerBlock;
      int max_threads = ctx.cuda_device_context().GetMaxPhysicalThreadCount();
      int max_blocks = std::max(max_threads / kThreadsPerBlock, 1);

      SparseSGDFunctorKernel<T, MPDType><<<
          max_blocks, thread_x, 0, ctx.cuda_device_context().stream()>>>(
          in_data, in_rows.CUDAData(ctx.GetPlace()), learning_rate->data<T>(),
          out_data, in_row_numel, in_rows.size(), master_in_data,
          master_out_data);

    } else {
      PADDLE_ENFORCE_EQ(false, true,
                        platform::errors::PermissionDenied(
                            "Unsupported Variable Type of Grad "
                            "in SgdOp. Excepted LodTensor or "
                            "SelectedRows, But received [%s]",
                            paddle::framework::ToTypeName(grad_var->Type())));
    }
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;
REGISTER_OP_CUDA_KERNEL(
    sgd, ops::SGDOpKernel<paddle::platform::CUDADeviceContext, float>,
    ops::SGDOpKernel<paddle::platform::CUDADeviceContext, double>,
    ops::SGDOpKernel<paddle::platform::CUDADeviceContext, plat::float16>);
