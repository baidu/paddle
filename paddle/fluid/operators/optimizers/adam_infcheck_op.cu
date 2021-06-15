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
#include "paddle/fluid/operators/amp/fp16_type_traits.h"
#include "paddle/fluid/operators/optimizers/adam_gpu_utils.h"
#include "paddle/fluid/operators/optimizers/adam_infcheck_op.h"
#include "paddle/fluid/platform/float16.h"
namespace paddle {
namespace operators {

template <typename T>
class AdamInfCheckOpCUDAKernel : public framework::OpKernel<T> {
 public:
  AdamInfCheckOpCUDAKernel()
      : _adam_op_cuda_kernel(new AdamOpCUDAKernel<T>()) {}
  void Compute(const framework::ExecutionContext& ctx) const override {
    using MPDType = typename details::MPTypeTrait<T>::Type;
    const auto infcheck_tensor = ctx.Input<framework::Tensor>("InfCheck");
    const auto infcheck_flag = GetDataToCPU<bool>(*infcheck_tensor);
    if (infcheck_flag) {
      VLOG(3) << "The GPU input tensor exit inf or nan";
      ctx.device_context().Wait();
      CopyTensorSameContext<T>(ctx, "Param", "ParamOut");
      CopyTensorSameContext<T>(ctx, "Moment1", "Moment1Out");
      CopyTensorSameContext<T>(ctx, "Moment2", "Moment2Out");
      CopyTensorSameContext<T>(ctx, "Beta1Pow", "Beta1PowOut");
      CopyTensorSameContext<T>(ctx, "Beta2Pow", "Beta2PowOut");
      return;
    }
    VLOG(3) << "The GPU input tensor not exit inf or nan";
    return _adam_op_cuda_kernel->Compute(ctx);
  }

 private:
  std::unique_ptr<AdamOpCUDAKernel<T>> _adam_op_cuda_kernel;
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_CUDA_KERNEL(adam_infcheck, ops::AdamInfCheckOpCUDAKernel<float>,
                        ops::AdamInfCheckOpCUDAKernel<double>,
                        ops::AdamInfCheckOpCUDAKernel<plat::float16>);
