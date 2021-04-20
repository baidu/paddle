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
#include <string>

#include "paddle/fluid/framework/op_registry.h"

namespace paddle {
namespace operators {

class CSyncCalcStreamOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {}

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(framework::proto::VarType::FP32,
                                   ctx.GetPlace());
  }
};

class CSyncCalcStreamOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() {
    AddInput("X", "(Tensor) Dependency of the variable need to sync");
    AddOutput("Out", "(Tensor) Dependency of the variable need to sync");
    AddComment(R"DOC(
CSyncCalcStream Operator

Call calculation stream synchronization.
)DOC");
  }
};

template <typename T>
class CSyncCalcStreamCudaKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
#if (defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)) && !defined(_WIN32)

    auto place = ctx.GetPlace();
    auto dev_ctx = static_cast<platform::CUDADeviceContext*>(
        platform::DeviceContextPool::Instance().Get(place));

#ifdef PADDLE_WITH_HIP
    PADDLE_ENFORCE_CUDA_SUCCESS(hipStreamSynchronize(dev_ctx->stream()));
#else
    PADDLE_ENFORCE_CUDA_SUCCESS(cudaStreamSynchronize(dev_ctx->stream()));
#endif

#elif defined(PADDLE_WITH_ASCEND_CL) && !defined(_WIN32)
    auto place = ctx.GetPlace();
    PADDLE_ENFORCE_EQ(is_npu_place(place), true,
                      platform::errors::PreconditionNotMet(
                          "Sync stream op can run on npu place only for now."));

    auto dev_ctx = static_cast<platform::NPUDeviceContext*>(
        platform::DeviceContextPool::Instance().Get(place));
    PADDLE_ENFORCE_NPU_SUCCESS(aclrtSynchronizeStream(dev_ctx->stream()));
#else
    PADDLE_THROW(platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU."));
#endif
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_WITHOUT_GRADIENT(c_sync_calc_stream, ops::CSyncCalcStreamOp,
                             ops::CSyncCalcStreamOpMaker);

REGISTER_OP_CUDA_KERNEL(c_sync_calc_stream,
                        ops::CSyncCalcStreamCudaKernel<float>);
