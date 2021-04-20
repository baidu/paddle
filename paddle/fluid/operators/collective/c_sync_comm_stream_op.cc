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

#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
#include "paddle/fluid/platform/nccl_helper.h"
#endif

#if defined(PADDLE_WITH_ASCEND_CL)
#include "paddle/fluid/platform/collective_helper.h"
#include "paddle/fluid/platform/hccl_helper.h"
#endif

namespace paddle {
namespace operators {

class CSyncCommStreamOp : public framework::OperatorWithKernel {
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

class CSyncCommStreamOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() {
    AddInput("X", "(Tensor) Dependency of the variable need to sync")
        .AsDuplicable();
    AddOutput("Out", "(Tensor) Dependency of the variable need to sync")
        .AsDuplicable();
    AddAttr<int>("ring_id", "(int default 0) ring id.").SetDefault(0);
    AddComment(R"DOC(
CSyncCommStream Operator

Call communication stream synchronization.
)DOC");
  }
};

template <typename T>
class CSyncCommStreamCudaKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)

    auto place = ctx.GetPlace();

    int ring_id = ctx.Attr<int>("ring_id");
    auto stream =
        platform::NCCLCommContext::Instance().Get(ring_id, place)->stream();

#ifdef PADDLE_WITH_RCCL
    PADDLE_ENFORCE_CUDA_SUCCESS(hipStreamSynchronize(stream));
#else
    PADDLE_ENFORCE_CUDA_SUCCESS(cudaStreamSynchronize(stream));
#endif

#elif defined(PADDLE_WITH_ASCEND_CL)
    auto place = ctx.GetPlace();
    PADDLE_ENFORCE_EQ(is_npu_place(place), true,
                      platform::errors::PreconditionNotMet(
                          "Sync stream op can run on npu place only for now."));
    int ring_id = Attr<int>("ring_id");
    auto stream =
        platform::HCCLCommContext::Instance().Get(ring_id, place)->stream();
    PADDLE_ENFORCE_NPU_SUCCESS(aclrtSynchronizeStream(stream));
#else
    PADDLE_THROW(platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU."));
#endif
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_WITHOUT_GRADIENT(c_sync_comm_stream, ops::CSyncCommStreamOp,
                             ops::CSyncCommStreamOpMaker);

REGISTER_OP_CUDA_KERNEL(c_sync_comm_stream,
                        ops::CSyncCommStreamCudaKernel<float>);
