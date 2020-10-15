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

#include "paddle/fluid/operators/collective/c_allgather_op.h"

#include <memory>

namespace paddle {
namespace operators {

class CAllGatherOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
  void InferShape(framework::InferShapeContext *ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "AllGather");
    OP_INOUT_CHECK(ctx->HasOutput("Out"), "Input", "Out", "AllGather");
    int nranks = ctx->Attrs().Get<int>("nranks");
    PADDLE_ENFORCE_GE(nranks, 2, platform::errors::InvalidArgument(
                                     "The value of nranks should be >=2."));
    framework::DDim dim = ctx->GetInputDim("X");
    dim[0] = dim[0] * nranks;
    if (dim[0] < 0) dim[0] = -1;
    ctx->SetOutputDim("Out", dim);
  }
};

class CAllGatherOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() {
    AddInput("X", "(Tensor) tensor to be allgather");
    AddOutput("Out", "(Tensor) the allgather result");
    AddAttr<int>("ring_id", "(int default 0) communication ring id.")
        .SetDefault(0);
    AddAttr<bool>(
        "use_calc_stream",
        "(bool default false) eject CUDA operations to calculation stream.")
        .SetDefault(false);
    AddAttr<int>("nranks",
                 "Total trainer count of the distributed training job");
    AddComment(R"DOC(
CAllGather Operator
each rank receives the aggregation of data from all ranks in the order of the ranks

reference: https://docs.nvidia.com/deeplearning/sdk/nccl-developer-guide/docs/usage/operations.html#allgather
)DOC");
  }
};

template <typename T>
class CAllGatherOpGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> retv) const override {
    retv->SetType("c_reducescatter");
    retv->SetInput("X", this->OutputGrad("Out"));
    retv->SetOutput("Out", this->InputGrad("X"));
    retv->SetAttrMap(this->Attrs());
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OPERATOR(c_allgather, ops::CAllGatherOp,
                  ops::CAllGatherOpGradMaker<paddle::framework::OpDesc>,
                  ops::CAllGatherOpGradMaker<paddle::imperative::OpBase>,
                  ops::CAllGatherOpMaker);

REGISTER_OP_CPU_KERNEL(c_allgather, ops::CAllGatherOpCPUKernel<float>,
                       ops::CAllGatherOpCPUKernel<double>,
                       ops::CAllGatherOpCPUKernel<int>,
                       ops::CAllGatherOpCPUKernel<int64_t>,
                       ops::CAllGatherOpCPUKernel<plat::float16>);
