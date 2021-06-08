// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/fluid/operators/meshgrid_op.h"

#include <memory>
#include <string>
#include <vector>

namespace paddle {
namespace operators {

using framework::Tensor;

class MeshgridOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE_GE(
        ctx->Inputs("X").size(), 1UL,
        platform::errors::InvalidArgument("Input(X) should not be empty."));
    PADDLE_ENFORCE_GE(
        ctx->Outputs("Out").size(), 1UL,
        platform::errors::InvalidArgument("Output(Out) should not be empty."));

    auto inputs_dims = ctx->GetInputsDim("X");
    const size_t inputs_num = inputs_dims.size();
    auto outs_names = ctx->Outputs("Out");
    const size_t outputs_num = outs_names.size();

    auto out_shape = std::vector<int>(inputs_num);

    for (size_t i = 0; i < inputs_num; i++) {
      out_shape[i] = inputs_dims[i][0];
    }
    auto out_dims = framework::make_ddim(std::vector<int>(out_shape));
    std::vector<framework::DDim> outs_dims(outputs_num, out_dims);
    ctx->SetOutputsDim("Out", outs_dims);
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    auto inputs = ctx.MultiInput<Tensor>("X");
    auto input_data_type = framework::proto::VarType::Type(0);
    bool flag = false;
    for (auto* input : inputs) {
      if (input->IsInitialized() && input->numel() > 0) {
        input_data_type = input->type();
        flag = true;
        break;
      }
    }
    if (flag == 0) {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "All Inputs of Meshgrid OP are Empty!"));
    }

    return framework::OpKernelType(input_data_type, ctx.GetPlace());
  }
};

class MeshgridOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "(Tensor, default Tensor<float>).").AsDuplicable();
    AddOutput("Out", "(Tensor, default Tensor<float>.)").AsDuplicable();

    AddComment(R"DOC(
Meshgrid Operator.
Take: N tensors, each of which can be either scalr or 1-dimensional vector, and create
N-dimensional grids.

Args:
  tensors (list of tensor): if the input k tensors has (N1,), (N2,),..., (Nk,), then 
  the output tensors are all of size (N1, N2, ...., Nk).

Example::
>>> x = fluid.data(name='x', shape=[10], dtype='float64')
>>> y = fluid.data(name='y', shape=[20], dtype='float64')
>>> grid_x, grid_y = fluid.layers.meshgrid([x, y])
>>> grid_x.shape
(10,20)
>>> grid_y.shape
(10,20)
)DOC");
  }
};

class MeshgridGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE_GT(ctx->Inputs(framework::GradVarName("Out")).size(), 1,
                      platform::errors::InvalidArgument(
                          "Number of Inputs(Out@Grad) should be larger than 1."
                          "But received Inputs(Out@Grad)' size = %d .",
                          ctx->Inputs(framework::GradVarName("Out")).size()));
    ctx->SetOutputsDim(framework::GradVarName("X"), ctx->GetInputsDim("X"));
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(OperatorWithKernel::IndicateVarDataType(
                                       ctx, framework::GradVarName("Out")),
                                   ctx.device_context());
  }
};

template <typename T>
class MeshgridGradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("meshgrid_grad");
    op->SetInput("X", this->Input("X"));
    op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    op->SetOutput(framework::GradVarName("X"), this->InputGrad("X", false));
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(meshgrid, ops::MeshgridOp, ops::MeshgridOpMaker,
                  ops::MeshgridGradOpMaker<paddle::framework::OpDesc>,
                  ops::MeshgridGradOpMaker<paddle::imperative::OpBase>);
REGISTER_OPERATOR(meshgrid_grad, ops::MeshgridGradOp);
REGISTER_OP_CPU_KERNEL(
    meshgrid, ops::MeshgridKernel<paddle::platform::CPUDeviceContext, float>,
    ops::MeshgridKernel<paddle::platform::CPUDeviceContext, double>,
    ops::MeshgridKernel<paddle::platform::CPUDeviceContext, int>,
    ops::MeshgridKernel<paddle::platform::CPUDeviceContext, int64_t>);

REGISTER_OP_CPU_KERNEL(
    meshgrid_grad,
    ops::MeshgridGradKernel<paddle::platform::CPUDeviceContext, float>,
    ops::MeshgridGradKernel<paddle::platform::CPUDeviceContext, int64_t>,
    ops::MeshgridGradKernel<paddle::platform::CPUDeviceContext, int>,
    ops::MeshgridGradKernel<paddle::platform::CPUDeviceContext, double>);
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
REGISTER_OP_CUDA_KERNEL(
    meshgrid, ops::MeshgridKernel<paddle::platform::CUDADeviceContext, float>,
    ops::MeshgridKernel<paddle::platform::CUDADeviceContext, double>,
    ops::MeshgridKernel<paddle::platform::CUDADeviceContext, int>,
    ops::MeshgridKernel<paddle::platform::CUDADeviceContext, int64_t>,
    ops::MeshgridKernel<paddle::platform::CUDADeviceContext, bool>);
REGISTER_OP_CUDA_KERNEL(
    meshgrid_grad,
    ops::MeshgridGradKernel<paddle::platform::CUDADeviceContext, float>,
    ops::MeshgridGradKernel<paddle::platform::CUDADeviceContext, double>,
    ops::MeshgridGradKernel<paddle::platform::CUDADeviceContext, int>,
    ops::MeshgridGradKernel<paddle::platform::CUDADeviceContext, int64_t>);
#endif
