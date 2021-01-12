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

#include "paddle/fluid/operators/pinned_memcpy_op.h"

#include <string>

namespace paddle {
namespace framework {
class OpDesc;
class Variable;
}  // namespace framework
namespace imperative {
class OpBase;
}  // namespace imperative
namespace platform {
struct CPUPlace;
struct CUDAPlace;
struct float16;
}  // namespace platform
}  // namespace paddle

namespace paddle {
namespace operators {

class PinnedMemcpyOp : public framework::OperatorWithKernel {
 public:
  PinnedMemcpyOp(const std::string &type,
                 const framework::VariableNameMap &inputs,
                 const framework::VariableNameMap &outputs,
                 const framework::AttributeMap &attrs)
      : OperatorWithKernel(type, inputs, outputs, attrs) {}

  void InferShape(framework::InferShapeContext *ctx) const override {
    if (ctx->HasInput("X")) {
      auto type = ctx->GetInputsVarType("X")[0];
      if (type == framework::proto::VarType::SELECTED_ROWS ||
          type == framework::proto::VarType::LOD_TENSOR) {
        ctx->SetOutputDim("Out", ctx->GetInputDim("X"));
        if (type == framework::proto::VarType::LOD_TENSOR) {
          ctx->ShareLoD("X", /*->*/ "Out");
        }
      } else if (type == framework::proto::VarType::LOD_TENSOR_ARRAY) {
        if (ctx->IsRuntime()) {
          // The runtime output shape is determined in kernel.
          return;
        } else {
          ctx->SetOutputDim("Out", ctx->GetInputDim("X"));
        }
      }
    }
  }

 protected:
  framework::OpKernelType GetKernelTypeForVar(
      const std::string &var_name, const framework::Tensor &tensor,
      const framework::OpKernelType &expected_kernel_type) const override {
    return framework::OpKernelType(expected_kernel_type.data_type_,
                                   expected_kernel_type.place_,
                                   tensor.layout());
  }

  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    const framework::Variable *var = ctx.InputVar("X");
    if (var->IsType<framework::LoDTensorArray>()) {
      auto t_arr = var->Get<framework::LoDTensorArray>();
      // NOTE(liym27): Support an empty tensor array as Input.
      // And set the kernel type is float.
      if (t_arr.size() == 0) {
        return framework::OpKernelType(framework::proto::VarType::FP32,
                                       ctx.device_context());
      }
    }

    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "X"),
        ctx.device_context());
  }
};

class PinnedMemcpyInferVarType : public framework::VarTypeInference {
 public:
  void operator()(framework::InferVarTypeContext *ctx) const override {
    ctx->SyncTypeAndDataType("X", "Out");
  }
};

class PinnedMemcpyKernel {
 public:
  void operator()(const framework::ExecutionContext &ctx) const {
    auto *x = ctx.InputVar("X");
    if (x == nullptr) {
      return;
    }
    PADDLE_ENFORCE_EQ(ctx.HasOutput("Out"), true,
                      platform::errors::NotFound(
                          "Output(Out) of pinned_memcpy_op is not found."));
    auto *out = ctx.OutputVar("Out");
    platform::DeviceContextPool &pool = platform::DeviceContextPool::Instance();
    auto &dev_ctx = *pool.Get(ctx.GetPlace());
    auto to_pinned = ctx.Attr<bool>("to_pinned");
    framework::VisitVarType(*x, PinnedMemcpyFunctor(out, dev_ctx, to_pinned));
  }
};

class PinnedMemcpyOpProtoMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X",
             "(LoDTensor, SelectedRows or LoDTensorArray) The input variable "
             "could be LoDTensor, SelectedRows or LoDTensorArray.")
        .AsDispensable();
    AddOutput("Out",
              "(LoDTensor, SelectedRows or LoDTensorArray) The type of output "
              "is the same as input X.");
    AddAttr<bool>(
        "to_pinned",
        "Determine the direction of tensor copy. "
        "True: the src is on CUDAPlace, dst is on CUDAPinnedPlace. "
        "False: the src is on CUDAPinnedPlace, dst is on CUDAPlace. ");
    AddComment(R"DOC(PinnedMemcpy Operator

Out = X,  when type in [LoDTensor/SelectedRows/LoDTensorArray]
raise error if the type is not listed above.
)DOC");
  }
};

template <typename T>
class PinnedMemcpyGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("pinned_memcpy");
    op->SetInput("X", this->OutputGrad("Out"));
    op->SetOutput("Out", this->InputGrad("X"));
  }
};

DECLARE_INPLACE_OP_INFERER(PinnedMemcpyOpInplaceInferer, {"X", "Out"});

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;
REGISTER_OPERATOR(pinned_memcpy, ops::PinnedMemcpyOp,
                  ops::PinnedMemcpyGradMaker<paddle::framework::OpDesc>,
                  ops::PinnedMemcpyGradMaker<paddle::imperative::OpBase>,
                  ops::PinnedMemcpyOpProtoMaker,
                  ops::PinnedMemcpyOpInplaceInferer,
                  ops::PinnedMemcpyInferVarType);

REGISTER_OP_CPU_KERNEL_FUNCTOR(pinned_memcpy, float, ops::PinnedMemcpyKernel,
                               double, ops::PinnedMemcpyKernel, int,
                               ops::PinnedMemcpyKernel, int64_t,
                               ops::PinnedMemcpyKernel, bool,
                               ops::PinnedMemcpyKernel, plat::float16,
                               ops::PinnedMemcpyKernel);

#ifdef PADDLE_WITH_CUDA
REGISTER_OP_CUDA_KERNEL_FUNCTOR(pinned_memcpy, float, ops::PinnedMemcpyKernel,
                                double, ops::PinnedMemcpyKernel, int,
                                ops::PinnedMemcpyKernel, int64_t,
                                ops::PinnedMemcpyKernel, bool,
                                ops::PinnedMemcpyKernel, plat::float16,
                                ops::PinnedMemcpyKernel);
#endif
