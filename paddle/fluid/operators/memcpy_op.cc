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

#include "paddle/fluid/operators/memcpy_op.h"

#include <string>

namespace paddle {
namespace framework {
class OpDesc;
class InferShapeContext;
template <typename T>
class EmptyGradOpMaker;
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

class MemcpyOp : public framework::OperatorWithKernel {
 public:
  MemcpyOp(const std::string &type, const framework::VariableNameMap &inputs,
           const framework::VariableNameMap &outputs,
           const framework::AttributeMap &attrs)
      : OperatorWithKernel(type, inputs, outputs, attrs) {}

  void InferShape(framework::InferShapeContext *ctx) const override {
    auto type = ctx->GetInputsVarType("X")[0];
    if (type == framework::proto::VarType::SELECTED_ROWS ||
        type == framework::proto::VarType::LOD_TENSOR) {
      ctx->SetOutputDim("Out", ctx->GetInputDim("X"));
      if (type == framework::proto::VarType::LOD_TENSOR) {
        ctx->ShareLoD("X", /*->*/ "Out");
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
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "X"),
        ctx.device_context());
  }
};

class MemcpyInferVarType : public framework::VarTypeInference {
 public:
  void operator()(framework::InferVarTypeContext *ctx) const override {
    ctx->SyncTypeAndDataType("X", "Out");
  }
};

class MemcpyKernel {
 public:
  void operator()(const framework::ExecutionContext &ctx) const {
    auto *x = ctx.InputVar("X");
    if (x == nullptr) {
      return;
    }
    PADDLE_ENFORCE_EQ(
        ctx.HasOutput("Out"), true,
        platform::errors::NotFound("Output(Out) of memcpy_op is not found."));
    auto *out = ctx.OutputVar("Out");
    platform::DeviceContextPool &pool = platform::DeviceContextPool::Instance();
    auto &dev_ctx = *pool.Get(ctx.GetPlace());
    auto dst_place_type = ctx.Attr<int>("dst_place_type");
    framework::VisitVarType(*x, MemcpyFunctor(out, dev_ctx, dst_place_type));
  }
};

class MemcpyOpProtoMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "(LoDTensor) The input variable ");
    AddOutput("Out",
              "(LoDTensor) The type of output "
              "is the same as input X.");
    AddAttr<int>("dst_place_type",
                 "Determine the dst place of tensor copy. "
                 "By Now it ONLY support CUDAPlace <-> CUDAPinnedPlace or "
                 "NPUPlace <-> CPUPlace. "
                 "Other place type is Unimplemented and will cause ERROR."
                 "0: dst is on CPUPlace. "
                 "1: dst is on CUDAPlace. "
                 "2: dst is on CUDAPinnedPlace. "
                 "3: dst is on XPUPlace. "
                 "4: dst is on NPUPlace. ");
    AddComment(R"DOC(
    Memcpy Operator.
    By now, it ONLY supports the memcopy between CUDAPinnedPlace <-> CUDAPlace or 
    NPUPlace <-> CPUPlace, and used as an internal op by Recompute-Offload.
    You would have to update it if you want other more capacities.

Out = X,  when type in [LoDTensor]
raise error if the type is not listed above.
)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;
REGISTER_OPERATOR(
    memcpy, ops::MemcpyOp, ops::MemcpyOpProtoMaker, ops::MemcpyInferVarType,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>);

REGISTER_OP_CPU_KERNEL_FUNCTOR(memcpy, float, ops::MemcpyKernel, double,
                               ops::MemcpyKernel, int, ops::MemcpyKernel,
                               int64_t, ops::MemcpyKernel, bool,
                               ops::MemcpyKernel, plat::float16,
                               ops::MemcpyKernel);

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
REGISTER_OP_CUDA_KERNEL_FUNCTOR(memcpy, float, ops::MemcpyKernel, double,
                                ops::MemcpyKernel, int, ops::MemcpyKernel,
                                int64_t, ops::MemcpyKernel, bool,
                                ops::MemcpyKernel, plat::float16,
                                ops::MemcpyKernel);
#endif

#ifdef PADDLE_WITH_ASCEND_CL
REGISTER_OP_NPU_KERNEL_FUNCTOR(memcpy, float, ops::MemcpyKernel, double,
                               ops::MemcpyKernel, int, ops::MemcpyKernel,
                               int64_t, ops::MemcpyKernel, bool,
                               ops::MemcpyKernel, plat::float16,
                               ops::MemcpyKernel);
#endif
