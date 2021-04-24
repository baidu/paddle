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

#include "paddle/fluid/operators/controlflow/compare_op.h"
#include <algorithm>
#include <string>
#include <vector>
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/fluid/operators/elementwise/elementwise_op_function.h"

namespace paddle {
namespace operators {

template <typename OpComment>
class CompareOpProtoMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    OpComment comment;
    AddInput("X", string::Sprintf("the left hand operand of %s operator",
                                  comment.type));
    AddInput("Y", string::Sprintf("the right hand operand of %s operator",
                                  comment.type));
    AddAttr<int>(
        "axis",
        "The start dimension index for broadcasting Y onto X. [default -1]")
        .SetDefault(-1)
        .EqualGreaterThan(-1);
    AddAttr<bool>("force_cpu",
                  "Force fill output variable to cpu "
                  "memory. Otherwise, fill output variable to the running "
                  "device [default true].")
        .SetDefault(false);
    AddOutput("Out", string::Sprintf("n-dim bool tensor. Each element is %s",
                                     comment.equation));
    AddComment(string::Sprintf(R"DOC(
It operates element-wise on X and Y, and returns the Out. Each of them is a
N-dim tensor. X and Y could be any type.  The each element of the Out tensor is
calculated by $%s$
)DOC",
                               comment.equation));
  }
};

template <typename OpComment>
class CompareOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContext* context) const override {
    OpComment comment;
    OP_INOUT_CHECK(context->HasInput("X"), "Input", "X", comment.type);
    OP_INOUT_CHECK(context->HasInput("Y"), "Input", "Y", comment.type);
    auto dim_x = context->GetInputDim("X");
    auto dim_y = context->GetInputDim("Y");

    if (context->GetInputDim("X") == context->GetInputDim("Y")) {
      context->ShareDim("X", /*->*/ "Out");
      context->ShareLoD("X", /*->*/ "Out");
    } else {
      int max_dim = std::max(dim_x.size(), dim_y.size());
      int axis = std::abs(dim_x.size() - dim_y.size());
      std::vector<int> x_dims_array(max_dim);
      std::vector<int> y_dims_array(max_dim);
      std::vector<int> out_dims_array(max_dim);
      GetBroadcastDimsArrays(dim_x, dim_y, x_dims_array.data(),
                             y_dims_array.data(), out_dims_array.data(),
                             max_dim, axis);
      context->SetOutputDim("Out", framework::make_ddim(out_dims_array));
      // to do
      context->ShareLoD("X", /*->*/ "Out");
    }
  }

  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    framework::OpKernelType kt = OperatorWithKernel::GetExpectedKernelType(ctx);
    // CompareOp kernel's device type is decided by input tensor place
    bool force_cpu = ctx.Attr<bool>("force_cpu");
    if (force_cpu) {
      kt.place_ = platform::CPUPlace();
    } else {
      if (ctx.Input<framework::LoDTensor>("X")->place().type() !=
          typeid(platform::CUDAPinnedPlace)) {
        kt.place_ = ctx.Input<framework::LoDTensor>("X")->place();
      } else {
        kt.place_ = ctx.GetPlace();
      }
    }
    return kt;
  }
};

}  // namespace operators
}  // namespace paddle

#define REGISTER_COMPARE_OP_VERSION(op_type)                               \
  REGISTER_OP_VERSION(op_type)                                             \
      .AddCheckpoint(                                                      \
          R"ROC(Upgrade compare ops, add a new attribute [force_cpu])ROC", \
          paddle::framework::compatible::OpVersionDesc().ModifyAttr(       \
              "force_cpu",                                                 \
              "In order to force fill output variable to gpu memory.",     \
              false));

#define REGISTER_COMPARE_OP(op_type, _equation)                           \
  struct _##op_type##Comment {                                            \
    static char type[];                                                   \
    static char equation[];                                               \
  };                                                                      \
  char _##op_type##Comment::type[]{#op_type};                             \
  char _##op_type##Comment::equation[]{_equation};                        \
  REGISTER_OPERATOR(                                                      \
      op_type, ::paddle::operators::CompareOp<_##op_type##Comment>,       \
      ::paddle::operators::CompareOpProtoMaker<_##op_type##Comment>,      \
      ::paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,   \
      ::paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>); \
  REGISTER_COMPARE_OP_VERSION(op_type);

REGISTER_COMPARE_OP(less_than, "Out = X < Y");
REGISTER_COMPARE_KERNEL(less_than, CPU, paddle::operators::LessThanFunctor,
                        paddle::operators::GreaterEqualFunctor);
REGISTER_COMPARE_OP(less_equal, "Out = X <= Y");
REGISTER_COMPARE_KERNEL(less_equal, CPU, paddle::operators::LessEqualFunctor,
                        paddle::operators::GreaterThanFunctor);
REGISTER_COMPARE_OP(greater_than, "Out = X > Y");
REGISTER_COMPARE_KERNEL(greater_than, CPU,
                        paddle::operators::GreaterThanFunctor,
                        paddle::operators::LessEqualFunctor);
REGISTER_COMPARE_OP(greater_equal, "Out = X >= Y");
REGISTER_COMPARE_KERNEL(greater_equal, CPU,
                        paddle::operators::GreaterEqualFunctor,
                        paddle::operators::LessThanFunctor);
REGISTER_COMPARE_OP(equal, "Out = X == Y");
REGISTER_COMPARE_KERNEL(equal, CPU, paddle::operators::EqualFunctor,
                        paddle::operators::EqualFunctor);
REGISTER_COMPARE_OP(not_equal, "Out = X != Y");
REGISTER_COMPARE_KERNEL(not_equal, CPU, paddle::operators::NotEqualFunctor,
                        paddle::operators::NotEqualFunctor);
