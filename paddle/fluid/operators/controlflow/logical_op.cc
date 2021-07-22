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

#include "paddle/fluid/operators/controlflow/logical_op.h"
#include <algorithm>
#include <string>
#include <vector>
#include "paddle/fluid/framework/op_registry.h"

namespace paddle {
namespace operators {
template <typename OpComment>
class BinaryLogicalOpProtoMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    OpComment comment;
    AddInput("X", string::Sprintf("Left hand operand of %s operator. Must be "
                                  "a Variable of type being one of bool, int8, "
                                  "int16, int32, int64, float32, float64.",
                                  comment.type));
    AddInput("Y", string::Sprintf("Right hand operand of %s operator. Must be "
                                  "a Variable of type being one of bool, int8, "
                                  "int16, int32, int64, float32, float64.",
                                  comment.type));
    AddOutput("Out", string::Sprintf("n-dim bool Variable"));
    AddComment(string::Sprintf(R"DOC(%s Operator
It operates element-wise on X and Y, and returns the Out. X, Y and Out are N-dim LoDTensor or Tensor.
Each element of Out is calculated by %s
)DOC",
                               comment.type, comment.equation));
  }
};

template <typename OpComment>
class UnaryLogicalOpProtoMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    OpComment comment;
    AddInput("X",
             string::Sprintf("Operand of %s operator. Must be "
                             "a LoDTensor or Tensor of type being one of bool, "
                             "int8, int16, int32, int64, float32, float64.",
                             comment.type));
    AddOutput("Out", string::Sprintf("n-dim bool LoDTensor or Tensor."));
    AddComment(string::Sprintf(R"DOC(%s Operator
It operates element-wise on X, and returns the Out. X and Out are N-dim LoDTensor or Tensor.
Each element of Out is calculated by %s
)DOC",
                               comment.type, comment.equation));
  }
};

class LogicalOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    framework::OpKernelType kt = OperatorWithKernel::GetExpectedKernelType(ctx);
    // LogicalOp kernel's device type is decided by input tensor place
    kt.place_ = ctx.Input<framework::LoDTensor>("X")->place();
    return kt;
  }
};

template <typename OpComment>
class UnaryLogicalOp : public LogicalOp {
 public:
  using LogicalOp::LogicalOp;

 protected:
  void InferShape(framework::InferShapeContext *context) const override {
    OpComment comment;
    OP_INOUT_CHECK(context->HasInput("X"), "Input", "X", comment.type);
    context->SetOutputDim("Out", context->GetInputDim("X"));
    context->ShareLoD("X", "Out");
  }
};

template <typename OpComment>
class BinaryLogicalOp : public LogicalOp {
 public:
  using LogicalOp::LogicalOp;

 protected:
  void InferShape(framework::InferShapeContext *context) const override {
    OpComment comment;
    OP_INOUT_CHECK(context->HasInput("X"), "Input", "X", comment.type);
    OP_INOUT_CHECK(context->HasInput("Y"), "Input", "Y", comment.type);
    auto dim_x = context->GetInputDim("X");
    auto dim_y = context->GetInputDim("Y");
    if (dim_x == dim_y) {
      context->SetOutputDim("Out", dim_x);
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
    }
    context->ShareLoD("X", "Out");
  }
};

}  // namespace operators
}  // namespace paddle

#define REGISTER_BINARY_LOGICAL_OP(op_type, _equation)                     \
  struct _##op_type##Comment {                                             \
    static char type[];                                                    \
    static char equation[];                                                \
  };                                                                       \
  char _##op_type##Comment::type[]{#op_type};                              \
  char _##op_type##Comment::equation[]{_equation};                         \
  REGISTER_OPERATOR(                                                       \
      op_type, ::paddle::operators::BinaryLogicalOp<_##op_type##Comment>,  \
      ::paddle::operators::BinaryLogicalOpProtoMaker<_##op_type##Comment>, \
      ::paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,    \
      ::paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>);

#define REGISTER_UNARY_LOGICAL_OP(op_type, _equation)                     \
  struct _##op_type##Comment {                                            \
    static char type[];                                                   \
    static char equation[];                                               \
  };                                                                      \
  char _##op_type##Comment::type[]{#op_type};                             \
  char _##op_type##Comment::equation[]{_equation};                        \
  REGISTER_OPERATOR(                                                      \
      op_type, ::paddle::operators::UnaryLogicalOp<_##op_type##Comment>,  \
      ::paddle::operators::UnaryLogicalOpProtoMaker<_##op_type##Comment>, \
      ::paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,   \
      ::paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>);

REGISTER_BINARY_LOGICAL_OP(logical_and, "$$Out = X \\&\\& Y$$");
REGISTER_BINARY_LOGICAL_KERNEL(logical_and, CPU,
                               paddle::operators::LogicalAndFunctor);
REGISTER_BINARY_LOGICAL_OP(logical_or, "$$Out = X || Y$$");
REGISTER_BINARY_LOGICAL_KERNEL(logical_or, CPU,
                               paddle::operators::LogicalOrFunctor);
REGISTER_UNARY_LOGICAL_OP(logical_not, "$$Out = !X$$");
REGISTER_UNARY_LOGICAL_KERNEL(logical_not, CPU,
                              paddle::operators::LogicalNotFunctor);
REGISTER_BINARY_LOGICAL_OP(logical_xor,
                           "$$Out = (X || Y) \\&\\& !(X \\&\\& Y)$$");
REGISTER_BINARY_LOGICAL_KERNEL(logical_xor, CPU,
                               paddle::operators::LogicalXorFunctor);
