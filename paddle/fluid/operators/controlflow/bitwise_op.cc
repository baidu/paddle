/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/controlflow/bitwise_op.h"
#include <algorithm>
#include <string>
#include <vector>
#include "paddle/fluid/framework/op_registry.h"

namespace paddle {
namespace operators {

template <typename OpComment>
class BinaryBitwiseOpProtoMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    OpComment comment;
    AddInput("X", string::Sprintf(
                      "Input Tensor of ``%s`` . It is "
                      "a N-D Tensor of bool, uint8, int8, int16, int32, int64.",
                      comment.type));
    AddInput("Y", string::Sprintf(
                      "Input Tensor of ``%s`` . It is "
                      "a N-D Tensor of bool, uint8, int8, int16, int32, int64.",
                      comment.type));
    AddOutput("Out",
              string::Sprintf("Result of ``%s`` . It is a N-D Tensor with "
                              "the same data type of input Tensor.",
                              comment.type));
    AddComment(string::Sprintf(R"DOC(
It operates ``%s`` on Tensor ``X`` and ``Y`` .

.. math::
        %s

.. note::
    ``paddle.%s`` supports broadcasting. If you want know more about broadcasting, please refer to :ref:`user_guide_broadcasting`.
)DOC",
                               comment.type, comment.equation, comment.type));
  }
};

template <typename OpComment>
class UnaryBitwiseOpProtoMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    OpComment comment;
    AddInput("X", string::Sprintf(
                      "Input Tensor of ``%s`` . It is "
                      "a N-D Tensor of bool, uint8, int8, int16, int32, int64.",
                      comment.type));
    AddOutput("Out",
              string::Sprintf("Result of ``%s`` . It is a N-D Tensor with "
                              "the same data type of input Tensor.",
                              comment.type));
    AddComment(string::Sprintf(R"DOC(
It operates ``%s`` on Tensor ``X`` .

.. math::
        %s

)DOC",
                               comment.type, comment.equation));
  }
};

class BitwiseOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    framework::OpKernelType kt = OperatorWithKernel::GetExpectedKernelType(ctx);
    // BitwiseOp kernel's device type is decided by input tensor place
    kt.place_ = ctx.Input<framework::LoDTensor>("X")->place();
    return kt;
  }
};

template <typename OpComment>
class UnaryBitwiseOp : public BitwiseOp {
 public:
  using BitwiseOp::BitwiseOp;

 protected:
  void InferShape(framework::InferShapeContext *context) const override {
    OpComment comment;
    OP_INOUT_CHECK(context->HasInput("X"), "Input", "X", comment.type);
    context->SetOutputDim("Out", context->GetInputDim("X"));
    context->ShareLoD("X", "Out");
  }
};

template <typename OpComment>
class BinaryBitwiseOp : public BitwiseOp {
 public:
  using BitwiseOp::BitwiseOp;

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

namespace ops = ::paddle::operators;

#define REGISTER_BINARY_BITWISE_OP(op_type, _equation)                  \
  struct _##op_type##Comment {                                          \
    static char type[];                                                 \
    static char equation[];                                             \
  };                                                                    \
  char _##op_type##Comment::type[]{#op_type};                           \
  char _##op_type##Comment::equation[]{_equation};                      \
  REGISTER_OPERATOR(                                                    \
      op_type, ops::BinaryBitwiseOp<_##op_type##Comment>,               \
      ops::BinaryBitwiseOpProtoMaker<_##op_type##Comment>,              \
      ::paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>, \
      ::paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>);

#define REGISTER_UNARY_BITWISE_OP(op_type, _equation)                   \
  struct _##op_type##Comment {                                          \
    static char type[];                                                 \
    static char equation[];                                             \
  };                                                                    \
  char _##op_type##Comment::type[]{#op_type};                           \
  char _##op_type##Comment::equation[]{_equation};                      \
  REGISTER_OPERATOR(                                                    \
      op_type, ops::UnaryBitwiseOp<_##op_type##Comment>,                \
      ops::UnaryBitwiseOpProtoMaker<_##op_type##Comment>,               \
      ::paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>, \
      ::paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>);

REGISTER_BINARY_BITWISE_OP(bitwise_and, "Out = X \\& Y");
REGISTER_BINARY_BITWISE_OP(bitwise_or, "Out = X | Y");
REGISTER_BINARY_BITWISE_OP(bitwise_xor, "Out = X ^\\wedge Y");
REGISTER_UNARY_BITWISE_OP(bitwise_not, "Out = \\sim X");

REGISTER_BINARY_BITWISE_KERNEL(bitwise_and, CPU, ops::BitwiseAndFunctor);
REGISTER_BINARY_BITWISE_KERNEL(bitwise_or, CPU, ops::BitwiseOrFunctor);
REGISTER_BINARY_BITWISE_KERNEL(bitwise_xor, CPU, ops::BitwiseXorFunctor);
REGISTER_UNARY_BITWISE_KERNEL(bitwise_not, CPU, ops::BitwiseNotFunctor);
