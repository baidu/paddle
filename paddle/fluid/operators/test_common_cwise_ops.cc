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

#include "gtest/gtest.h"
#include "paddle/fluid/framework/ddim.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/var_type.h"
#include "paddle/fluid/imperative/infer_shape_context.h"
#include "paddle/fluid/imperative/layer.h"
#include "paddle/fluid/operators/common_cwise_functors.h"
#include "paddle/fluid/operators/common_cwise_ops.h"
#include "paddle/fluid/operators/common_infer_shape_functions.h"

namespace paddle {
namespace operators {

class TestNegOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "input of test op");
    AddOutput("Out", "output of test op");
    AddComment("Out = -X");
  }
};

template <typename T>
class TestNegOpGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType(this->ForwardOpType() + "_grad");
    op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    op->SetAttrMap(this->Attrs());
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace functors = paddle::operators::functors;

REGISTER_OPERATOR(test_neg, ops::UnaryOp, ops::TestNegOpMaker,
                  ops::TestNegOpGradMaker<paddle::framework::OpDesc>,
                  ops::TestNegOpGradMaker<paddle::imperative::OpBase>);
REGISTER_OPERATOR(test_neg_grad, ops::UnaryGradOp);
REGISTER_OP_KERNEL_4(test_neg, ops::UnaryOpKernel, CPU, functors::Neg, int,
                     int64_t, float, double);
REGISTER_OP_KERNEL_4(test_neg_grad, ops::UnaryGradOpKernel, CPU, functors::Neg,
                     int, int64_t, float, double);

USE_OP(test_neg);

namespace paddle {
namespace operators {

TEST(test_neg, test_run) {
  paddle::platform::CPUPlace cpu_place;
  paddle::framework::Scope scope;

  auto op = framework::OpRegistry::CreateOp(
      "test_neg", {{"X", {"X"}}}, {{"Out", {"Out"}}}, {}, {"X"}, {"Out"});

  auto in_tensor = scope.Var("X")->GetMutable<framework::LoDTensor>();
  in_tensor->Resize({2, 10});
  size_t numel = 2 * 10;
  auto in_data = in_tensor->mutable_data<float>(cpu_place);
  std::uniform_real_distribution<float> dist(static_cast<float>(-10.0),
                                             static_cast<float>(10.0));
  std::mt19937 engine;
  std::vector<float> expected_out;
  expected_out.reserve(numel);
  for (size_t i = 0; i < numel; ++i) {
    in_data[i] = dist(engine);
    expected_out[i] = -in_data[i];
  }
  auto out_tensor = scope.Var("Out")->GetMutable<framework::LoDTensor>();
  op->Run(scope, cpu_place);
  auto out_data = out_tensor->data<float>();
  auto is_equal = std::equal(out_data, out_data + numel, expected_out.data());
  ASSERT_TRUE(is_equal);
}

TEST(test_neg_grad, test_run) {
  paddle::platform::CPUPlace cpu_place;
  paddle::framework::Scope scope;

  auto op =
      framework::OpRegistry::CreateOp("test_neg_grad", {{"DOut", {"DOut"}}},
                                      {{"DX", {"DX"}}}, {}, {"DOut"}, {"DX"});

  auto in_tensor = scope.Var("DOut")->GetMutable<framework::LoDTensor>();
  in_tensor->Resize({2, 10});
  size_t numel = 2 * 10;
  auto in_data = in_tensor->mutable_data<float>(cpu_place);
  std::uniform_real_distribution<float> dist(static_cast<float>(-10.0),
                                             static_cast<float>(10.0));
  std::mt19937 engine;
  std::vector<float> expected_out;
  expected_out.reserve(numel);
  for (size_t i = 0; i < numel; ++i) {
    in_data[i] = dist(engine);
    expected_out[i] = -in_data[i];
  }
  auto out_tensor = scope.Var("DX")->GetMutable<framework::LoDTensor>();
  op->Run(scope, cpu_place);
  auto out_data = out_tensor->data<float>();
  auto is_equal = std::equal(out_data, out_data + numel, expected_out.data());
  ASSERT_TRUE(is_equal);
}

}  // namespace operators
}  // namespace paddle
