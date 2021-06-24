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
#include "paddle/fluid/operators/common_infer_shape_functions.h"

USE_OP(relu);
USE_OP(elementwise_add);
USE_OP(softmax);

namespace paddle {
namespace operators {
namespace details {

class DygraphInferShapeTest {
 public:
  void AddInput(const std::string& name, const framework::DDim& dim) {
    std::shared_ptr<imperative::VarBase> vin(
        new imperative::VarBase(false, name));
    vin->MutableVar()->GetMutable<framework::LoDTensor>()->Resize(dim);
    ins_[name] = {vin};
  }
  void AddOutput(const std::string& name, const framework::DDim& expected_dim) {
    std::shared_ptr<imperative::VarBase> vout(
        new imperative::VarBase(false, name));
    vout->MutableVar()
        ->GetMutable<framework::LoDTensor>();  // InitializeVariable
    outs_[name] = {vout};
    expected_dims_[name] = expected_dim;
  }
  void AddAttrs(const framework::AttributeMap& attrs) { attrs_ = attrs; }
  void SetOpType(const std::string& op_type) { op_type_ = op_type; }
  void Run(std::function<void(framework::InferShapeContext* ctx)> infer_shape) {
    imperative::DygraphInferShapeContext<imperative::VarBase> ctx(
        &ins_, &outs_, &attrs_, {}, op_type_);
    infer_shape(&ctx);
    for (const auto& pair : expected_dims_) {
      auto out = outs_[pair.first][0];
      ASSERT_EQ(pair.second,
                out->MutableVar()->GetMutable<framework::LoDTensor>()->dims());
    }
  }

 private:
  imperative::NameVarBaseMap ins_;
  imperative::NameVarBaseMap outs_;
  framework::AttributeMap attrs_;
  std::string op_type_;
  std::map<std::string, framework::DDim> expected_dims_;
};
}  // namespace details

TEST(test_UnaryOpUnchangedInferShape, test_shape) {
  details::DygraphInferShapeTest test;
  test.AddInput("X", {2, 10});
  test.AddOutput("Out", {2, 10});
  test.SetOpType("relu");
  test.Run(UnaryOpUnchangedInferShape);
}

TEST(test_BinaryOpBroadcastInferShape, test_same_shape) {
  details::DygraphInferShapeTest test;
  test.AddInput("X", {2, 3, 4, 5});
  test.AddInput("Y", {2, 3, 4, 5});
  test.AddOutput("Out", {2, 3, 4, 5});
  test.SetOpType("elementwise_add");
  test.Run(BinaryOpBroadcastInferShape);
}

TEST(test_BinaryOpBroadcastInferShape, test_broadcast1) {
  details::DygraphInferShapeTest test;
  test.AddInput("X", {2, 3, 4, 5});
  test.AddInput("Y", {4, 5});
  test.AddOutput("Out", {2, 3, 4, 5});
  test.AddAttrs({
      {"axis", -1},
  });
  test.SetOpType("elementwise_add");
  test.Run(BinaryOpBroadcastInferShape);
}

TEST(test_BinaryOpBroadcastInferShape, test_broadcast2) {
  details::DygraphInferShapeTest test;
  test.AddInput("X", {2, 10, 5, 1});
  test.AddInput("Y", {10, 1, 1});
  test.AddOutput("Out", {2, 10, 5, 1});
  test.AddAttrs({
      {"axis", -1},
  });
  test.SetOpType("elementwise_add");
  test.Run(BinaryOpBroadcastInferShape);
}

TEST(test_BinaryOpBroadcastInferShape, test_broadcast3) {
  details::DygraphInferShapeTest test;
  test.AddInput("X", {10, 1, 1});
  test.AddInput("Y", {2, 10, 5, 5});
  test.AddOutput("Out", {2, 10, 5, 5});
  test.AddAttrs({
      {"axis", -1},
  });
  test.SetOpType("elementwise_add");
  test.Run(BinaryOpBroadcastInferShape);
}

TEST(test_UnaryOpUnchangedInferShapeCheckAxis, test_shape) {
  details::DygraphInferShapeTest test;
  test.AddInput("X", {2, 10});
  test.AddOutput("Out", {2, 10});
  test.AddAttrs({
      {"axis", -1},
  });
  test.SetOpType("softmax");
  test.Run(UnaryOpUnchangedInferShapeCheckAxis);
}

TEST(test_UnaryOpUnchangedInferShapeCheckAxis, test_axis_exception) {
  details::DygraphInferShapeTest test;
  test.AddInput("X", {2, 10});
  test.AddOutput("Out", {2, 10});
  test.AddAttrs({
      {"axis", 2},
  });
  test.SetOpType("softmax");
  ASSERT_ANY_THROW(test.Run(UnaryOpUnchangedInferShapeCheckAxis));
}

}  // namespace operators
}  // namespace paddle
