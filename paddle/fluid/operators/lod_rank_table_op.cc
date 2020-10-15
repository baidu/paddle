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
#include "paddle/fluid/framework/op_registry.h"

namespace paddle {
namespace framework {
class InferShapeContext;
class OpDesc;
class Scope;
template <typename T>
class EmptyGradOpMaker;
}  // namespace framework
namespace imperative {
class OpBase;
}  // namespace imperative
}  // namespace paddle

namespace paddle {
namespace operators {

class LoDRankTableOp : public framework::OperatorBase {
 public:
  LoDRankTableOp(const std::string &type,
                 const framework::VariableNameMap &inputs,
                 const framework::VariableNameMap &outputs,
                 const framework::AttributeMap &attrs)
      : OperatorBase(type, inputs, outputs, attrs) {}

 private:
  void RunImpl(const framework::Scope &scope,
               const platform::Place &dev_place) const override {
    auto x = scope.FindVar(Input("X"))->Get<framework::LoDTensor>();
    auto *out =
        scope.FindVar(Output("Out"))->GetMutable<framework::LoDRankTable>();
    VLOG(10) << "Level = " << static_cast<size_t>(Attr<int>("level"));
    out->Reset(x.lod(), static_cast<size_t>(Attr<int>("level")));
    VLOG(10) << Input("X") << "'s lod information is " << *out;
  }
};

class LoDRankTableOpProtoMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X",
             "(LoDTensor) input lod tensor, must contain lod information.");
    AddOutput("Out", "(LoDRankTable) The rank table of specific level.");
    AddAttr<int>("level", "(int) the specific lod level to rank.")
        .SetDefault(0)
        .EqualGreaterThan(0);
    AddComment(R"DOC(Create LoDRanTable by LoDTensor

LoD Rank Table stores the `level` of `lod` which is ordered by sequence
length in descending order. It is useful when implement dynamic RNN and is
shared by dynamic RNN memory, dynamic RNN slice input and dynamic RNN slice
output operators.
)DOC");
  }
};

class LoDRankTableInferShape : public framework::InferShapeBase {
 public:
  void operator()(framework::InferShapeContext *context) const override {
    PADDLE_ENFORCE_EQ(
        context->HasInput("X"), true,
        platform::errors::NotFound("LoDRankTable must have input X."));
  }
};

class LoDRankTableInferVarType : public framework::VarTypeInference {
 public:
  void operator()(framework::InferVarTypeContext *ctx) const override {
    ctx->SetOutputType("Out", framework::proto::VarType::LOD_RANK_TABLE,
                       framework::ALL_ELEMENTS);
  }
};

}  // namespace operators
}  // namespace paddle

REGISTER_OPERATOR(
    lod_rank_table, paddle::operators::LoDRankTableOp,
    paddle::operators::LoDRankTableOpProtoMaker,
    paddle::operators::LoDRankTableInferShape,
    paddle::operators::LoDRankTableInferVarType,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>);
