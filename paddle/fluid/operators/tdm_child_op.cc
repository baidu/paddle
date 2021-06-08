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

#include "paddle/fluid/operators/tdm_child_op.h"

#include <vector>

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/math/sampler.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace operators {
class TDMChildOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X",
             "X(Tensor), dtype support int32/int64, X variable is the "
             "node id of TDM-Tree");
    AddInput(
        "TreeInfo",
        "TreeInfo(Tensor), dtype support int32/int64, it stores the node "
        "information in the following format: item_id(shape=1), "
        "layer_id(shape=1), parent_id(shape=1), child_id(shape=child_nums)");
    AddAttr<int>("child_nums", "child_nums(int)",
                 "The child nums of one node, if the node hasn't enough child, "
                 "it should padding 0 until child nums equal to child_nums");
    AddOutput("Child",
              "Return the children's node_id of input node, "
              "if input don't have child, return 0");
    AddOutput("LeafMask",
              "LeafMask has the same shape with Child"
              "If child is leaf node, LeafMask value = 1, else = 0");
    AddAttr<int>("dtype",
                 "(int, default INT32) "
                 "Output data type.")
        .SetDefault(2);
    AddComment(R"DOC("
     **Tdm Child**
     According to the input node_id on the given tree, return the corresponding child node_id and 
      whether child is a leaf node by LeafMask.")DOC");
  }
};

class TDMChildOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE_EQ(ctx->HasInput("X"), true,
                      platform::errors::InvalidArgument(
                          "Inputs(X) of TdmChild should not be null."));
    PADDLE_ENFORCE_EQ(ctx->HasInput("TreeInfo"), true,
                      platform::errors::InvalidArgument(
                          "Inputs(TreeInfo) of TdmChild should not be null."));

    int child_nums = ctx->Attrs().Get<int>("child_nums");
    PADDLE_ENFORCE_GT(
        child_nums, 0,
        platform::errors::InvalidArgument(
            "ValueError: The value of the 'child_nums' must greater than 0. "
            "But received child_nums value = %d, ",
            child_nums));

    auto info_dims = ctx->GetInputDim("TreeInfo");
    auto input_dims = ctx->GetInputDim("X");

    PADDLE_ENFORCE_EQ(
        info_dims.size(), 2,
        platform::errors::InvalidArgument(
            "ShapeError: The dimensions of the 'tree info' must be 2. "
            "But received tree info's dimensions = %d, "
            "tree info's shape = [%s].",
            info_dims.size(), info_dims));

    auto output_dims = framework::vectorize(input_dims);
    output_dims.push_back(child_nums);
    ctx->SetOutputDim("Child", framework::make_ddim(output_dims));
    ctx->SetOutputDim("LeafMask", framework::make_ddim(output_dims));

    if (ctx->GetOutputsVarType("Child")[0] ==
        framework::proto::VarType::LOD_TENSOR) {
      ctx->ShareLoD("X", /*->*/ "Child");
      ctx->ShareLoD("X", /*->*/ "LeafMask");
    }
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    auto data_type = OperatorWithKernel::IndicateVarDataType(ctx, "X");
    return framework::OpKernelType(data_type, ctx.device_context());
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(
    tdm_child, ops::TDMChildOp, ops::TDMChildOpMaker,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>);
REGISTER_OP_CPU_KERNEL(
    tdm_child, ops::TDMChildKernel<paddle::platform::CPUPlace, float>,
    ops::TDMChildKernel<paddle::platform::CPUPlace, double>,
    ops::TDMChildKernel<paddle::platform::CPUPlace, int>,
    ops::TDMChildKernel<paddle::platform::CPUPlace, int64_t>);
