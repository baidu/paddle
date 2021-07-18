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

#include "paddle/fluid/operators/tdm_sampler_op.h"
#include <vector>
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/math/sampler.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace operators {

class TDMSamplerOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() {
    AddInput("X",
             "X(Tensor), Input variable which"
             "mapping the leaf node idx of tdm tree,"
             "dtype support int32/int64");
    AddInput("Travel",
             "Travel(Tensor), must has the same dtype with Layer"
             "Contains path information of all leaf nodes to root node,"
             " dtype support int32/64");
    AddInput("Layer",
             "Layer(Tensor), must has the same dtype with Travel "
             "Indicates which nodes are in each layer");
    AddAttr<bool>("output_positive",
                  "output_positive(bool)"
                  "Whether positive samples are included in the output")
        .SetDefault(true);
    AddAttr<std::vector<int>>(
        "neg_samples_num_list",
        "neg_samples_num_list(python:list[int], C++:vector<int>)"
        "The num of negative samples in each layer")
        .SetDefault({});
    AddAttr<std::vector<int>>(
        "neg_samples_radius_list",
        "neg_samples_radius_list(python:list[int], C++:vector<int>)"
        "The num of negative samples radius in each layer")
        .SetDefault({});
    AddAttr<std::vector<int>>("layer_offset_lod",
                              "offset lod information of Layer")
        .SetDefault({});
    AddAttr<int>("seed",
                 "(int) The seed used in sampler. If it is 0, "
                 "the sampler will generate a seed randomly.")
        .SetDefault(0);
    AddAttr<int>("dtype",
                 "(int, default INT32) "
                 "Output data type.")
        .SetDefault(2);
    AddOutput("Out",
              "Sampling result lodTensor, with shape [batch_size, layer_num, "
              "neg_num_of_layer]");
    AddOutput("Labels",
              "Labels of sampling result, has the same shape with Out."
              "pos samples mapping value 1, neg sample mapping value 0")
        .AsDispensable();
    AddOutput(
        "Mask",
        "Padding flag of Sampling result, if sampling res comes from padding,"
        "it will be 0, else 1, lodTensor, with shape [batch_size, "
        "layer_num, neg_num_of_layer]");
    AddComment(R"DOC("
        **TDM Sampler**
        According to the input positive samples at leaf node, do negative sampling layer by layer on the given tree.")DOC");
  }
};

class TDMSamplerOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE_EQ(ctx->HasInput("X"), true,
                      platform::errors::InvalidArgument(
                          "Inputs(Input) of TdmSampler should not be null."));
    PADDLE_ENFORCE_EQ(ctx->HasInput("Travel"), true,
                      platform::errors::InvalidArgument(
                          "Inputs(Travel) of TdmSampler should not be null."));
    PADDLE_ENFORCE_EQ(ctx->HasInput("Layer"), true,
                      platform::errors::InvalidArgument(
                          "Inputs(Layer) of TdmSampler should not be null."));
    auto neg_samples_num_vec =
        ctx->Attrs().Get<std::vector<int>>("neg_samples_num_list");
    auto output_positive_flag = ctx->Attrs().Get<bool>("output_positive");

    int64_t sample_res_length = 0;
    for (auto sample_nums : neg_samples_num_vec) {
      sample_res_length += sample_nums + (int64_t)output_positive_flag;
    }

    auto input_dims = ctx->GetInputDim("X");
    auto ddim = framework::make_ddim({-1, sample_res_length});
    if (ctx->IsRuntime()) {
      auto output_dims = framework::vectorize(input_dims);
      auto batch_size = output_dims[0];
      ctx->SetOutputDim("Out",
                        framework::make_ddim({batch_size, sample_res_length}));
      ctx->SetOutputDim("Labels",
                        framework::make_ddim({batch_size, sample_res_length}));
      ctx->SetOutputDim("Mask",
                        framework::make_ddim({batch_size, sample_res_length}));
    } else {
      ctx->SetOutputDim("Out", ddim);
      ctx->SetOutputDim("Labels", ddim);
      ctx->SetOutputDim("Mask", ddim);
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
    tdm_sampler, ops::TDMSamplerOp, ops::TDMSamplerOpMaker,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>);
REGISTER_OP_CPU_KERNEL(
    tdm_sampler, ops::TDMSamplerKernel<paddle::platform::CPUPlace, float>,
    ops::TDMSamplerKernel<paddle::platform::CPUPlace, double>,
    ops::TDMSamplerKernel<paddle::platform::CPUPlace, int>,
    ops::TDMSamplerKernel<paddle::platform::CPUPlace, int64_t>);
