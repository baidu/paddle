//   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/fluid/operators/class_center_sample_op.h"

namespace paddle {
namespace operators {

class ClassCenterSampleOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("Label"), "Input", "Label",
                   "ClassCenterSample");
    OP_INOUT_CHECK(ctx->HasOutput("RemappedLabel"), "Output", "RemappedLabel",
                   "ClassCenterSample");
    OP_INOUT_CHECK(ctx->HasOutput("SampledLocalClassCenter"), "Output",
                   "SampledLocalClassCenter", "ClassCenterSample");

    auto x_dims = ctx->GetInputDim("Label");
    PADDLE_ENFORCE_EQ(x_dims.size(), 1,
                      platform::errors::InvalidArgument(
                          "Rank of Input(Label) should be equal to 1, "
                          "but the value given is %d.",
                          x_dims.size()));

    ctx->SetOutputDim("RemappedLabel", x_dims);
    auto num_samples = ctx->Attrs().Get<int>("num_samples");
    ctx->SetOutputDim("SampledLocalClassCenter",
                      framework::make_ddim({num_samples}));
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "Label"),
        ctx.device_context());
  }
};

class ClassCenterSampleOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput(
        "Label",
        "(Tensor<int|int64>) The input of ClassCenterSample op. Each value "
        "of Label is an integer label.");
    AddOutput("RemappedLabel",
              "(Tensor<int|int64>) Output tensor with same shape as Label. "
              "Each label is remap using sampled class.");
    AddOutput("SampledLocalClassCenter",
              "(Tensor<int|int64>) The sampled class center for local rank,"
              "value in [0, num_classes).");
    AddAttr<int>(
        "num_classes",
        "A positive integer to specify the number of classes at local rank. "
        "Note that num_classes of each GPU can be different.");
    AddAttr<int>(
        "num_samples",
        "A positive integer to specify the number of class center to sample.");
    AddAttr<int>("ring_id", "(int default 0) nccl communication ring id.")
        .SetDefault(0);
    AddAttr<int>("nranks", "(int default 1) The total number of GPUs.")
        .SetDefault(1);
    AddAttr<int>("rank", "(int default 0) The rank id in nranks.")
        .SetDefault(0);
    AddAttr<bool>("fix_seed",
                  "A flag indicating whether to use a fixed seed to generate "
                  "random negative class center. NOTE: DO NOT set this flag to"
                  "true in training. Setting this flag to true is only useful "
                  "in unittest or for debug")
        .SetDefault(false);
    AddAttr<int>("seed",
                 "Random seed used to generate random negative class center. "
                 "[default 0].")
        .SetDefault(0);
    AddComment(R"DOC(
    Class center sample method is proposed from the paper PartialFC that only sample a subset of the class centers.
    The process of sampling subset class centers is straightforward: 1) First select the positive class centers;
    2) Randomly sample negative class centers. Specifically, given a Label tensor, shape [batch_size], select all
    the positive class centers and randomly sample negative class centers, then remap the input label tensor using
    the sampled class centers. Note that if the number of the positive class centers is greater than the input 
    num_samples, it keeps all the positive class centers and the shape of SampledLocalClassCenter will be 
    [num_positive_class_centers]. The op supports CPU, single GPU and multi GPU.

    For more information, Partial FC: Training 10 Million Identities on a Single Machine
    arxiv: https://arxiv.org/abs/2010.05222

    Examples:
      For CPU or only one GPU
      Given:
        Label: [11, 5 , 1 , 3 , 12, 2 , 15, 19, 18, 19]
        num_classes = 20
        num_samples = 6
      Then:
        RemappedLabel: [4, 3, 0, 2, 5, 1, 6, 8, 7, 8]
        SampledLocalClassCenter: [1 , 2 , 3 , 5 , 11, 12, 15, 18, 19]

      For multi GPU
      Given:
        rank0:
            Label: [10, 17, 15, 11, 9 , 12, 18, 18, 17, 18, 19, 2 , 8 , 13, 11, 13, 9 , 10, 0 , 4 ]
            num_classes = 10
            num_samples = 6
            ring_id = 0
            nranks = 2
            rank = 0
        rank1:
            Label: [10, 17, 15, 11, 9 , 12, 18, 18, 17, 18, 19, 2 , 8 , 13, 11, 13, 9 , 10, 0 , 4 ]
            num_classes = 10
            num_samples = 6
            ring_id = 0
            nranks = 2
            rank = 1
      Then:
        rank0:
            RemappedLabel: [6 , 11, 10, 7 , 4 , 8 , 12, 12, 11, 12, 13, 1 , 3 , 9 , 7 , 9 , 4 , 6 , 0 , 2 ]
            SampledLocalClassCenter: [0, 2, 4, 8, 9, 3]
        rank1:
            RemappedLabel: [6 , 11, 10, 7 , 4 , 8 , 12, 12, 11, 12, 13, 1 , 3 , 9 , 7 , 9 , 4 , 6 , 0 , 2 ]
            SampledLocalClassCenter: [0, 1, 2, 3, 5, 7, 8]
)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;
REGISTER_OP_WITHOUT_GRADIENT(class_center_sample, ops::ClassCenterSampleOp,
                             ops::ClassCenterSampleOpMaker);
REGISTER_OP_CPU_KERNEL(class_center_sample,
                       ops::ClassCenterSampleCPUKernel<int64_t>,
                       ops::ClassCenterSampleCPUKernel<int>);
