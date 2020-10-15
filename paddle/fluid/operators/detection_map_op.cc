/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/detection_map_op.h"
#include <string>

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

class DetectionMAPOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("DetectRes"), "Input", "DetectRes",
                   "DetectionMAP");
    OP_INOUT_CHECK(ctx->HasInput("Label"), "Input", "Label", "DetectionMAP");
    OP_INOUT_CHECK(ctx->HasOutput("AccumPosCount"), "Output", "AccumPosCount",
                   "DetectionMAP");
    OP_INOUT_CHECK(ctx->HasOutput("AccumTruePos"), "Output", "AccumTruePos",
                   "DetectionMAP");
    OP_INOUT_CHECK(ctx->HasOutput("AccumFalsePos"), "Output", "AccumFalsePos",
                   "DetectionMAP");
    OP_INOUT_CHECK(ctx->HasOutput("MAP"), "Output", "MAP", "DetectionMAP");

    auto det_dims = ctx->GetInputDim("DetectRes");
    PADDLE_ENFORCE_EQ(
        det_dims.size(), 2UL,
        platform::errors::InvalidArgument(
            "Input(DetectRes) ndim must be 2, the shape is [N, 6],"
            "but received the ndim is %d",
            det_dims.size()));
    PADDLE_ENFORCE_EQ(
        det_dims[1], 6UL,
        platform::errors::InvalidArgument(
            "The shape is of Input(DetectRes) [N, 6], but received"
            " shape is [N, %d]",
            det_dims[1]));
    auto label_dims = ctx->GetInputDim("Label");
    PADDLE_ENFORCE_EQ(label_dims.size(), 2,
                      platform::errors::InvalidArgument(
                          "The ndim of Input(Label) must be 2, but received %d",
                          label_dims.size()));
    if (ctx->IsRuntime() || label_dims[1] > 0) {
      PADDLE_ENFORCE_EQ(
          (label_dims[1] == 6 || label_dims[1] == 5), true,
          platform::errors::InvalidArgument(
              "The shape of Input(Label) is [N, 6] or [N, 5], but received "
              "[N, %d]",
              label_dims[1]));
    }

    if (ctx->HasInput("PosCount")) {
      PADDLE_ENFORCE(
          ctx->HasInput("TruePos"),
          platform::errors::InvalidArgument(
              "Input(TruePos) of DetectionMAPOp should not be null when "
              "Input(PosCount) is not null."));
      PADDLE_ENFORCE(
          ctx->HasInput("FalsePos"),
          platform::errors::InvalidArgument(
              "Input(FalsePos) of DetectionMAPOp should not be null when "
              "Input(PosCount) is not null."));
    }

    ctx->SetOutputDim("MAP", framework::make_ddim({1}));
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "DetectRes"),
        platform::CPUPlace());
  }
};

class DetectionMAPOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("DetectRes",
             "(LoDTensor) A 2-D LoDTensor with shape [M, 6] represents the "
             "detections. Each row has 6 values: "
             "[label, confidence, xmin, ymin, xmax, ymax], M is the total "
             "number of detect results in this mini-batch. For each instance, "
             "the offsets in first dimension are called LoD, the number of "
             "offset is N + 1, if LoD[i + 1] - LoD[i] == 0, means there is "
             "no detected data.");
    AddInput("Label",
             "(LoDTensor) A 2-D LoDTensor represents the"
             "Labeled ground-truth data. Each row has 6 values: "
             "[label, xmin, ymin, xmax, ymax, is_difficult] or 5 values: "
             "[label, xmin, ymin, xmax, ymax], where N is the total "
             "number of ground-truth data in this mini-batch. For each "
             "instance, the offsets in first dimension are called LoD, "
             "the number of offset is N + 1, if LoD[i + 1] - LoD[i] == 0, "
             "means there is no ground-truth data.");
    AddInput("HasState",
             "(Tensor<int>) A tensor with shape [1], 0 means ignoring input "
             "states, which including PosCount, TruePos, FalsePos.")
        .AsDispensable();
    AddInput("PosCount",
             "(Tensor) A tensor with shape [Ncls, 1], store the "
             "input positive example count of each class, Ncls is the count of "
             "input classification. "
             "This input is used to pass the AccumPosCount generated by the "
             "previous mini-batch when the multi mini-batches cumulative "
             "calculation carried out. "
             "When the input(PosCount) is empty, the cumulative "
             "calculation is not carried out, and only the results of the "
             "current mini-batch are calculated.")
        .AsDispensable();
    AddInput("TruePos",
             "(LoDTensor) A 2-D LoDTensor with shape [Ntp, 2], store the "
             "input true positive example of each class."
             "This input is used to pass the AccumTruePos generated by the "
             "previous mini-batch when the multi mini-batches cumulative "
             "calculation carried out. ")
        .AsDispensable();
    AddInput("FalsePos",
             "(LoDTensor) A 2-D LoDTensor with shape [Nfp, 2], store the "
             "input false positive example of each class."
             "This input is used to pass the AccumFalsePos generated by the "
             "previous mini-batch when the multi mini-batches cumulative "
             "calculation carried out. ")
        .AsDispensable();
    AddOutput("AccumPosCount",
              "(Tensor) A tensor with shape [Ncls, 1], store the "
              "positive example count of each class. It combines the input "
              "input(PosCount) and the positive example count computed from "
              "input(Detection) and input(Label).");
    AddOutput("AccumTruePos",
              "(LoDTensor) A LoDTensor with shape [Ntp', 2], store the "
              "true positive example of each class. It combines the "
              "input(TruePos) and the true positive examples computed from "
              "input(Detection) and input(Label).");
    AddOutput("AccumFalsePos",
              "(LoDTensor) A LoDTensor with shape [Nfp', 2], store the "
              "false positive example of each class. It combines the "
              "input(FalsePos) and the false positive examples computed from "
              "input(Detection) and input(Label).");
    AddOutput("MAP",
              "(Tensor) A tensor with shape [1], store the mAP evaluate "
              "result of the detection.");
    AddAttr<int>("class_num",
                 "(int) "
                 "The class number.");
    AddAttr<int>(
        "background_label",
        "(int, default: 0) "
        "The index of background label, the background label will be ignored. "
        "If set to -1, then all categories will be considered.")
        .SetDefault(0);
    AddAttr<float>(
        "overlap_threshold",
        "(float) "
        "The lower bound jaccard overlap threshold of detection output and "
        "ground-truth data.")
        .SetDefault(.5f);
    AddAttr<bool>("evaluate_difficult",
                  "(bool, default true) "
                  "Switch to control whether the difficult data is evaluated.")
        .SetDefault(true);
    AddAttr<std::string>("ap_type",
                         "(string, default 'integral') "
                         "The AP algorithm type, 'integral' or '11point'.")
        .SetDefault("integral")
        .InEnum({"integral", "11point"})
        .AddCustomChecker([](const std::string& ap_type) {
          PADDLE_ENFORCE_NE(
              GetAPType(ap_type), APType::kNone,
              platform::errors::InvalidArgument(
                  "The ap_type should be 'integral' or '11point."));
        });
    AddComment(R"DOC(
Detection mAP evaluate operator.
The general steps are as follows. First, calculate the true positive and
false positive according to the input of detection and labels, then
calculate the mAP evaluate value.
Supporting '11 point' and 'integral' mAP algorithm. Please get more information
from the following articles:
https://sanchom.wordpress.com/tag/average-precision/
https://arxiv.org/abs/1512.02325

)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(
    detection_map, ops::DetectionMAPOp, ops::DetectionMAPOpMaker,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>);
REGISTER_OP_CPU_KERNEL(
    detection_map, ops::DetectionMAPOpKernel<paddle::platform::CPUPlace, float>,
    ops::DetectionMAPOpKernel<paddle::platform::CPUPlace, double>);
