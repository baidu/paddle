/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at
   http://www.apache.org/licenses/LICENSE-2.0
   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#include "paddle/fluid/operators/detection/yolo_box_op.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/op_version_registry.h"

namespace paddle {
namespace operators {

using framework::Tensor;

class YoloBoxOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "YoloBoxOp");
    OP_INOUT_CHECK(ctx->HasInput("ImgSize"), "Input", "ImgSize", "YoloBoxOp");
    OP_INOUT_CHECK(ctx->HasOutput("Boxes"), "Output", "Boxes", "YoloBoxOp");
    OP_INOUT_CHECK(ctx->HasOutput("Scores"), "Output", "Scores", "YoloBoxOp");

    auto dim_x = ctx->GetInputDim("X");
    auto dim_imgsize = ctx->GetInputDim("ImgSize");
    auto anchors = ctx->Attrs().Get<std::vector<int>>("anchors");
    int anchor_num = anchors.size() / 2;
    auto class_num = ctx->Attrs().Get<int>("class_num");
    auto iou_aware = ctx->Attrs().Get<bool>("iou_aware");
    auto iou_aware_factor = ctx->Attrs().Get<float>("iou_aware_factor");

    PADDLE_ENFORCE_EQ(dim_x.size(), 4, platform::errors::InvalidArgument(
                                           "Input(X) should be a 4-D tensor."
                                           "But received X dimension(%s)",
                                           dim_x.size()));
    if (iou_aware) {
      PADDLE_ENFORCE_EQ(
          dim_x[1], anchor_num * (6 + class_num),
          platform::errors::InvalidArgument(
              "Input(X) dim[1] should be equal to (anchor_mask_number * (6 "
              "+ class_num)) while iou_aware is true."
              "But received dim[1](%s) != (anchor_mask_number * "
              "(6+class_num)(%s).",
              dim_x[1], anchor_num * (6 + class_num)));
      PADDLE_ENFORCE_GE(
          iou_aware_factor, 0,
          platform::errors::InvalidArgument(
              "Attr(iou_aware_factor) should greater than or equal to 0."
              "But received iou_aware_factor (%s)",
              iou_aware_factor));
      PADDLE_ENFORCE_LE(
          iou_aware_factor, 1,
          platform::errors::InvalidArgument(
              "Attr(iou_aware_factor) should less than or equal to 1."
              "But received iou_aware_factor (%s)",
              iou_aware_factor));
    } else {
      PADDLE_ENFORCE_EQ(
          dim_x[1], anchor_num * (5 + class_num),
          platform::errors::InvalidArgument(
              "Input(X) dim[1] should be equal to (anchor_mask_number * (5 "
              "+ class_num))."
              "But received dim[1](%s) != (anchor_mask_number * "
              "(5+class_num)(%s).",
              dim_x[1], anchor_num * (5 + class_num)));
    }
    PADDLE_ENFORCE_EQ(dim_imgsize.size(), 2,
                      platform::errors::InvalidArgument(
                          "Input(ImgSize) should be a 2-D tensor."
                          "But received Imgsize size(%s)",
                          dim_imgsize.size()));
    if ((dim_imgsize[0] > 0 && dim_x[0] > 0) || ctx->IsRuntime()) {
      PADDLE_ENFORCE_EQ(
          dim_imgsize[0], dim_x[0],
          platform::errors::InvalidArgument(
              "Input(ImgSize) dim[0] and Input(X) dim[0] should be same."));
    }
    PADDLE_ENFORCE_EQ(
        dim_imgsize[1], 2,
        platform::errors::InvalidArgument("Input(ImgSize) dim[1] should be 2."
                                          "But received imgsize dim[1](%s).",
                                          dim_imgsize[1]));
    PADDLE_ENFORCE_GT(anchors.size(), 0,
                      platform::errors::InvalidArgument(
                          "Attr(anchors) length should be greater than 0."
                          "But received anchors length(%s).",
                          anchors.size()));
    PADDLE_ENFORCE_EQ(anchors.size() % 2, 0,
                      platform::errors::InvalidArgument(
                          "Attr(anchors) length should be even integer."
                          "But received anchors length (%s)",
                          anchors.size()));
    PADDLE_ENFORCE_GT(class_num, 0,
                      platform::errors::InvalidArgument(
                          "Attr(class_num) should be an integer greater than 0."
                          "But received class_num (%s)",
                          class_num));

    int box_num = dim_x[2] * dim_x[3] * anchor_num;
    std::vector<int64_t> dim_boxes({dim_x[0], box_num, 4});
    ctx->SetOutputDim("Boxes", framework::make_ddim(dim_boxes));

    std::vector<int64_t> dim_scores({dim_x[0], box_num, class_num});
    ctx->SetOutputDim("Scores", framework::make_ddim(dim_scores));
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "X"), ctx.GetPlace());
  }
};

class YoloBoxOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X",
             "The input tensor of YoloBox operator is a 4-D tensor with "
             "shape of [N, C, H, W]. The second dimension(C) stores "
             "box locations, confidence score and classification one-hot "
             "keys of each anchor box. Generally, X should be the output "
             "of YOLOv3 network.");
    AddInput("ImgSize",
             "The image size tensor of YoloBox operator, "
             "This is a 2-D tensor with shape of [N, 2]. This tensor holds "
             "height and width of each input image used for resizing output "
             "box in input image scale.");
    AddOutput("Boxes",
              "The output tensor of detection boxes of YoloBox operator, "
              "This is a 3-D tensor with shape of [N, M, 4], N is the "
              "batch num, M is output box number, and the 3rd dimension "
              "stores [xmin, ymin, xmax, ymax] coordinates of boxes.");
    AddOutput("Scores",
              "The output tensor of detection boxes scores of YoloBox "
              "operator, This is a 3-D tensor with shape of "
              "[N, M, :attr:`class_num`], N is the batch num, M is "
              "output box number.");

    AddAttr<int>("class_num", "The number of classes to predict.");
    AddAttr<std::vector<int>>("anchors",
                              "The anchor width and height, "
                              "it will be parsed pair by pair.")
        .SetDefault(std::vector<int>{});
    AddAttr<int>("downsample_ratio",
                 "The downsample ratio from network input to YoloBox operator "
                 "input, so 32, 16, 8 should be set for the first, second, "
                 "and thrid YoloBox operators.")
        .SetDefault(32);
    AddAttr<float>("conf_thresh",
                   "The confidence scores threshold of detection boxes. "
                   "Boxes with confidence scores under threshold should "
                   "be ignored.")
        .SetDefault(0.01);
    AddAttr<bool>("clip_bbox",
                  "Whether clip output bonding box in Input(ImgSize) "
                  "boundary. Default true.")
        .SetDefault(true);
    AddAttr<float>("scale_x_y",
                   "Scale the center point of decoded bounding "
                   "box. Default 1.0")
        .SetDefault(1.);
    AddAttr<bool>("iou_aware", "Whether use iou aware. Default false.")
        .SetDefault(false);
    AddAttr<float>("iou_aware_factor", "iou aware factor. Default 0.5.")
        .SetDefault(0.5);
    AddComment(R"DOC(
         This operator generates YOLO detection boxes from output of YOLOv3 network.
         
         The output of previous network is in shape [N, C, H, W], while H and W
         should be the same, H and W specify the grid size, each grid point predict 
         given number boxes, this given number, which following will be represented as S,
         is specified by the number of anchors. In the second dimension(the channel
         dimension), C should be equal to S * (5 + class_num) if :attr:`iou_aware` is false,
         otherwise C should be equal to S * (6 + class_num). class_num is the object
         category number of source dataset(such as 80 in coco dataset), so the 
         second(channel) dimension, apart from 4 box location coordinates x, y, w, h, 
         also includes confidence score of the box and class one-hot key of each anchor 
         box.

         Assume the 4 location coordinates are :math:`t_x, t_y, t_w, t_h`, the box 
         predictions should be as follows:

         $$
         b_x = \\sigma(t_x) + c_x
         $$
         $$
         b_y = \\sigma(t_y) + c_y
         $$
         $$
         b_w = p_w e^{t_w}
         $$
         $$
         b_h = p_h e^{t_h}
         $$

         in the equation above, :math:`c_x, c_y` is the left top corner of current grid
         and :math:`p_w, p_h` is specified by anchors.

         The logistic regression value of the 5th channel of each anchor prediction boxes
         represents the confidence score of each prediction box, and the logistic
         regression value of the last :attr:`class_num` channels of each anchor prediction 
         boxes represents the classifcation scores. Boxes with confidence scores less than
         :attr:`conf_thresh` should be ignored, and box final scores is the product of 
         confidence scores and classification scores.

         $$
         score_{pred} = score_{conf} * score_{class}
         $$

         where the confidence scores follow the formula bellow

         .. math::

            score_{conf} = \begin{case}
                             obj, \text{if } iou_aware == flase \\
                             obj^{1 - iou_aware_factor} * iou^{iou_aware_factor}, \text{otherwise}
                           \end{case}

         )DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(
    yolo_box, ops::YoloBoxOp, ops::YoloBoxOpMaker,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>);
REGISTER_OP_CPU_KERNEL(yolo_box, ops::YoloBoxKernel<float>,
                       ops::YoloBoxKernel<double>);

REGISTER_OP_VERSION(yolo_box)
    .AddCheckpoint(
        R"ROC(
      Upgrade yolo box to add new attribute [iou_aware, iou_aware_factor].
    )ROC",
        paddle::framework::compatible::OpVersionDesc()
            .NewAttr("iou_aware", "Whether use iou aware", false)
            .NewAttr("iou_aware_factor", "iou aware factor", 0.5f));
