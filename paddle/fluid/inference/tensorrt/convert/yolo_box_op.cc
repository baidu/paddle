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

#include <vector>
#include "paddle/fluid/inference/tensorrt/convert/op_converter.h"
#include "paddle/fluid/inference/tensorrt/plugin/yolo_box_op_plugin.h"

namespace paddle {
namespace framework {
class Scope;
namespace proto {
class OpDesc;
}  // namespace proto
}  // namespace framework
}  // namespace paddle

namespace paddle {
namespace inference {
namespace tensorrt {

class YoloBoxOpConverter : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope, bool test_mode) override {
    VLOG(3) << "convert a fluid yolo box op to tensorrt plugin";

    framework::OpDesc op_desc(op, nullptr);
    std::string X = op_desc.Input("X").front();
    std::string img_size = op_desc.Input("ImgSize").front();

    auto* X_tensor = engine_->GetITensor(X);
    auto* img_size_tensor = engine_->GetITensor(img_size);

    int class_num = BOOST_GET_CONST(int, op_desc.GetAttr("class_num"));
    std::vector<int> anchors =
        BOOST_GET_CONST(std::vector<int>, op_desc.GetAttr("anchors"));

    int downsample_ratio =
        BOOST_GET_CONST(int, op_desc.GetAttr("downsample_ratio"));
    float conf_thresh = BOOST_GET_CONST(float, op_desc.GetAttr("conf_thresh"));
    bool clip_bbox = BOOST_GET_CONST(bool, op_desc.GetAttr("clip_bbox"));
    float scale_x_y = BOOST_GET_CONST(float, op_desc.GetAttr("scale_x_y"));
    bool iou_aware = op_desc.HasAttr("iou_aware")
                         ? BOOST_GET_CONST(bool, op_desc.GetAttr("iou_aware"))
                         : false;
    float iou_aware_factor =
        op_desc.HasAttr("iou_aware_factor")
            ? BOOST_GET_CONST(float, op_desc.GetAttr("iou_aware_factor"))
            : 0.5;

    int type_id = static_cast<int>(engine_->WithFp16());
    auto input_dim = X_tensor->getDimensions();
    auto* yolo_box_plugin = new plugin::YoloBoxPlugin(
        type_id ? nvinfer1::DataType::kHALF : nvinfer1::DataType::kFLOAT,
        anchors, class_num, conf_thresh, downsample_ratio, clip_bbox, scale_x_y,
        iou_aware, iou_aware_factor, input_dim.d[1], input_dim.d[2]);

    std::vector<nvinfer1::ITensor*> yolo_box_inputs;
    yolo_box_inputs.push_back(X_tensor);
    yolo_box_inputs.push_back(img_size_tensor);

    auto* yolo_box_layer = engine_->network()->addPluginV2(
        yolo_box_inputs.data(), yolo_box_inputs.size(), *yolo_box_plugin);

    std::vector<std::string> output_names;
    output_names.push_back(op_desc.Output("Boxes").front());
    output_names.push_back(op_desc.Output("Scores").front());

    RreplenishLayerAndOutput(yolo_box_layer, "yolo_box", output_names,
                             test_mode);
  }
};

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle

REGISTER_TRT_OP_CONVERTER(yolo_box, YoloBoxOpConverter);
