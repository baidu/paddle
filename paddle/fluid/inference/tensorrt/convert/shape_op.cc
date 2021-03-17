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

#include "paddle/fluid/inference/tensorrt/convert/op_converter.h"

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

/*
 * ShapeOp: Only used in dynamic shape mode.
 */
class ShapeOpConverter : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope, bool test_mode) override {
    VLOG(3) << "convert a paddle shape op to tensorrt shape layer";

    framework::OpDesc op_desc(op, nullptr);
    // Declare inputs
    std::string input_name = op_desc.Input("Input").front();
    std::string out_name = op_desc.Output("Out").front();

    auto input = engine_->GetITensor(input_name);

    nvinfer1::ILayer* layer = nullptr;

    layer = TRT_ENGINE_ADD_LAYER(engine_, Shape, *input);
    layer->setOutputType(0, nvinfer1::DataType::kINT32);

    PADDLE_ENFORCE_EQ(layer != nullptr, true,
                      platform::errors::Fatal("Create shape layer failed."));

    RreplenishLayerAndOutput(layer, "shape", {out_name}, test_mode);
    layer->getOutput(0)->setType(nvinfer1::DataType::kINT32);
  }
};

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle

REGISTER_TRT_OP_CONVERTER(shape, ShapeOpConverter);
