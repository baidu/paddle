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
 * ConcatOp
 */
class ShuffleChannelOpConverter : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope, bool test_mode) override {
    framework::OpDesc op_desc(op, nullptr);
    // Declare inputs
    auto* input = engine_->GetITensor(op_desc.Input("X")[0]);
    auto input_dims = input->getDimensions();
    PADDLE_ENFORCE_EQ(
        input_dims.nbDims, 3,
        platform::errors::InvalidArgument("ShuffleChannel TRT op converter "
                                          "input dims is invalid. The input "
                                          "dims size should be 3, but got %d.",
                                          input_dims.nbDims));
    int c = input_dims.d[0];
    int h = input_dims.d[1];
    int w = input_dims.d[2];
    int group = BOOST_GET_CONST(int, op_desc.GetAttr("group"));

    auto* layer = TRT_ENGINE_ADD_LAYER(engine_, Shuffle, *input);
    nvinfer1::Dims4 reshape_dim(group, c / group, h, w);
    layer->setReshapeDimensions(reshape_dim);
    layer->setSecondTranspose({1, 0, 2, 3});
    auto* output = layer->getOutput(0);

    auto* reshape_layer = TRT_ENGINE_ADD_LAYER(engine_, Shuffle, *output);
    nvinfer1::Dims3 reshape_dim2(c, h, w);
    reshape_layer->setReshapeDimensions(reshape_dim2);

    auto output_name = op_desc.Output("Out")[0];
    RreplenishLayerAndOutput(reshape_layer, "concat", {output_name}, test_mode);
  }
};

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle

REGISTER_TRT_OP_CONVERTER(shuffle_channel, ShuffleChannelOpConverter);
