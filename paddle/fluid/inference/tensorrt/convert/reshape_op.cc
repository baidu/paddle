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
 * ReshapeOp
 */
class ReshapeOpConverter : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope, bool test_mode) override {
    framework::OpDesc op_desc(op, nullptr);
    // Declare inputs
    auto* input = engine_->GetITensor(op_desc.Input("X")[0]);
    std::vector<int> shape =
        BOOST_GET_CONST(std::vector<int>, op_desc.GetAttr("shape"));
    int nbDims_num = shape.size();
    nvinfer1::Dims reshape_dim;
    if (engine_->with_dynamic_shape()) {  // running the TRT Dynamic Shape mode
      reshape_dim.nbDims = nbDims_num;
      for (int i = 0; i < nbDims_num; ++i) {
        reshape_dim.d[i] = shape[i];
      }
    } else {  // running the TRT Static Shape mode
      reshape_dim.nbDims = nbDims_num - 1;
      for (int i = 0; i < nbDims_num - 1; ++i) {
        reshape_dim.d[i] = shape[i + 1];
      }
    }
    auto* layer = TRT_ENGINE_ADD_LAYER(engine_, Shuffle, *input);
    layer->setReshapeDimensions(reshape_dim);
    auto output_name = op_desc.Output("Out")[0];
    RreplenishLayerAndOutput(layer, "reshape", {output_name}, test_mode);
  }
};

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle

REGISTER_TRT_OP_CONVERTER(reshape, ReshapeOpConverter);
