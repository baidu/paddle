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

#include <algorithm>
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
 * SoftMaxOp, ISoftMaxLayer in TRT. This Layer doesn't has weights.
 */
class SoftMaxOpConverter : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope, bool test_mode) override {
    VLOG(3)
        << "convert a fluid softmax op to tensorrt softmax layer without bias";
    framework::OpDesc op_desc(op, nullptr);
    // Declare inputs
    auto* input1 = engine_->GetITensor(op_desc.Input("X")[0]);
    nvinfer1::Dims input_shape = input1->getDimensions();
    int input_dims = input_shape.nbDims;
    int axis = op_desc.HasAttr("axis")
                   ? BOOST_GET_CONST(int, op_desc.GetAttr("axis"))
                   : -1;

    auto* layer = TRT_ENGINE_ADD_LAYER(engine_, SoftMax,
                                       *const_cast<nvinfer1::ITensor*>(input1));
    uint32_t axes = std::max(0, input_dims - 3);
    // TODO(cryoco): Poor workaround. Fix padded dims problem when TRT layers
    // support Nd.
    int padded_dims = 0;
    int explicit_batch = 0;
    if (engine_->with_dynamic_shape()) explicit_batch = 1;
    for (int i = input_dims - 1; i > explicit_batch; i--) {
      if (input_shape.d[i] == 1) {
        padded_dims += 1;
      } else {
        break;
      }
    }
    if (!engine_->with_dynamic_shape()) {
      if (axis == -1) {
        axes = input_dims - 1 - padded_dims;
      } else {
        axes = axis;
      }
    } else {
      if (axis == -1) {
        axes = input_dims - 1 - padded_dims;
      } else {
        axes = axis + 1;
      }
    }
    layer->setAxes(1 << axes);
    auto output_name = op_desc.Output("Out")[0];
    RreplenishLayerAndOutput(layer, "softmax", {output_name}, test_mode);

    // The trt will not run int for softmax.
    engine_->SetTensorDynamicRange(input1, 1.0);
  }
};

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle

USE_OP(softmax);
REGISTER_TRT_OP_CONVERTER(softmax, SoftMaxOpConverter);
