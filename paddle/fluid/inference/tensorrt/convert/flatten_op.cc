/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
 * FlattenOp trt converter
 */
class FlattenOpConverter : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope, bool test_mode) override {
    framework::OpDesc op_desc(op, nullptr);
    // Declare inputs
    auto* input = engine_->GetITensor(op_desc.Input("X")[0]);
    int dims = input->getDimensions().nbDims;
    nvinfer1::IShuffleLayer* layer = nullptr;
    if (!engine_->with_dynamic_shape()) {
      int dim_prod = 1;
      for (int i = 0; i < dims; i++) {
        int dim_i = input->getDimensions().d[i];
        PADDLE_ENFORCE_GT(
            dim_i, 0,
            platform::errors::InvalidArgument(
                "flatten input dim should be > 0, but got %d.", dim_i));
        dim_prod *= dim_i;
      }
      nvinfer1::Dims flatten_dim;
      flatten_dim.nbDims = 1;
      flatten_dim.d[0] = dim_prod;
      layer = TRT_ENGINE_ADD_LAYER(engine_, Shuffle, *input);
      layer->setReshapeDimensions(flatten_dim);
    } else {
      auto* shape_layer = TRT_ENGINE_ADD_LAYER(engine_, Shape, *input);
      uint32_t reduce_dim = 1;

      auto* reduce_prod_layer = TRT_ENGINE_ADD_LAYER(
          engine_, Reduce, *(shape_layer->getOutput(0)),
          nvinfer1::ReduceOperation::kPROD, reduce_dim, true);
      int32_t* constant_weight_data = new int32_t[1];
      constant_weight_data[0] = -1;
      TensorRTEngine::Weight constant_weight{
          nvinfer1::DataType::kINT32, static_cast<void*>(constant_weight_data),
          1};
      nvinfer1::Dims constant_dims;
      constant_dims.nbDims = 1;
      constant_dims.d[0] = 1;
      auto* constant_layer = TRT_ENGINE_ADD_LAYER(
          engine_, Constant, constant_dims, constant_weight.get());
      std::vector<nvinfer1::ITensor*> itensors;
      itensors.push_back(constant_layer->getOutput(0));
      itensors.push_back(reduce_prod_layer->getOutput(0));
      auto* concat_layer =
          TRT_ENGINE_ADD_LAYER(engine_, Concatenation, itensors.data(), 2);
      concat_layer->setAxis(0);
      layer = TRT_ENGINE_ADD_LAYER(engine_, Shuffle, *input);
      layer->setInput(1, *(concat_layer->getOutput(0)));
    }
    auto output_name = op_desc.Output("Out")[0];
    RreplenishLayerAndOutput(layer, "flatten", {output_name}, test_mode);
  }
};

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle

REGISTER_TRT_OP_CONVERTER(flatten, FlattenOpConverter);
