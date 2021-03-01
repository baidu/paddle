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
class ScaleOpConverter : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope, bool test_mode) override {
    VLOG(3) << "convert a fluid scale op to tensorrt mul layer without bias";

    framework::OpDesc op_desc(op, nullptr);
    // Declare inputs
    std::vector<nvinfer1::ITensor*> itensors;
    std::string input_name = op_desc.Input("X").front();
    std::string out_name = op_desc.Output("Out").front();

    auto input = engine_->GetITensor(input_name);
    bool bias_after_scale =
        BOOST_GET_CONST(bool, op_desc.GetAttr("bias_after_scale"));
    float bias = BOOST_GET_CONST(float, op_desc.GetAttr("bias"));
    float scale = BOOST_GET_CONST(float, op_desc.GetAttr("scale"));
    auto create_weights = [&](float data, std::string type) -> float* {
      std::unique_ptr<framework::Tensor> tmp_tensor(new framework::Tensor());
      tmp_tensor->Resize({1});
      auto* tmp_data = tmp_tensor->mutable_data<float>(platform::CPUPlace());
      tmp_data[0] = data;
      engine_->SetWeights(out_name + "_scale_op_" + type,
                          std::move(tmp_tensor));
      return tmp_data;
    };

    float* bias_ptr = create_weights(bias, "bias");
    float* scale_ptr = create_weights(scale, "scale");

    TensorRTEngine::Weight scale_weights{nvinfer1::DataType::kFLOAT,
                                         static_cast<void*>(scale_ptr), 1};
    TensorRTEngine::Weight shift_weights{nvinfer1::DataType::kFLOAT,
                                         static_cast<void*>(bias_ptr), 1};
    TensorRTEngine::Weight power_weights{nvinfer1::DataType::kFLOAT, nullptr,
                                         0};
    nvinfer1::ILayer* layer = nullptr;

    auto input_dim = input->getDimensions();
    nvinfer1::IShuffleLayer* expand_layer = nullptr;
    nvinfer1::IShuffleLayer* squeeze_layer = nullptr;
    if (engine_->with_dynamic_shape()) {
      PADDLE_ENFORCE_GE(
          input_dim.nbDims, 3,
          platform::errors::Fatal("Paddle-TRT scale converter only support "
                                  "dimension >= 3 with dynamic shape"));

      if (input_dim.nbDims == 3) {
        // TensorRT scale layer is not supporting input dims < 4 when using
        // explicit batch
        expand_layer = TRT_ENGINE_ADD_LAYER(engine_, Shuffle, *input);
        nvinfer1::Dims4 target_shape(0, 0, 0, 1);  // expand 1 dims
        expand_layer->setReshapeDimensions(target_shape);
        input = expand_layer->getOutput(0);
      }
    } else {
      PADDLE_ENFORCE_GE(
          input_dim.nbDims, 1,
          platform::errors::Fatal("Paddle-TRT scale converter only support "
                                  "dimension >= 1 without dynamic shape"));
      if (input_dim.nbDims < 3) {
        expand_layer = TRT_ENGINE_ADD_LAYER(engine_, Shuffle, *input);
        int expand_dims[4];
        for (int i = 0; i < 3; i++) {
          if (i < input_dim.nbDims) {
            expand_dims[i] = 0;
          } else {
            expand_dims[i] = 1;
          }
        }
        expand_layer = TRT_ENGINE_ADD_LAYER(engine_, Shuffle, *input);
        nvinfer1::Dims3 target_shape(expand_dims[0], expand_dims[1],
                                     expand_dims[2]);
        expand_layer->setReshapeDimensions(target_shape);
        input = expand_layer->getOutput(0);
      }
    }
    PADDLE_ENFORCE_NOT_NULL(
        input,
        platform::errors::NotFound(
            "input after expand layer in scale converter should not be null"));

    if (bias_after_scale) {
      layer = TRT_ENGINE_ADD_LAYER(
          engine_, Scale, *input, nvinfer1::ScaleMode::kUNIFORM,
          shift_weights.get(), scale_weights.get(), power_weights.get());
    } else {
      // add bias
      layer = TRT_ENGINE_ADD_LAYER(
          engine_, Scale, *(input), nvinfer1::ScaleMode::kUNIFORM,
          shift_weights.get(), power_weights.get(), power_weights.get());
      // mul scale
      layer = TRT_ENGINE_ADD_LAYER(
          engine_, Scale, *(layer->getOutput(0)), nvinfer1::ScaleMode::kUNIFORM,
          power_weights.get(), scale_weights.get(), power_weights.get());
    }

    PADDLE_ENFORCE_EQ(layer != nullptr, true,
                      platform::errors::Fatal("Create scale layer failed."));
    if (engine_->with_dynamic_shape()) {
      if (input_dim.nbDims == 3) {
        // TensorRT scale layer is not supporting input dims < 4 when using
        // explicit batch
        squeeze_layer =
            TRT_ENGINE_ADD_LAYER(engine_, Shuffle, *(layer->getOutput(0)));
        nvinfer1::Dims3 target_shape(0, 0, 0);  // expand 1 dims
        squeeze_layer->setReshapeDimensions(target_shape);
        layer = static_cast<nvinfer1::ILayer*>(squeeze_layer);
      }
    } else {
      squeeze_layer =
          TRT_ENGINE_ADD_LAYER(engine_, Shuffle, *(layer->getOutput(0)));
      squeeze_layer->setReshapeDimensions(input_dim);
      layer = static_cast<nvinfer1::ILayer*>(squeeze_layer);
    }

    RreplenishLayerAndOutput(layer, "scale", {out_name}, test_mode);
  }
};

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle

REGISTER_TRT_OP_CONVERTER(scale, ScaleOpConverter);
