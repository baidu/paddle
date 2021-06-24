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
#include "paddle/fluid/operators/layer_norm_op.h"
#include "paddle/fluid/inference/tensorrt/convert/op_converter.h"
#include "paddle/fluid/inference/tensorrt/plugin/layer_norm_op_plugin.h"

namespace paddle {
namespace inference {
namespace tensorrt {

class LayerNormOpConverter : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope, bool test_mode) override {
    VLOG(4) << "convert a fluid layer_norm op to tensorrt layer_norm plugin";
    framework::OpDesc op_desc(op, nullptr);

    auto* X = engine_->GetITensor(op_desc.Input("X").front());
    auto* Bias_v = scope.FindVar(op_desc.Input("Bias").front());
    auto* Scale_v = scope.FindVar(op_desc.Input("Scale").front());
    const int begin_norm_axis =
        op_desc.HasAttr("begin_norm_axis")
            ? BOOST_GET_CONST(int, op_desc.GetAttr("begin_norm_axis"))
            : 1;
    const float eps = op_desc.HasAttr("epsilon")
                          ? BOOST_GET_CONST(float, op_desc.GetAttr("epsilon"))
                          : 1e-5f;
    PADDLE_ENFORCE_NOT_NULL(
        Bias_v, platform::errors::InvalidArgument(
                    "Input(Bias) of layer_norm should not be null."));
    PADDLE_ENFORCE_NOT_NULL(
        Scale_v, platform::errors::InvalidArgument(
                     "Input(Scale) of layer_norm should not be null."));

    auto* Bias_t = Bias_v->GetMutable<framework::LoDTensor>();
    auto* Scale_t = Scale_v->GetMutable<framework::LoDTensor>();

    std::unique_ptr<framework::LoDTensor> bias_tensor(
        new framework::LoDTensor());
    std::unique_ptr<framework::LoDTensor> scale_tensor(
        new framework::LoDTensor());

    bias_tensor->Resize(Bias_t->dims());
    scale_tensor->Resize(Scale_t->dims());

    platform::CPUPlace cpu_place;
    TensorCopySync((*Bias_t), cpu_place, &(*bias_tensor));
    TensorCopySync((*Scale_t), cpu_place, &(*scale_tensor));

    auto* bias_data = bias_tensor->mutable_data<float>(platform::CPUPlace());
    auto* scale_data = scale_tensor->mutable_data<float>(platform::CPUPlace());

    nvinfer1::ILayer* layernorm_layer = nullptr;
    if (engine_->with_dynamic_shape()) {
      int input_num = 1;
      for (int i = begin_norm_axis; i < X->getDimensions().nbDims; i++) {
        input_num *= X->getDimensions().d[i];
      }
      std::vector<int64_t> mean_shape{input_num};
      std::vector<int64_t> variance_shape{input_num};
      plugin::LayerNormPluginDynamic* plugin =
          new plugin::LayerNormPluginDynamic(bias_data, bias_tensor->numel(),
                                             scale_data, scale_tensor->numel(),
                                             begin_norm_axis, eps, mean_shape,
                                             variance_shape);
      layernorm_layer = engine_->AddDynamicPlugin(&X, 1, plugin);
    } else {
      int input_num = 1;
      for (int i = begin_norm_axis - 1; i < X->getDimensions().nbDims; i++) {
        input_num *= X->getDimensions().d[i];
      }
      std::vector<int64_t> mean_shape{input_num};
      std::vector<int64_t> variance_shape{input_num};
      plugin::LayerNormPlugin* plugin = new plugin::LayerNormPlugin(
          bias_data, bias_tensor->numel(), scale_data, scale_tensor->numel(),
          begin_norm_axis, eps, mean_shape, variance_shape);
      layernorm_layer = engine_->AddPlugin(
          &X, 1, reinterpret_cast<plugin::PluginTensorRT*>(plugin));
    }

    auto output_name = op_desc.Output("Y").front();
    engine_->SetWeights(op_desc.Input("Bias").front(), std::move(bias_tensor));
    engine_->SetWeights(op_desc.Input("Scale").front(),
                        std::move(scale_tensor));
    RreplenishLayerAndOutput(layernorm_layer, "layer_norm", {output_name},
                             test_mode);
  }
};

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle

REGISTER_TRT_OP_CONVERTER(layer_norm, LayerNormOpConverter);
