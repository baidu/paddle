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
#include "paddle/fluid/inference/tensorrt/plugin/mish_op_plugin.h"

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
 * Mish converter from fluid to tensorRT.
 */
class MishOpConverter : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope, bool test_mode) override {
    VLOG(4) << "convert fluid Mish op to tensorrt Mish plugin";

    framework::OpDesc op_desc(op, nullptr);
    // Declare inputs
    int input_num = op_desc.Input("X").size();
    PADDLE_ENFORCE_EQ(input_num, 1,
                      platform::errors::InvalidArgument(
                          "Mish op has only 1 input, but got %d", input_num));
    auto* input = engine_->GetITensor(op_desc.Input("X")[0]);
    // Get output
    size_t output_num = op_desc.Output("Out").size();
    PADDLE_ENFORCE_EQ(output_num, 1,
                      platform::errors::InvalidArgument(
                          "Mish op has only 1 output, but got %d", output_num));

    const float threshold =
        op_desc.HasAttr("threshold")
            ? BOOST_GET_CONST(float, op_desc.GetAttr("threshold"))
            : 20.0f;

    nvinfer1::ILayer* layer = nullptr;
    if (engine_->with_dynamic_shape()) {
#if IS_TRT_VERSION_GE(6000)
      bool with_fp16 =
          engine_->WithFp16() && !engine_->disable_trt_plugin_fp16();
      plugin::MishPluginDynamic* plugin =
          new plugin::MishPluginDynamic(threshold, with_fp16);
      layer = engine_->AddDynamicPlugin(&input, input_num, plugin);
#else
      PADDLE_THROW(platform::errors::Fatal(
          "You are running the TRT Dynamic Shape mode, need to confirm that "
          "your TRT version is no less than 6.0"));
#endif
    } else {
      bool with_fp16 =
          engine_->WithFp16() && !engine_->disable_trt_plugin_fp16();
      plugin::MishPlugin* plugin = new plugin::MishPlugin(threshold, with_fp16);
      layer = engine_->AddPlugin(&input, input_num, plugin);
    }

    auto output_name = op_desc.Output("Out")[0];
    RreplenishLayerAndOutput(layer, "mish", {output_name}, test_mode);
  }
};

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle

REGISTER_TRT_OP_CONVERTER(mish, MishOpConverter);
