/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  See
the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/inference/tensorrt/convert/op_converter.h"
#include "paddle/fluid/inference/tensorrt/plugin/qkv_to_context_plugin.h"

namespace paddle {
namespace inference {
namespace tensorrt {

class MultiheadMatMulOpConverter : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope, bool test_mode) override {
#if IS_TRT_VERSION_GE(6000)
    VLOG(3) << "convert a fluid multihead_mamul op to a corresponding tensorrt "
               "network structure";
    framework::OpDesc op_desc(op, nullptr);
    // Declare inputs
    auto* input = engine_->GetITensor(op_desc.Input("Input").front());

    // fc weights and fc bias
    auto weight_name = op_desc.Input("W").front();
    auto bias_name = op_desc.Input("Bias").front();

    auto* weight_v = scope.FindVar(weight_name);
    auto* weight_t = weight_v->GetMutable<framework::LoDTensor>();

    auto* bias_v = scope.FindVar(bias_name);
    auto* bias_t = bias_v->GetMutable<framework::LoDTensor>();

    float* weight_data = nullptr;
    bool enable_int8 = op_desc.HasAttr("enable_int8");
    float in_scale = 0.;

    if (enable_int8) {
      PADDLE_ENFORCE_EQ(
          op_desc.HasAttr("Input_scale"), true,
          platform::errors::InvalidArgument(
              "must have input scale in multihead layers in int8 mode"));
      in_scale = BOOST_GET_CONST(float, op_desc.GetAttr("Input_scale")) * 127;
      auto weight_scale =
          BOOST_GET_CONST(std::vector<float>, op_desc.GetAttr("weight_scale"));
      weight_data =
          engine_->GetWeightCPUData(weight_name, weight_t, true, weight_scale);
      engine_->SetTensorDynamicRange(input, in_scale);
    } else {
      weight_data = engine_->GetWeightCPUData(weight_name, weight_t, false);
    }

    float* bias_data = engine_->GetWeightCPUData(bias_name, bias_t, false);
    std::vector<float> weight_data_tmp;
    weight_data_tmp.reserve(weight_t->numel());
    memcpy(weight_data_tmp.data(), weight_data,
           weight_t->numel() * sizeof(float));

    // (hidden_in, 3, hidden_out)
    auto weight_dims = weight_t->dims();

    int hidden_in = weight_dims[0];   // channels_in
    int three = weight_dims[1];       // channels_out
    int hidden_out = weight_dims[2];  // channels_out
    int m = hidden_in;
    int n = three * hidden_out;
    auto tranpose_weight = [](const float* src, float* dst, int m, int n) {
      for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
          dst[j * m + i] = src[i * n + j];
        }
      }
    };
    tranpose_weight(weight_data_tmp.data(), weight_data, m, n);

    int head_number = BOOST_GET_CONST(int, op_desc.GetAttr("head_number"));

    nvinfer1::ILayer* layer = nullptr;
    auto output_name = op_desc.Output("Out")[0];

    if (engine_->with_dynamic_shape()) {
      if (engine_->use_oss()) {
        int head_size = hidden_out / head_number;
        // [3, head_number, head_size, hidden_in] -> [head_number, 3, head_size,
        // hidden_in]
        auto transpose_weight_v2 = [](const float* src, float* dst, int three,
                                      int head_number, int head_size,
                                      int hidden_in) {
          const int HH = head_size * hidden_in;
          for (int i = 0; i < three; ++i) {
            for (int n = 0; n < head_number; ++n) {
              for (int hh = 0; hh < HH; ++hh) {
                dst[n * three * HH + i * HH + hh] =
                    src[i * head_number * HH + n * HH + hh];
              }
            }
          }
        };
        // [3, head_number, head_size] -> [head_number, 3, head_size]
        auto transpose_bias_v2 = [](const float* src, float* dst, int N,
                                    int H) {
          for (int i = 0; i < 3; ++i) {
            for (int n = 0; n < N; ++n) {
              for (int h = 0; h < H; ++h) {
                dst[n * 3 * H + i * H + h] = src[i * N * H + n * H + h];
              }
            }
          }
        };
        memcpy(weight_data_tmp.data(), weight_data,
               weight_t->numel() * sizeof(float));
        transpose_weight_v2(weight_data_tmp.data(), weight_data, three,
                            head_number, head_size, hidden_in);
        nvinfer1::Weights weight{nvinfer1::DataType::kFLOAT,
                                 static_cast<void*>(weight_data),
                                 static_cast<int32_t>(weight_t->numel())};

        std::vector<float> bias_data_tmp;
        bias_data_tmp.reserve(bias_t->numel());
        memcpy(bias_data_tmp.data(), bias_data,
               bias_t->numel() * sizeof(float));
        transpose_bias_v2(bias_data_tmp.data(), bias_data, head_number,
                          head_size);
        nvinfer1::Weights bias{nvinfer1::DataType::kFLOAT,
                               static_cast<void*>(bias_data),
                               static_cast<int32_t>(bias_t->numel())};

        nvinfer1::ILayer* fc_layer = nullptr;
        float dp_probs = 1.0 / 127.0;
        if (enable_int8) {
          nvinfer1::DimsHW nv_ksize(1, 1);
          fc_layer = TRT_ENGINE_ADD_LAYER(engine_, Convolution, *input, n,
                                          nv_ksize, weight, bias);
        } else {
          fc_layer = TRT_ENGINE_ADD_LAYER(engine_, FullyConnected, *input, n,
                                          weight, bias);
        }

        if (enable_int8) {
          PADDLE_ENFORCE_EQ(
              op_desc.HasAttr("out_threshold"), true,
              platform::errors::InvalidArgument(
                  "must have out threshold in multihead layers in int8 mode"));
          float out_scale =
              BOOST_GET_CONST(float, op_desc.GetAttr("out_threshold"));
          engine_->SetTensorDynamicRange(fc_layer->getOutput(0), out_scale);
          dp_probs = out_scale / 127.0;
        }

        auto mask_tensor = engine_->GetITensor("qkv_plugin_mask");

        auto creator = GetPluginRegistry()->getPluginCreator(
            "CustomQKVToContextPluginDynamic", "2");
        assert(creator != nullptr);
        int type = static_cast<int>((engine_->WithFp16() == 1)
                                        ? nvinfer1::DataType::kHALF
                                        : nvinfer1::DataType::kFLOAT);
        if (enable_int8) {
          type = static_cast<int>(nvinfer1::DataType::kHALF);
        }
        bool has_mask = true;
        int var_seqlen = 1;
        const std::vector<nvinfer1::PluginField> fields{
            {"type_id", &type, nvinfer1::PluginFieldType::kINT32, 1},
            {"hidden_size", &hidden_out, nvinfer1::PluginFieldType::kINT32, 1},
            {"num_heads", &head_number, nvinfer1::PluginFieldType::kINT32, 1},
            {"has_mask", &has_mask, nvinfer1::PluginFieldType::kINT32, 1},
            {"var_seqlen", &var_seqlen, nvinfer1::PluginFieldType::kINT32, 1},
            { "dq_probs", &dp_probs, nvinfer1::PluginFieldType::kFLOAT32, 1 }};
        nvinfer1::PluginFieldCollection* plugin_collection =
            static_cast<nvinfer1::PluginFieldCollection*>(
                malloc(sizeof(*plugin_collection) +
                       fields.size() *
                           sizeof(nvinfer1::PluginField)));  // remember to free
        plugin_collection->nbFields = static_cast<int>(fields.size());
        plugin_collection->fields = fields.data();

        auto plugin = creator->createPlugin("CustomQKVToContextPluginDynamic",
                                            plugin_collection);
        free(plugin_collection);

        std::vector<nvinfer1::ITensor*> plugin_inputs;
        plugin_inputs.emplace_back(fc_layer->getOutput(0));
        plugin_inputs.emplace_back(mask_tensor);
        if (engine_->Has("ernie_pos_name")) {
          plugin_inputs.emplace_back(
              engine_->GetITensor(engine_->Get<std::string>("ernie_pos_name")));
        } else {
          plugin_inputs.emplace_back(engine_->GetITensor(
              engine_->network()
                  ->getInput(2)
                  ->getName()));  // cu_seqlens, eval_placeholder_2
        }
        auto max_seqlen_tensor =
            engine_->GetITensor(engine_->network()->getInput(3)->getName());
        auto* shuffle_layer = TRT_ENGINE_ADD_LAYER(
            engine_, Shuffle,
            *const_cast<nvinfer1::ITensor*>(max_seqlen_tensor));
        nvinfer1::Dims shape_dim;
        shape_dim.nbDims = 1;
        shape_dim.d[0] = -1;
        shuffle_layer->setReshapeDimensions(shape_dim);
        plugin_inputs.emplace_back(
            shuffle_layer->getOutput(0));  // max_seqlen, eval_placeholder_3

        auto plugin_layer = engine_->network()->addPluginV2(
            plugin_inputs.data(), plugin_inputs.size(), *plugin);
        layer = plugin_layer;
      } else {
        PADDLE_ENFORCE_EQ(
            input->getDimensions().nbDims, 3,
            platform::errors::InvalidArgument(
                "The Input dim of the MultiheadMatMul should be 3, "
                "but it's (%d) now.",
                input->getDimensions().nbDims));
        // transpose weight_data from m * n to  n * m
        auto* input_bias_qk =
            engine_->GetITensor(op_desc.Input("BiasQK").front());

        TensorRTEngine::Weight weight{nvinfer1::DataType::kFLOAT,
                                      static_cast<void*>(weight_data),
                                      static_cast<size_t>(weight_t->numel())};
        weight.dims.assign({n, m});

        TensorRTEngine::Weight bias{nvinfer1::DataType::kFLOAT,
                                    static_cast<void*>(bias_data),
                                    static_cast<size_t>(bias_t->numel())};

        // add shuffle before fc
        nvinfer1::Dims reshape_before_fc_dim;
        reshape_before_fc_dim.nbDims = 5;
        reshape_before_fc_dim.d[0] = 0;
        reshape_before_fc_dim.d[1] = 0;
        reshape_before_fc_dim.d[2] = 0;
        reshape_before_fc_dim.d[3] = 1;
        reshape_before_fc_dim.d[4] = 1;
        auto* reshape_before_fc_layer =
            TRT_ENGINE_ADD_LAYER(engine_, Shuffle, *input);
        reshape_before_fc_layer->setReshapeDimensions(reshape_before_fc_dim);
        reshape_before_fc_layer->setName(
            ("shuffle_before_multihead_mamul(Output: " + output_name + ")")
                .c_str());

        // add layer fc
        auto* fc_layer = TRT_ENGINE_ADD_LAYER(
            engine_, FullyConnected, *reshape_before_fc_layer->getOutput(0), n,
            weight.get(), bias.get());
        fc_layer->setName(
            ("multihead_mamul_fc(Output: " + output_name + ")").c_str());

        // no need to add shuffle after fc, just change it in
        // QkvToContextPluginDynamic

        // add qkv to context
        int head_size = hidden_out / head_number;
        float scale = BOOST_GET_CONST(float, op_desc.GetAttr("alpha"));

        std::vector<nvinfer1::ITensor*> plugin_inputs;
        plugin_inputs.push_back(fc_layer->getOutput(0));
        plugin_inputs.push_back(input_bias_qk);
        bool with_fp16 =
            engine_->WithFp16() && !engine_->disable_trt_plugin_fp16();
        plugin::DynamicPluginTensorRT* plugin =
            new plugin::QkvToContextPluginDynamic(hidden_in, head_number,
                                                  head_size, scale, with_fp16);
        layer = engine_->AddDynamicPlugin(plugin_inputs.data(), 2, plugin);
      }
    } else {
      PADDLE_THROW(platform::errors::Fatal(
          "You are running the Ernie(Bert) model in static shape mode, which "
          "is not supported for the time being.\n"
          "You can use the config.SetTRTDynamicShapeInfo(...) interface to set "
          "the shape information to run the dynamic shape mode."));
    }
    RreplenishLayerAndOutput(layer, "multihead_matmul", {output_name},
                             test_mode);
#else
    PADDLE_THROW(platform::errors::Fatal(
        "You are running the TRT Dynamic Shape mode, need to confirm that "
        "your TRT version is no less than 6.0"));
#endif
  }
};

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle

REGISTER_TRT_OP_CONVERTER(multihead_matmul, MultiheadMatMulOpConverter);
