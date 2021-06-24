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
#include "paddle/fluid/inference/tensorrt/helper.h"
#include "paddle/fluid/inference/tensorrt/plugin/emb_eltwise_layernorm_plugin.h"

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

class EmbEltwiseLayerNormOpConverter : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope, bool test_mode) override {
#if IS_TRT_VERSION_GE(6000)
    VLOG(4) << "convert fluid EmbEltwiseLayerNorm op to tensorrt layer";

    framework::OpDesc op_desc(op, nullptr);
    auto word_id_name = op_desc.Input("WordId").front();
    auto pos_id_name = op_desc.Input("PosId").front();
    engine_->Set("ernie_pos_name", new std::string(pos_id_name));

    auto sent_id_name = op_desc.Input("SentId").front();
    auto word_emb_name = op_desc.Input("WordEmbedding").front();
    auto pos_emb_name = op_desc.Input("PosEmbedding").front();
    auto sent_emb_name = op_desc.Input("SentEmbedding").front();

    std::vector<std::string> id_names;
    std::vector<std::string> emb_names;

    if (engine_->use_oss()) {
      id_names =
          std::vector<std::string>{word_id_name, pos_id_name, sent_id_name};
      emb_names =
          std::vector<std::string>{word_emb_name, pos_emb_name, sent_emb_name};
    } else {
      id_names = op_desc.Input("Ids");
      emb_names = op_desc.Input("Embs");
    }

    int input_num = id_names.size();

    // Declare inputs
    std::vector<nvinfer1::ITensor*> input_ids;
    for (int i = 0; i < input_num; i++) {
      input_ids.push_back(engine_->GetITensor(id_names[i]));
    }

    // input_embs[0]: word_embedding
    // input_embs[1]: pos_embedding
    // input_embs[2]: sent_embedding
    std::vector<float*> input_embs;
    std::vector<int> emb_sizes;

    // get the presistable var's data
    auto get_persistable_data = [&](const std::string& var_name,
                                    framework::DDim* dims) -> float* {
      auto* temp_var = scope.FindVar(var_name);
      auto* temp_tensor = temp_var->GetMutable<framework::LoDTensor>();
      (*dims) = temp_tensor->dims();

      auto* temp_data = engine_->GetWeightCPUData(var_name, temp_tensor, false);
      return temp_data;
    };

    int hidden = 0;
    for (int i = 0; i < input_num; i++) {
      framework::DDim emb_dims;
      float* emb_data = get_persistable_data(emb_names[i], &emb_dims);
      int64_t emb_size = framework::product(emb_dims);
      input_embs.push_back(emb_data);
      emb_sizes.push_back(emb_size);
      PADDLE_ENFORCE_EQ(
          emb_dims.size(), 2,
          platform::errors::InvalidArgument(
              "The fused EmbEltwiseLayerNorm's emb should be 2 dims."));
      hidden = emb_dims[1];
    }

    framework::DDim bias_dims, scale_dims;

    auto* bias =
        get_persistable_data(op_desc.Input("Bias").front(), &bias_dims);
    auto* scale =
        get_persistable_data(op_desc.Input("Scale").front(), &scale_dims);
    int64_t bias_size = framework::product(bias_dims);
    int64_t scale_size = framework::product(scale_dims);
    nvinfer1::ILayer* layer = nullptr;
    bool enable_int8 = op_desc.HasAttr("enable_int8");

    if (engine_->use_oss()) {
      int output_fp16 = static_cast<int>((engine_->WithFp16() == 1) ? 1 : 0);
      if (enable_int8) {
        output_fp16 = 1;
      }
      PADDLE_ENFORCE_EQ(
          input_num, 3,
          platform::errors::InvalidArgument(
              "When using oss and var-len, embedding_eltwise_layernorm op"
              "should have 3 inputs only, but got %d.",
              input_num));
      PADDLE_ENFORCE_EQ(
          output_fp16, 1,
          platform::errors::InvalidArgument(
              "Only Precision::KHalf(fp16) is supported when infering "
              "ernie(bert) model with config.EnableTensorRtOSS(). "
              "But Precision::KFloat32 is setted."));
      const std::vector<nvinfer1::PluginField> fields{
          {"bert_embeddings_layernorm_beta", bias,
           nvinfer1::PluginFieldType::kFLOAT32,
           static_cast<int32_t>(bias_size)},
          {"bert_embeddings_layernorm_gamma", scale,
           nvinfer1::PluginFieldType::kFLOAT32,
           static_cast<int32_t>(scale_size)},
          {"bert_embeddings_word_embeddings", input_embs[0],
           nvinfer1::PluginFieldType::kFLOAT32,
           static_cast<int32_t>(emb_sizes[0])},
          {"bert_embeddings_token_type_embeddings", input_embs[2],
           nvinfer1::PluginFieldType::kFLOAT32,
           static_cast<int32_t>(emb_sizes[2])},
          {"bert_embeddings_position_embeddings", input_embs[1],
           nvinfer1::PluginFieldType::kFLOAT32,
           static_cast<int32_t>(emb_sizes[1])},
          {"output_fp16", &output_fp16, nvinfer1::PluginFieldType::kINT32, 1},
      };

      // remember to free
      nvinfer1::PluginFieldCollection* plugin_ptr =
          static_cast<nvinfer1::PluginFieldCollection*>(
              malloc(sizeof(*plugin_ptr) +
                     fields.size() * sizeof(nvinfer1::PluginField)));
      plugin_ptr->nbFields = static_cast<int>(fields.size());
      plugin_ptr->fields = fields.data();

      std::vector<nvinfer1::ITensor*> plugin_inputs;
      plugin_inputs.emplace_back(
          engine_->GetITensor(word_id_name));  // word_embedding,
                                               // eval_placeholder_0
      plugin_inputs.emplace_back(
          engine_->GetITensor(sent_id_name));  // sent_embedding,
                                               // eval_placeholder_1
      plugin_inputs.emplace_back(
          engine_->GetITensor(pos_id_name));  // cu_seqlens,
                                              // eval_placeholder_2
      auto max_seqlen_tensor =
          engine_->GetITensor(engine_->network()->getInput(3)->getName());
      auto* shuffle_layer =
          TRT_ENGINE_ADD_LAYER(engine_, Shuffle, *max_seqlen_tensor);
      nvinfer1::Dims shape_dim;
      shape_dim.nbDims = 1;
      shape_dim.d[0] = -1;
      shuffle_layer->setReshapeDimensions(shape_dim);
      plugin_inputs.emplace_back(
          shuffle_layer->getOutput(0));  // max_seqlen, eval_placeholder_3

      auto creator = GetPluginRegistry()->getPluginCreator(
          "CustomEmbLayerNormPluginDynamic", "2");

      auto plugin_obj =
          creator->createPlugin("CustomEmbLayerNormPluginDynamic", plugin_ptr);
      auto plugin_layer = engine_->network()->addPluginV2(
          plugin_inputs.data(), plugin_inputs.size(), *plugin_obj);
      layer = plugin_layer;
      free(plugin_ptr);
      auto output_name = op_desc.Output("Out")[0];
      RreplenishLayerAndOutput(layer, "emb_eltwise_layernorm",
                               {output_name, std::string("qkv_plugin_mask")},
                               test_mode);
    } else {
      bool with_fp16 =
          engine_->WithFp16() && !engine_->disable_trt_plugin_fp16();
      float eps = BOOST_GET_CONST(float, op_desc.GetAttr("epsilon"));
      plugin::DynamicPluginTensorRT* plugin = nullptr;
      plugin = new plugin::EmbEltwiseLayernormPluginDynamic(
          input_embs, bias, scale, emb_sizes, bias_size, scale_size, hidden,
          eps, with_fp16);
      layer = engine_->AddDynamicPlugin(input_ids.data(), input_num, plugin);
      auto output_name = op_desc.Output("Out")[0];
      RreplenishLayerAndOutput(layer, "emb_eltwise_layernorm", {output_name},
                               test_mode);
    }

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

REGISTER_TRT_OP_CONVERTER(fused_embedding_eltwise_layernorm,
                          EmbEltwiseLayerNormOpConverter);
