// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <algorithm>
#include <string>
#include <vector>
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/fluid/inference/tensorrt/engine.h"
#include "paddle/fluid/inference/tensorrt/plugin/trt_plugin.h"

namespace paddle {
namespace inference {
namespace tensorrt {
namespace plugin {

class LayerNormPlugin : public PluginTensorRT {
  std::vector<float> bias_;
  std::vector<float> scale_;
  framework::Tensor scale_t;
  framework::Tensor bias_t;
  framework::Tensor mean_t;
  framework::Tensor variance_t;
  int begin_norm_axis_;
  float eps_;
  std::vector<int64_t> mean_shape_;
  std::vector<int64_t> variance_shape_;

 protected:
  size_t getSerializationSize() override {
    return getBaseSerializationSize() + SerializedSize(bias_) +
           SerializedSize(scale_) + SerializedSize(begin_norm_axis_) +
           SerializedSize(eps_) + SerializedSize(mean_shape_) +
           SerializedSize(variance_shape_) + SerializedSize(getPluginType());
  }

  // TRT will call this func when we need to serialize the configuration of
  // tensorrt.
  // It should not be called by users.
  void serialize(void* buffer) override {
    SerializeValue(&buffer, getPluginType());
    serializeBase(buffer);
    SerializeValue(&buffer, bias_);
    SerializeValue(&buffer, scale_);
    SerializeValue(&buffer, begin_norm_axis_);
    SerializeValue(&buffer, eps_);
    SerializeValue(&buffer, mean_shape_);
    SerializeValue(&buffer, variance_shape_);
  }

 public:
  LayerNormPlugin(const float* bias, const int bias_num, const float* scale,
                  const int scale_num, int begin_norm_axis, float eps,
                  std::vector<int64_t> mean_shape,
                  std::vector<int64_t> variance_shape)
      : begin_norm_axis_(begin_norm_axis),
        eps_(eps),
        mean_shape_(mean_shape),
        variance_shape_(variance_shape) {
    bias_.resize(bias_num);
    scale_.resize(scale_num);
    std::copy(bias, bias + bias_num, bias_.data());
    std::copy(scale, scale + scale_num, scale_.data());
  }

  // It was used for tensorrt deserialization.
  // It should not be called by users.
  LayerNormPlugin(void const* serialData, size_t serialLength) {
    deserializeBase(serialData, serialLength);
    DeserializeValue(&serialData, &serialLength, &bias_);
    DeserializeValue(&serialData, &serialLength, &scale_);
    DeserializeValue(&serialData, &serialLength, &begin_norm_axis_);
    DeserializeValue(&serialData, &serialLength, &eps_);
    DeserializeValue(&serialData, &serialLength, &mean_shape_);
    DeserializeValue(&serialData, &serialLength, &variance_shape_);
  }
  ~LayerNormPlugin() {}
  int initialize() override;

  LayerNormPlugin* clone() const override {
    return new LayerNormPlugin(bias_.data(), bias_.size(), scale_.data(),
                               scale_.size(), begin_norm_axis_, eps_,
                               mean_shape_, variance_shape_);
  }

  const char* getPluginType() const override { return "layer_norm_plugin"; }
  int getNbOutputs() const override { return 1; }
  nvinfer1::Dims getOutputDimensions(int index, const nvinfer1::Dims* inputs,
                                     int nbInputDims) override;
#if IS_TRT_VERSION_LT(8000)
  int enqueue(int batchSize, const void* const* inputs, void** outputs,
#else
  int enqueue(int batchSize, const void* const* inputs, void* const* outputs,
#endif
              void* workspace, cudaStream_t stream) override;
};

class LayerNormPluginDynamic : public DynamicPluginTensorRT {
 public:
  LayerNormPluginDynamic(const float* bias, const int bias_num,
                         const float* scale, const int scale_num,
                         int begin_norm_axis, float eps,
                         std::vector<int64_t> mean_shape,
                         std::vector<int64_t> variance_shape)
      : begin_norm_axis_(begin_norm_axis),
        eps_(eps),
        mean_shape_(mean_shape),
        variance_shape_(variance_shape) {
    bias_.resize(bias_num);
    scale_.resize(scale_num);
    std::copy(bias, bias + bias_num, bias_.data());
    std::copy(scale, scale + scale_num, scale_.data());
  }

  LayerNormPluginDynamic(void const* serialData, size_t serialLength) {
    DeserializeValue(&serialData, &serialLength, &bias_);
    DeserializeValue(&serialData, &serialLength, &scale_);
    DeserializeValue(&serialData, &serialLength, &begin_norm_axis_);
    DeserializeValue(&serialData, &serialLength, &eps_);
    DeserializeValue(&serialData, &serialLength, &mean_shape_);
    DeserializeValue(&serialData, &serialLength, &variance_shape_);
  }
  nvinfer1::IPluginV2DynamicExt* clone() const override {
    return new LayerNormPluginDynamic(bias_.data(), bias_.size(), scale_.data(),
                                      scale_.size(), begin_norm_axis_, eps_,
                                      mean_shape_, variance_shape_);
  }

  const char* getPluginType() const override { return "layernorm_plugin"; }
  int getNbOutputs() const override { return 1; }
  int initialize() override { return 0; }

  size_t getSerializationSize() const override {
    return SerializedSize(bias_) + SerializedSize(scale_) +
           SerializedSize(begin_norm_axis_) + SerializedSize(eps_) +
           SerializedSize(mean_shape_) + SerializedSize(variance_shape_);
  }

  void serialize(void* buffer) const override {
    SerializeValue(&buffer, bias_);
    SerializeValue(&buffer, scale_);
    SerializeValue(&buffer, begin_norm_axis_);
    SerializeValue(&buffer, eps_);
    SerializeValue(&buffer, mean_shape_);
    SerializeValue(&buffer, variance_shape_);
  }

  nvinfer1::DimsExprs getOutputDimensions(
      int output_index, const nvinfer1::DimsExprs* inputs, int nb_inputs,
      nvinfer1::IExprBuilder& expr_builder) override;

  bool supportsFormatCombination(int pos,
                                 const nvinfer1::PluginTensorDesc* inOut,
                                 int nbInputs, int nbOutputs) override;

  void configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in,
                       int nbInputs,
                       const nvinfer1::DynamicPluginTensorDesc* out,
                       int nbOutputs) override {}

  size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs,
                          int nbInputs,
                          const nvinfer1::PluginTensorDesc* outputs,
                          int nbOutputs) const override {
    return 0;
  }

  int enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
              const nvinfer1::PluginTensorDesc* outputDesc,
              const void* const* inputs, void* const* outputs, void* workspace,
              cudaStream_t stream) override;
  nvinfer1::DataType getOutputDataType(int index,
                                       const nvinfer1::DataType* inputTypes,
                                       int nbInputs) const override;

  void destroy() override { delete this; }

 private:
  std::vector<float> bias_;
  std::vector<float> scale_;
  framework::Tensor scale_t;
  framework::Tensor bias_t;
  framework::Tensor mean_t;
  framework::Tensor variance_t;
  int begin_norm_axis_;
  float eps_;
  std::vector<int64_t> mean_shape_;
  std::vector<int64_t> variance_shape_;
};

class LayerNormPluginDynamicCreator : public nvinfer1::IPluginCreator {
 public:
  LayerNormPluginDynamicCreator() {}
  const char* getPluginName() const override { return "layernorm_plugin"; }

  const char* getPluginVersion() const override { return "1"; }

  const nvinfer1::PluginFieldCollection* getFieldNames() override {
    return &field_collection_;
  }

  nvinfer1::IPluginV2* createPlugin(
      const char* name, const nvinfer1::PluginFieldCollection* fc) override {
    return nullptr;
  }

  nvinfer1::IPluginV2* deserializePlugin(const char* name,
                                         const void* serial_data,
                                         size_t serial_length) override {
    auto plugin = new LayerNormPluginDynamic(serial_data, serial_length);
    return plugin;
  }

  void setPluginNamespace(const char* lib_namespace) override {
    plugin_namespace_ = lib_namespace;
  }

  const char* getPluginNamespace() const override {
    return plugin_namespace_.c_str();
  }

 private:
  std::string plugin_namespace_;
  std::string plugin_name_;
  nvinfer1::PluginFieldCollection field_collection_{0, nullptr};
  std::vector<nvinfer1::PluginField> plugin_attributes_;
};

REGISTER_TRT_PLUGIN_V2(LayerNormPluginDynamicCreator);

}  // namespace plugin
}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
