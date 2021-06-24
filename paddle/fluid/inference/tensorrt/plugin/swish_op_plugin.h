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

#include "paddle/fluid/inference/tensorrt/engine.h"
#include "paddle/fluid/inference/tensorrt/plugin/trt_plugin.h"

namespace paddle {
namespace inference {
namespace tensorrt {
namespace plugin {

class SwishPlugin : public PluginTensorRT {
 private:
  float beta_;

 protected:
  size_t getSerializationSize() override {
    return SerializedSize(getPluginType()) + getBaseSerializationSize() +
           SerializedSize(beta_);
  }

  // TRT will call this func when we need to serialize the configuration of
  // tensorrt.
  // It should not be called by users.
  void serialize(void* buffer) override {
    SerializeValue(&buffer, getPluginType());
    serializeBase(buffer);
    SerializeValue(&buffer, beta_);
  }

 public:
  explicit SwishPlugin(const float beta, const bool with_fp16) : beta_(beta) {
    with_fp16_ = with_fp16;
  }

  // It was used for tensorrt deserialization.
  // It should not be called by users.
  SwishPlugin(void const* serialData, size_t serialLength) {
    deserializeBase(serialData, serialLength);
    DeserializeValue(&serialData, &serialLength, &beta_);
  }
  ~SwishPlugin() {}
  int initialize() override;

  SwishPlugin* clone() const override {
    return new SwishPlugin(beta_, with_fp16_);
  }

  const char* getPluginType() const override { return "swish_plugin"; }
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

#if IS_TRT_VERSION_GE(6000)
class SwishPluginDynamic : public DynamicPluginTensorRT {
 public:
  explicit SwishPluginDynamic(const float beta, const bool with_fp16)
      : beta_(beta) {
    with_fp16_ = with_fp16;
  }
  SwishPluginDynamic(void const* serialData, size_t serialLength) {
    DeserializeValue(&serialData, &serialLength, &beta_);
    DeserializeValue(&serialData, &serialLength, &with_fp16_);
  }
  nvinfer1::IPluginV2DynamicExt* clone() const override {
    return new SwishPluginDynamic(beta_, with_fp16_);
  }

  const char* getPluginType() const override { return "swish_plugin"; }
  int getNbOutputs() const override { return 1; }
  int initialize() override;

  size_t getSerializationSize() const override;
  void serialize(void* buffer) const override;

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
  float beta_;
};

class SwishPluginDynamicCreator : public nvinfer1::IPluginCreator {
 public:
  SwishPluginDynamicCreator() {}
  const char* getPluginName() const override { return "swish_plugin"; }

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
    auto plugin = new SwishPluginDynamic(serial_data, serial_length);
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

REGISTER_TRT_PLUGIN_V2(SwishPluginDynamicCreator);
#endif

}  // namespace plugin
}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
