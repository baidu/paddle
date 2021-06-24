// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
#include <stdio.h>
#include <cassert>
#include <string>
#include <vector>
#include "paddle/fluid/inference/tensorrt/plugin/trt_plugin.h"

namespace paddle {
namespace inference {
namespace tensorrt {
namespace plugin {

class GeluPlugin : public PluginTensorRT {
 public:
  explicit GeluPlugin(const bool with_fp16) { with_fp16_ = with_fp16; }

  // It was used for tensorrt deserialization.
  // It should not be called by users.
  GeluPlugin(void const* serial_data, size_t serial_length) {
    deserializeBase(serial_data, serial_length);
  }

  ~GeluPlugin() {}
  GeluPlugin* clone() const override { return new GeluPlugin(with_fp16_); }

  const char* getPluginType() const override { return "gelu_plugin"; }
  int getNbOutputs() const override { return 1; }
  int initialize() override { return 0; }
  bool supportsFormat(nvinfer1::DataType type,
                      nvinfer1::PluginFormat format) const override;
  nvinfer1::Dims getOutputDimensions(int index, const nvinfer1::Dims* inputs,
                                     int nb_input_dims) override;
#if IS_TRT_VERSION_LT(8000)
  int enqueue(int batch_size, const void* const* inputs, void** outputs,
#else
  int enqueue(int batch_size, const void* const* inputs, void* const* outputs,
#endif
              void* workspace, cudaStream_t stream) override;

 protected:
  size_t getSerializationSize() override {
    return getBaseSerializationSize() + SerializedSize(getPluginType());
  }

  // TRT will call this func  to serialize the configuration of TRT
  // It should not be called by users.
  void serialize(void* buffer) override {
    SerializeValue(&buffer, getPluginType());
    serializeBase(buffer);
  }
};

#if IS_TRT_VERSION_GE(6000)
class GeluPluginDynamic : public DynamicPluginTensorRT {
 public:
  explicit GeluPluginDynamic(const bool with_fp16) { with_fp16_ = with_fp16; }
  GeluPluginDynamic(void const* serial_data, size_t serial_length) {
    DeserializeValue(&serial_data, &serial_length, &with_fp16_);
  }

  ~GeluPluginDynamic() {}
  nvinfer1::IPluginV2DynamicExt* clone() const override {
    return new GeluPluginDynamic(with_fp16_);
  }

  const char* getPluginType() const override { return "gelu_plugin"; }
  int getNbOutputs() const override { return 1; }
  int initialize() override { return 0; }

  size_t getSerializationSize() const override {
    return SerializedSize(with_fp16_);
  }
  void serialize(void* buffer) const override {
    SerializeValue(&buffer, with_fp16_);
  }

  nvinfer1::DimsExprs getOutputDimensions(
      int output_index, const nvinfer1::DimsExprs* inputs, int nb_inputs,
      nvinfer1::IExprBuilder& expr_builder) override;

  bool supportsFormatCombination(int pos,
                                 const nvinfer1::PluginTensorDesc* in_out,
                                 int nb_inputs, int nb_outputs) override;

  void configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in,
                       int nb_inputs,
                       const nvinfer1::DynamicPluginTensorDesc* out,
                       int nb_outputs) override {}

  size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs,
                          int nb_inputs,
                          const nvinfer1::PluginTensorDesc* outputs,
                          int nb_outputs) const override {
    return 0;
  }

  int enqueue(const nvinfer1::PluginTensorDesc* input_desc,
              const nvinfer1::PluginTensorDesc* output_desc,
              const void* const* inputs, void* const* outputs, void* workspace,
              cudaStream_t stream) override;
  nvinfer1::DataType getOutputDataType(int index,
                                       const nvinfer1::DataType* input_types,
                                       int nb_inputs) const override;

  void destroy() override { delete this; }
};

class GeluPluginDynamicCreator : public nvinfer1::IPluginCreator {
 public:
  GeluPluginDynamicCreator() {}
  const char* getPluginName() const override { return "gelu_plugin"; }

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
    auto plugin = new GeluPluginDynamic(serial_data, serial_length);
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

REGISTER_TRT_PLUGIN_V2(GeluPluginDynamicCreator);
#endif

}  // namespace plugin
}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
