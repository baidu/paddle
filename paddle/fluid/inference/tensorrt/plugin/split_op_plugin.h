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

#include <thrust/device_vector.h>
#include <string>
#include <utility>
#include <vector>
#include "paddle/fluid/inference/tensorrt/plugin/trt_plugin.h"

namespace paddle {
namespace inference {
namespace tensorrt {
namespace plugin {

class SplitPlugin : public PluginTensorRTV2Ext {
 public:
  SplitPlugin() {}
  SplitPlugin(int axis, std::vector<int> const& output_lengths, bool with_fp16)
      : axis_(axis), same_shape_(true), output_length_(output_lengths) {
    with_fp16_ = with_fp16;
  }

  SplitPlugin(void const* serial_data, size_t serial_length) {
    deserializeBase(serial_data, serial_length);
    DeserializeValue(&serial_data, &serial_length, &axis_);
    DeserializeValue(&serial_data, &serial_length, &output_length_);
  }

  nvinfer1::IPluginV2Ext* clone() const override {
    SplitPlugin* ptr = new SplitPlugin(axis_, output_length_, with_fp16_);
    ptr->setPluginNamespace(this->getPluginNamespace());
    ptr->shareData(this);
    return ptr;
  }

  nvinfer1::DataType getOutputDataType(int index,
                                       const nvinfer1::DataType* input_types,
                                       int nb_inputs) const override {
    return input_types[0];
  }

  const char* getPluginType() const override { return "split_plugin_v2ext"; }
  int getNbOutputs() const override { return output_length_.size(); }
  nvinfer1::Dims getOutputDimensions(int index,
                                     const nvinfer1::Dims* input_dims,
                                     int num_inputs) override;

  int initialize() override;
  void terminate() override;
#if IS_TRT_VERSION_LT(8000)
  int enqueue(int batch_size, const void* const* inputs, void** outputs,
#else
  int enqueue(int batch_size, const void* const* inputs, void* const* outputs,
#endif
              void* workspace, cudaStream_t stream) override;

  void destroy() override { delete this; }

 protected:
  size_t getSerializationSize() const override {
    return SerializedSize(axis_) + SerializedSize(output_length_) +
           getBaseSerializationSize();
  }

  void serialize(void* buffer) const override {
    serializeBase(buffer);
    SerializeValue(&buffer, axis_);
    SerializeValue(&buffer, output_length_);
  }

  int axis_;
  int outer_rows_;
  int inner_cols_;
  int axis_shape_;
  bool same_shape_;
  std::vector<int> output_length_;
  std::vector<int> segment_offsets_;
  thrust::device_vector<int> d_segment_offsets_;
  thrust::device_vector<float*> d_output_ptrs_;

 private:
  void shareData(const SplitPlugin* another);
};

class SplitPluginCreator : public nvinfer1::IPluginCreator {
 public:
  SplitPluginCreator() {}
  const char* getPluginName() const override { return "split_plugin_v2ext"; }

  const char* getPluginVersion() const override { return "1"; }

  const nvinfer1::PluginFieldCollection* getFieldNames() override {
    return &field_collection_;
  }

  nvinfer1::IPluginV2* createPlugin(
      const char* name, const nvinfer1::PluginFieldCollection* fc) override {
    // not implemented
    return nullptr;
  }

  nvinfer1::IPluginV2* deserializePlugin(const char* name,
                                         const void* serial_data,
                                         size_t serial_length) override {
    auto plugin = new SplitPlugin(serial_data, serial_length);
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

REGISTER_TRT_PLUGIN_V2(SplitPluginCreator);

#if IS_TRT_VERSION_GE(6000)
class SplitPluginDynamic : public DynamicPluginTensorRT {
 public:
  SplitPluginDynamic(int axis, std::vector<int> const& output_lengths,
                     bool with_fp16)
      : axis_(axis), output_length_(output_lengths) {
    with_fp16_ = with_fp16;
  }

  SplitPluginDynamic(void const* serial_data, size_t serial_length) {
    DeserializeValue(&serial_data, &serial_length, &axis_);
    DeserializeValue(&serial_data, &serial_length, &output_length_);
    DeserializeValue(&serial_data, &serial_length, &with_fp16_);
  }

  nvinfer1::IPluginV2DynamicExt* clone() const override {
    return new SplitPluginDynamic(axis_, output_length_, with_fp16_);
  }

  const char* getPluginType() const override { return "split_plugin"; }
  int getNbOutputs() const override { return output_length_.size(); }
  int initialize() override;

  size_t getSerializationSize() const override;
  void serialize(void* buffer) const override;

  nvinfer1::DimsExprs getOutputDimensions(
      int outputIndex, const nvinfer1::DimsExprs* inputs, int nbInputs,
      nvinfer1::IExprBuilder& exprBuilder) override;

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
  int axis_;
  std::vector<int> output_length_;
};

class SplitPluginDynamicCreator : public nvinfer1::IPluginCreator {
 public:
  SplitPluginDynamicCreator() {}
  const char* getPluginName() const override { return "split_plugin"; }

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
    auto plugin = new SplitPluginDynamic(serial_data, serial_length);
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

REGISTER_TRT_PLUGIN_V2(SplitPluginDynamicCreator);
#endif

}  // namespace plugin
}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
