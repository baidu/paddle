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

class SlicePlugin : public PluginTensorRT {
 public:
  explicit SlicePlugin(std::vector<int> starts, std::vector<int> ends,
                       std::vector<int> axes, bool with_fp16);

  // It was used for tensorrt deserialization.
  // It should not be called by users.
  SlicePlugin(void const* serial_data, size_t serial_length);
  ~SlicePlugin();
  SlicePlugin* clone() const override;

  const char* getPluginType() const override { return "slice_plugin"; }
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
  size_t getSerializationSize() override;

  // TRT will call this func  to serialize the configuration of TRT
  // It should not be called by users.
  void serialize(void* buffer) override;

 private:
  std::vector<int> starts_;
  std::vector<int> ends_;
  std::vector<int> axes_;
  int* offset_temp_data_{nullptr};
  cudaEvent_t copy_event_;
  cudaStream_t copy_stream_;
};

#if IS_TRT_VERSION_GE(6000)
class SlicePluginDynamic : public DynamicPluginTensorRT {
 public:
  explicit SlicePluginDynamic(std::vector<int> starts, std::vector<int> ends,
                              std::vector<int> axes, bool with_fp16);

  nvinfer1::IPluginV2DynamicExt* clone() const override {
    return new SlicePluginDynamic(starts_, ends_, axes_, with_fp16_);
  }

  SlicePluginDynamic(void const* serialData, size_t serialLength);

  const char* getPluginType() const override { return "slice_plugin"; }
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

  void destroy() override;

 private:
  std::vector<int> starts_;
  std::vector<int> ends_;
  std::vector<int> axes_;
  int* offset_temp_data_{nullptr};
  cudaEvent_t copy_event_;
  cudaStream_t copy_stream_;
};

class SlicePluginDynamicCreator : public nvinfer1::IPluginCreator {
 public:
  SlicePluginDynamicCreator() {}
  const char* getPluginName() const override { return "slice_plugin"; }

  const char* getPluginVersion() const override { return "1"; }

  const nvinfer1::PluginFieldCollection* getFieldNames() override {
    return &field_collection_;
  }

  nvinfer1::IPluginV2* createPlugin(
      const char* name, const nvinfer1::PluginFieldCollection* fc) override {
    return nullptr;
  }

  nvinfer1::IPluginV2* deserializePlugin(const char* name,
                                         const void* serialData,
                                         size_t serialLength) override {
    auto plugin = new SlicePluginDynamic(serialData, serialLength);
    return plugin;
  }

  void setPluginNamespace(const char* libNamespace) override {
    namespace_ = libNamespace;
  }

  const char* getPluginNamespace() const override { return namespace_.c_str(); }

 private:
  std::string namespace_;
  nvinfer1::PluginFieldCollection field_collection_;
};

REGISTER_TRT_PLUGIN_V2(SlicePluginDynamicCreator);

#endif

}  // namespace plugin
}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
