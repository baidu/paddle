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

#include <string>
#include <vector>

#include "paddle/fluid/inference/tensorrt/engine.h"
#include "paddle/fluid/inference/tensorrt/plugin/trt_plugin.h"

namespace paddle {
namespace inference {
namespace tensorrt {
namespace plugin {

class AnchorGeneratorPlugin : public nvinfer1::IPluginV2Ext {
 public:
  explicit AnchorGeneratorPlugin(
      const nvinfer1::DataType, const std::vector<float>& anchor_sizes,
      const std::vector<float>& aspect_ratios, const std::vector<float>& stride,
      const std::vector<float>& variances, const float offset, const int height,
      const int width, const int num_anchors, const int box_num);
  AnchorGeneratorPlugin(const void* data, size_t length);
  ~AnchorGeneratorPlugin() override;
  const char* getPluginType() const override;
  const char* getPluginVersion() const override;
  int getNbOutputs() const override;
  nvinfer1::Dims getOutputDimensions(int index, const nvinfer1::Dims* inputs,
                                     int nb_input_dims) override;
  bool supportsFormat(nvinfer1::DataType type,
                      nvinfer1::TensorFormat format) const override;
  size_t getWorkspaceSize(int max_batch_size) const override;
#if IS_TRT_VERSION_LT(8000)
  int enqueue(int batch_size, const void* const* inputs, void** outputs,
#else
  int enqueue(int batch_size, const void* const* inputs, void* const* outputs,
#endif
              void* workspace, cudaStream_t stream) override;
  int initialize() override;
  void terminate() override;
  size_t getSerializationSize() const override;
  void serialize(void* buffer) const override;
  void destroy() override;
  void setPluginNamespace(const char* lib_namespace) override;
  const char* getPluginNamespace() const override;
  nvinfer1::DataType getOutputDataType(int index,
                                       const nvinfer1::DataType* input_type,
                                       int nb_inputs) const override;
  bool isOutputBroadcastAcrossBatch(int output_index,
                                    const bool* input_is_broadcast,
                                    int nb_inputs) const override;
  bool canBroadcastInputAcrossBatch(int input_index) const override;
  void configurePlugin(const nvinfer1::Dims* input_dims, int nb_inputs,
                       const nvinfer1::Dims* output_dims, int nb_outputs,
                       const nvinfer1::DataType* input_types,
                       const nvinfer1::DataType* output_types,
                       const bool* input_is_broadcast,
                       const bool* output_is_broadcast,
                       nvinfer1::PluginFormat float_format,
                       int max_batct_size) override;
  nvinfer1::IPluginV2Ext* clone() const override;

 private:
  template <typename T>
  int enqueue_impl(int batch_size, const void* const* inputs, void** outputs,
                   void* workspace, cudaStream_t stream);
  nvinfer1::DataType data_type_;
  std::vector<float> anchor_sizes_;
  std::vector<float> aspect_ratios_;
  std::vector<float> stride_;
  std::vector<float> variances_;
  float offset_;
  void* anchor_sizes_device_;
  void* aspect_ratios_device_;
  void* stride_device_;
  void* variances_device_;
  int height_;
  int width_;
  int num_anchors_;
  int box_num_;
  std::string namespace_;
};

class AnchorGeneratorPluginCreator : public nvinfer1::IPluginCreator {
 public:
  AnchorGeneratorPluginCreator() = default;
  ~AnchorGeneratorPluginCreator() override = default;
  void setPluginNamespace(const char* lib_namespace) override;
  const char* getPluginNamespace() const override;
  const char* getPluginName() const override;
  const char* getPluginVersion() const override;
  const nvinfer1::PluginFieldCollection* getFieldNames() override;
  nvinfer1::IPluginV2Ext* createPlugin(
      const char* name, const nvinfer1::PluginFieldCollection* fc) override;
  nvinfer1::IPluginV2Ext* deserializePlugin(const char* name,
                                            const void* serial_data,
                                            size_t serial_length) override;

 private:
  std::string namespace_;
  nvinfer1::PluginFieldCollection field_collection_;
};

REGISTER_TRT_PLUGIN_V2(AnchorGeneratorPluginCreator);

#if IS_TRT_VERSION_GE(6000)
class AnchorGeneratorPluginDynamic : public DynamicPluginTensorRT {
 public:
  explicit AnchorGeneratorPluginDynamic(const nvinfer1::DataType data_type,
                                        const std::vector<float>& anchor_sizes,
                                        const std::vector<float>& aspect_ratios,
                                        const std::vector<float>& stride,
                                        const std::vector<float>& variances,
                                        const float offset,
                                        const int num_anchors);
  AnchorGeneratorPluginDynamic(void const* data, size_t length);
  ~AnchorGeneratorPluginDynamic();
  nvinfer1::IPluginV2DynamicExt* clone() const override;
  nvinfer1::DimsExprs getOutputDimensions(
      int outputIndex, const nvinfer1::DimsExprs* inputs, int nbInputs,
      nvinfer1::IExprBuilder& exprBuilder) override;
  bool supportsFormatCombination(int pos,
                                 const nvinfer1::PluginTensorDesc* inOut,
                                 int nbInputs, int nbOutputs) override;
  void configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in,
                       int nbInputs,
                       const nvinfer1::DynamicPluginTensorDesc* out,
                       int nbOutputs) override;
  size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs,
                          int nbInputs,
                          const nvinfer1::PluginTensorDesc* outputs,
                          int nbOutputs) const override;
  int enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
              const nvinfer1::PluginTensorDesc* outputDesc,
              const void* const* inputs, void* const* outputs, void* workspace,
              cudaStream_t stream) override;
  nvinfer1::DataType getOutputDataType(int index,
                                       const nvinfer1::DataType* inputTypes,
                                       int nbInputs) const override;
  const char* getPluginType() const override;
  int getNbOutputs() const override;
  int initialize() override;
  void terminate() override;
  size_t getSerializationSize() const override;
  void serialize(void* buffer) const override;
  void destroy() override;

 private:
  template <typename T>
  int enqueue_impl(const nvinfer1::PluginTensorDesc* inputDesc,
                   const nvinfer1::PluginTensorDesc* outputDesc,
                   const void* const* inputs, void* const* outputs,
                   void* workspace, cudaStream_t stream);
  nvinfer1::DataType data_type_;
  std::vector<float> anchor_sizes_;
  std::vector<float> aspect_ratios_;
  std::vector<float> stride_;
  std::vector<float> variances_;
  float offset_;
  void* anchor_sizes_device_;
  void* aspect_ratios_device_;
  void* stride_device_;
  void* variances_device_;
  int num_anchors_;
  std::string namespace_;
};

class AnchorGeneratorPluginDynamicCreator : public nvinfer1::IPluginCreator {
 public:
  AnchorGeneratorPluginDynamicCreator() = default;
  ~AnchorGeneratorPluginDynamicCreator() override = default;
  void setPluginNamespace(const char* lib_namespace) override;
  const char* getPluginNamespace() const override;
  const char* getPluginName() const override;
  const char* getPluginVersion() const override;
  const nvinfer1::PluginFieldCollection* getFieldNames() override;
  nvinfer1::IPluginV2Ext* createPlugin(
      const char* name, const nvinfer1::PluginFieldCollection* fc) override;
  nvinfer1::IPluginV2Ext* deserializePlugin(const char* name,
                                            const void* serial_data,
                                            size_t serial_length) override;

 private:
  std::string namespace_;
  nvinfer1::PluginFieldCollection field_collection_;
};
REGISTER_TRT_PLUGIN_V2(AnchorGeneratorPluginDynamicCreator);
#endif

}  // namespace plugin
}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
