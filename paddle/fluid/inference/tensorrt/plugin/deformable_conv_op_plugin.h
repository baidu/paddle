// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include <cstdio>
#include <string>
#include <vector>

#include "paddle/fluid/inference/tensorrt/engine.h"
#include "paddle/fluid/inference/tensorrt/plugin/trt_plugin.h"
#include "paddle/fluid/platform/dynload/cublas.h"

namespace paddle {
namespace inference {
namespace tensorrt {
namespace plugin {

class DeformableConvPlugin : public nvinfer1::IPluginV2Ext {
 public:
  explicit DeformableConvPlugin(
      const nvinfer1::DataType data_type, const nvinfer1::Weights& weights,
      const std::vector<int>& kernel_dims, const std::vector<int>& strides,
      const std::vector<int>& paddings, const std::vector<int>& dilations,
      const int groups, const int deformable_groups, const int im2col_step);
  explicit DeformableConvPlugin(
      const nvinfer1::DataType data_type, const nvinfer1::Weights& weights,
      const std::vector<int>& kernel_dims, const std::vector<int>& strides,
      const std::vector<int>& paddings, const std::vector<int>& dilations,
      const int groups, const int deformable_groups, const int im2col_step,
      const std::vector<int>& input_dim, const std::vector<int>& offset_dim,
      const std::vector<int>& mask_dim, const std::vector<int>& output_dim);
  DeformableConvPlugin(const void* data, size_t length);
  ~DeformableConvPlugin() override;

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
  template <typename T>
  int enqueue_impl(int batch_size, const void* const* inputs, void** outputs,
                   void* workspace, cudaStream_t stream);
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

  void attachToContext(cudnnContext* cudnnContext, cublasContext* cublasContext,
                       nvinfer1::IGpuAllocator* gpuAllocator) override;

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
  nvinfer1::Weights copyToDevice(const void* hostData, size_t count);
  void serializeFromDevice(void** hostBuffer,
                           const nvinfer1::Weights& deviceWeights) const;
  nvinfer1::Weights deserializeToDevice(const void** hostBuffer, size_t count);

  nvinfer1::DataType data_type_;
  nvinfer1::Weights weights_;
  std::vector<int> kernel_dims_;
  std::vector<int> strides_;
  std::vector<int> paddings_;
  std::vector<int> dilations_;
  int groups_;
  int deformable_groups_;
  int im2col_step_;
  std::string namespace_;

  std::vector<int> input_dim_;
  std::vector<int> offset_dim_;
  std::vector<int> mask_dim_;
  std::vector<int> output_dim_;

  cublasHandle_t cublasHandle_;
};

class DeformableConvPluginCreator : public nvinfer1::IPluginCreator {
 public:
  DeformableConvPluginCreator();
  ~DeformableConvPluginCreator() override = default;

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

REGISTER_TRT_PLUGIN_V2(DeformableConvPluginCreator);

}  // namespace plugin
}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
