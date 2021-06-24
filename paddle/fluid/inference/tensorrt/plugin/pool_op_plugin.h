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
#include <stdio.h>
#include <cassert>
#include <string>
#include <vector>
#include "paddle/fluid/inference/tensorrt/plugin/trt_plugin.h"

namespace paddle {
namespace inference {
namespace tensorrt {
namespace plugin {

static std::vector<int> CalcOutputSize(const std::vector<int>& input_shape,
                                       const bool& ceil_mode,
                                       const bool& adaptive,
                                       const std::vector<int>& ksize,
                                       const std::vector<int>& strides,
                                       const std::vector<int>& paddings) {
  std::vector<int> output_shape = input_shape;
  if (adaptive) {
    output_shape[0] = ksize[0];
    output_shape[1] = ksize[1];
  } else {
    int output_h, output_w;
    if (!ceil_mode) {
      output_h = (input_shape[0] - ksize[0] + 2 * paddings[0]) / strides[0] + 1;
      output_w = (input_shape[1] - ksize[1] + 2 * paddings[1]) / strides[1] + 1;
    } else {
      output_h =
          (input_shape[0] - ksize[0] + 2 * paddings[0] + strides[0] - 1) /
              strides[0] +
          1;
      output_w =
          (input_shape[1] - ksize[1] + 2 * paddings[1] + strides[1] - 1) /
              strides[1] +
          1;
    }
    output_shape[0] = output_h;
    output_shape[1] = output_w;
  }
  return output_shape;
}

class PoolPlugin : public PluginTensorRT {
 protected:
  size_t getSerializationSize() override {
    return SerializedSize(getPluginType()) + SerializedSize(ceil_mode_) +
           SerializedSize(pool_type_) + SerializedSize(adaptive_) +
           SerializedSize(ksize_) + SerializedSize(strides_) +
           SerializedSize(paddings_) + SerializedSize(input_shape_) +
           SerializedSize(output_shape_) + getBaseSerializationSize();
  }

  // TRT will call this func when we need to serialize the configuration of
  // tensorrt.
  void serialize(void* buffer) override {
    SerializeValue(&buffer, getPluginType());
    serializeBase(buffer);
    SerializeValue(&buffer, ceil_mode_);
    SerializeValue(&buffer, pool_type_);
    SerializeValue(&buffer, adaptive_);
    SerializeValue(&buffer, ksize_);
    SerializeValue(&buffer, strides_);
    SerializeValue(&buffer, paddings_);
    SerializeValue(&buffer, input_shape_);
    SerializeValue(&buffer, output_shape_);
  }

 public:
  enum class PoolType {
    max = 0,
    avg,
  };
  PoolPlugin() {}
  PoolPlugin(bool ceil_mode, PoolType pool_type, bool adaptive,
             std::vector<int> ksize, std::vector<int> strides,
             std::vector<int> paddings, std::vector<int> input_shape)
      : ceil_mode_(ceil_mode),
        pool_type_(pool_type),
        adaptive_(adaptive),
        ksize_(ksize),
        strides_(strides),
        paddings_(paddings),
        input_shape_(input_shape) {
    output_shape_ = input_shape_;
    std::vector<int> output_shape =
        CalcOutputSize({input_shape_[1], input_shape_[2]}, ceil_mode_,
                       adaptive_, ksize_, strides_, paddings_);
    output_shape_[1] = output_shape[0];
    output_shape_[2] = output_shape[1];
  }

  // It was used for tensorrt deserialization.
  // It should not be called by users.
  PoolPlugin(void const* serialData, size_t serialLength) {
    deserializeBase(serialData, serialLength);
    DeserializeValue(&serialData, &serialLength, &ceil_mode_);
    DeserializeValue(&serialData, &serialLength, &pool_type_);
    DeserializeValue(&serialData, &serialLength, &adaptive_);
    DeserializeValue(&serialData, &serialLength, &ksize_);
    DeserializeValue(&serialData, &serialLength, &strides_);
    DeserializeValue(&serialData, &serialLength, &paddings_);
    DeserializeValue(&serialData, &serialLength, &input_shape_);
    DeserializeValue(&serialData, &serialLength, &output_shape_);
  }

  PoolPlugin* clone() const override {
    return new PoolPlugin(ceil_mode_, pool_type_, adaptive_, ksize_, strides_,
                          paddings_, input_shape_);
  }

  const char* getPluginType() const override { return "pool_plugin"; }
  int getNbOutputs() const override { return 1; }
  nvinfer1::Dims getOutputDimensions(int index, const nvinfer1::Dims* inputs,
                                     int nbInputDims) override;
  int initialize() override { return 0; }
#if IS_TRT_VERSION_LT(8000)
  int enqueue(int batchSize, const void* const* inputs, void** outputs,
#else
  int enqueue(int batchSize, const void* const* inputs, void* const* outputs,
#endif
              void* workspace, cudaStream_t stream) override;

 private:
  bool ceil_mode_;
  PoolType pool_type_;
  bool adaptive_;
  std::vector<int> ksize_;
  std::vector<int> strides_;
  std::vector<int> paddings_;
  std::vector<int> input_shape_;
  std::vector<int> output_shape_;
};

#if IS_TRT_VERSION_GE(6000)
class PoolPluginDynamic : public DynamicPluginTensorRT {
 public:
  PoolPluginDynamic() {}
  PoolPluginDynamic(const bool& ceil_mode, const std::string& pool_type,
                    const bool& adaptive, const std::vector<int>& ksize,
                    const std::vector<int>& strides,
                    const std::vector<int>& paddings, const bool& is_global)
      : ceil_mode_(ceil_mode),
        pool_type_(pool_type),
        adaptive_(adaptive),
        ksize_(ksize),
        strides_(strides),
        paddings_(paddings),
        is_global_(is_global) {}

  PoolPluginDynamic(void const* serialData, size_t serialLength) {
    deserializeBase(serialData, serialLength);
    DeserializeValue(&serialData, &serialLength, &ceil_mode_);
    const char* pool_type;
    DeserializeValue(&serialData, &serialLength, &pool_type);
    pool_type_ = std::string(pool_type);
    DeserializeValue(&serialData, &serialLength, &adaptive_);
    DeserializeValue(&serialData, &serialLength, &ksize_);
    DeserializeValue(&serialData, &serialLength, &strides_);
    DeserializeValue(&serialData, &serialLength, &paddings_);
    DeserializeValue(&serialData, &serialLength, &is_global_);
  }
  ~PoolPluginDynamic() {}
  nvinfer1::IPluginV2DynamicExt* clone() const override {
    return new PoolPluginDynamic(ceil_mode_, pool_type_, adaptive_, ksize_,
                                 strides_, paddings_, is_global_);
  }

  const char* getPluginType() const override { return "pool_plugin"; }
  int getNbOutputs() const override { return 1; }
  int initialize() override { return 0; }

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
  bool ceil_mode_;
  std::string pool_type_;
  bool adaptive_;
  std::vector<int> ksize_;
  std::vector<int> strides_;
  std::vector<int> paddings_;
  bool is_global_;
};
#endif

}  // namespace plugin
}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
