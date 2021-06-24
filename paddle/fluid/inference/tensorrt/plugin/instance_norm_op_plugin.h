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
#include "paddle/fluid/inference/tensorrt/engine.h"
#include "paddle/fluid/inference/tensorrt/plugin/trt_plugin.h"

namespace paddle {
namespace inference {
namespace tensorrt {
namespace plugin {

class InstanceNormPlugin : public PluginTensorRT {
 private:
  float eps_;
  std::vector<float> scale_;
  std::vector<float> bias_;

  framework::Tensor scale_t;
  framework::Tensor bias_t;
  cudnnHandle_t handle_;
  cudnnTensorDescriptor_t x_desc_, y_desc_, b_desc_;

 protected:
  size_t getSerializationSize() override {
    return getBaseSerializationSize() + SerializedSize(eps_) +
           SerializedSize(scale_) + SerializedSize(bias_) +
           SerializedSize(getPluginType());
  }

  // TRT will call this func when we need to serialize the configuration of
  // tensorrt.
  // It should not be called by users.
  void serialize(void *buffer) override {
    SerializeValue(&buffer, getPluginType());
    serializeBase(buffer);
    SerializeValue(&buffer, eps_);
    SerializeValue(&buffer, scale_);
    SerializeValue(&buffer, bias_);
  }

 public:
  explicit InstanceNormPlugin(const float eps, const std::vector<float> scale,
                              const std::vector<float> bias)
      : eps_(eps), scale_(scale), bias_(bias) {
    PADDLE_ENFORCE_EQ(scale.size(), bias.size(),
                      platform::errors::InvalidArgument(
                          "The instanceNorm's scale and bias should be the "
                          "same size. Got scale size = %d, but bias size = %d",
                          scale.size(), bias.size()));
    platform::dynload::cudnnCreate(&handle_);
    platform::dynload::cudnnCreateTensorDescriptor(&x_desc_);
    platform::dynload::cudnnCreateTensorDescriptor(&y_desc_);
    platform::dynload::cudnnCreateTensorDescriptor(&b_desc_);
  }

  // It was used for tensorrt deserialization.
  // It should not be called by users.
  InstanceNormPlugin(void const *serialData, size_t serialLength) {
    deserializeBase(serialData, serialLength);
    DeserializeValue(&serialData, &serialLength, &eps_);
    DeserializeValue(&serialData, &serialLength, &scale_);
    DeserializeValue(&serialData, &serialLength, &bias_);

    platform::dynload::cudnnCreate(&handle_);
    platform::dynload::cudnnCreateTensorDescriptor(&x_desc_);
    platform::dynload::cudnnCreateTensorDescriptor(&y_desc_);
    platform::dynload::cudnnCreateTensorDescriptor(&b_desc_);
  }

  ~InstanceNormPlugin() {
    platform::dynload::cudnnDestroy(handle_);
    platform::dynload::cudnnDestroyTensorDescriptor(x_desc_);
    platform::dynload::cudnnDestroyTensorDescriptor(y_desc_);
    platform::dynload::cudnnDestroyTensorDescriptor(b_desc_);
  }
  int initialize() override;

  InstanceNormPlugin *clone() const override {
    return new InstanceNormPlugin(eps_, scale_, bias_);
  }

  const char *getPluginType() const override { return "instance_norm_plugin"; }
  int getNbOutputs() const override { return 1; }
  nvinfer1::Dims getOutputDimensions(int index, const nvinfer1::Dims *inputs,
                                     int nbInputDims) override;
#if IS_TRT_VERSION_LT(8000)
  int enqueue(int batchSize, const void *const *inputs, void **outputs,
#else
  int enqueue(int batchSize, const void *const *inputs, void *const *outputs,
#endif
              void *workspace, cudaStream_t stream) override;

  bool supportsFormat(nvinfer1::DataType type,
                      nvinfer1::PluginFormat format) const override {
    return ((type == nvinfer1::DataType::kFLOAT ||
             type == nvinfer1::DataType::kHALF) &&
            (format == nvinfer1::PluginFormat::kNCHW));
  }
};

}  // namespace plugin
}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
