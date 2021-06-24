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
#include "paddle/fluid/inference/tensorrt/engine.h"
#include "paddle/fluid/inference/tensorrt/plugin/trt_plugin.h"

namespace paddle {
namespace inference {
namespace tensorrt {
namespace plugin {

class HardSwishPlugin : public PluginTensorRT {
 public:
  HardSwishPlugin(const float threshold, const float scale, const float offset)
      : threshold_(threshold), scale_(scale), offset_(offset) {}

  // It was used for tensorrt deserialization.
  // It should not be called by users.
  HardSwishPlugin(void const* serialData, size_t serialLength) {
    deserializeBase(serialData, serialLength);
    DeserializeValue(&serialData, &serialLength, &threshold_);
    DeserializeValue(&serialData, &serialLength, &scale_);
    DeserializeValue(&serialData, &serialLength, &offset_);
  }

  ~HardSwishPlugin() {}
  HardSwishPlugin* clone() const override {
    return new HardSwishPlugin(threshold_, scale_, offset_);
  }

  const char* getPluginType() const override { return "hard_swish_plugin"; }
  int getNbOutputs() const override { return 1; }
  int initialize() override { return 0; }
  nvinfer1::Dims getOutputDimensions(int index, const nvinfer1::Dims* inputs,
                                     int nbInputDims) override;
#if IS_TRT_VERSION_LT(8000)
  int enqueue(int batchSize, const void* const* inputs, void** outputs,
#else
  int enqueue(int batchSize, const void* const* inputs, void* const* outputs,
#endif
              void* workspace, cudaStream_t stream) override;

 protected:
  float threshold_;
  float scale_;
  float offset_;

  size_t getSerializationSize() override {
    return getBaseSerializationSize() + SerializedSize(threshold_) +
           SerializedSize(scale_) + SerializedSize(offset_) +
           SerializedSize(getPluginType());
  }

  // TRT will call this func  to serialize the configuration of TRT
  // It should not be called by users.
  void serialize(void* buffer) override {
    SerializeValue(&buffer, getPluginType());
    serializeBase(buffer);
    SerializeValue(&buffer, threshold_);
    SerializeValue(&buffer, scale_);
    SerializeValue(&buffer, offset_);
  }
};

}  // namespace plugin
}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
