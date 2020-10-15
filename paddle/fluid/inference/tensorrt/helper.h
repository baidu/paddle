/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#pragma once

#include <NvInfer.h>
#include <cuda.h>
#include <glog/logging.h>
#include <string>
#include <utility>
#include <vector>
#include "paddle/fluid/platform/dynload/tensorrt.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace inference {
namespace tensorrt {

#define IS_TRT_VERSION_GE(version)                       \
  ((NV_TENSORRT_MAJOR * 1000 + NV_TENSORRT_MINOR * 100 + \
    NV_TENSORRT_PATCH * 10 + NV_TENSORRT_BUILD) >= version)

#define TRT_VERSION                                    \
  NV_TENSORRT_MAJOR * 1000 + NV_TENSORRT_MINOR * 100 + \
      NV_TENSORRT_PATCH * 10 + NV_TENSORRT_BUILD

namespace dy = paddle::platform::dynload;

// TensorRT data type to size
const int kDataTypeSize[] = {
    4,  // kFLOAT
    2,  // kHALF
    1,  // kINT8
    4   // kINT32
};

// The following two API are implemented in TensorRT's header file, cannot load
// from the dynamic library. So create our own implementation and directly
// trigger the method from the dynamic library.
static nvinfer1::IBuilder* createInferBuilder(nvinfer1::ILogger* logger) {
  return static_cast<nvinfer1::IBuilder*>(
      dy::createInferBuilder_INTERNAL(logger, NV_TENSORRT_VERSION));
}
static nvinfer1::IRuntime* createInferRuntime(nvinfer1::ILogger* logger) {
  return static_cast<nvinfer1::IRuntime*>(
      dy::createInferRuntime_INTERNAL(logger, NV_TENSORRT_VERSION));
}
#if IS_TRT_VERSION_GE(6000)
static nvinfer1::IPluginRegistry* GetPluginRegistry() {
  return static_cast<nvinfer1::IPluginRegistry*>(dy::getPluginRegistry());
}
#endif

// A logger for create TensorRT infer builder.
class NaiveLogger : public nvinfer1::ILogger {
 public:
  void log(nvinfer1::ILogger::Severity severity, const char* msg) override {
    switch (severity) {
      case Severity::kINFO:
        VLOG(3) << msg;
        break;
      case Severity::kWARNING:
        LOG(WARNING) << msg;
        break;
      case Severity::kINTERNAL_ERROR:
      case Severity::kERROR:
        LOG(ERROR) << msg;
        break;
      default:
        break;
    }
  }

  static nvinfer1::ILogger& Global() {
    static nvinfer1::ILogger* x = new NaiveLogger;
    return *x;
  }

  ~NaiveLogger() override {}
};

class NaiveProfiler : public nvinfer1::IProfiler {
 public:
  typedef std::pair<std::string, float> Record;
  std::vector<Record> mProfile;

  virtual void reportLayerTime(const char* layerName, float ms) {
    auto record =
        std::find_if(mProfile.begin(), mProfile.end(),
                     [&](const Record& r) { return r.first == layerName; });
    if (record == mProfile.end())
      mProfile.push_back(std::make_pair(layerName, ms));
    else
      record->second += ms;
  }

  void printLayerTimes() {
    float totalTime = 0;
    for (size_t i = 0; i < mProfile.size(); i++) {
      printf("%-40.40s %4.3fms\n", mProfile[i].first.c_str(),
             mProfile[i].second);
      totalTime += mProfile[i].second;
    }
    printf("Time over all layers: %4.3f\n", totalTime);
  }
};

inline size_t ProductDim(const nvinfer1::Dims& dims) {
  size_t v = 1;
  for (int i = 0; i < dims.nbDims; i++) {
    v *= dims.d[i];
  }
  return v;
}

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
