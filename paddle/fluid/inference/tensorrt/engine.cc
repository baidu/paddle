/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use
this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/inference/tensorrt/engine.h"

#include <NvInfer.h>
#include <glog/logging.h>
#include <string>

#include "cuda_runtime_api.h"  // NOLINT
#include "paddle/fluid/inference/tensorrt/helper.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/gpu_info.h"

namespace paddle {
namespace inference {
namespace tensorrt {

int TensorRTEngine::runtime_batch_ = 1;

void TensorRTEngine::InitNetwork() {
  freshDeviceId();
  infer_builder_.reset(createInferBuilder(&logger_));

  if (with_dynamic_shape_) {
    infer_network_.reset(infer_builder_->createNetworkV2(
        1U << static_cast<int>(
            nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH)));
  } else {
    infer_network_.reset(infer_builder_->createNetworkV2(0U));
  }

  infer_builder_config_.reset(infer_builder_->createBuilderConfig());
  optim_profile_ = infer_builder_->createOptimizationProfile();
}

void TensorRTEngine::Execute(int batch_size, std::vector<void *> *buffers,
                             cudaStream_t stream) {
  freshDeviceId();
  auto infer_context = context();
  if (!with_dynamic_shape()) {
    infer_context->enqueue(batch_size, buffers->data(), stream, nullptr);
  } else {
#if IS_TRT_VERSION_GE(6000)
    infer_context->enqueueV2(buffers->data(), stream, nullptr);
#endif
  }
  SetRuntimeBatch(batch_size);
}

void TensorRTEngine::FreezeNetwork() {
  freshDeviceId();
  VLOG(3) << "TRT to freeze network";
  PADDLE_ENFORCE_NOT_NULL(infer_builder_,
                          platform::errors::InvalidArgument(
                              "Inference builder of TRT is null. Please make "
                              "sure you call InitNetwork first."));
  PADDLE_ENFORCE_NOT_NULL(network(),
                          platform::errors::InvalidArgument(
                              "Call InitNetwork first to initialize network."));
  // build engine.
  infer_builder_->setMaxBatchSize(max_batch_);
  infer_builder_config_->setMaxWorkspaceSize(max_workspace_);

  bool enable_fp16 = (precision_ == AnalysisConfig::Precision::kHalf);
  if (enable_fp16) {
    bool support_fp16 = infer_builder_->platformHasFastFp16();
    infer_builder_config_->setFlag(nvinfer1::BuilderFlag::kFP16);
    if (!support_fp16) {
      LOG(INFO) << "You specify FP16 mode, but the hardware do not support "
                   "FP16 speed up, use FP32 instead.";
    } else {
      LOG(INFO) << "Run Paddle-TRT FP16 mode";
    }
  }

  bool enable_int8 = (precision_ == AnalysisConfig::Precision::kInt8);
  if (enable_int8) {
    infer_builder_config_->setFlag(nvinfer1::BuilderFlag::kFP16);
    infer_builder_config_->setFlag(nvinfer1::BuilderFlag::kINT8);
    infer_builder_config_->setFlag(nvinfer1::BuilderFlag::kSTRICT_TYPES);

    if (calibrator_) {
      infer_builder_config_->setInt8Calibrator(calibrator_);
    } else {
      infer_builder_config_->setInt8Calibrator(nullptr);

#if IS_TRT_VERSION_GE(5000)
      for (auto &quant_range : quant_dynamic_range_) {
        auto tensor = quant_range.first;
        float range = quant_range.second;
        tensor->setDynamicRange(-range, range);
      }

      std::unordered_set<nvinfer1::ITensor *> all_t;
      for (int i = 0; i < network()->getNbLayers(); i++) {
        auto layer = network()->getLayer(i);
        for (int j = 0; j < layer->getNbOutputs(); j++) {
          all_t.insert(layer->getOutput(j));
        }
      }

      for (int i = 0; i < network()->getNbInputs(); i++) {
        all_t.insert(network()->getInput(i));
      }

      for (auto &t : all_t) {
        if (!quant_dynamic_range_.count(t)) {
          VLOG(3) << "We are in trt int8 mode(not calibration), scale not set"
                  << " for tensor " << t->getName()
                  << ", this might be ok when trt does not need this range";
        }
      }

#if IS_TRT_VERSION_GE(5122)
      auto is_layer_int8 = [&](nvinfer1::ILayer *layer) -> bool {
        for (int j = 0; j < layer->getNbInputs(); j++) {
          auto *temp_in = layer->getInput(j);
          if (!temp_in->dynamicRangeIsSet()) {
            VLOG(1) << "Layer(Name: " << layer->getName()
                    << ") is set to float32 because its input("
                    << temp_in->getName() << ") doesn't have dynamic range.";
            return false;
          }
        }
        for (int j = 0; j < layer->getNbOutputs(); j++) {
          auto *temp_out = layer->getOutput(j);
          if (temp_out->isNetworkOutput()) {
            VLOG(1) << "Layer(Name: " << layer->getName()
                    << ") is set to float32 because its output("
                    << temp_out->getName() << ") is the output of the network.";
            return false;
          }
          if (!temp_out->dynamicRangeIsSet()) {
            VLOG(1) << "Layer(Name: " << layer->getName()
                    << ") is set to float32 because its output("
                    << temp_out->getName() << ") doesn't have dynamic range.";
            return false;
          }
        }
        return true;
      };
      // If a layer's output is the network's output, or not all of its inputs
      // and outputs have scales,
      // this layer's precision and output type are set to float32.
      // This step has no effect if this layer is fused during TRT optimization.
      for (int i = 0; i < network()->getNbLayers(); i++) {
        auto layer = network()->getLayer(i);
        if (!is_layer_int8(layer)) {
          layer->setPrecision(nvinfer1::DataType::kFLOAT);
        }
      }
#else
      LOG(WARNING) << "If your TensorRT version is lower than 5.1.2.2, you "
                      "must provide quantization scales for all tensors using "
                      "TRT to run.";
#endif
#endif
    }
  }

  if (use_dla_) {
    if (!enable_int8 && !enable_fp16) {
      LOG(WARNING) << "TensorRT DLA must be used with int8 or fp16, but you "
                      "set float32, so DLA is not used.";
    } else if (infer_builder_->getNbDLACores() == 0) {
      LOG(WARNING)
          << "TensorRT DLA is set by config, but your device does not have "
             "DLA, so DLA is not used.";
    } else {
      if (dla_core_ < 0 || dla_core_ >= infer_builder_->getNbDLACores()) {
        dla_core_ = 0;
        LOG(WARNING) << "Invalid DLACore, must be 0 < DLACore < "
                     << infer_builder_->getNbDLACores() << ", but got "
                     << dla_core_ << ", so use use 0 as default.";
      }
      infer_builder_config_->setDefaultDeviceType(nvinfer1::DeviceType::kDLA);
      infer_builder_config_->setDLACore(dla_core_);
      infer_builder_config_->setFlag(nvinfer1::BuilderFlag::kGPU_FALLBACK);
      LOG(INFO) << "TensorRT DLA enabled in FreezeNetwork(), DLACore "
                << dla_core_;
    }
  }

  if (with_dynamic_shape_) {
#if IS_TRT_VERSION_GE(6000)
    LOG(INFO) << "Run Paddle-TRT Dynamic Shape mode.";
    for (auto &input : min_input_shape_) {
      optim_profile_->setDimensions(
          input.first.c_str(), nvinfer1::OptProfileSelector::kMIN,
          Vec2TRT_Dims(input.second, input.first, true));
      optim_profile_->setDimensions(
          input.first.c_str(), nvinfer1::OptProfileSelector::kMAX,
          Vec2TRT_Dims(max_input_shape_[input.first], input.first, true));
      optim_profile_->setDimensions(
          input.first.c_str(), nvinfer1::OptProfileSelector::kOPT,
          Vec2TRT_Dims(optim_input_shape_[input.first], input.first, true));
    }
    infer_builder_config_->addOptimizationProfile(optim_profile_);
    if (WithFp16() && disable_trt_plugin_fp16()) {
      LOG(INFO) << "NOTE: In order to achieve higher accuracy, you have "
                   "disabled the fp16 mode of TRT Plugin,\n"
                << "you can reopen it with "
                   "'config.SetDynamicShapeInfo(min_shape, max_shape, "
                   "opt_shape, false /*disable_trt_plugin_fp16*/)'";
    }
#endif
  }
  infer_engine_.reset(infer_builder_->buildEngineWithConfig(
      *network(), *infer_builder_config_));

  PADDLE_ENFORCE_NOT_NULL(
      infer_engine_, platform::errors::Fatal(
                         "Build TensorRT cuda engine failed! Please recheck "
                         "you configurations related to paddle-TensorRT."));
}

nvinfer1::ITensor *TensorRTEngine::DeclareInput(const std::string &name,
                                                nvinfer1::DataType dtype,
                                                const nvinfer1::Dims &dims) {
  PADDLE_ENFORCE_EQ(network() != nullptr, true,
                    platform::errors::InvalidArgument(
                        "The TRT network should be initialized first."));
  auto *input = network()->addInput(name.c_str(), dtype, dims);
  PADDLE_ENFORCE_NOT_NULL(
      input, platform::errors::InvalidArgument("Adding input %s failed in "
                                               "TensorRT inference network. "
                                               "Please recheck your input.",
                                               name));
  PADDLE_ENFORCE_EQ(input->isNetworkInput(), true,
                    platform::errors::InvalidArgument(
                        "Input %s is not the input of TRT inference network. "
                        "Please recheck your input.",
                        name));
  TensorRTEngine::SetITensor(name, input);
  return input;
}

void TensorRTEngine::DeclareOutput(const nvinfer1::ILayer *layer, int offset,
                                   const std::string &name) {
  auto *output = layer->getOutput(offset);
  SetITensor(name, output);
  PADDLE_ENFORCE_NOT_NULL(
      output, platform::errors::InvalidArgument(
                  "The output %s of TRT engine should not be null.", name));
  output->setName(name.c_str());
  PADDLE_ENFORCE_EQ(output->isNetworkInput(), false,
                    platform::errors::InvalidArgument(
                        "The output %s of TRT engine should not be the input "
                        "of the network at the same time.",
                        name));
  network()->markOutput(*output);
  PADDLE_ENFORCE_EQ(
      output->isNetworkOutput(), true,
      platform::errors::InvalidArgument(
          "The output %s of TRT engine should be the output of the network.",
          name));
}

void TensorRTEngine::DeclareOutput(const std::string &name) {
  auto *output = TensorRTEngine::GetITensor(name);
  PADDLE_ENFORCE_NOT_NULL(
      output, platform::errors::InvalidArgument(
                  "The output %s of TRT engine should not be null.", name));
  output->setName(name.c_str());
  PADDLE_ENFORCE_EQ(output->isNetworkInput(), false,
                    platform::errors::InvalidArgument(
                        "The output %s of TRT engine should not be the input "
                        "of the network at the same time.",
                        name));
  network()->markOutput(*output);
}

void TensorRTEngine::SetITensor(const std::string &name,
                                nvinfer1::ITensor *tensor) {
  PADDLE_ENFORCE_NOT_NULL(
      tensor, platform::errors::InvalidArgument(
                  "Tensor named %s of TRT engine should not be null.", name));
  PADDLE_ENFORCE_EQ(
      0, itensor_map_.count(name),
      platform::errors::InvalidArgument(
          "Tensor named %s of TRT engine should not be duplicated", name));
  itensor_map_[name] = tensor;
}

nvinfer1::ITensor *TensorRTEngine::GetITensor(const std::string &name) {
  PADDLE_ENFORCE_EQ(itensor_map_.count(name), true,
                    platform::errors::NotFound(
                        "Tensor named %s is not found in TRT engine", name));
  return itensor_map_[name];
}

void TensorRTEngine::SetRuntimeBatch(size_t batch_size) {
  runtime_batch_ = batch_size;
}

float *TensorRTEngine::GetWeightCPUData(const std::string &name,
                                        framework::Tensor *weight_tensor,
                                        bool enable_int8,
                                        const std::vector<float> &scale) {
  static int name_suffix_counter = 0;
  std::string name_suffix = std::to_string(name_suffix_counter);
  std::string splitter = "__";
  std::string name_with_suffix = name + splitter + name_suffix;
  platform::CPUPlace cpu_place;
  PADDLE_ENFORCE_EQ(weight_map.count(name_with_suffix), 0,
                    platform::errors::AlreadyExists(
                        "The weight named %s is set into the weight map "
                        "twice in TRT OP converter.",
                        name_with_suffix));
  weight_map[name_with_suffix].reset(new framework::Tensor());
  weight_map[name_with_suffix]->Resize(weight_tensor->dims());
  TensorCopySync(*weight_tensor, cpu_place, weight_map[name_with_suffix].get());
  float *weight_data =
      weight_map[name_with_suffix]->mutable_data<float>(cpu_place);
  name_suffix_counter += 1;
  return weight_data;
}

int TensorRTEngine::GetRuntimeBatch() { return runtime_batch_; }

nvinfer1::IPluginLayer *TensorRTEngine::AddPlugin(
    nvinfer1::ITensor *const *inputs, int num_inputs,
    plugin::PluginTensorRT *plugin) {
  owned_plugin_.emplace_back(plugin);
  return network()->addPluginExt(inputs, num_inputs, *plugin);
}

nvinfer1::IPluginV2Layer *TensorRTEngine::AddPluginV2Ext(
    nvinfer1::ITensor *const *inputs, int num_inputs,
    plugin::PluginTensorRTV2Ext *plugin) {
  owned_plugin_v2ext_.emplace_back(plugin);
  return network()->addPluginV2(inputs, num_inputs, *plugin);
}

void TensorRTEngine::freshDeviceId() {
  int count;
  cudaGetDeviceCount(&count);
  PADDLE_ENFORCE_LT(device_id_, count,
                    platform::errors::OutOfRange(
                        "Device id %d exceeds the current device count: %d.",
                        device_id_, count));
  platform::SetDeviceId(device_id_);
}

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
