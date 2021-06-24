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

#include <stdio.h>

#include <cassert>
#include <vector>

#include "glog/logging.h"
#include "paddle/fluid/inference/tensorrt/plugin/prelu_op_plugin.h"
#include "paddle/fluid/inference/tensorrt/plugin/trt_plugin_factory.h"
#include "paddle/fluid/operators/math/prelu.h"

namespace paddle {
namespace inference {
namespace tensorrt {
namespace plugin {

PReluPlugin *CreatePreluPluginDeserialize(const void *buffer, size_t length) {
  return new PReluPlugin(buffer, length);
}
REGISTER_TRT_PLUGIN("prelu_plugin", CreatePreluPluginDeserialize);

int PReluPlugin::initialize() {
  cudaMalloc(&p_gpu_weight_, sizeof(float) * weight_.size());
  cudaMemcpy(p_gpu_weight_, weight_.data(), weight_.size() * sizeof(float),
             cudaMemcpyHostToDevice);
  return 0;
}

void PReluPlugin::terminate() {
  if (p_gpu_weight_) {
    cudaFree(p_gpu_weight_);
    p_gpu_weight_ = nullptr;
  }
}

nvinfer1::Dims PReluPlugin::getOutputDimensions(int index,
                                                const nvinfer1::Dims *inputDims,
                                                int nbInputs) {
  assert(nbInputs == 1);
  assert(index < this->getNbOutputs());
  nvinfer1::Dims const &input_dims = inputDims[0];
  nvinfer1::Dims output_dims = input_dims;
  return output_dims;
}

int PReluPlugin::enqueue(int batch_size, const void *const *inputs,
#if IS_TRT_VERSION_LT(8000)
                         void **outputs, void *workspace, cudaStream_t stream) {
#else
                         void *const *outputs, void *workspace,
                         cudaStream_t stream) {
#endif
  // input dims is CHW.
  const auto &input_dims = this->getInputDims(0);
  const float *input = reinterpret_cast<const float *>(inputs[0]);
  // const float *alpha = reinterpret_cast<const float *>(alpha_.get().values);
  const float *alpha = p_gpu_weight_;
  float *output = reinterpret_cast<float **>(outputs)[0];
  int numel = 1;
  for (int i = 0; i < input_dims.nbDims; i++) {
    numel *= input_dims.d[i];
  }

  if (mode_ == "channel") {
    operators::math::PreluChannelWiseDirectCUDAFunctor<float>
        prelu_channel_wise;
    prelu_channel_wise(stream, input, alpha, output, input_dims.d[0],
                       input_dims.d[1], numel);
  } else if (mode_ == "element") {
    operators::math::PreluElementWiseDirectCUDAFunctor<float>
        prelu_element_wise;
    prelu_element_wise(stream, input, alpha, output, input_dims.d[0], numel);
  } else {
    operators::math::PreluScalarDirectCUDAFunctor<float> prelu_scalar;
    prelu_scalar(stream, input, alpha, output, numel);
  }
  return cudaGetLastError() != cudaSuccess;
}

#if IS_TRT_VERSION_GE(6000)

void PReluPluginDynamic::terminate() {
  if (p_gpu_weight_) {
    cudaFree(p_gpu_weight_);
  }
}

int PReluPluginDynamic::initialize() {
  cudaMalloc(&p_gpu_weight_, sizeof(float) * weight_.size());
  cudaMemcpy(p_gpu_weight_, weight_.data(), weight_.size() * sizeof(float),
             cudaMemcpyHostToDevice);
  return 0;
}
size_t PReluPluginDynamic::getSerializationSize() const { return 0; }

void PReluPluginDynamic::serialize(void *buffer) const {}

nvinfer1::DimsExprs PReluPluginDynamic::getOutputDimensions(
    int output_index, const nvinfer1::DimsExprs *inputs, int nb_inputs,
    nvinfer1::IExprBuilder &expr_builder) {
  return inputs[0];
}

bool PReluPluginDynamic::supportsFormatCombination(
    int pos, const nvinfer1::PluginTensorDesc *in_out, int nb_inputs,
    int nb_outputs) {
  PADDLE_ENFORCE_NOT_NULL(
      in_out, platform::errors::InvalidArgument(
                  "The input of swish plugin shoule not be nullptr."));

  PADDLE_ENFORCE_LT(
      pos, nb_inputs + nb_outputs,
      platform::errors::InvalidArgument("The pos(%d) should be less than the "
                                        "num(%d) of the input and the output.",
                                        pos, nb_inputs + nb_outputs));
  (in_out && pos < (nb_inputs + nb_outputs));

  return ((in_out[pos].type == nvinfer1::DataType::kFLOAT) &&
          in_out[pos].format == nvinfer1::PluginFormat::kNCHW);
}

nvinfer1::DataType PReluPluginDynamic::getOutputDataType(
    int index, const nvinfer1::DataType *input_types, int nb_inputs) const {
  PADDLE_ENFORCE_EQ(index, 0, platform::errors::InvalidArgument(
                                  "The PRelu Plugin only has one input, so the "
                                  "index value should be 0, but get %d.",
                                  index));
  PADDLE_ENFORCE_EQ((input_types[0] == nvinfer1::DataType::kFLOAT), true,
                    platform::errors::InvalidArgument(
                        "The input type should be half or float"));
  return input_types[0];
}

int PReluPluginDynamic::enqueue(const nvinfer1::PluginTensorDesc *input_desc,
                                const nvinfer1::PluginTensorDesc *output_desc,
                                const void *const *inputs, void *const *outputs,
                                void *workspace, cudaStream_t stream) {
  auto input_dims = input_desc[0].dims;
  const float *alpha = p_gpu_weight_;
  const float *input = static_cast<const float *>(inputs[0]);
  float *output = static_cast<float *>(outputs[0]);
  int numel = 1;
  for (int i = 0; i < input_dims.nbDims; i++) {
    numel *= input_dims.d[i];
  }

  if (mode_ == "channel") {
    operators::math::PreluChannelWiseDirectCUDAFunctor<float>
        prelu_channel_wise;
    prelu_channel_wise(stream, input, alpha, output, input_dims.d[0],
                       input_dims.d[1], numel);
  } else if (mode_ == "element") {
    operators::math::PreluElementWiseDirectCUDAFunctor<float>
        prelu_element_wise;
    prelu_element_wise(stream, input, alpha, output, input_dims.d[0], numel);
  } else {
    operators::math::PreluScalarDirectCUDAFunctor<float> prelu_scalar;
    prelu_scalar(stream, input, alpha, output, numel);
  }
  return cudaGetLastError() != cudaSuccess;
}
#endif

}  // namespace plugin
}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
