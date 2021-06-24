// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#include <cassert>
#include <cstring>
#include "paddle/fluid/inference/tensorrt/plugin/hard_swish_op_plugin.h"
#include "paddle/fluid/inference/tensorrt/plugin/trt_plugin_factory.h"

namespace paddle {
namespace inference {
namespace tensorrt {
namespace plugin {

HardSwishPlugin* CreateHardSwishPluginDeserialize(const void* buffer,
                                                  size_t length) {
  return new HardSwishPlugin(buffer, length);
}

REGISTER_TRT_PLUGIN("hard_swish_plugin", CreateHardSwishPluginDeserialize);

nvinfer1::Dims HardSwishPlugin::getOutputDimensions(
    int index, const nvinfer1::Dims* in_dims, int nb_inputs) {
  assert(nb_inputs == 1);
  assert(index < this->getNbOutputs());
  nvinfer1::Dims const& input_dims = in_dims[0];
  nvinfer1::Dims output_dims = input_dims;
  return output_dims;
}

template <typename T>
__device__ T kMax(T a, T b) {
  return a > b ? a : b;
}

template <typename T>
__device__ T kMin(T a, T b) {
  return a < b ? a : b;
}

template <typename T, unsigned TPB>
__global__ void hard_swish_kernel(float threshold, float scale, float offset,
                                  int n, const T* input, T* output) {
  const int idx = blockIdx.x * TPB + threadIdx.x;
  if (idx < n) {
    const T in = input[idx];
    output[idx] = in / scale * kMin<T>(kMax<T>(in + offset, 0), threshold);
  }
}

int HardSwishPlugin::enqueue(int batch_size, const void* const* inputs,
#if IS_TRT_VERSION_LT(8000)
                             void** outputs, void*, cudaStream_t stream) {
#else
                             void* const* outputs, void*, cudaStream_t stream) {
#endif
  const auto& input_dims = this->getInputDims(0);
  int num = batch_size;
  for (int i = 0; i < input_dims.nbDims; i++) {
    num *= input_dims.d[i];
  }
  float threshold = threshold_;
  float scale = scale_;
  float offset = offset_;

  const int block_size = 256;
  const int grid_size = (num + block_size - 1) / block_size;

  const float* input = static_cast<const float*>(inputs[0]);
  float* output = static_cast<float*>(outputs[0]);
  hard_swish_kernel<float, block_size><<<grid_size, block_size, 0, stream>>>(
      threshold, scale, offset, num, input, output);

  return cudaGetLastError() != cudaSuccess;
}

}  // namespace plugin
}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
