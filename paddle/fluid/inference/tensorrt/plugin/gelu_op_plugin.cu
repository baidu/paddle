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

#include <cassert>
#include <cstring>
#include <vector>
#include "paddle/fluid/inference/tensorrt/plugin/gelu_op_plugin.h"
#include "paddle/fluid/inference/tensorrt/plugin/trt_plugin_factory.h"
#include "paddle/fluid/platform/float16.h"

namespace paddle {
namespace inference {
namespace tensorrt {
namespace plugin {

// constants for approximating the normal cdf
static const float kA = 1.41421356237309504;  // sqrt(2)

static const float kAT = 0.5;
static const float kBT = 0.7978845608028654;    // sqrt(2.0/M_PI)
static const float kCT = 0.035677408136300125;  // 0.044715 * sqrt(2.0/M_PI)

GeluPlugin* CreateGeluPluginDeserialize(const void* buffer, size_t length) {
  return new GeluPlugin(buffer, length);
}

REGISTER_TRT_PLUGIN("gelu_plugin", CreateGeluPluginDeserialize);

bool GeluPlugin::supportsFormat(nvinfer1::DataType type,
                                nvinfer1::PluginFormat format) const {
  if (with_fp16_) {
    return ((type == nvinfer1::DataType::kFLOAT ||
             type == nvinfer1::DataType::kHALF) &&
            (format == nvinfer1::PluginFormat::kNCHW));
  } else {
    return ((type == nvinfer1::DataType::kFLOAT) &&
            (format == nvinfer1::PluginFormat::kNCHW));
  }
}

nvinfer1::Dims GeluPlugin::getOutputDimensions(int index,
                                               const nvinfer1::Dims* in_dims,
                                               int nb_inputs) {
  assert(nb_inputs == 1);
  assert(index < this->getNbOutputs());
  nvinfer1::Dims const& input_dims = in_dims[0];
  nvinfer1::Dims output_dims = input_dims;
  return output_dims;
}

template <typename T, unsigned TPB>
__global__ void gelu_kernel(const T a, int n, const T* input, T* output) {
  const int idx = blockIdx.x * TPB + threadIdx.x;
  if (idx < n) {
    const T in = input[idx];
    const T cdf = 0.5f * (1.0f + erff(in * 0.5f * a));
    output[idx] = in * cdf;
  }
}

template <typename T>
__device__ T do_tanh(T a);

template <>
__device__ float do_tanh<float>(float a) {
  return tanf(a);
}

template <>
__device__ half do_tanh<half>(half a) {
  const float tmp = tanhf(__half2float(a));
  return __float2half(tmp);
}

// the kernel below is not aligned with fluid fp32 forwrad ones, use it for
// fp16.
template <typename T, unsigned TPB>
__global__ void no_exact_gelu_kernel(const T a, const T b, const T c, int n,
                                     const T* input, T* output) {
#if CUDA_ARCH_FP16_SUPPORTED(__CUDA_ARCH__)
  const int idx = blockIdx.x * TPB + threadIdx.x;
  if (idx < n) {
    const T in = input[idx];
    const T tmp = in * (c * in * in + b);
    const T cdf = a + a * do_tanh<T>(tmp);
    output[idx] = in * cdf;
  }
#endif
}

int GeluPlugin::enqueue(int batch_size, const void* const* inputs,
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
  const int block_size = 256;
  const int grid_size = (num + block_size - 1) / block_size;

  auto type = getDataType();
  if (type == nvinfer1::DataType::kFLOAT) {
    VLOG(1) << "TRT Plugin DataType selected. Gelu-->fp32";
    const float* input = static_cast<const float*>(inputs[0]);
    float* output = static_cast<float*>(outputs[0]);
    gelu_kernel<float, block_size><<<grid_size, block_size, 0, stream>>>(
        kA, num, input, output);
  } else if (type == nvinfer1::DataType::kHALF) {
    VLOG(1) << "TRT Plugin DataType selected. Gelu-->fp16";
    const half* input = static_cast<const half*>(inputs[0]);
    half* output = static_cast<half*>(outputs[0]);
    no_exact_gelu_kernel<half,
                         block_size><<<grid_size, block_size, 0, stream>>>(
        kAT, kBT, kCT, num, input, output);
  } else {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "The Gelu TRT Plugin's input type should be float or half."));
  }
  return cudaGetLastError() != cudaSuccess;
}

// Dynamic Plugin below.
#if IS_TRT_VERSION_GE(6000)

nvinfer1::DimsExprs GeluPluginDynamic::getOutputDimensions(
    int output_index, const nvinfer1::DimsExprs* inputs, int nb_inputs,
    nvinfer1::IExprBuilder& expr_builder) {
  return inputs[0];
}

bool GeluPluginDynamic::supportsFormatCombination(
    int pos, const nvinfer1::PluginTensorDesc* in_out, int nb_inputs,
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

  const nvinfer1::PluginTensorDesc& in = in_out[pos];
  if (pos == 0) {
    if (with_fp16_) {
      return (in.type == nvinfer1::DataType::kFLOAT ||
              in.type == nvinfer1::DataType::kHALF) &&
             (in.format == nvinfer1::TensorFormat::kLINEAR);
    } else {
      return (in.type == nvinfer1::DataType::kFLOAT) &&
             (in.format == nvinfer1::TensorFormat::kLINEAR);
    }
  }
  const nvinfer1::PluginTensorDesc& prev = in_out[pos - 1];
  // output
  return in.type == prev.type && in.format == prev.format;
}

nvinfer1::DataType GeluPluginDynamic::getOutputDataType(
    int index, const nvinfer1::DataType* input_types, int nb_inputs) const {
  PADDLE_ENFORCE_EQ(index, 0, platform::errors::InvalidArgument(
                                  "The Gelu Plugin only has one input, so the "
                                  "index value should be 0, but get %d.",
                                  index));
  return input_types[0];
}

int GeluPluginDynamic::enqueue(const nvinfer1::PluginTensorDesc* input_desc,
                               const nvinfer1::PluginTensorDesc* output_desc,
                               const void* const* inputs, void* const* outputs,
                               void* workspace, cudaStream_t stream) {
  auto input_dims = input_desc[0].dims;
  size_t num = ProductDim(input_dims);
  const int block_size = 256;
  const int grid_size = (num + block_size - 1) / block_size;

  auto input_type = input_desc[0].type;
  if (input_type == nvinfer1::DataType::kFLOAT) {
    VLOG(1) << "TRT Plugin DataType selected. Gelu-->fp32";
    const float* input = static_cast<const float*>(inputs[0]);
    float* output = static_cast<float*>(outputs[0]);
    gelu_kernel<float, block_size><<<grid_size, block_size, 0, stream>>>(
        kA, num, input, output);
  } else if (input_type == nvinfer1::DataType::kHALF) {
    VLOG(1) << "TRT Plugin DataType selected. Gelu-->fp16";
    const half* input = static_cast<const half*>(inputs[0]);
    half* output = static_cast<half*>(outputs[0]);
    no_exact_gelu_kernel<half,
                         block_size><<<grid_size, block_size, 0, stream>>>(
        kAT, kBT, kCT, num, input, output);
  } else {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "The Gelu TRT Plugin's input type should be float or half."));
  }
  return cudaGetLastError() != cudaSuccess;
}
#endif

}  // namespace plugin
}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
