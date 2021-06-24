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

#include <cuda_fp16.h>

#include <algorithm>
#include <cstdint>
#include <functional>
#include <numeric>
#include <sstream>

#include "NvInferRuntimeCommon.h"
#include "paddle/fluid/inference/tensorrt/plugin/gather_nd_op_plugin.h"
#include "paddle/fluid/inference/tensorrt/plugin/trt_plugin_factory.h"
#include "paddle/fluid/platform/place.h"

namespace paddle {
namespace inference {
namespace tensorrt {
namespace plugin {

#if IS_TRT_VERSION_GE(6000)

template <typename T, typename IndexT = int>
__global__ void GatherNdCUDAKernel(const T* input, const int32_t* input_dims,
                                   const IndexT* indices, T* output,
                                   int32_t remain_size, int32_t slice_size,
                                   int32_t end_size) {
  CUDA_KERNEL_LOOP(i, remain_size * slice_size) {
    int indices_i = i / slice_size;
    int slice_i = i - indices_i * slice_size;  // offset inside the slice
    IndexT gather_i = 0;
    int32_t temp = slice_size;
    for (int32_t j = end_size - 1; j >= 0; --j) {
      auto index_value = indices[indices_i * end_size + j];
      PADDLE_ENFORCE(
          index_value >= 0 && index_value < input_dims[j],
          "The index is out of bounds, "
          "please check whether the dimensions of index and "
          "input meet the requirements. It should "
          "be less than [%d] and greater or equal to 0, but received [%d]",
          input_dims[j], index_value);
      gather_i += (index_value * temp);
      temp *= input_dims[j];
    }
    IndexT input_i = gather_i + slice_i;
    *(output + i) = *(input + input_i);
  }
}

int GatherNdPluginDynamic::initialize() { return 0; }

size_t GatherNdPluginDynamic::getSerializationSize() const {
  return SerializedSize(with_fp16_);
}

void GatherNdPluginDynamic::serialize(void* buffer) const {
  SerializeValue(&buffer, with_fp16_);
}

nvinfer1::DimsExprs GatherNdPluginDynamic::getOutputDimensions(
    int output_index, const nvinfer1::DimsExprs* inputs, int nb_inputs,
    nvinfer1::IExprBuilder& expr_builder) {
  PADDLE_ENFORCE_EQ(
      nb_inputs, 2,
      platform::errors::InvalidArgument(
          "The gather_nd plugin should have 2 input, but got %d.", nb_inputs));
  PADDLE_ENFORCE_EQ(output_index, 0,
                    platform::errors::InvalidArgument(
                        "When GetOutputDimensions in gather_nd "
                        "plugin, the output_index should be 0."));

  nvinfer1::DimsExprs x_dims = inputs[0];
  nvinfer1::DimsExprs index_dims = inputs[1];

  int32_t x_dims_size = x_dims.nbDims;
  int32_t index_dims_size = index_dims.nbDims;

  // TODO(wilber): The result dims shoule be Index.shape[:-1] +
  // X.shape[Index.shape[-1]:], but the trt DimsExprs is an expression we can't
  // get the actual value. So we only support one scenario: input_dims.size ==
  // index_dims.size.
  nvinfer1::DimsExprs ret(x_dims);
  for (int i = 0; i < index_dims_size - 1; ++i) {
    ret.d[i] = index_dims.d[i];
  }

  return ret;
}

bool GatherNdPluginDynamic::supportsFormatCombination(
    int pos, const nvinfer1::PluginTensorDesc* in_out, int nb_inputs,
    int nb_outputs) {
  PADDLE_ENFORCE_NOT_NULL(
      in_out, platform::errors::InvalidArgument(
                  "The input of gather_nd plugin should not be nullptr."));

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
  } else if (pos == 1) {
    return in.type == nvinfer1::DataType::kINT32 &&
           in.format == nvinfer1::TensorFormat::kLINEAR;
  } else if (pos == 2) {
    return in.type == in_out[0].type &&
           in.format == nvinfer1::TensorFormat::kLINEAR;
  }

  return true;
}

nvinfer1::DataType GatherNdPluginDynamic::getOutputDataType(
    int index, const nvinfer1::DataType* input_types, int nb_inputs) const {
  return input_types[0];
}

int GatherNdPluginDynamic::enqueue(
    const nvinfer1::PluginTensorDesc* input_desc,
    const nvinfer1::PluginTensorDesc* output_desc, const void* const* inputs,
    void* const* outputs, void* workspace, cudaStream_t stream) {
  auto input_dims = input_desc[0].dims;
  auto index_dims = input_desc[1].dims;
  auto input_dims_size = input_dims.nbDims;
  auto index_dims_size = index_dims.nbDims;

  std::vector<int32_t> input_shape, index_shape, out_shape;
  for (int i = 0; i < input_dims.nbDims; i++)
    input_shape.push_back(input_dims.d[i]);
  for (int i = 0; i < index_dims.nbDims; i++)
    index_shape.push_back(index_dims.d[i]);
  // The out_shape is
  //   Index.shape[:-1] + X.shape[Index.shape[-1]:]
  for (int i = 0; i < index_dims_size - 1; ++i) {
    out_shape.emplace_back(index_shape[i]);
  }
  for (int i = index_shape[index_dims_size - 1]; i < input_dims_size; ++i) {
    out_shape.emplace_back(input_shape[i]);
  }

  // final dim
  int end_size = index_shape[index_dims_size - 1];
  // remain dim
  std::vector<int> remain_ddim(index_shape.begin(), index_shape.end() - 1);
  int remain_numel = std::accumulate(remain_ddim.begin(), remain_ddim.end(), 1,
                                     std::multiplies<int>());
  // slice size
  int slice_size = 1;
  for (int i = end_size; i < input_dims_size; ++i) {
    slice_size *= input_shape[i];
  }

  auto input_type = input_desc[0].type;
  if (input_type == nvinfer1::DataType::kFLOAT) {
    VLOG(1) << "TRT Plugin DataType selected. gather_nd-->fp32";

    const float* p_input = static_cast<const float*>(inputs[0]);
    const int32_t* p_index = static_cast<const int32_t*>(inputs[1]);
    float* p_output = static_cast<float*>(outputs[0]);

    if (input_dims_data_ == nullptr) {
      cudaMalloc(&input_dims_data_, input_shape.size() * sizeof(int));
    }
    cudaMemcpyAsync(input_dims_data_, input_shape.data(),
                    sizeof(int) * input_shape.size(), cudaMemcpyHostToDevice,
                    stream);

    int block = 512;
    int n = slice_size * remain_numel;
    int grid = (n + block - 1) / block;

    GatherNdCUDAKernel<float, int32_t><<<grid, block, 0, stream>>>(
        p_input, input_dims_data_, p_index, p_output, remain_numel, slice_size,
        end_size);
  } else if (input_type == nvinfer1::DataType::kHALF) {
    VLOG(1) << "TRT Plugin DataType selected. gather_nd-->fp16";

    const half* p_input = static_cast<const half*>(inputs[0]);
    const int32_t* p_index = static_cast<const int32_t*>(inputs[1]);
    half* p_output = static_cast<half*>(outputs[0]);

    if (input_dims_data_ == nullptr) {
      cudaMalloc(&input_dims_data_, input_shape.size() * sizeof(int));
    }
    cudaMemcpyAsync(input_dims_data_, input_shape.data(),
                    sizeof(int) * input_shape.size(), cudaMemcpyHostToDevice,
                    stream);

    int block = 512;
    int n = slice_size * remain_numel;
    int grid = (n + block - 1) / block;

    GatherNdCUDAKernel<half, int32_t><<<grid, block, 0, stream>>>(
        p_input, input_dims_data_, p_index, p_output, remain_numel, slice_size,
        end_size);
  }

  return cudaGetLastError() != cudaSuccess;
}
#endif

}  // namespace plugin
}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
