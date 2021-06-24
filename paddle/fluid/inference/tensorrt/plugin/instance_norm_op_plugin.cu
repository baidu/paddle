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
#include "paddle/fluid/inference/tensorrt/plugin/instance_norm_op_plugin.h"
#include "paddle/fluid/inference/tensorrt/plugin/trt_plugin_factory.h"
#include "paddle/fluid/platform/cudnn_helper.h"

namespace paddle {
namespace inference {
namespace tensorrt {
namespace plugin {

cudnnStatus_t convert_trt2cudnn_dtype(nvinfer1::DataType trt_dtype,
                                      cudnnDataType_t *cudnn_dtype) {
  switch (trt_dtype) {
    case nvinfer1::DataType::kFLOAT:
      *cudnn_dtype = CUDNN_DATA_FLOAT;
      break;
    case nvinfer1::DataType::kHALF:
      *cudnn_dtype = CUDNN_DATA_HALF;
      break;
    default:
      return CUDNN_STATUS_BAD_PARAM;
  }
  return CUDNN_STATUS_SUCCESS;
}

InstanceNormPlugin *CreateInstanceNormPluginDeserialize(const void *buffer,
                                                        size_t length) {
  return new InstanceNormPlugin(buffer, length);
}
REGISTER_TRT_PLUGIN("instance_norm_plugin",
                    CreateInstanceNormPluginDeserialize);

int InstanceNormPlugin::initialize() { return 0; }

nvinfer1::Dims InstanceNormPlugin::getOutputDimensions(
    int index, const nvinfer1::Dims *inputDims, int nbInputs) {
  assert(nbInputs == 1);
  assert(index < this->getNbOutputs());
  nvinfer1::Dims const &input_dims = inputDims[0];
  nvinfer1::Dims output_dims = input_dims;
  return output_dims;
}

int InstanceNormPlugin::enqueue(int batch_size, const void *const *inputs,
#if IS_TRT_VERSION_LT(8000)
                                void **outputs, void *workspace,
#else
                                void *const *outputs, void *workspace,
#endif
                                cudaStream_t stream) {
  const auto &input_dims = this->getInputDims(0);

  PADDLE_ENFORCE_EQ(input_dims.nbDims, 3,
                    platform::errors::InvalidArgument(
                        "Input Dims should be 3 (except the batch), got %d",
                        input_dims.nbDims));
  int n = batch_size;
  int c = input_dims.d[0];
  int h = input_dims.d[1];
  int w = input_dims.d[2];

  scale_t.Resize(framework::make_ddim({batch_size, c}));
  bias_t.Resize(framework::make_ddim({batch_size, c}));
  int device_id;
  cudaGetDevice(&device_id);
  float *scale_d = scale_t.mutable_data<float>(platform::CUDAPlace(device_id));
  float *bias_d = bias_t.mutable_data<float>(platform::CUDAPlace(device_id));

  for (int i = 0; i < batch_size; i++) {
    cudaMemcpyAsync(scale_d + i * c, scale_.data(), sizeof(float) * c,
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(bias_d + i * c, bias_.data(), sizeof(float) * c,
                    cudaMemcpyHostToDevice, stream);
  }
  platform::dynload::cudnnSetTensor4dDescriptor(
      b_desc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, n * c, 1, 1);

  cudnnDataType_t cudnn_dtype;
  nvinfer1::DataType data_type = getDataType();
  convert_trt2cudnn_dtype(data_type, &cudnn_dtype);
  platform::dynload::cudnnSetTensor4dDescriptor(x_desc_, CUDNN_TENSOR_NCHW,
                                                cudnn_dtype, 1, n * c, h, w);
  platform::dynload::cudnnSetTensor4dDescriptor(y_desc_, CUDNN_TENSOR_NCHW,
                                                cudnn_dtype, 1, n * c, h, w);
  float alpha = 1;
  float beta = 0;
  platform::dynload::cudnnSetStream(handle_, stream);

  void const *x_ptr = inputs[0];
  void *y_ptr = outputs[0];
  platform::dynload::cudnnBatchNormalizationForwardTraining(
      handle_, CUDNN_BATCHNORM_SPATIAL_PERSISTENT, &alpha, &beta, x_desc_,
      x_ptr, y_desc_, y_ptr, b_desc_, scale_d, bias_d, 1., nullptr, nullptr,
      eps_, nullptr, nullptr);
  return cudaGetLastError() != cudaSuccess;
}

}  // namespace plugin
}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
