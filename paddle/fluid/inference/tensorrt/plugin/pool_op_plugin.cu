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

#include "paddle/fluid/inference/tensorrt/plugin/pool_op_plugin.h"
#include "paddle/fluid/inference/tensorrt/plugin/trt_plugin_factory.h"
#include "paddle/fluid/operators/math/pooling.h"

namespace paddle {
namespace inference {
namespace tensorrt {
namespace plugin {

PoolPlugin *CreatePoolPluginDeserialize(const void *buffer, size_t length) {
  return new PoolPlugin(buffer, length);
}
REGISTER_TRT_PLUGIN("pool_plugin", CreatePoolPluginDeserialize);

nvinfer1::Dims PoolPlugin::getOutputDimensions(int index,
                                               const nvinfer1::Dims *inputDims,
                                               int nbInputs) {
  assert(nbInputs == 1);
  assert(index == 0);
  assert(inputDims[0].nbDims == 3);
  nvinfer1::Dims const &input_dims = inputDims[0];

  nvinfer1::Dims output_dims = input_dims;

  output_dims.d[1] = output_shape_[1];
  output_dims.d[2] = output_shape_[2];
  return output_dims;
}

int PoolPlugin::enqueue(int batchSize, const void *const *inputs,
#if IS_TRT_VERSION_LT(8000)
                        void **outputs, void *workspace, cudaStream_t stream) {
#else
                        void *const *outputs, void *workspace,
                        cudaStream_t stream) {
#endif
  auto const &input_dims = this->getInputDims(0);
  int input_size = 0;
  float const *idata = reinterpret_cast<float const *>(inputs[0]);
  float **odatas = reinterpret_cast<float **>(outputs);

  std::vector<int> input_shape = input_shape_;
  std::vector<int> output_shape = output_shape_;
  input_shape.insert(input_shape.begin(), batchSize);
  output_shape.insert(output_shape.begin(), batchSize);

  if (pool_type_ == PoolType::max) {
    paddle::operators::math::MaxPool<float> pool_process;
    paddle::operators::math::Pool2dDirectCUDAFunctor<
        paddle::operators::math::MaxPool<float>, float>
        pool2d_forward;
    pool2d_forward(idata, input_shape, output_shape, ksize_, strides_,
                   paddings_, true, adaptive_, odatas[0], stream, pool_process);
  } else if (pool_type_ == PoolType::avg) {
    paddle::operators::math::AvgPool<float> pool_process;
    paddle::operators::math::Pool2dDirectCUDAFunctor<
        paddle::operators::math::AvgPool<float>, float>
        pool2d_forward;
    pool2d_forward(idata, input_shape, output_shape, ksize_, strides_,
                   paddings_, true, adaptive_, odatas[0], stream, pool_process);
  }

  return cudaGetLastError() != cudaSuccess;
}

// Dynamic Plugin below.
#if IS_TRT_VERSION_GE(6000)

size_t PoolPluginDynamic::getSerializationSize() const { return 0; }

void PoolPluginDynamic::serialize(void *buffer) const {}

nvinfer1::DimsExprs PoolPluginDynamic::getOutputDimensions(
    int output_index, const nvinfer1::DimsExprs *inputs, int nb_inputs,
    nvinfer1::IExprBuilder &expr_builder) {
  PADDLE_ENFORCE_EQ(nb_inputs, 1,
                    platform::errors::InvalidArgument(
                        "The Split plugin should be only one input."));

  PADDLE_ENFORCE_EQ(
      inputs[0].d[1]->isConstant(), true,
      platform::errors::InvalidArgument("The channel dimension should be "
                                        "static, but we found it's dynamic."));
  nvinfer1::DimsExprs output(inputs[0]);
  if (is_global_) {
    output.d[2] = expr_builder.constant(1);
    output.d[3] = expr_builder.constant(1);
    return output;
  }
  if (adaptive_) {
    output.d[2] = expr_builder.constant(ksize_[0]);
    output.d[3] = expr_builder.constant(ksize_[1]);
    return output;
  }

  auto stri_0 = expr_builder.constant(strides_[0]);
  auto stri_1 = expr_builder.constant(strides_[1]);
  auto one_value = expr_builder.constant(1);

  auto v0_tmp = expr_builder.constant(-ksize_[0] + 2 * paddings_[0]);
  auto v1_tmp = expr_builder.constant(-ksize_[1] + 2 * paddings_[1]);

  auto ceil_tmp =
      expr_builder.constant(-ksize_[0] + 2 * paddings_[0] + strides_[0] - 1);
  auto ceil1_tmp =
      expr_builder.constant(-ksize_[1] + 2 * paddings_[1] + strides_[1] - 1);

  if (!ceil_mode_) {
    output.d[2] = expr_builder.operation(
        nvinfer1::DimensionOperation::kSUM,
        *expr_builder.operation(
            nvinfer1::DimensionOperation::kFLOOR_DIV,
            *expr_builder.operation(nvinfer1::DimensionOperation::kSUM,
                                    *inputs[0].d[2], *v0_tmp),
            *stri_0),
        *one_value);
    output.d[3] = expr_builder.operation(
        nvinfer1::DimensionOperation::kSUM,
        *expr_builder.operation(
            nvinfer1::DimensionOperation::kFLOOR_DIV,
            *expr_builder.operation(nvinfer1::DimensionOperation::kSUM,
                                    *inputs[0].d[3], *v1_tmp),
            *stri_1),
        *one_value);

  } else {
    output.d[2] = expr_builder.operation(
        nvinfer1::DimensionOperation::kSUM,
        *expr_builder.operation(
            nvinfer1::DimensionOperation::kFLOOR_DIV,
            *expr_builder.operation(nvinfer1::DimensionOperation::kSUM,
                                    *inputs[0].d[2], *ceil_tmp),
            *stri_0),
        *one_value);
    output.d[3] = expr_builder.operation(
        nvinfer1::DimensionOperation::kSUM,
        *expr_builder.operation(
            nvinfer1::DimensionOperation::kFLOOR_DIV,
            *expr_builder.operation(nvinfer1::DimensionOperation::kSUM,
                                    *inputs[0].d[3], *ceil1_tmp),
            *stri_1),
        *one_value);
  }

  return output;
}

bool PoolPluginDynamic::supportsFormatCombination(
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

nvinfer1::DataType PoolPluginDynamic::getOutputDataType(
    int index, const nvinfer1::DataType *input_types, int nb_inputs) const {
  PADDLE_ENFORCE_EQ(index, 0, platform::errors::InvalidArgument(
                                  "The Pool Plugin only has one input, so the "
                                  "index value should be 0, but get %d.",
                                  index));
  PADDLE_ENFORCE_EQ((input_types[0] == nvinfer1::DataType::kFLOAT), true,
                    platform::errors::InvalidArgument(
                        "The input type should be half or float"));
  return input_types[0];
}

int PoolPluginDynamic::enqueue(const nvinfer1::PluginTensorDesc *input_desc,
                               const nvinfer1::PluginTensorDesc *output_desc,
                               const void *const *inputs, void *const *outputs,
                               void *workspace, cudaStream_t stream) {
  auto input_dims = input_desc[0].dims;
  int n = input_dims.d[0];
  int c = input_dims.d[1];
  int h = input_dims.d[2];
  int w = input_dims.d[3];

  const float *input = static_cast<const float *>(inputs[0]);
  float *output = static_cast<float *>(outputs[0]);

  std::vector<int> input_shape, output_shape;
  for (int i = 0; i < input_dims.nbDims; i++)
    input_shape.push_back(input_dims.d[i]);
  output_shape = input_shape;

  std::vector<int> ksize = ksize_;
  std::vector<int> paddings = paddings_;
  if (is_global_) {
    ksize[0] = h;
    ksize[1] = w;
    paddings[0] = 0;
    paddings[1] = 0;
    output_shape[2] = 1;
    output_shape[3] = 1;
  } else {
    auto data_dim = CalcOutputSize({h, w}, ceil_mode_, adaptive_, ksize_,
                                   strides_, paddings_);
    output_shape[2] = data_dim[0];
    output_shape[3] = data_dim[1];
  }

  if (pool_type_ == "max") {
    paddle::operators::math::MaxPool<float> pool_process;
    paddle::operators::math::Pool2dDirectCUDAFunctor<
        paddle::operators::math::MaxPool<float>, float>
        pool2d_forward;
    pool2d_forward(input, input_shape, output_shape, ksize, strides_, paddings,
                   true, adaptive_, output, stream, pool_process);
  } else if (pool_type_ == "avg") {
    paddle::operators::math::AvgPool<float> pool_process;
    paddle::operators::math::Pool2dDirectCUDAFunctor<
        paddle::operators::math::AvgPool<float>, float>
        pool2d_forward;
    pool2d_forward(input, input_shape, output_shape, ksize, strides_, paddings,
                   true, adaptive_, output, stream, pool_process);
  }

  return cudaGetLastError() != cudaSuccess;
}
#endif

}  // namespace plugin
}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
