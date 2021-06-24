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
#include <cub/cub.cuh>  // NOLINT
#include <vector>
#include "glog/logging.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/fluid/inference/tensorrt/plugin/emb_eltwise_layernorm_plugin.h"
#include "paddle/fluid/inference/tensorrt/plugin/trt_plugin_factory.h"
#include "paddle/fluid/operators/math/bert_encoder_functor.h"

namespace paddle {
namespace inference {
namespace tensorrt {
namespace plugin {

// Dynamic shape plugin requires TRT version greater than 6.0.
#if IS_TRT_VERSION_GE(6000)

template <typename T>
EmbEltwiseLayernormPluginDynamicImpl<
    T>::~EmbEltwiseLayernormPluginDynamicImpl() {
  this->terminate();
}

inline half fp32tofp16(float x) { return static_cast<half>(x); }

template <typename T>
void EmbEltwiseLayernormPluginDynamicImpl<T>::shareGPUData(
    const EmbEltwiseLayernormPluginDynamicImplBase *anthor) {
  auto *ptr =
      dynamic_cast<const EmbEltwiseLayernormPluginDynamicImpl<T> *>(anthor);
  if (!ptr->is_initialized_) {
    return;
  }
  embs_gpu_ = ptr->embs_gpu_;
  scale_gpu_ = ptr->scale_gpu_;
  bias_gpu_ = ptr->bias_gpu_;
  int input_num = embs_.size();
  in_ptr_tensor_.Resize({input_num});
  emb_ptr_tensor_.ShareDataWith(ptr->emb_ptr_tensor_);
}

template <typename T>
int EmbEltwiseLayernormPluginDynamicImpl<T>::initialize() {
  if (is_initialized_) {
    return 0;
  }
  embs_gpu_.resize(embs_.size());
  for (int i = 0; i < embs_.size(); i++) {
    if (embs_[i]) {
      T *host_ptr;
      auto size = emb_sizes_[i];

      if (std::is_same<T, half>::value) {
        host_ptr = new T[size];
        std::transform(embs_[i], (embs_[i] + size), host_ptr, fp32tofp16);
      } else {
        host_ptr = reinterpret_cast<T *>(embs_[i]);
      }

      cudaMalloc(&embs_gpu_[i], sizeof(T) * size);
      cudaMemcpy(embs_gpu_[i], host_ptr, size * sizeof(T),
                 cudaMemcpyHostToDevice);
      if (std::is_same<T, half>::value) {
        delete[] host_ptr;
      }
    }
  }

  if (bias_) {
    cudaMalloc(&bias_gpu_, sizeof(float) * bias_size_);
    cudaMemcpy(bias_gpu_, bias_, bias_size_ * sizeof(float),
               cudaMemcpyHostToDevice);
  }
  if (scale_) {
    cudaMalloc(&scale_gpu_, sizeof(float) * scale_size_);
    cudaMemcpy(scale_gpu_, scale_, scale_size_ * sizeof(float),
               cudaMemcpyHostToDevice);
  }

  int input_num = embs_.size();
  in_ptr_tensor_.Resize({input_num});
  emb_ptr_tensor_.Resize({input_num});
  cudaGetDevice(&device_id_);
  auto emb_ptr_gpu_d =
      emb_ptr_tensor_.mutable_data<int64_t>(platform::CUDAPlace(device_id_));
  cudaMemcpy(emb_ptr_gpu_d, embs_gpu_.data(), sizeof(uintptr_t) * input_num,
             cudaMemcpyHostToDevice);
  is_initialized_ = true;
  return 0;
}

template <typename T>
void EmbEltwiseLayernormPluginDynamicImpl<T>::terminate() {
  for (int i = 0; i < embs_gpu_.size(); ++i) {
    if (embs_gpu_[i]) {
      cudaFree(embs_gpu_[i]);
      embs_gpu_[i] = nullptr;
    }
  }

  if (bias_gpu_) {
    cudaFree(bias_gpu_);
    bias_gpu_ = nullptr;
  }

  if (scale_gpu_) {
    cudaFree(scale_gpu_);
    scale_gpu_ = nullptr;
  }
}

template <typename T>
int EmbEltwiseLayernormPluginDynamicImpl<T>::enqueue(
    const nvinfer1::PluginTensorDesc *input_desc,
    const nvinfer1::PluginTensorDesc *output_desc, const void *const *inputs,
    void *const *outputs, void *workspace, cudaStream_t stream) {
  auto id_dims = input_desc[0].dims;
  int batch = id_dims.d[0];
  int seq_len = id_dims.d[1];
  int input_num = embs_.size();
  cudaGetDevice(&device_id_);
  auto in_ptr_gpu_d =
      in_ptr_tensor_.mutable_data<int64_t>(platform::CUDAPlace(device_id_));
  auto emb_ptr_gpu_d =
      emb_ptr_tensor_.mutable_data<int64_t>(platform::CUDAPlace(device_id_));

  auto new_input_ptr = reinterpret_cast<uintptr_t>(inputs[0]);

  if (old_input_ptr_ != new_input_ptr) {
    old_input_ptr_ = new_input_ptr;

    cudaMemcpyAsync(in_ptr_gpu_d, reinterpret_cast<const void *>(inputs),
                    sizeof(uintptr_t) * input_num, cudaMemcpyHostToDevice,
                    stream);
  }

  auto out_type = output_desc[0].type;

  if (std::is_same<T, float>::value) {
    PADDLE_ENFORCE_EQ(
        out_type == nvinfer1::DataType::kFLOAT, true,
        platform::errors::InvalidArgument(
            "The EmbEltwiseLayernorm Plugin only support fp32 input."));
  } else if (std::is_same<T, half>::value) {
    PADDLE_ENFORCE_EQ(
        out_type == nvinfer1::DataType::kHALF, true,
        platform::errors::InvalidArgument(
            "The EmbEltwiseLayernorm Plugin only support fp16 input."));
  } else {
    PADDLE_THROW(platform::errors::Fatal(
        "Unsupport data type, the out type of EmbEltwiseLayernorm should be "
        "float or half."));
  }

  auto *output_d = reinterpret_cast<T *>(outputs[0]);

  operators::math::EmbEltwiseLayerNormFunctor<T> emb_eltwise_layernorm_func;
  emb_eltwise_layernorm_func(batch, seq_len, hidden_size_, in_ptr_gpu_d,
                             scale_gpu_, bias_gpu_, emb_ptr_gpu_d, output_d,
                             eps_, input_num, stream);
  return cudaGetLastError() != cudaSuccess;
}

template class EmbEltwiseLayernormPluginDynamicImpl<float>;
#ifdef TRT_PLUGIN_FP16_AVALIABLE
template class EmbEltwiseLayernormPluginDynamicImpl<half>;
#endif

int EmbEltwiseLayernormPluginDynamic::initialize() {
  impl_->initialize();

  return 0;
}

void EmbEltwiseLayernormPluginDynamic::terminate() { impl_->terminate(); }

nvinfer1::DimsExprs EmbEltwiseLayernormPluginDynamic::getOutputDimensions(
    int output_index, const nvinfer1::DimsExprs *inputs, int nb_inputs,
    nvinfer1::IExprBuilder &expr_builder) {  // NOLINT
  PADDLE_ENFORCE_EQ(output_index, 0,
                    platform::errors::InvalidArgument(
                        "There is only one output of the EmbEltwiseLayernorm, "
                        "so the index should be zero,"
                        "but it's (%d)",
                        output_index));
  nvinfer1::DimsExprs ret;
  ret.nbDims = 3;
  ret.d[0] = inputs[0].d[0];
  ret.d[1] = inputs[0].d[1];
  ret.d[2] = expr_builder.constant(hidden_size_);
  return ret;
}

bool EmbEltwiseLayernormPluginDynamic::supportsFormatCombination(
    int pos, const nvinfer1::PluginTensorDesc *in_out, int nb_inputs,
    int nb_outputs) {
  PADDLE_ENFORCE_NOT_NULL(
      in_out, platform::errors::InvalidArgument(
                  "The input of swish plugin shoule not be nullptr."));
  PADDLE_ENFORCE_EQ(nb_outputs, 1,
                    platform::errors::InvalidArgument(
                        "The EmbEltwiseLayerNorm's output should be one"
                        "but it's (%d) outputs.",
                        nb_outputs));
  PADDLE_ENFORCE_EQ(nb_outputs, 1,
                    platform::errors::InvalidArgument(
                        "The EmbEltwiseLayerNorm's output should be one"
                        "but it's (%d) outputs.",
                        nb_outputs));
  PADDLE_ENFORCE_LT(
      pos, nb_inputs + nb_outputs,
      platform::errors::InvalidArgument("The pos(%d) should be less than the "
                                        "num(%d) of the input and the output.",
                                        pos, nb_inputs + nb_outputs));

  int all_nums = nb_inputs + nb_outputs;

  const nvinfer1::PluginTensorDesc &desc = in_out[pos];
  if (desc.format != nvinfer1::TensorFormat::kLINEAR) {
    return false;
  }

  if (pos == 0) {
    return desc.type == nvinfer1::DataType::kINT32;
  }

  const nvinfer1::PluginTensorDesc &prev = in_out[pos - 1];
  if (pos < all_nums - 1) {
    return desc.type == nvinfer1::DataType::kINT32 &&
           desc.dims.d[0] == prev.dims.d[0] && desc.dims.d[1] == prev.dims.d[1];
  }

  if (pos == all_nums - 1) {
    if (with_fp16_ == false) {
      return desc.type == nvinfer1::DataType::kFLOAT;
    } else {
      return desc.type == nvinfer1::DataType::kHALF;
    }
  }
  return false;
}

nvinfer1::DataType EmbEltwiseLayernormPluginDynamic::getOutputDataType(
    int index, const nvinfer1::DataType *input_types, int nb_inputs) const {
  PADDLE_ENFORCE_EQ(
      index, 0, platform::errors::InvalidArgument(
                    "The EmbEltwiseLayernorm Plugin only has one input, so the "
                    "index value should be 0, but get %d.",
                    index));
  if (with_fp16_)
    return nvinfer1::DataType::kHALF;
  else
    return nvinfer1::DataType::kFLOAT;
}

int EmbEltwiseLayernormPluginDynamic::enqueue(
    const nvinfer1::PluginTensorDesc *input_desc,
    const nvinfer1::PluginTensorDesc *output_desc, const void *const *inputs,
    void *const *outputs, void *workspace, cudaStream_t stream) {
  impl_->enqueue(input_desc, output_desc, inputs, outputs, workspace, stream);
  return cudaGetLastError() != cudaSuccess;
}

#endif

}  // namespace plugin
}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
