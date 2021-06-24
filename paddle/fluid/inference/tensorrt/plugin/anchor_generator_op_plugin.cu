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

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cassert>

#include "paddle/fluid/inference/tensorrt/plugin/anchor_generator_op_plugin.h"
#include "paddle/fluid/inference/tensorrt/plugin/trt_plugin_factory.h"

#include "paddle/fluid/operators/detection/anchor_generator_op.h"

namespace paddle {
namespace inference {
namespace tensorrt {
namespace plugin {

#define PrepareParamsOnDevice()                                          \
  constexpr int data_size = 4;                                           \
  cudaMalloc(&anchor_sizes_device_, anchor_sizes_.size() * data_size);   \
  cudaMalloc(&aspect_ratios_device_, aspect_ratios_.size() * data_size); \
  cudaMalloc(&stride_device_, stride_.size() * data_size);               \
  cudaMalloc(&variances_device_, variances_.size() * data_size);         \
  cudaMemcpy(anchor_sizes_device_, anchor_sizes_.data(),                 \
             anchor_sizes_.size() * data_size, cudaMemcpyHostToDevice);  \
  cudaMemcpy(aspect_ratios_device_, aspect_ratios_.data(),               \
             aspect_ratios_.size() * data_size, cudaMemcpyHostToDevice); \
  cudaMemcpy(stride_device_, stride_.data(), stride_.size() * data_size, \
             cudaMemcpyHostToDevice);                                    \
  cudaMemcpy(variances_device_, variances_.data(),                       \
             variances_.size() * data_size, cudaMemcpyHostToDevice);

AnchorGeneratorPlugin::AnchorGeneratorPlugin(
    const nvinfer1::DataType data_type, const std::vector<float>& anchor_sizes,
    const std::vector<float>& aspect_ratios, const std::vector<float>& stride,
    const std::vector<float>& variances, const float offset, const int height,
    const int width, const int num_anchors, const int box_num)
    : data_type_(data_type),
      anchor_sizes_(anchor_sizes),
      aspect_ratios_(aspect_ratios),
      stride_(stride),
      variances_(variances),
      offset_(offset),
      height_(height),
      width_(width),
      num_anchors_(num_anchors),
      box_num_(box_num) {
  // anchors must be float32, which is the generator proposals' input
  PADDLE_ENFORCE_EQ(data_type_, nvinfer1::DataType::kFLOAT,
                    platform::errors::InvalidArgument(
                        "TRT anchor generator plugin only accepts float32."));
  PADDLE_ENFORCE_GE(height_, 0,
                    platform::errors::InvalidArgument(
                        "TRT anchor generator plugin only accepts height "
                        "greater than 0, but receive height = %d.",
                        height_));
  PADDLE_ENFORCE_GE(width_, 0,
                    platform::errors::InvalidArgument(
                        "TRT anchor generator plugin only accepts width "
                        "greater than 0, but receive width = %d.",
                        width_));
  PADDLE_ENFORCE_GE(
      num_anchors_, 0,
      platform::errors::InvalidArgument(
          "TRT anchor generator plugin only accepts number of anchors greater "
          "than 0, but receive number of anchors = %d.",
          num_anchors_));
  PADDLE_ENFORCE_GE(box_num_, 0,
                    platform::errors::InvalidArgument(
                        "TRT anchor generator plugin only accepts box_num "
                        "greater than 0, but receive box_num = %d.",
                        box_num_));
  PrepareParamsOnDevice();
}

AnchorGeneratorPlugin::~AnchorGeneratorPlugin() {
  auto release_device_ptr = [](void* ptr) {
    if (ptr) {
      cudaFree(ptr);
      ptr = nullptr;
    }
  };
  release_device_ptr(anchor_sizes_device_);
  release_device_ptr(aspect_ratios_device_);
  release_device_ptr(stride_device_);
  release_device_ptr(variances_device_);
}

AnchorGeneratorPlugin::AnchorGeneratorPlugin(const void* data, size_t length) {
  DeserializeValue(&data, &length, &data_type_);
  DeserializeValue(&data, &length, &anchor_sizes_);
  DeserializeValue(&data, &length, &aspect_ratios_);
  DeserializeValue(&data, &length, &stride_);
  DeserializeValue(&data, &length, &variances_);
  DeserializeValue(&data, &length, &offset_);
  DeserializeValue(&data, &length, &height_);
  DeserializeValue(&data, &length, &width_);
  DeserializeValue(&data, &length, &num_anchors_);
  DeserializeValue(&data, &length, &box_num_);
  PrepareParamsOnDevice();
}

const char* AnchorGeneratorPlugin::getPluginType() const {
  return "anchor_generator_plugin";
}

const char* AnchorGeneratorPlugin::getPluginVersion() const { return "1"; }

int AnchorGeneratorPlugin::getNbOutputs() const { return 2; }

nvinfer1::Dims AnchorGeneratorPlugin::getOutputDimensions(
    int index, const nvinfer1::Dims* inputs, int nb_input_dims) {
  nvinfer1::Dims dims{};
  dims.nbDims = 4;
  dims.d[0] = height_;
  dims.d[1] = width_;
  dims.d[2] = num_anchors_;
  dims.d[3] = 4;
  return dims;
}

bool AnchorGeneratorPlugin::supportsFormat(
    nvinfer1::DataType type, nvinfer1::TensorFormat format) const {
  // static shape plugin can't support different type between input/out
  // it may cause addition overhead in half mode
  return (type == data_type_ && format == nvinfer1::TensorFormat::kLINEAR);
}

size_t AnchorGeneratorPlugin::getWorkspaceSize(int max_batch_size) const {
  return 0;
}

template <typename T>
int AnchorGeneratorPlugin::enqueue_impl(int batch_size,
                                        const void* const* inputs,
                                        void** outputs, void* workspace,
                                        cudaStream_t stream) {
  const int block = 512;
  const int gen_anchor_grid = (box_num_ + block - 1) / block;
  T* anchors = static_cast<T*>(outputs[0]);
  T* vars = static_cast<T*>(outputs[1]);
  const T* anchor_sizes_device = static_cast<const T*>(anchor_sizes_device_);
  const T* aspect_ratios_device = static_cast<const T*>(aspect_ratios_device_);
  const T* stride_device = static_cast<const T*>(stride_device_);
  const T* variances_device = static_cast<const T*>(variances_device_);
  paddle::operators::GenAnchors<T><<<gen_anchor_grid, block, 0, stream>>>(
      anchors, aspect_ratios_device, aspect_ratios_.size(), anchor_sizes_device,
      anchor_sizes_.size(), stride_device, stride_.size(), height_, width_,
      offset_);
  const int var_grid = (box_num_ * 4 + block - 1) / block;
  paddle::operators::SetVariance<T><<<var_grid, block, 0, stream>>>(
      vars, variances_device, variances_.size(), box_num_ * 4);
  return cudaGetLastError() != cudaSuccess;
}

int AnchorGeneratorPlugin::enqueue(int batch_size, const void* const* inputs,
#if IS_TRT_VERSION_LT(8000)
                                   void** outputs, void* workspace,
#else
                                   void* const* outputs, void* workspace,
#endif
                                   cudaStream_t stream) {
  return enqueue_impl<float>(batch_size, inputs, outputs, workspace, stream);
}

int AnchorGeneratorPlugin::initialize() { return 0; }

void AnchorGeneratorPlugin::terminate() {}

size_t AnchorGeneratorPlugin::getSerializationSize() const {
  size_t serialize_size = 0;
  serialize_size += SerializedSize(data_type_);
  serialize_size += SerializedSize(anchor_sizes_);
  serialize_size += SerializedSize(aspect_ratios_);
  serialize_size += SerializedSize(stride_);
  serialize_size += SerializedSize(variances_);
  serialize_size += SerializedSize(offset_);
  serialize_size += SerializedSize(height_);
  serialize_size += SerializedSize(width_);
  serialize_size += SerializedSize(num_anchors_);
  serialize_size += SerializedSize(box_num_);
  return serialize_size;
}

void AnchorGeneratorPlugin::serialize(void* buffer) const {
  SerializeValue(&buffer, data_type_);
  SerializeValue(&buffer, anchor_sizes_);
  SerializeValue(&buffer, aspect_ratios_);
  SerializeValue(&buffer, stride_);
  SerializeValue(&buffer, variances_);
  SerializeValue(&buffer, offset_);
  SerializeValue(&buffer, height_);
  SerializeValue(&buffer, width_);
  SerializeValue(&buffer, num_anchors_);
  SerializeValue(&buffer, box_num_);
}

void AnchorGeneratorPlugin::destroy() {}

void AnchorGeneratorPlugin::setPluginNamespace(const char* lib_namespace) {
  namespace_ = std::string(lib_namespace);
}

const char* AnchorGeneratorPlugin::getPluginNamespace() const {
  return namespace_.c_str();
}

nvinfer1::DataType AnchorGeneratorPlugin::getOutputDataType(
    int index, const nvinfer1::DataType* input_type, int nb_inputs) const {
  return data_type_;
}

bool AnchorGeneratorPlugin::isOutputBroadcastAcrossBatch(
    int output_index, const bool* input_is_broadcast, int nb_inputs) const {
  return true;
}

bool AnchorGeneratorPlugin::canBroadcastInputAcrossBatch(
    int input_index) const {
  return false;
}

void AnchorGeneratorPlugin::configurePlugin(
    const nvinfer1::Dims* input_dims, int nb_inputs,
    const nvinfer1::Dims* output_dims, int nb_outputs,
    const nvinfer1::DataType* input_types,
    const nvinfer1::DataType* output_types, const bool* input_is_broadcast,
    const bool* output_is_broadcast, nvinfer1::PluginFormat float_format,
    int max_batct_size) {}

nvinfer1::IPluginV2Ext* AnchorGeneratorPlugin::clone() const {
  auto plugin = new AnchorGeneratorPlugin(
      data_type_, anchor_sizes_, aspect_ratios_, stride_, variances_, offset_,
      height_, width_, num_anchors_, box_num_);
  plugin->setPluginNamespace(namespace_.c_str());
  return plugin;
}

void AnchorGeneratorPluginCreator::setPluginNamespace(
    const char* lib_namespace) {
  namespace_ = std::string(lib_namespace);
}

const char* AnchorGeneratorPluginCreator::getPluginNamespace() const {
  return namespace_.c_str();
}

const char* AnchorGeneratorPluginCreator::getPluginName() const {
  return "anchor_generator_plugin";
}

const char* AnchorGeneratorPluginCreator::getPluginVersion() const {
  return "1";
}

const nvinfer1::PluginFieldCollection*
AnchorGeneratorPluginCreator::getFieldNames() {
  return &field_collection_;
}

nvinfer1::IPluginV2Ext* AnchorGeneratorPluginCreator::createPlugin(
    const char* name, const nvinfer1::PluginFieldCollection* fc) {
  const nvinfer1::PluginField* fields = fc->fields;
  int type_id = -1;
  std::vector<float> anchor_sizes, aspect_ratios, stride, variances;
  float offset = .5;
  int height = -1, width = -1;
  int num_anchors = -1;
  int box_num = -1;

  for (int i = 0; i < fc->nbFields; ++i) {
    const std::string field_name(fc->fields[i].name);
    const auto length = fc->fields[i].length;
    if (field_name.compare("type_id") == 0) {
      type_id = *static_cast<const int*>(fc->fields[i].data);
    } else if (field_name.compare("anchor_sizes")) {
      const auto* data = static_cast<const float*>(fc->fields[i].data);
      anchor_sizes.insert(anchor_sizes.end(), data, data + length);
    } else if (field_name.compare("aspect_ratios")) {
      const auto* data = static_cast<const float*>(fc->fields[i].data);
      aspect_ratios.insert(aspect_ratios.end(), data, data + length);
    } else if (field_name.compare("stride")) {
      const auto* data = static_cast<const float*>(fc->fields[i].data);
      stride.insert(stride.end(), data, data + length);
    } else if (field_name.compare("variances")) {
      const auto* data = static_cast<const float*>(fc->fields[i].data);
      variances.insert(variances.end(), data, data + length);
    } else if (field_name.compare("offset")) {
      offset = *static_cast<const float*>(fc->fields[i].data);
    } else if (field_name.compare("height")) {
      height = *static_cast<const int*>(fc->fields[i].data);
    } else if (field_name.compare("width")) {
      width = *static_cast<const int*>(fc->fields[i].data);
    } else if (field_name.compare("num_anchors")) {
      num_anchors = *static_cast<const int*>(fc->fields[i].data);
    } else if (field_name.compare("box_num")) {
      box_num = *static_cast<const int*>(fc->fields[i].data);
    } else {
      assert(false && "unknown plugin field name.");
    }
  }
  return new AnchorGeneratorPlugin(nvinfer1::DataType::kFLOAT, anchor_sizes,
                                   aspect_ratios, stride, variances, offset,
                                   height, width, num_anchors, box_num);
}

nvinfer1::IPluginV2Ext* AnchorGeneratorPluginCreator::deserializePlugin(
    const char* name, const void* serial_data, size_t serial_length) {
  auto plugin = new AnchorGeneratorPlugin(serial_data, serial_length);
  plugin->setPluginNamespace(namespace_.c_str());
  return plugin;
}

#if IS_TRT_VERSION_GE(6000)
AnchorGeneratorPluginDynamic::AnchorGeneratorPluginDynamic(
    const nvinfer1::DataType data_type, const std::vector<float>& anchor_sizes,
    const std::vector<float>& aspect_ratios, const std::vector<float>& stride,
    const std::vector<float>& variances, const float offset,
    const int num_anchors)
    : data_type_(data_type),
      anchor_sizes_(anchor_sizes),
      aspect_ratios_(aspect_ratios),
      stride_(stride),
      variances_(variances),
      offset_(offset),
      num_anchors_(num_anchors) {
  // data_type_ is used to determine the output data type
  // data_type_ can only be float32
  // height, width, num_anchors are calculated at configurePlugin
  PADDLE_ENFORCE_EQ(data_type_, nvinfer1::DataType::kFLOAT,
                    platform::errors::InvalidArgument(
                        "TRT anchor generator plugin only accepts float32."));
  PADDLE_ENFORCE_GE(
      num_anchors_, 0,
      platform::errors::InvalidArgument(
          "TRT anchor generator plugin only accepts number of anchors greater "
          "than 0, but receive number of anchors = %d.",
          num_anchors_));
  PrepareParamsOnDevice();
}

AnchorGeneratorPluginDynamic::~AnchorGeneratorPluginDynamic() {
  auto release_device_ptr = [](void* ptr) {
    if (ptr) {
      cudaFree(ptr);
      ptr = nullptr;
    }
  };
  release_device_ptr(anchor_sizes_device_);
  release_device_ptr(aspect_ratios_device_);
  release_device_ptr(stride_device_);
  release_device_ptr(variances_device_);
}

AnchorGeneratorPluginDynamic::AnchorGeneratorPluginDynamic(void const* data,
                                                           size_t length) {
  DeserializeValue(&data, &length, &data_type_);
  DeserializeValue(&data, &length, &anchor_sizes_);
  DeserializeValue(&data, &length, &aspect_ratios_);
  DeserializeValue(&data, &length, &stride_);
  DeserializeValue(&data, &length, &variances_);
  DeserializeValue(&data, &length, &offset_);
  DeserializeValue(&data, &length, &num_anchors_);
  PrepareParamsOnDevice();
}

nvinfer1::IPluginV2DynamicExt* AnchorGeneratorPluginDynamic::clone() const {
  auto plugin = new AnchorGeneratorPluginDynamic(
      data_type_, anchor_sizes_, aspect_ratios_, stride_, variances_, offset_,
      num_anchors_);
  plugin->setPluginNamespace(namespace_.c_str());
  return plugin;
}

nvinfer1::DimsExprs AnchorGeneratorPluginDynamic::getOutputDimensions(
    int outputIndex, const nvinfer1::DimsExprs* inputs, int nbInputs,
    nvinfer1::IExprBuilder& exprBuilder) {
  nvinfer1::DimsExprs ret{};
  ret.nbDims = 4;
  ret.d[0] = inputs[0].d[2];  // feature height
  ret.d[1] = inputs[0].d[3];  // feature width
  ret.d[2] = exprBuilder.constant(num_anchors_);
  ret.d[3] = exprBuilder.constant(4);
  return ret;
}

bool AnchorGeneratorPluginDynamic::supportsFormatCombination(
    int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs,
    int nbOutputs) {
  // input can be any, doesn't matter
  // anchor generator doesn't read input raw data, only need the shape info
  auto type = inOut[pos].type;
  auto format = inOut[pos].format;
#if IS_TRT_VERSION_GE(7234)
  if (pos == 0) return true;
#else
  if (pos == 0) return format == nvinfer1::TensorFormat::kLINEAR;
#endif
  return (type == nvinfer1::DataType::kFLOAT &&
          format == nvinfer1::TensorFormat::kLINEAR);
}

void AnchorGeneratorPluginDynamic::configurePlugin(
    const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
    const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) {}

size_t AnchorGeneratorPluginDynamic::getWorkspaceSize(
    const nvinfer1::PluginTensorDesc* inputs, int nbInputs,
    const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const {
  return 0;
}

template <typename T>
int AnchorGeneratorPluginDynamic::enqueue_impl(
    const nvinfer1::PluginTensorDesc* inputDesc,
    const nvinfer1::PluginTensorDesc* outputDesc, const void* const* inputs,
    void* const* outputs, void* workspace, cudaStream_t stream) {
  const int height = inputDesc[0].dims.d[2];
  const int width = inputDesc[0].dims.d[3];
  const int box_num = height * width * num_anchors_;
  const int block = 512;
  const int gen_anchor_grid = (box_num + block - 1) / block;
  T* anchors = static_cast<T*>(outputs[0]);
  T* vars = static_cast<T*>(outputs[1]);
  const T* anchor_sizes_device = static_cast<const T*>(anchor_sizes_device_);
  const T* aspect_ratios_device = static_cast<const T*>(aspect_ratios_device_);
  const T* stride_device = static_cast<const T*>(stride_device_);
  const T* variances_device = static_cast<const T*>(variances_device_);
  paddle::operators::GenAnchors<T><<<gen_anchor_grid, block, 0, stream>>>(
      anchors, aspect_ratios_device, aspect_ratios_.size(), anchor_sizes_device,
      anchor_sizes_.size(), stride_device, stride_.size(), height, width,
      offset_);
  const int var_grid = (box_num * 4 + block - 1) / block;
  paddle::operators::SetVariance<T><<<var_grid, block, 0, stream>>>(
      vars, variances_device, variances_.size(), box_num * 4);
  return cudaGetLastError() != cudaSuccess;
}

int AnchorGeneratorPluginDynamic::enqueue(
    const nvinfer1::PluginTensorDesc* inputDesc,
    const nvinfer1::PluginTensorDesc* outputDesc, const void* const* inputs,
    void* const* outputs, void* workspace, cudaStream_t stream) {
  assert(outputDesc[0].type == nvinfer1::DataType::kFLOAT);
  assert(outputDesc[1].type == nvinfer1::DataType::kFLOAT);
  return enqueue_impl<float>(inputDesc, outputDesc, inputs, outputs, workspace,
                             stream);
}

nvinfer1::DataType AnchorGeneratorPluginDynamic::getOutputDataType(
    int index, const nvinfer1::DataType* inputTypes, int nbInputs) const {
  return data_type_;
}

const char* AnchorGeneratorPluginDynamic::getPluginType() const {
  return "anchor_generator_plugin_dynamic";
}

int AnchorGeneratorPluginDynamic::getNbOutputs() const { return 2; }

int AnchorGeneratorPluginDynamic::initialize() { return 0; }

void AnchorGeneratorPluginDynamic::terminate() {}

size_t AnchorGeneratorPluginDynamic::getSerializationSize() const {
  size_t serialize_size = 0;
  serialize_size += SerializedSize(data_type_);
  serialize_size += SerializedSize(anchor_sizes_);
  serialize_size += SerializedSize(aspect_ratios_);
  serialize_size += SerializedSize(stride_);
  serialize_size += SerializedSize(variances_);
  serialize_size += SerializedSize(offset_);
  serialize_size += SerializedSize(num_anchors_);
  return serialize_size;
}

void AnchorGeneratorPluginDynamic::serialize(void* buffer) const {
  SerializeValue(&buffer, data_type_);
  SerializeValue(&buffer, anchor_sizes_);
  SerializeValue(&buffer, aspect_ratios_);
  SerializeValue(&buffer, stride_);
  SerializeValue(&buffer, variances_);
  SerializeValue(&buffer, offset_);
  SerializeValue(&buffer, num_anchors_);
}

void AnchorGeneratorPluginDynamic::destroy() {}

void AnchorGeneratorPluginDynamicCreator::setPluginNamespace(
    const char* lib_namespace) {
  namespace_ = std::string(lib_namespace);
}

const char* AnchorGeneratorPluginDynamicCreator::getPluginNamespace() const {
  return namespace_.c_str();
}

const char* AnchorGeneratorPluginDynamicCreator::getPluginName() const {
  return "anchor_generator_plugin_dynamic";
}

const char* AnchorGeneratorPluginDynamicCreator::getPluginVersion() const {
  return "1";
}

const nvinfer1::PluginFieldCollection*
AnchorGeneratorPluginDynamicCreator::getFieldNames() {
  return &field_collection_;
}

nvinfer1::IPluginV2Ext* AnchorGeneratorPluginDynamicCreator::createPlugin(
    const char* name, const nvinfer1::PluginFieldCollection* fc) {
  const nvinfer1::PluginField* fields = fc->fields;
  int type_id = -1;
  std::vector<float> anchor_sizes, aspect_ratios, stride, variances;
  float offset = .5;
  int num_anchors = -1;
  for (int i = 0; i < fc->nbFields; ++i) {
    const std::string field_name(fc->fields[i].name);
    const auto length = fc->fields[i].length;
    if (field_name.compare("type_id") == 0) {
      type_id = *static_cast<const int*>(fc->fields[i].data);
    } else if (field_name.compare("anchor_sizes")) {
      const auto* data = static_cast<const float*>(fc->fields[i].data);
      anchor_sizes.insert(anchor_sizes.end(), data, data + length);
    } else if (field_name.compare("aspect_ratios")) {
      const auto* data = static_cast<const float*>(fc->fields[i].data);
      aspect_ratios.insert(aspect_ratios.end(), data, data + length);
    } else if (field_name.compare("stride")) {
      const auto* data = static_cast<const float*>(fc->fields[i].data);
      stride.insert(stride.end(), data, data + length);
    } else if (field_name.compare("variances")) {
      const auto* data = static_cast<const float*>(fc->fields[i].data);
      variances.insert(variances.end(), data, data + length);
    } else if (field_name.compare("offset")) {
      offset = *static_cast<const float*>(fc->fields[i].data);
    } else if (field_name.compare("num_anchors")) {
      num_anchors = *static_cast<const int*>(fc->fields[i].data);
    } else {
      assert(false && "unknown plugin field name.");
    }
  }
  return new AnchorGeneratorPluginDynamic(nvinfer1::DataType::kFLOAT,
                                          anchor_sizes, aspect_ratios, stride,
                                          variances, offset, num_anchors);
}

nvinfer1::IPluginV2Ext* AnchorGeneratorPluginDynamicCreator::deserializePlugin(
    const char* name, const void* serial_data, size_t serial_length) {
  auto plugin = new AnchorGeneratorPluginDynamic(serial_data, serial_length);
  plugin->setPluginNamespace(namespace_.c_str());
  return plugin;
}
#endif

}  // namespace plugin
}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
