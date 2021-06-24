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

#include "paddle/fluid/inference/tensorrt/plugin/trt_plugin_factory.h"
#include "paddle/fluid/inference/tensorrt/plugin/yolo_box_op_plugin.h"
#include "paddle/fluid/operators/detection/yolo_box_op.h"

namespace paddle {
namespace inference {
namespace tensorrt {
namespace plugin {

YoloBoxPlugin::YoloBoxPlugin(const nvinfer1::DataType data_type,
                             const std::vector<int>& anchors,
                             const int class_num, const float conf_thresh,
                             const int downsample_ratio, const bool clip_bbox,
                             const float scale_x_y, const int input_h,
                             const int input_w)
    : data_type_(data_type),
      class_num_(class_num),
      conf_thresh_(conf_thresh),
      downsample_ratio_(downsample_ratio),
      clip_bbox_(clip_bbox),
      scale_x_y_(scale_x_y),
      input_h_(input_h),
      input_w_(input_w) {
  anchors_.insert(anchors_.end(), anchors.cbegin(), anchors.cend());
  assert(data_type_ == nvinfer1::DataType::kFLOAT ||
         data_type_ == nvinfer1::DataType::kHALF);
  assert(class_num_ > 0);
  assert(input_h_ > 0);
  assert(input_w_ > 0);

  cudaMalloc(&anchors_device_, anchors.size() * sizeof(int));
  cudaMemcpy(anchors_device_, anchors.data(), anchors.size() * sizeof(int),
             cudaMemcpyHostToDevice);
}

YoloBoxPlugin::YoloBoxPlugin(const void* data, size_t length) {
  DeserializeValue(&data, &length, &data_type_);
  DeserializeValue(&data, &length, &anchors_);
  DeserializeValue(&data, &length, &class_num_);
  DeserializeValue(&data, &length, &conf_thresh_);
  DeserializeValue(&data, &length, &downsample_ratio_);
  DeserializeValue(&data, &length, &clip_bbox_);
  DeserializeValue(&data, &length, &scale_x_y_);
  DeserializeValue(&data, &length, &input_h_);
  DeserializeValue(&data, &length, &input_w_);
}

YoloBoxPlugin::~YoloBoxPlugin() {
  if (anchors_device_ != nullptr) {
    cudaFree(anchors_device_);
    anchors_device_ = nullptr;
  }
}

const char* YoloBoxPlugin::getPluginType() const { return "yolo_box_plugin"; }

const char* YoloBoxPlugin::getPluginVersion() const { return "1"; }

int YoloBoxPlugin::getNbOutputs() const { return 2; }

nvinfer1::Dims YoloBoxPlugin::getOutputDimensions(int index,
                                                  const nvinfer1::Dims* inputs,
                                                  int nb_input_dims) {
  const int anchor_num = anchors_.size() / 2;
  const int box_num = inputs[0].d[1] * inputs[0].d[2] * anchor_num;

  assert(index <= 1);

  if (index == 0) {
    return nvinfer1::Dims2(box_num, 4);
  }
  return nvinfer1::Dims2(box_num, class_num_);
}

bool YoloBoxPlugin::supportsFormat(nvinfer1::DataType type,
                                   nvinfer1::TensorFormat format) const {
  return ((type == data_type_ || type == nvinfer1::DataType::kINT32) &&
          format == nvinfer1::TensorFormat::kLINEAR);
}

size_t YoloBoxPlugin::getWorkspaceSize(int max_batch_size) const { return 0; }

template <typename T>
__device__ inline T sigmoid(T x) {
  return 1. / (1. + exp(-x));
}

template <>
__device__ inline float sigmoid(float x) {
  return 1.f / (1.f + expf(-x));
}

template <typename T>
__device__ inline void GetYoloBox(float* box, const T* x, const int* anchors,
                                  int i, int j, int an_idx, int grid_size_h,
                                  int grid_size_w, int input_size_h,
                                  int input_size_w, int index, int stride,
                                  int img_height, int img_width, float scale,
                                  float bias) {
  box[0] = static_cast<float>(
      (i + sigmoid(static_cast<float>(x[index]) * scale + bias)) * img_width /
      grid_size_w);
  box[1] = static_cast<float>(
      (j + sigmoid(static_cast<float>(x[index + stride]) * scale + bias)) *
      img_height / grid_size_h);
  box[2] = static_cast<float>(expf(static_cast<float>(x[index + 2 * stride])) *
                              anchors[2 * an_idx] * img_width / input_size_w);
  box[3] =
      static_cast<float>(expf(static_cast<float>(x[index + 3 * stride])) *
                         anchors[2 * an_idx + 1] * img_height / input_size_h);
}

__device__ inline int GetEntryIndex(int batch, int an_idx, int hw_idx,
                                    int an_num, int an_stride, int stride,
                                    int entry) {
  return (batch * an_num + an_idx) * an_stride + entry * stride + hw_idx;
}

template <typename T>
__device__ inline void CalcDetectionBox(T* boxes, const float* box,
                                        const int box_idx, const int img_height,
                                        const int img_width, bool clip_bbox) {
  float tmp_box_0, tmp_box_1, tmp_box_2, tmp_box_3;
  tmp_box_0 = box[0] - box[2] / 2;
  tmp_box_1 = box[1] - box[3] / 2;
  tmp_box_2 = box[0] + box[2] / 2;
  tmp_box_3 = box[1] + box[3] / 2;

  if (clip_bbox) {
    tmp_box_0 = max(tmp_box_0, 0.f);
    tmp_box_1 = max(tmp_box_1, 0.f);
    tmp_box_2 = min(tmp_box_2, static_cast<float>(img_width - 1));
    tmp_box_3 = min(tmp_box_3, static_cast<float>(img_height - 1));
  }

  boxes[box_idx + 0] = static_cast<T>(tmp_box_0);
  boxes[box_idx + 1] = static_cast<T>(tmp_box_1);
  boxes[box_idx + 2] = static_cast<T>(tmp_box_2);
  boxes[box_idx + 3] = static_cast<T>(tmp_box_3);
}

template <typename T>
__device__ inline void CalcLabelScore(T* scores, const T* input,
                                      const int label_idx, const int score_idx,
                                      const int class_num, const float conf,
                                      const int stride) {
  for (int i = 0; i < class_num; i++) {
    scores[score_idx + i] = static_cast<T>(
        conf * sigmoid(static_cast<float>(input[label_idx + i * stride])));
  }
}

template <typename T>
__global__ void KeYoloBoxFw(const T* const input, const int* const imgsize,
                            T* boxes, T* scores, const float conf_thresh,
                            const int* anchors, const int n, const int h,
                            const int w, const int an_num, const int class_num,
                            const int box_num, int input_size_h,
                            int input_size_w, bool clip_bbox, const float scale,
                            const float bias) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  float box[4];
  for (; tid < n * box_num; tid += stride) {
    int grid_num = h * w;
    int i = tid / box_num;
    int j = (tid % box_num) / grid_num;
    int k = (tid % grid_num) / w;
    int l = tid % w;

    int an_stride = (5 + class_num) * grid_num;
    int img_height = imgsize[2 * i];
    int img_width = imgsize[2 * i + 1];

    int obj_idx =
        GetEntryIndex(i, j, k * w + l, an_num, an_stride, grid_num, 4);
    float conf = sigmoid(static_cast<float>(input[obj_idx]));
    int box_idx =
        GetEntryIndex(i, j, k * w + l, an_num, an_stride, grid_num, 0);

    if (conf < conf_thresh) {
      for (int i = 0; i < 4; ++i) {
        box[i] = 0.f;
      }
    } else {
      GetYoloBox<T>(box, input, anchors, l, k, j, h, w, input_size_h,
                    input_size_w, box_idx, grid_num, img_height, img_width,
                    scale, bias);
    }

    box_idx = (i * box_num + j * grid_num + k * w + l) * 4;
    CalcDetectionBox<T>(boxes, box, box_idx, img_height, img_width, clip_bbox);

    int label_idx =
        GetEntryIndex(i, j, k * w + l, an_num, an_stride, grid_num, 5);
    int score_idx = (i * box_num + j * grid_num + k * w + l) * class_num;
    CalcLabelScore<T>(scores, input, label_idx, score_idx, class_num, conf,
                      grid_num);
  }
}

template <typename T>
int YoloBoxPlugin::enqueue_impl(int batch_size, const void* const* inputs,
                                void** outputs, void* workspace,
                                cudaStream_t stream) {
  const int n = batch_size;
  const int h = input_h_;
  const int w = input_w_;
  const int an_num = anchors_.size() / 2;
  const int box_num = h * w * an_num;
  int input_size_h = downsample_ratio_ * h;
  int input_size_w = downsample_ratio_ * w;

  float bias = -0.5 * (scale_x_y_ - 1.);
  constexpr int threads = 256;

  KeYoloBoxFw<T><<<(n * box_num + threads - 1) / threads, threads, 0, stream>>>(
      reinterpret_cast<const T* const>(inputs[0]),
      reinterpret_cast<const int* const>(inputs[1]),
      reinterpret_cast<T*>(outputs[0]), reinterpret_cast<T*>(outputs[1]),
      conf_thresh_, anchors_device_, n, h, w, an_num, class_num_, box_num,
      input_size_h, input_size_w, clip_bbox_, scale_x_y_, bias);
  return cudaGetLastError() != cudaSuccess;
}

int YoloBoxPlugin::enqueue(int batch_size, const void* const* inputs,
#if IS_TRT_VERSION_LT(8000)
                           void** outputs, void* workspace,
#else
                           void* const* outputs, void* workspace,
#endif
                           cudaStream_t stream) {
  if (data_type_ == nvinfer1::DataType::kFLOAT) {
    return enqueue_impl<float>(batch_size, inputs, outputs, workspace, stream);
  } else if (data_type_ == nvinfer1::DataType::kHALF) {
    return enqueue_impl<half>(batch_size, inputs, outputs, workspace, stream);
  }
  assert("unsupported type.");
}

int YoloBoxPlugin::initialize() { return 0; }

void YoloBoxPlugin::terminate() {}

size_t YoloBoxPlugin::getSerializationSize() const {
  size_t serialize_size = 0;
  serialize_size += SerializedSize(data_type_);
  serialize_size += SerializedSize(anchors_);
  serialize_size += SerializedSize(class_num_);
  serialize_size += SerializedSize(conf_thresh_);
  serialize_size += SerializedSize(downsample_ratio_);
  serialize_size += SerializedSize(clip_bbox_);
  serialize_size += SerializedSize(scale_x_y_);
  serialize_size += SerializedSize(input_h_);
  serialize_size += SerializedSize(input_w_);
  return serialize_size;
}

void YoloBoxPlugin::serialize(void* buffer) const {
  SerializeValue(&buffer, data_type_);
  SerializeValue(&buffer, anchors_);
  SerializeValue(&buffer, class_num_);
  SerializeValue(&buffer, conf_thresh_);
  SerializeValue(&buffer, downsample_ratio_);
  SerializeValue(&buffer, clip_bbox_);
  SerializeValue(&buffer, scale_x_y_);
  SerializeValue(&buffer, input_h_);
  SerializeValue(&buffer, input_w_);
}

void YoloBoxPlugin::destroy() {}

void YoloBoxPlugin::setPluginNamespace(const char* lib_namespace) {
  namespace_ = std::string(lib_namespace);
}

const char* YoloBoxPlugin::getPluginNamespace() const {
  return namespace_.c_str();
}

nvinfer1::DataType YoloBoxPlugin::getOutputDataType(
    int index, const nvinfer1::DataType* input_type, int nb_inputs) const {
  return data_type_;
}

bool YoloBoxPlugin::isOutputBroadcastAcrossBatch(int output_index,
                                                 const bool* input_is_broadcast,
                                                 int nb_inputs) const {
  return false;
}

bool YoloBoxPlugin::canBroadcastInputAcrossBatch(int input_index) const {
  return false;
}

void YoloBoxPlugin::configurePlugin(
    const nvinfer1::Dims* input_dims, int nb_inputs,
    const nvinfer1::Dims* output_dims, int nb_outputs,
    const nvinfer1::DataType* input_types,
    const nvinfer1::DataType* output_types, const bool* input_is_broadcast,
    const bool* output_is_broadcast, nvinfer1::PluginFormat float_format,
    int max_batct_size) {}

nvinfer1::IPluginV2Ext* YoloBoxPlugin::clone() const {
  return new YoloBoxPlugin(data_type_, anchors_, class_num_, conf_thresh_,
                           downsample_ratio_, clip_bbox_, scale_x_y_, input_h_,
                           input_w_);
}

YoloBoxPluginCreator::YoloBoxPluginCreator() {}

void YoloBoxPluginCreator::setPluginNamespace(const char* lib_namespace) {
  namespace_ = std::string(lib_namespace);
}

const char* YoloBoxPluginCreator::getPluginNamespace() const {
  return namespace_.c_str();
}

const char* YoloBoxPluginCreator::getPluginName() const {
  return "yolo_box_plugin";
}

const char* YoloBoxPluginCreator::getPluginVersion() const { return "1"; }

const nvinfer1::PluginFieldCollection* YoloBoxPluginCreator::getFieldNames() {
  return &field_collection_;
}

nvinfer1::IPluginV2Ext* YoloBoxPluginCreator::createPlugin(
    const char* name, const nvinfer1::PluginFieldCollection* fc) {
  const nvinfer1::PluginField* fields = fc->fields;

  int type_id = -1;
  std::vector<int> anchors;
  int class_num = -1;
  float conf_thresh = 0.01;
  int downsample_ratio = 32;
  bool clip_bbox = true;
  float scale_x_y = 1.;
  int h = -1;
  int w = -1;

  for (int i = 0; i < fc->nbFields; ++i) {
    const std::string field_name(fc->fields[i].name);
    if (field_name.compare("type_id") == 0) {
      type_id = *static_cast<const int*>(fc->fields[i].data);
    } else if (field_name.compare("anchors")) {
      const int length = fc->fields[i].length;
      const int* data = static_cast<const int*>(fc->fields[i].data);
      anchors.insert(anchors.end(), data, data + length);
    } else if (field_name.compare("class_num")) {
      class_num = *static_cast<const int*>(fc->fields[i].data);
    } else if (field_name.compare("conf_thresh")) {
      conf_thresh = *static_cast<const float*>(fc->fields[i].data);
    } else if (field_name.compare("downsample_ratio")) {
      downsample_ratio = *static_cast<const int*>(fc->fields[i].data);
    } else if (field_name.compare("clip_bbox")) {
      clip_bbox = *static_cast<const bool*>(fc->fields[i].data);
    } else if (field_name.compare("scale_x_y")) {
      scale_x_y = *static_cast<const float*>(fc->fields[i].data);
    } else if (field_name.compare("h")) {
      h = *static_cast<const int*>(fc->fields[i].data);
    } else if (field_name.compare("w")) {
      w = *static_cast<const int*>(fc->fields[i].data);
    } else {
      assert(false && "unknown plugin field name.");
    }
  }

  return new YoloBoxPlugin(
      type_id ? nvinfer1::DataType::kHALF : nvinfer1::DataType::kFLOAT, anchors,
      class_num, conf_thresh, downsample_ratio, clip_bbox, scale_x_y, h, w);
}

nvinfer1::IPluginV2Ext* YoloBoxPluginCreator::deserializePlugin(
    const char* name, const void* serial_data, size_t serial_length) {
  auto plugin = new YoloBoxPlugin(serial_data, serial_length);
  plugin->setPluginNamespace(namespace_.c_str());
  return plugin;
}

}  // namespace plugin
}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
