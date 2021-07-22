// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <numeric>
#include <vector>
#include "paddle/fluid/inference/api/paddle_api.h"
#include "paddle/fluid/inference/capi/c_api_internal.h"
#include "paddle/fluid/inference/capi/paddle_c_api.h"
#include "paddle/fluid/platform/enforce.h"

using paddle::ConvertToACPrecision;
using paddle::ConvertToPaddleDType;
using paddle::ConvertToPDDataType;

namespace {
#define _DataTypeHelper_(CALLBACK, CPP_TYPE, PD_TYPE) \
  CALLBACK(CPP_TYPE, PD_DataType::PD_TYPE);

#define _DataType_(CALLBACK)                     \
  _DataTypeHelper_(CALLBACK, float, PD_FLOAT32); \
  _DataTypeHelper_(CALLBACK, int32_t, PD_INT32); \
  _DataTypeHelper_(CALLBACK, int64_t, PD_INT64); \
  _DataTypeHelper_(CALLBACK, uint8_t, PD_UINT8);

template <typename Visitor>
inline void VisitDataType(PD_DataType type, Visitor visitor) {
#define VisitDataTypeCallback(CPP_TYPE, PD_TYPE) \
  do {                                           \
    if (type == PD_TYPE) {                       \
      visitor.template apply<CPP_TYPE>();        \
      return;                                    \
    }                                            \
  } while (0)

  _DataType_(VisitDataTypeCallback);
#undef VisitDataTypeCallback
  PADDLE_THROW(
      paddle::platform::errors::InvalidArgument("Unsupported data type."));
}

struct PD_ZeroCopyFunctor {
  PD_ZeroCopyData* output_i;
  paddle::ZeroCopyTensor* output_t;

  PD_ZeroCopyFunctor(PD_ZeroCopyData* output_i_,
                     paddle::ZeroCopyTensor* output_t_)
      : output_i(output_i_), output_t(output_t_) {}

  template <typename OutT>
  void apply() {
    std::vector<OutT> out_data;
    int out_num =
        std::accumulate(output_i->shape, output_i->shape + output_i->shape_size,
                        1, std::multiplies<int>());
    out_data.resize(out_num);
    output_t->copy_to_cpu(out_data.data());
    output_i->data = reinterpret_cast<void*>(malloc(out_num * sizeof(OutT)));
    memmove(static_cast<OutT*>(output_i->data), out_data.data(),
            out_num * sizeof(OutT));
  }
};

}  // namespace

extern "C" {
bool PD_PredictorRun(const PD_AnalysisConfig* config, PD_Tensor* inputs,
                     int in_size, PD_Tensor** output_data, int* out_size,
                     int batch_size) {
  PADDLE_ENFORCE_NOT_NULL(
      config,
      paddle::platform::errors::InvalidArgument(
          "The pointer of analysis configuration shouldn't be nullptr"));
  VLOG(3) << "Predoctor: PD_PredictorRun. ";
  static std::map<std::string, std::unique_ptr<paddle::PaddlePredictor>>
      predictors;
  std::string val = config->config.model_dir();
  // fallback to prog_file_
  if (val.empty()) {
    val = config->config.prog_file();
  }
  if (!predictors.count(val)) {
    predictors[val] = paddle::CreatePaddlePredictor(config->config);
  }
  auto& predictor = predictors[val];
  std::vector<paddle::PaddleTensor> in;
  for (int i = 0; i < in_size; ++i) {
    in.emplace_back(inputs->tensor);
  }
  std::vector<paddle::PaddleTensor> out;
  VLOG(3) << "Run predictor in CAPI encapsulation. ";
  if (predictor->Run(in, &out, batch_size)) {
    int osize = out.size();
    *output_data = new PD_Tensor[osize];
    for (int i = 0; i < osize; ++i) {
      output_data[i]->tensor = out[i];
    }
    *out_size = osize;
    return true;
  }
  return false;
}

bool PD_PredictorZeroCopyRun(const PD_AnalysisConfig* config,
                             PD_ZeroCopyData* inputs, int in_size,
                             PD_ZeroCopyData** output, int* out_size) {
  PADDLE_ENFORCE_NOT_NULL(
      config,
      paddle::platform::errors::InvalidArgument(
          "The pointer of analysis configuration shouldn't be nullptr"));
  static std::map<std::string, std::unique_ptr<paddle::PaddlePredictor>>
      predictors;
  if (!predictors.count(config->config.model_dir())) {
    predictors[config->config.model_dir()] =
        paddle::CreatePaddlePredictor(config->config);
  }
  auto& predictor = predictors[config->config.model_dir()];
  auto input_names = predictor->GetInputNames();
  VLOG(3) << "The inputs' size is " << input_names.size();
  PADDLE_ENFORCE_EQ(
      input_names.size(), in_size,
      paddle::platform::errors::InvalidArgument(
          "The number of input and the number of model's input must match. The "
          "number of input is %d, the number of model's input is %d.",
          input_names.size(), in_size));
  for (int i = 0; i < in_size; ++i) {
    auto input_t = predictor->GetInputTensor(inputs[i].name);
    std::vector<int> tensor_shape;
    tensor_shape.assign(inputs[i].shape,
                        inputs[i].shape + inputs[i].shape_size);
    input_t->Reshape(tensor_shape);
    switch (inputs[i].dtype) {
      case PD_FLOAT32:
        input_t->copy_from_cpu(static_cast<float*>(inputs[i].data));
        break;
      case PD_INT32:
        input_t->copy_from_cpu(static_cast<int32_t*>(inputs[i].data));
        break;
      case PD_INT64:
        input_t->copy_from_cpu(static_cast<int64_t*>(inputs[i].data));
        break;
      case PD_UINT8:
        input_t->copy_from_cpu(static_cast<uint8_t*>(inputs[i].data));
        break;
      default:
        PADDLE_THROW(paddle::platform::errors::InvalidArgument(
            "Unsupported data type."));
        break;
    }
  }
  VLOG(3) << "Run ZeroCopyRun() in CAPI encapsulation. ";
  CHECK(predictor->ZeroCopyRun());
  auto output_names = predictor->GetOutputNames();
  int osize = output_names.size();
  *out_size = osize;
  *output = new PD_ZeroCopyData[osize];
  VLOG(3) << "The output size is " << osize;
  for (int i = 0; i < *out_size; ++i) {
    auto& output_i = (*output)[i];
    output_i.name = new char[output_names[i].length() + 1];
    snprintf(output_i.name, output_names[i].length() + 1, "%s",
             output_names[i].c_str());
    auto output_t = predictor->GetOutputTensor(output_names[i]);
    output_i.dtype = ConvertToPDDataType(output_t->type());
    std::vector<int> output_shape = output_t->shape();
    output_i.shape = new int[output_shape.size()];
    memmove(output_i.shape, output_shape.data(),
            output_shape.size() * sizeof(int));
    output_i.shape_size = output_shape.size();
    VisitDataType(output_i.dtype,
                  PD_ZeroCopyFunctor(&output_i, std::move(output_t.get())));
  }
  return true;
}

PD_Predictor* PD_NewPredictor(const PD_AnalysisConfig* config) {
  PD_Predictor* predictor = new PD_Predictor;
  predictor->predictor = paddle::CreatePaddlePredictor(config->config);
  return predictor;
}

void PD_DeletePredictor(PD_Predictor* predictor) {
  if (predictor) {
    predictor->predictor = nullptr;
    delete predictor;
    predictor = nullptr;
  }
}

int PD_GetInputNum(const PD_Predictor* predictor) {
  return static_cast<int>(predictor->predictor->GetInputNames().size());
}

int PD_GetOutputNum(const PD_Predictor* predictor) {
  return static_cast<int>(predictor->predictor->GetOutputNames().size());
}

const char* PD_GetInputName(const PD_Predictor* predictor, int n) {
  static std::vector<std::string> names;
  names.resize(predictor->predictor->GetInputNames().size());
  names[n] = predictor->predictor->GetInputNames()[n];
  return names[n].c_str();
}

const char* PD_GetOutputName(const PD_Predictor* predictor, int n) {
  static std::vector<std::string> names;
  names.resize(predictor->predictor->GetOutputNames().size());
  names[n] = predictor->predictor->GetOutputNames()[n];
  return names[n].c_str();
}

void PD_SetZeroCopyInput(PD_Predictor* predictor,
                         const PD_ZeroCopyTensor* tensor) {
  auto input = predictor->predictor->GetInputTensor(tensor->name);
  auto* shape_ptr = static_cast<int*>(tensor->shape.data);
  std::vector<int> shape(shape_ptr,
                         shape_ptr + tensor->shape.length / sizeof(int));
  input->Reshape(std::move(shape));
  switch (tensor->dtype) {
    case PD_FLOAT32:
      input->copy_from_cpu(static_cast<float*>(tensor->data.data));
      break;
    case PD_INT32:
      input->copy_from_cpu(static_cast<int32_t*>(tensor->data.data));
      break;
    case PD_INT64:
      input->copy_from_cpu(static_cast<int64_t*>(tensor->data.data));
      break;
    case PD_UINT8:
      input->copy_from_cpu(static_cast<uint8_t*>(tensor->data.data));
      break;
    default:
      PADDLE_THROW(
          paddle::platform::errors::InvalidArgument("Unsupported data type."));
      break;
  }

  if (tensor->lod.length) {
    auto* lod_ptr = reinterpret_cast<size_t*>(tensor->lod.data);
    std::vector<size_t> lod;
    lod.assign(lod_ptr, lod_ptr + tensor->lod.length / sizeof(size_t));
    input->SetLoD({std::move(lod)});
  }
}

void PD_GetZeroCopyOutput(PD_Predictor* predictor, PD_ZeroCopyTensor* tensor) {
  auto output = predictor->predictor->GetOutputTensor(tensor->name);
  tensor->dtype = ConvertToPDDataType(output->type());
  auto shape = output->shape();
  size_t shape_size = shape.size();
  if (tensor->shape.capacity < shape_size * sizeof(int)) {
    if (tensor->shape.data || tensor->shape.capacity) {
      std::free(tensor->shape.data);
    }
    tensor->shape.data = std::malloc(shape_size * sizeof(int));
    tensor->shape.capacity = shape_size * sizeof(int);
  }
  tensor->shape.length = shape_size * sizeof(int);
  std::copy(shape.begin(), shape.end(), static_cast<int*>(tensor->shape.data));

  int n =
      std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
  size_t length = n * paddle::PaddleDtypeSize(output->type());
  if (tensor->data.capacity < length) {
    if (tensor->data.data) {
      std::free(tensor->data.data);
    }
    tensor->data.data = std::malloc(length);
    tensor->data.capacity = std::move(length);
  }
  tensor->data.length = length;

  auto lod = output->lod();
  if (!lod.empty()) {
    tensor->lod.length = lod.front().size() * sizeof(size_t);
    if (tensor->lod.capacity < lod.front().size()) {
      if (tensor->lod.data) {
        std::free(tensor->lod.data);
      }

      tensor->lod.data = std::malloc(lod.front().size() * sizeof(size_t));
      tensor->lod.capacity = lod.front().size() * sizeof(size_t);
    }
    std::copy(lod.front().begin(), lod.front().end(),
              reinterpret_cast<size_t*>(tensor->lod.data));
  }
  switch (tensor->dtype) {
    case PD_FLOAT32:
      output->copy_to_cpu(reinterpret_cast<float*>(tensor->data.data));
      break;
    case PD_INT32:
      output->copy_to_cpu(reinterpret_cast<int32_t*>(tensor->data.data));
      break;
    case PD_INT64:
      output->copy_to_cpu(reinterpret_cast<int64_t*>(tensor->data.data));
      break;
    case PD_UINT8:
      output->copy_to_cpu(reinterpret_cast<uint8_t*>(tensor->data.data));
      break;
    default:
      PADDLE_THROW(
          paddle::platform::errors::InvalidArgument("Unsupported data type."));
      break;
  }
}

void PD_ZeroCopyRun(PD_Predictor* predictor) {
  predictor->predictor->ZeroCopyRun();
}
}  // extern "C"
