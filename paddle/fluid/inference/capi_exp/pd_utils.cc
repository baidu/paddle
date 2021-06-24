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

#include <string>

#include "paddle/fluid/inference/api/paddle_inference_api.h"
#include "paddle/fluid/inference/capi_exp/pd_utils.h"
#include "paddle/fluid/inference/capi_exp/utils_internal.h"
#include "paddle/fluid/platform/enforce.h"

#define DESTROY_ONE_DIM_ARRAY(type)                                           \
  void PD_OneDimArray##type##Destroy(__pd_take PD_OneDimArray##type* array) { \
    if (array != NULL) {                                                      \
      delete[] array->data;                                                   \
      delete array;                                                           \
    }                                                                         \
  }
#define CONVERT_VEC_TO_ONE_DIM_ARRAY(type, Type, vec_type)   \
  __pd_give PD_OneDimArray##Type* CvtVecToOneDimArray##Type( \
      const std::vector<vec_type>& vec) {                    \
    PD_OneDimArray##Type* array = new PD_OneDimArray##Type;  \
    array->size = vec.size();                                \
    array->data = vec.empty() ? NULL : new type[vec.size()]; \
    for (size_t index = 0; index < vec.size(); ++index) {    \
      array->data[index] = vec[index];                       \
    }                                                        \
    return array;                                            \
  }
#define CONVERT_ONE_DIM_ARRAY_TO_VEC(type, Type, vec_type)   \
  std::vector<vec_type> CvtOneDimArrayToVec##Type(           \
      __pd_keep const PD_OneDimArray##Type* array) {         \
    std::vector<vec_type> vec;                               \
    if (array != NULL) {                                     \
      vec.resize(array->size);                               \
      for (size_t index = 0; index < array->size; ++index) { \
        vec[index] = array->data[index];                     \
      }                                                      \
    }                                                        \
    return vec;                                              \
  }

#define ONE_DIM_ARRAY_UTILS_FUNC_IMPL(type, Type, vec_type) \
  extern "C" {                                              \
  DESTROY_ONE_DIM_ARRAY(Type);                              \
  }                                                         \
  namespace paddle_infer {                                  \
  CONVERT_VEC_TO_ONE_DIM_ARRAY(type, Type, vec_type)        \
  CONVERT_ONE_DIM_ARRAY_TO_VEC(type, Type, vec_type)        \
  }

ONE_DIM_ARRAY_UTILS_FUNC_IMPL(int32_t, Int32, int)
ONE_DIM_ARRAY_UTILS_FUNC_IMPL(size_t, Size, size_t)

#undef ONE_DIM_ARRAY_UTILS_FUNC_IMPL
#undef CONVERT_ONE_DIM_ARRAY_TO_VEC
#undef CONVERT_VEC_TO_ONE_DIM_ARRAY
#undef DESTROY_ONE_DIM_ARRAY

void PD_OneDimArrayCstrDestroy(__pd_take PD_OneDimArrayCstr* array) {
  if (array != NULL) {
    if (array->size != 0) {
      for (size_t index = 0; index < array->size; ++index) {
        delete[] array->data[index];
      }
    }
    delete[] array->data;
    delete array;
  }
}
namespace paddle_infer {

__pd_give PD_OneDimArrayCstr* CvtVecToOneDimArrayCstr(
    const std::vector<std::string>& vec) {
  PD_OneDimArrayCstr* array = new PD_OneDimArrayCstr;
  array->size = vec.size();
  array->data = vec.empty() ? NULL : new char*[vec.size()];
  for (size_t index = 0u; index < vec.size(); ++index) {
    array->data[index] = new char[vec[index].size() + 1];
    memcpy(array->data[index], vec[index].c_str(), vec[index].size() + 1);
  }
  return array;
}

std::vector<std::string> CvtOneDimArrayToVecCstr(
    __pd_keep const PD_OneDimArrayCstr* array) {
  std::vector<std::string> vec;
  for (size_t index = 0; index < array->size; ++index) {
    vec.emplace_back(array->data[index]);
  }
  return vec;
}

}  // namespace paddle_infer

#define DESTROY_TWO_DIM_ARRAY(type)                                           \
  void PD_TwoDimArray##type##Destroy(__pd_take PD_TwoDimArray##type* array) { \
    if (array != NULL) {                                                      \
      if (array->size != 0) {                                                 \
        for (size_t index = 0; index < array->size; ++index) {                \
          PD_OneDimArray##type##Destroy(array->data[index]);                  \
        }                                                                     \
      }                                                                       \
      delete[] array->data;                                                   \
      delete array;                                                           \
    }                                                                         \
  }
#define CONVERT_VEC_TO_TWO_DIM_ARRAY(type, Type, vec_type)                    \
  __pd_give PD_TwoDimArray##Type* CvtVecToTwoDimArray##Type(                  \
      const std::vector<std::vector<vec_type>>& vec) {                        \
    PD_TwoDimArray##Type* array = new PD_TwoDimArray##Type;                   \
    array->size = vec.size();                                                 \
    array->data = vec.empty() ? NULL : new PD_OneDimArray##Type*[vec.size()]; \
    for (size_t index = 0; index < vec.size(); ++index) {                     \
      array->data[index] = CvtVecToOneDimArray##Type(vec[index]);             \
    }                                                                         \
    return array;                                                             \
  }
#define CONVERT_TWO_DIM_ARRAY_TO_VEC(type, Type, vec_type)            \
  std::vector<std::vector<vec_type>> CvtTwoDimArrayToVec##Type(       \
      __pd_keep const PD_TwoDimArray##Type* array) {                  \
    std::vector<std::vector<vec_type>> vec;                           \
    if (array != NULL && array->size != 0) {                          \
      vec.resize(array->size);                                        \
      for (size_t index = 0; index < array->size; ++index) {          \
        vec[index] = CvtOneDimArrayToVec##Type((array->data)[index]); \
      }                                                               \
    }                                                                 \
    return vec;                                                       \
  }
#define TWO_DIM_ARRAY_UTILS_FUNC_IMPL(type, Type, vec_type) \
  extern "C" {                                              \
  DESTROY_TWO_DIM_ARRAY(Type);                              \
  }                                                         \
  namespace paddle_infer {                                  \
  CONVERT_VEC_TO_TWO_DIM_ARRAY(type, Type, vec_type)        \
  CONVERT_TWO_DIM_ARRAY_TO_VEC(type, Type, vec_type)        \
  }

TWO_DIM_ARRAY_UTILS_FUNC_IMPL(size_t, Size, size_t)

#undef TWO_DIM_ARRAY_UTILS_FUNC_IMPL
#undef CONVERT_TWO_DIM_ARRAY_TO_VEC
#undef CONVERT_VEC_TO_TWO_DIM_ARRAY
#undef DESTROY_TWO_DIM_ARRAY

namespace paddle_infer {

PlaceType CvtToCxxPlaceType(PD_PlaceType place_type) {
  switch (place_type) {
    case PD_PLACE_UNK:
      return PlaceType::kUNK;
    case PD_PLACE_CPU:
      return PlaceType::kCPU;
    case PD_PLACE_GPU:
      return PlaceType::kGPU;
    case PD_PLACE_XPU:
      return PlaceType::kXPU;
    default:
      PADDLE_THROW(paddle::platform::errors::InvalidArgument(
          "Unsupport paddle place type %d.", place_type));
      return PlaceType::kUNK;
  }
}

PD_PlaceType CvtFromCxxPlaceType(PlaceType place_type) {
  switch (place_type) {
    case PlaceType::kCPU:
      return PD_PLACE_CPU;
    case PlaceType::kGPU:
      return PD_PLACE_GPU;
    case PlaceType::kXPU:
      return PD_PLACE_XPU;
    default:
      return PD_PLACE_UNK;
  }
}

DataType CvtToCxxDatatype(PD_DataType data_type) {
  switch (data_type) {
    case PD_DATA_FLOAT32:
      return DataType::FLOAT32;
    case PD_DATA_INT64:
      return DataType::INT64;
    case PD_DATA_INT32:
      return DataType::INT32;
    case PD_DATA_UINT8:
      return DataType::UINT8;
    case PD_DATA_INT8:
      return DataType::INT8;
    default:
      PADDLE_THROW(paddle::platform::errors::InvalidArgument(
          "Unsupport paddle data type %d.", data_type));
      return DataType::FLOAT32;
  }
}

PD_DataType CvtFromCxxDatatype(DataType data_type) {
  switch (data_type) {
    case DataType::FLOAT32:
      return PD_DATA_FLOAT32;
    case DataType::INT64:
      return PD_DATA_INT64;
    case DataType::INT32:
      return PD_DATA_INT32;
    case DataType::UINT8:
      return PD_DATA_UINT8;
    default:
      return PD_DATA_UNK;
  }
}

}  // namespace paddle_infer
