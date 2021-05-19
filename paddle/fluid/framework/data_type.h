/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once
#include <string>
#include <typeindex>

#include "paddle/fluid/framework/framework.pb.h"
#include "paddle/fluid/platform/bfloat16.h"
#include "paddle/fluid/platform/complex.h"
#include "paddle/fluid/platform/complex128.h"
#include "paddle/fluid/platform/complex64.h"
#include "paddle/fluid/platform/eigen_ext.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/float16.h"

namespace paddle {
namespace platform {
struct bfloat16;
struct complex128;
struct complex64;
struct float16;
template <typename T>
struct complex;
}  // namespace platform
}  // namespace paddle

namespace paddle {
namespace framework {

template <typename T>
struct DataTypeTrait {};

// Stub handle for void
template <>
struct DataTypeTrait<void> {
  constexpr static proto::VarType::Type DataType() {
    return proto::VarType::RAW;
  }
};

#define _ForEachDataTypeHelper_(callback, cpp_type, proto_type) \
  callback(cpp_type, ::paddle::framework::proto::VarType::proto_type);

#define _ForEachDataType_(callback)                                            \
  _ForEachDataTypeHelper_(callback, float, FP32);                              \
  _ForEachDataTypeHelper_(callback, ::paddle::platform::float16, FP16);        \
  _ForEachDataTypeHelper_(callback, ::paddle::platform::bfloat16, BF16);       \
  _ForEachDataTypeHelper_(callback, double, FP64);                             \
  _ForEachDataTypeHelper_(callback, int, INT32);                               \
  _ForEachDataTypeHelper_(callback, int64_t, INT64);                           \
  _ForEachDataTypeHelper_(callback, bool, BOOL);                               \
  _ForEachDataTypeHelper_(callback, uint8_t, UINT8);                           \
  _ForEachDataTypeHelper_(callback, int16_t, INT16);                           \
  _ForEachDataTypeHelper_(callback, int8_t, INT8);                             \
  _ForEachDataTypeHelper_(callback, ::paddle::platform::complex<float>,        \
                          COMPLEX64);                                          \
  _ForEachDataTypeHelper_(callback, ::paddle::platform::complex<double>,       \
                          COMPLEX128);                                         \
  _ForEachDataTypeHelper_(callback, ::paddle::platform::complex64, COMPLEX64); \
  _ForEachDataTypeHelper_(callback, ::paddle::platform::complex128, COMPLEX128);

#define _ForEachDataTypeSmall_(callback)                                       \
  _ForEachDataTypeHelper_(callback, float, FP32);                              \
  _ForEachDataTypeHelper_(callback, double, FP64);                             \
  _ForEachDataTypeHelper_(callback, int, INT32);                               \
  _ForEachDataTypeHelper_(callback, int64_t, INT64);                           \
  _ForEachDataTypeHelper_(callback, ::paddle::platform::complex<float>,        \
                          COMPLEX64);                                          \
  _ForEachDataTypeHelper_(callback, ::paddle::platform::complex<double>,       \
                          COMPLEX128);                                         \
  _ForEachDataTypeHelper_(callback, ::paddle::platform::complex64, COMPLEX64); \
  _ForEachDataTypeHelper_(callback, ::paddle::platform::complex128, COMPLEX128);

// For the use of thrust, as index-type elements can be only integers.
#define _ForEachDataTypeTiny_(callback)          \
  _ForEachDataTypeHelper_(callback, int, INT32); \
  _ForEachDataTypeHelper_(callback, int64_t, INT64);

#define DefineDataTypeTrait(cpp_type, proto_type)                           \
  template <>                                                               \
  struct DataTypeTrait<cpp_type> {                                          \
    constexpr static proto::VarType::Type DataType() { return proto_type; } \
  }

_ForEachDataType_(DefineDataTypeTrait);

#undef DefineDataTypeTrait

extern proto::VarType::Type ToDataType(std::type_index type);
extern std::type_index ToTypeIndex(proto::VarType::Type type);

template <typename Visitor>
inline void VisitDataType(proto::VarType::Type type, Visitor visitor) {
#define VisitDataTypeCallback(cpp_type, proto_type) \
  do {                                              \
    if (type == proto_type) {                       \
      visitor.template apply<cpp_type>();           \
      return;                                       \
    }                                               \
  } while (0)

  _ForEachDataType_(VisitDataTypeCallback);
#undef VisitDataTypeCallback
  PADDLE_THROW(platform::errors::Unimplemented(
      "Not supported proto::VarType::Type(%d) as data type.",
      static_cast<int>(type)));
}

template <typename Visitor>
inline void VisitDataTypeSmall(proto::VarType::Type type, Visitor visitor) {
#define VisitDataTypeCallbackSmall(cpp_type, proto_type) \
  do {                                                   \
    if (type == proto_type) {                            \
      visitor.template apply<cpp_type>();                \
      return;                                            \
    }                                                    \
  } while (0)

  _ForEachDataTypeSmall_(VisitDataTypeCallbackSmall);
#undef VisitDataTypeCallbackSmall
}

template <typename Visitor>
inline void VisitDataTypeTiny(proto::VarType::Type type, Visitor visitor) {
#define VisitDataTypeCallbackTiny(cpp_type, proto_type) \
  do {                                                  \
    if (type == proto_type) {                           \
      visitor.template apply<cpp_type>();               \
      return;                                           \
    }                                                   \
  } while (0)

  _ForEachDataTypeTiny_(VisitDataTypeCallbackTiny);
#undef VisitDataTypeCallbackTiny
}

extern std::string DataTypeToString(const proto::VarType::Type type);
extern size_t SizeOfType(proto::VarType::Type type);
inline std::ostream& operator<<(std::ostream& out,
                                const proto::VarType::Type& type) {
  out << DataTypeToString(type);
  return out;
}

extern inline bool IsComplexType(const proto::VarType::Type type) {
  return (type == proto::VarType::COMPLEX64 ||
          type == proto::VarType::COMPLEX128);
}

extern proto::VarType::Type PromoteTypesIfComplexExists(
    const proto::VarType::Type type_a, const proto::VarType::Type type_b);

extern inline proto::VarType::Type ToComplexType(proto::VarType::Type t) {
  switch (t) {
    case proto::VarType::FP32:
      return proto::VarType::COMPLEX64;
    case proto::VarType::FP64:
      return proto::VarType::COMPLEX128;
    default:
      PADDLE_THROW(platform::errors::Unimplemented(
          "Unknown complex value data type (%s), now only support float32 and "
          "float64.",
          DataTypeToString(t)));
  }
}

}  // namespace framework
}  // namespace paddle
