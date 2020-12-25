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

#pragma once

#include <map>
#include <string>
#include <tuple>
#include <typeindex>
#include <vector>

#include "paddle/fluid/framework/feed_fetch_type.h"
#include "paddle/fluid/framework/framework.pb.h"
#include "paddle/fluid/framework/lod_tensor_array.h"
#include "paddle/fluid/platform/place.h"
#ifdef PADDLE_WITH_CUDA
#include <cudnn.h>
#if defined(PADDLE_WITH_NCCL)
#include <nccl.h>
#endif
#endif

#if defined(PADDLE_WITH_XPU_BKCL)
#include "xpu/bkcl.h"
#endif

// Users should add forward declarations here
namespace paddle {

namespace platform {
#ifdef PADDLE_WITH_CUDA
#if defined(PADDLE_WITH_NCCL)
class Communicator;
class NCCLCommunicator;
#endif
#endif

#if defined(PADDLE_WITH_XPU_BKCL)
class BKCLCommunicator;
#endif
}  // namespace platform

namespace framework {
class LoDRankTable;
class LoDTensor;
class ReaderHolder;
class Scope;
class SelectedRows;
class Tensor;
}  // namespace framework

namespace operators {

class CudnnRNNCache;

namespace reader {
class LoDTensorBlockingQueueHolder;
class OrderedMultiDeviceLoDTensorBlockingQueueHolder;
}  // namespace reader
}  // namespace operators

}  // namespace paddle

namespace paddle {
namespace framework {

const char *ToTypeName(int var_id);
const std::type_index &VarTraitIdToTypeIndex(int var_id);
int TypeIndexToVarTraitId(const std::type_index &type);

namespace detail {

template <bool kStop, int kStart, int kEnd, typename T1, typename T2,
          typename... Args>
struct TypePosFinderImpl {
  static constexpr int kPos =
      std::is_same<T1, T2>::value
          ? kStart
          : TypePosFinderImpl<kStart + 2 == kEnd, kStart + 1, kEnd, T1,
                              Args...>::kPos;
};

template <int kStart, int kEnd, typename T1, typename T2>
struct TypePosFinderImpl<true, kStart, kEnd, T1, T2> {
  static constexpr int kPos = std::is_same<T1, T2>::value ? kStart : -1;
};

// TypePosFinder helps to find the position in which T is inside Args...
// If T is not inside Args..., kPos would be -1
template <typename T, typename... Args>
struct TypePosFinder {
  static constexpr int kPos =
      TypePosFinderImpl<sizeof...(Args) == 1, 0, sizeof...(Args), T,
                        Args...>::kPos;
};

template <typename... Args>
struct VarTypeRegistryImpl {
  static constexpr size_t kRegisteredTypeNum = sizeof...(Args);
  using ArgTuple = std::tuple<Args...>;

  // TypePos() returns the position in which T is inside Args...
  // If T is not inside Args..., return -1
  template <typename T>
  static constexpr int TypePos() {
    return TypePosFinder<T, Args...>::kPos;
  }

  // IsRegistered() returns whether T is registered inside RegistryImpl
  template <typename T>
  static constexpr bool IsRegistered() {
    return TypePos<T>() >= 0;
  }
};

}  // namespace detail

#define REG_PROTO_VAR_TYPE_TRAIT(type, proto_id)           \
  template <>                                              \
  struct VarTypeTrait<type> {                              \
    static_assert(VarTypeRegistry::IsRegistered<type>(),   \
                  "Must be registered type");              \
    using Type = type;                                     \
    static constexpr int kId = static_cast<int>(proto_id); \
  }

/**
 * The following codes are designed to register variable types.
 * Only registered types can be stored in Variable.
 * This registry mechanism is designed to speed up Variable.
 *
 * Caution: If you want to add more var types, please consider carefully
 * whether you really need to add it.
 */

// Users should add other variable types below.
// Paddle would generate unique Ids for each registered variable types.
using VarTypeRegistry = detail::VarTypeRegistryImpl<
    Tensor, LoDTensor, SelectedRows, std::vector<Scope *>, LoDRankTable,
    LoDTensorArray, platform::PlaceList, ReaderHolder, std::string, Scope *,
    operators::reader::LoDTensorBlockingQueueHolder, FetchList,
    operators::reader::OrderedMultiDeviceLoDTensorBlockingQueueHolder,
#ifdef PADDLE_WITH_CUDA
#if defined(PADDLE_WITH_NCCL)
    ncclUniqueId, platform::Communicator, platform::NCCLCommunicator,
#endif
    operators::CudnnRNNCache,
#endif
#if defined(PADDLE_WITH_XPU_BKCL)
    BKCLUniqueId, platform::BKCLCommunicator,
#endif
    int, float>;

template <typename T>
struct VarTypeTrait {
  static_assert(VarTypeRegistry::IsRegistered<T>(), "Must be registered type");
  using Type = T;
  /**
   * Unique VarType Id generation.
   *
   * The auto-generated id should not be the same as any protobuf id defined in
   * framework.proto. Therefore, we generate id by adding the type pos and
   * maximum protobuf id (i.e., proto::VarType::TUPLE).
   *
   * However, we may need more protobuf id in the future.
   * To avoid changing this auto id generation algorithm frequently, we
   * generate id by adding the type pos and twice of maximum protobuf id (i.e.,
   * proto::VarType::TUPLE).
   */
  static constexpr int kId = VarTypeRegistry::TypePos<T>() +
                             static_cast<int>(proto::VarType::TUPLE) * 2;
};

// Users should set some of variable type ids to be what is defined in
// framework.proto below
REG_PROTO_VAR_TYPE_TRAIT(LoDTensor, proto::VarType::LOD_TENSOR);
REG_PROTO_VAR_TYPE_TRAIT(SelectedRows, proto::VarType::SELECTED_ROWS);
REG_PROTO_VAR_TYPE_TRAIT(std::vector<Scope *>, proto::VarType::STEP_SCOPES);
REG_PROTO_VAR_TYPE_TRAIT(LoDRankTable, proto::VarType::LOD_RANK_TABLE);
REG_PROTO_VAR_TYPE_TRAIT(LoDTensorArray, proto::VarType::LOD_TENSOR_ARRAY);
REG_PROTO_VAR_TYPE_TRAIT(platform::PlaceList, proto::VarType::PLACE_LIST);
REG_PROTO_VAR_TYPE_TRAIT(ReaderHolder, proto::VarType::READER);
REG_PROTO_VAR_TYPE_TRAIT(FetchList, proto::VarType::FETCH_LIST);
REG_PROTO_VAR_TYPE_TRAIT(int, proto::VarType::INT32);
REG_PROTO_VAR_TYPE_TRAIT(float, proto::VarType::FP32);

/** End of variable type registration */

template <typename T>
inline constexpr bool IsRegisteredVarType() {
  return VarTypeRegistry::IsRegistered<T>();
}

#undef REG_PROTO_VAR_TYPE_TRAIT
}  // namespace framework
}  // namespace paddle
