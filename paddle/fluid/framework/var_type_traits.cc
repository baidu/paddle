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

#include "paddle/fluid/framework/var_type_traits.h"
#include <unordered_map>
#include "paddle/fluid/framework/lod_rank_table.h"
#include "paddle/fluid/framework/reader.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/framework/selected_rows.h"
#include "paddle/fluid/operators/reader/lod_tensor_blocking_queue.h"
#include "paddle/fluid/platform/macros.h"
#ifdef PADDLE_WITH_CUDA
#if defined(PADDLE_WITH_NCCL)
#include "paddle/fluid/operators/nccl/nccl_gpu_common.h"
#include "paddle/fluid/platform/nccl_helper.h"
#endif
#include <cudnn.h>
#include "paddle/fluid/operators/conv_cudnn_op_cache.h"
#include "paddle/fluid/operators/cudnn_rnn_cache.h"
#endif

#if defined(PADDLE_WITH_XPU_BKCL)
#include "paddle/fluid/platform/bkcl_helper.h"
#endif

namespace paddle {
namespace framework {

// Besides registering variable type id, it is helpful to register a
// var_id -> std::type_index map (for example, get type names according to id)
namespace detail {

template <int kStart, int kEnd, bool kStop>
struct VarIdToTypeIndexMapInitializerImpl {
  template <typename MapType1, typename MapType2>
  static void Init(MapType1 *id_to_type, MapType2 *type_to_id) {
    using Type =
        typename std::tuple_element<kStart, VarTypeRegistry::ArgTuple>::type;
    static_assert(!std::is_same<Type, void>::value, "Type cannot be void");
    constexpr int kId = VarTypeTrait<Type>::kId;
    auto type = std::type_index(typeid(Type));
    PADDLE_ENFORCE_EQ(
        id_to_type->count(kId), 0,
        platform::errors::AlreadyExists(
            "Registered duplicate type id %d for type %s.", kId, type.name()));
    PADDLE_ENFORCE_EQ(
        type_to_id->count(type), 0,
        platform::errors::AlreadyExists(
            "Registered duplicate type index %s for id %d.", type.name(), kId));
    id_to_type->emplace(kId, type);
    type_to_id->emplace(type, kId);
    VarIdToTypeIndexMapInitializerImpl<kStart + 1, kEnd,
                                       kStart + 1 == kEnd>::Init(id_to_type,
                                                                 type_to_id);
  }
};

template <int kStart, int kEnd>
struct VarIdToTypeIndexMapInitializerImpl<kStart, kEnd, true> {
  template <typename MapType1, typename MapType2>
  static void Init(MapType1 *, MapType2 *) {}
};

// VarIdToTypeIndexMapInitializer is designed to initialize var_id ->
// std::type_index map and std::type_index -> var_id map
using VarIdToTypeIndexMapInitializer =
    VarIdToTypeIndexMapInitializerImpl<0, VarTypeRegistry::kRegisteredTypeNum,
                                       VarTypeRegistry::kRegisteredTypeNum ==
                                           0>;

struct VarIdToTypeIndexMapHolder {
  DISABLE_COPY_AND_ASSIGN(VarIdToTypeIndexMapHolder);

 public:
  static const std::type_index &ToTypeIndex(int var_id) {
    auto it = Instance().id_to_type_map_.find(var_id);
    PADDLE_ENFORCE_NE(it, Instance().id_to_type_map_.end(),
                      platform::errors::NotFound(
                          "Variable Id %d is not registered.", var_id));
    return it->second;
  }

  static int ToTypeId(const std::type_index &type) {
    auto it = Instance().type_to_id_map_.find(type);
    PADDLE_ENFORCE_NE(it, Instance().type_to_id_map_.end(),
                      platform::errors::NotFound(
                          "Variable Type %s is not registered.", type.name()));
    return it->second;
  }

 private:
  VarIdToTypeIndexMapHolder() {
    VarIdToTypeIndexMapInitializer::Init(&id_to_type_map_, &type_to_id_map_);
  }

  static const VarIdToTypeIndexMapHolder &Instance() {
    static const VarIdToTypeIndexMapHolder instance;
    return instance;
  }

  std::unordered_map<int, std::type_index> id_to_type_map_;
  std::unordered_map<std::type_index, int> type_to_id_map_;
};

}  // namespace detail

const std::type_index &VarTraitIdToTypeIndex(int var_id) {
  return detail::VarIdToTypeIndexMapHolder::ToTypeIndex(var_id);
}

const char *ToTypeName(int var_id) {
  return VarTraitIdToTypeIndex(var_id).name();
}

int TypeIndexToVarTraitId(const std::type_index &type) {
  return detail::VarIdToTypeIndexMapHolder::ToTypeId(type);
}

}  // namespace framework
}  // namespace paddle
