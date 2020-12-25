//   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "paddle/fluid/framework/details/op_handle_base.h"
#include "paddle/fluid/framework/details/var_handle.h"
#include "paddle/fluid/framework/ir/graph.h"
#include "paddle/fluid/framework/ir/pass.h"
#include "paddle/fluid/framework/op_desc.h"
#include "paddle/fluid/framework/op_proto_maker.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/platform/place.h"

namespace paddle {
namespace framework {
class OpDesc;
}  // namespace framework
}  // namespace paddle

namespace paddle {
namespace framework {
namespace details {

// all variable in each devices.
// The outside vector is the device vector. Each element of this vector is a
// map from variable name to variables. The variables, who have the same name,
// will have a differsent version. The offset in the
// `std::vector<VarHandle*>` is the version of varaibles.
typedef std::vector<std::unordered_map<std::string, std::vector<VarHandle *>>>
    GraphVars;
constexpr char kGraphVars[] = "vars";

constexpr char kNRanks[] = "nranks";

constexpr char kPlaces[] = "places";
constexpr char kGlobalScope[] = "global_scope";
constexpr char kLocalScopes[] = "local_scopes";
constexpr char kNCCLCtxs[] = "nccl_ctxs";
constexpr char kBKCLCtxs[] = "bkcl_ctxs";
constexpr char kUseHierarchicalAllReduce[] = "use_hierarchical_allreduce";

// aux variables to represent dependency. Useful to resolve data hazard.
typedef std::unordered_set<VarHandleBase *> GraphDepVars;
constexpr char kGraphDepVars[] = "dep_vars";

typedef std::unordered_set<std::string> FusedVars;
constexpr char kFusedVars[] = "fused_vars";
constexpr char kFusedVarNamePrefix[] = "@FUSEDVAR@";

typedef std::string FusedOptType;
constexpr char kFusedOptType[] = "fused_opt_type";

typedef std::vector<std::string> FusedGrads;
constexpr char kFusedGrads[] = "fused_gradients";

typedef std::vector<std::pair<std::string, std::string>> ParamsAndGrads;
constexpr char kParamsAndDenseGrads[] = "params_and_dense_grads";
constexpr char kParamsAndSparseGrads[] = "params_and_sparse_grads";

typedef std::vector<ProgramDesc> ProgramDescs;
constexpr char kProgramDescs[] = "program_descs";

typedef std::unordered_set<std::string> PinnedVars;
constexpr char kPinnedVars[] = "pinned_vars";

typedef std::vector<std::vector<std::pair<std::string, std::string>>>
    GroupParamsAndGrads;
constexpr char kGroupParamsAndDenseGrads[] = "group_params_dense_grads";

inline bool IsOpRole(const OpDesc &op, OpRole role) {
  const auto &attrs = op.GetAttrMap();
  auto iter = attrs.find(OpProtoAndCheckerMaker::OpRoleAttrName());
  if (iter == attrs.end()) return false;
  return static_cast<bool>(BOOST_GET_CONST(int, iter->second) &
                           static_cast<int>(role));
}

inline std::vector<std::string> GetOpRoleVarsOrEmpty(const OpDesc &op) {
  const auto &attrs = op.GetAttrMap();
  auto iter = attrs.find(OpProtoAndCheckerMaker::OpRoleVarAttrName());
  if (iter == attrs.end()) return {};
  auto &ret = BOOST_GET_CONST(std::vector<std::string>, iter->second);
  PADDLE_ENFORCE_EQ(
      ret.size() % 2, 0,
      platform::errors::InvalidArgument(
          "The size of attribute %s must be an even number, but got %d",
          OpProtoAndCheckerMaker::OpRoleVarAttrName(), ret.size()));
  return BOOST_GET_CONST(std::vector<std::string>, iter->second);
}

bool IsDataParallelInferenceGraph(const ir::Graph &graph);

std::vector<std::unique_ptr<ir::Graph>> TrySeparateToMultipleSingleDeviceGraphs(
    ir::Graph *graph);

bool HasDropLastReadOp(const ir::Graph &graph);

bool HasKeepLastReadOp(const ir::Graph &graph);

template <typename T>
void CopyGraphAttrIfExists(const ir::Graph &src, ir::Graph *dst,
                           const std::string &name) {
  if (src.Has(name)) {
    auto &attr = src.Get<T>(name);
    dst->Set(name, new T(attr));
  }
}

}  // namespace details
}  // namespace framework
}  // namespace paddle
