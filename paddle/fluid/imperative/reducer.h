// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
#include <ThreadPool.h>
#include <algorithm>
#include <iostream>
#include <map>
#include <memory>
#include <queue>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/framework/details/exception_holder.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/fluid/framework/variable.h"
#include "paddle/fluid/operators/math/math_function.h"
#include "paddle/fluid/platform/for_range.h"

namespace paddle {
namespace platform {
class DeviceContext;

}  // namespace platform

namespace imperative {
class ParallelContext;
class VarBase;
class VariableWrapper;
}  // namespace imperative
}  // namespace paddle

namespace paddle {
namespace imperative {

#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL) || \
    defined(PADDLE_WITH_XPU_BKCL)

template <typename T>
struct DivNRanksFunctor {
  DivNRanksFunctor(int64_t nranks, T* output)
      : nranks_(nranks), output_(output) {}
  HOSTDEVICE void operator()(size_t idx) const {
    output_[idx] /= static_cast<T>(nranks_);
  }
  int64_t nranks_;
  T* output_;
};

template <typename Dex>
struct DivNRanksForAllReduce {
  framework::Tensor* in_;
  int64_t nranks_;
  const platform::DeviceContext& ctx_;
  DivNRanksForAllReduce(framework::Tensor* in, int64_t nranks,
                        const platform::DeviceContext& ctx)
      : in_(in), nranks_(nranks), ctx_(ctx) {}

  template <typename T>
  void apply() const {
    T* data = in_->mutable_data<T>(ctx_.GetPlace());
    platform::ForRange<Dex> for_range(static_cast<const Dex&>(ctx_),
                                      static_cast<size_t>(in_->numel()));
    DivNRanksFunctor<T> functor(nranks_, data);
    for_range(functor);
  }
};

class Group {
 public:
  // Here, we use dense_contents_ & sparse_contents_ to
  // achieve the tensor fuse. When is_sparse_ is true, sparse_contents_ work,
  // conversely, dense_contents_ works. It is mutex relationship.
  framework::Variable dense_contents_;
  framework::Variable* sparse_contents_ = nullptr;
  bool is_sparse_ = false;

  // for concat kernel
  std::vector<framework::Tensor> dense_tensors_;

  std::vector<size_t> length_;

  int64_t all_length_{0};
  // Global indices of participating variables in the group
  std::vector<size_t> variable_indices_;

  // Number of params that haven't been ready. When it is 0, it means
  // the group is ready.
  size_t pending_ = -1;

  // external message of group
  framework::proto::VarType::Type dtype_;

  // context is used to select the stream for concat
  void ConcatTensors(const platform::DeviceContext& context);

  // context is used to select the stream for split
  void SplitTensors(const platform::DeviceContext& context);

  // use it in CUDA
  void DivNRanks(framework::Tensor* tensor, int64_t nranks,
                 const platform::DeviceContext& context);

  void DivNRanks(const platform::DeviceContext& context, int64_t nranks);

  friend std::ostream& operator<<(std::ostream&, const Group&);
};

struct VariableLocator {
  // record the index in groups_
  size_t group_index;
  size_t inside_group_index;
};

class Reducer {
 public:
  explicit Reducer(
      const std::vector<std::shared_ptr<imperative::VarBase>>& vars,
      const std::vector<std::vector<size_t>>& group_indices,
      const std::vector<bool>& is_sparse_gradient,
      std::shared_ptr<imperative::ParallelContext> parallel_ctx,
      const std::vector<size_t>& group_size_limits, bool find_unused_vars);

  virtual ~Reducer() {}

  void InitializeGroups(const std::vector<std::vector<size_t>>& group_indices);

  void InitializeDenseGroups(const std::vector<size_t>& variable_indices_,
                             Group* p_group);

  void PrepareDeps(const std::unordered_set<GradOpNode*>& init_nodes);

  void PrepareForBackward(
      const std::vector<std::shared_ptr<imperative::VarBase>>& outputs);

  void AddDistHook(size_t var_index);

  void MarkVarReady(const size_t var_index, const bool is_used_var);

  void MarkGroupReady(size_t group_index);

  void FusedAllReduceSchedule(const int run_order, Group& group,  // NOLINT
                              const int curr_group_index);

  void FinalizeBackward();

  std::vector<std::vector<size_t>> RebuildGruops();

  inline bool NeedRebuildGroup() {
    return !has_rebuilt_group_ && !find_unused_vars_each_step_;
  }

  void ProcessUnusedDenseVars();

  bool HasGrad(size_t var_index);

  void TraverseBackwardGraph(
      const std::vector<std::shared_ptr<imperative::VarBase>>& outputs);

 private:
  std::vector<std::shared_ptr<imperative::VarBase>> vars_;
  std::vector<std::vector<size_t>> group_indices_;
  std::vector<Group> groups_;
  size_t next_group_ = 0;
  platform::Place place_;
  std::once_flag once_flag_;
  std::vector<bool> is_sparse_gradient_;
  std::shared_ptr<imperative::ParallelContext> parallel_ctx_;
  std::vector<VariableLocator> variable_locators_;

  int nrings_ = 1;
  int64_t nranks_ = -1;

  // Following variables are to help rebuild group
  // TODO(shenliang03): Support rebuild in the future.
  bool has_rebuilt_group_{true};
  std::vector<std::shared_ptr<imperative::VarBase>> rebuild_vars_;
  std::vector<int64_t> rebuild_var_indices_;
  const std::vector<size_t> group_size_limits_;

  // Following variables are to help unused vars
  std::unordered_map<GradOpNode*, size_t> node_deps_;
  std::unordered_map<VariableWrapper*, size_t> var_index_map_;
  std::vector<size_t> unused_vars_;
  bool has_marked_unused_vars_{false};
  bool find_unused_vars_each_step_{false};
  bool find_unused_vars_once_{true};
  bool groups_need_finalize_{false};
#ifdef PADDLE_WITH_XPU_BKCL
  // comm_pool_ is used for scheduling allreduce in multi Kunlun cards training.
  framework::details::ExceptionHolder exception_;
  std::unique_ptr<::ThreadPool> comm_pool_{nullptr};
  uint32_t comm_op_count_;
  std::mutex mutex_;
  std::condition_variable cv_;
#endif

  // it just for checking hook, each parameter can only trigger one hook
  std::vector<bool> vars_marked_ready_;

  // Following variables are to help control flow.
  // local_used_vars_ uses 0/1 to indicate whether the
  // var is used in iteration. After the end of the
  // iteration, global_used_vars_ is obtained synchronously
  // globally. Choose whether to update the local
  // gradient according to the global_used_vars_.
  std::vector<int> local_used_vars_;
  // global_used_vars_ is used in comm stream to avoid wait
  framework::Variable global_used_vars_;
};

std::vector<std::vector<size_t>> AssignGroupBySize(
    const std::vector<std::shared_ptr<imperative::VarBase>>& tensors,
    const std::vector<bool>& is_sparse_gradient,
    const std::vector<size_t>& group_size_limits,
    const std::vector<int64_t>& tensor_indices = {});
#endif

}  // namespace imperative
}  // namespace paddle
