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

#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "paddle/fluid/framework/details/build_strategy.h"
#include "paddle/fluid/framework/details/execution_strategy.h"
#include "paddle/fluid/framework/details/op_handle_base.h"
#include "paddle/fluid/framework/executor.h"
#include "paddle/fluid/framework/feed_fetch_type.h"
#include "paddle/fluid/framework/op_info.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/platform/device_context.h"

#if defined(PADDLE_WITH_NCCL)
#include "paddle/fluid/platform/nccl_helper.h"
#endif

namespace paddle {
namespace framework {

class ParallelExecutorPrivate;

using details::BuildStrategy;
using details::ExecutionStrategy;
namespace p = paddle::platform;
using DeviceType = paddle::platform::DeviceType;

class ParallelExecutor {
  DISABLE_COPY_AND_ASSIGN(ParallelExecutor);

 public:
  explicit ParallelExecutor(const std::vector<platform::Place> &places,
                            const std::vector<std::string> &bcast_vars,
                            const std::string &loss_var_name, Scope *scope,
                            const std::vector<Scope *> &local_scopes,
                            const ExecutionStrategy &exec_strategy,
                            const BuildStrategy &build_strategy,
                            ir::Graph *graph);

  ~ParallelExecutor();

  size_t DeviceCount() const;

  std::vector<Scope *> &GetLocalScopes();

  void DropLocalExeScopes();

  // This API is used to check whether DropLocalExeScopes work.
  bool NeedCreateLocalExeScope();

  /**
   * Feed tensors to local scopes. The size of tensors should be equal to the
   * size of local scopes.
   */
  void FeedTensorsIntoLocalScopes(
      const std::vector<std::unordered_map<std::string, LoDTensor>> &tensors);

  void FeedAndSplitTensorIntoLocalScopes(
      const std::unordered_map<std::string, LoDTensor> &tensors);

  FetchResultType Run(const std::vector<std::string> &fetch_tensors,
                      bool return_merged = true);

  const ir::Graph &Graph() const;

 private:
  // broadcast the parameters from the 0th device.
  // trainer_id the trainer index in nccl distributed training.
  void BCastParamsToDevices(const std::vector<std::string> &vars,
                            int trainer_id = 0) const;
  bool EnableParallelGraphExecution(const ir::Graph &graph,
                                    const ExecutionStrategy &exec_strategy,
                                    const BuildStrategy &build_strategy) const;

  ParallelExecutorPrivate *member_;
  std::vector<std::unique_ptr<ir::Graph>> async_graphs_;
};
}  // namespace framework
}  // namespace paddle
