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

#include "paddle/fluid/framework/executor_cache.h"
#include "paddle/fluid/framework/op_info.h"

namespace paddle {
namespace framework {
class ProgramDesc;
}  // namespace framework
}  // namespace paddle

namespace paddle {
namespace framework {

namespace details {

static ExecutionStrategy GetExecutionStrategy(const platform::Place &place) {
  framework::ExecutionStrategy execution_strategy;

  auto device_type = platform::Place2DeviceType(place);
  switch (device_type) {
    case platform::DeviceType::CPU: {
      execution_strategy.num_threads_ = 2;
      break;
    }
    case platform::DeviceType::CUDA: {
      // NOTE: According experiments, one thread is faster in
      // most model training.
      execution_strategy.num_threads_ = 1;
      break;
    }
    case platform::DeviceType::XPU: {
      execution_strategy.num_threads_ = 1;
      break;
    }
    default:
      PADDLE_THROW(platform::errors::Unavailable("Unsupported Device type %d.",
                                                 device_type));
  }
  execution_strategy.use_device_ = device_type;

  return execution_strategy;
}

void AppendSkipDeletionVars(const std::vector<std::string> &append_vars,
                            std::vector<std::string> *all_vars) {
  for (auto &var : append_vars) {
    all_vars->emplace_back(var);
  }
}

/*
 * NOTE(Aurelius84): In ParallelExecutor, memory optimized pass will be applied.
 * To avoid eagerly deleting last alive variables which are necessary in
 * backward program, we firstly parse these variable names as
 * skip_eager_vars. While executing pe.run skip_eager_vars are used to
 * skip memory optimization.
 *
 * Variables satisfying the following rules are considered as skip_eager_var:
 *
 *   1. it is an output var in run_program_op
 *   2. it is an input var used in backward_op
 */
void ParseSafeEagerDeletionSkipVars(
    const ProgramDesc &program, int64_t forward_op_nums,
    const std::vector<std::string> &output_var_names,
    std::vector<std::string> *skip_eager_delete_vars) {
  auto all_ops = program.Block(0).AllOps();
  // NOTE: skip `shape` and `fill_constant` op created by
  // fluid.backward.gradients, one forward output will generate one `shape`
  // and `fill_constant`.
  size_t backward_op_start_index =
      forward_op_nums + (output_var_names.size() * 2);

  auto &op_info_map = OpInfoMap::Instance();

  // step 2: parse the necessary variable of backward op
  std::unordered_set<std::string> op_outputs;
  std::unordered_set<std::string> op_inputs;
  std::unordered_set<std::string> no_need_buffer_ins;

  for (auto i = backward_op_start_index; i < all_ops.size(); ++i) {
    framework::OpDesc *op = all_ops[i];
    // NOTE: skip NoNeedBufferVars of grad_op and GC its memory in advance.
    auto &op_info = op_info_map.Get(op->Type());
    auto &inferer = op_info.NoNeedBufferVarsInferer();
    no_need_buffer_ins.clear();
    if (inferer) {
      no_need_buffer_ins =
          inferer(op->Inputs(), op->Outputs(), op->GetAttrMap());
    }
    for (auto &in_names : op->Inputs()) {
      if (no_need_buffer_ins.count(in_names.first) == 0) {
        for (auto &in_name : in_names.second) {
          op_inputs.emplace(in_name);
        }
      } else {
        VLOG(2) << op->Type() << " has no_need_buffer_in: " << in_names.first
                << " , skip it.";
      }
    }
    for (const std::string &out_arg_name : op->OutputArgumentNames()) {
      op_outputs.emplace(out_arg_name);
    }
  }

  // For the grad op input variables, if it is not output of grad_op, it may
  // be output of forward op and we should set the variables as skip_var to
  // prevent it being deleted when grad op is called multiple times.
  for (const std::string &var_name : op_inputs) {
    if (op_outputs.find(var_name) == op_outputs.end()) {
      VLOG(2) << "skip eager var: " << var_name;
      skip_eager_delete_vars->emplace_back(var_name);
    }
  }
  VLOG(3) << "Found skip_eager_delete_vars: " << skip_eager_delete_vars->size();
}

}  // namespace details

// C++11 removes the need for manual locking. Concurrent execution shall wait if
// a static local variable is already being initialized.
// https://stackoverflow.com/questions/11711920/how-to-implement-multithread-safe-singleton-in-c11-without-using-mutex
ExecutorInfoCache &ExecutorInfoCache::Instance() {
  static ExecutorInfoCache g_exe_cache_info_map;
  return g_exe_cache_info_map;
}

CacheInfo GetExecutorInfoFromCache(const ProgramDesc *program_desc,
                                   const platform::Place &place,
                                   int64_t start_op_index, int64_t end_op_index,
                                   bool is_grad, int64_t program_id,
                                   framework::Scope *scope) {
  auto &cached_exe_info = framework::ExecutorInfoCache::Instance();

  if (!cached_exe_info.Has(program_id, is_grad)) {
    VLOG(1) << "create exe_info for " << program_id << " is_grad: " << is_grad;
    auto execution_strategy = details::GetExecutionStrategy(place);
    auto &build_strategy = cached_exe_info.GetBuildStrategy(program_id);

    // 2. Construct Graph and ParallelExecutor.
    auto graph = std::make_shared<framework::ir::Graph>(
        *program_desc, start_op_index, end_op_index);
    auto parallel_executor = std::make_shared<framework::ParallelExecutor>(
        place, scope, execution_strategy, build_strategy, graph.get());
    parallel_executor->PrepareVariables(scope);

    // 3. Insert value into cached map.
    auto &cached_value = cached_exe_info.GetMutable(program_id, is_grad);
    cached_value.executor_ = parallel_executor;
    cached_value.graph_ = std::move(graph);
    return std::make_pair(parallel_executor, /*is_new_created=*/true);
  } else {
    VLOG(1) << "get exe_info from cache by: " << program_id
            << " is_grad: " << is_grad;
    auto &cached_value = cached_exe_info.Get(program_id, is_grad);

    auto &parallel_executor = cached_value.executor_;
  }
}

}  // namespace framework
}  // namespace paddle
