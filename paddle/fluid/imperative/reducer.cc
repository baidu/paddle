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

#include "paddle/fluid/imperative/reducer.h"
#include <algorithm>
#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include "paddle/fluid/framework/data_type.h"

namespace paddle {
namespace imperative {

std::shared_ptr<Reducer> Reducer::s_instance_ = NULL;

Reducer::Reducer(const std::vector<std::shared_ptr<imperative::VarBase>> &vars,
                 const std::vector<std::vector<size_t>> &group_indices,
                 const std::vector<bool> &is_sparse_gradient,
                 std::shared_ptr<imperative::ParallelContext> parallel_ctx)
    : vars_(vars),
      group_indices_(group_indices),
      is_sparse_gradient_(is_sparse_gradient),
      parallel_ctx_(parallel_ctx) {
  VLOG(3) << "Start construct the Reducer ...";
  // initialize groups
  InitializeGroups(group_indices);

  // initialize varname2index_
  {
    for (size_t group_index = 0; group_index < group_indices.size();
         ++group_index) {
      for (size_t var_index = 0; var_index < group_indices[group_index].size();
           ++var_index) {
        size_t global_var_index = group_indices[group_index][var_index];
        const auto variable_index = VariableIndex{
            .group_index = group_index, .variable_index = var_index,
        };
        VLOG(0) << "add hook for var[" << vars_[global_var_index]->GradVarName()
                << "]";
        vars_[global_var_index]->SharedVar()->AddGradVarLeafBackwardHook(
            std::unique_ptr<LambdaGradAccumulatorPostHook>(
                new LambdaGradAccumulatorPostHook([=](VariableWrapper *grad) {
                  this->AddDistHook(grad, variable_index);
                })));
      }
    }
  }
  compute_stream_ = static_cast<platform::CUDADeviceContext *>(
                        platform::DeviceContextPool::Instance().Get(place_))
                        ->stream();
  comm_stream_ = platform::NCCLCommContext::Instance().Get(0, place_)->stream();
  events_.resize(group_indices.size());
  for (auto &event : events_) {
    event = platform::CudaEventResourcePool::Instance().New(
        BOOST_GET_CONST(platform::CUDAPlace, place_).device);
  }
  // just for copy back
  copy_enent_ = platform::CudaEventResourcePool::Instance().New(
      BOOST_GET_CONST(platform::CUDAPlace, place_).device);

  std::call_once(once_flag_, []() {
    std::atexit([]() { Reducer::GetInstance()->ReleaseReducer(); });
  });
}

void Reducer::ReleaseReducer() {
  for (auto &event : events_) {
    event.reset();
  }
  copy_enent_.reset();
}

void Reducer::InitializeGroups(
    const std::vector<std::vector<size_t>> &group_indices) {
  VLOG(3) << "Start initialize groups ..";
  // clear the group
  groups_.clear();
  groups_.reserve(group_indices.size());

  auto group_nums = group_indices.size();
  for (size_t group_index = 0; group_index < group_nums; ++group_index) {
    const auto &variable_indices_ = group_indices[group_index];
    PADDLE_ENFORCE_GT(
        variable_indices_.size(), 0,
        platform::errors::PreconditionNotMet(
            "The number of group_index[`%d`]'s elements is 0.", group_index));
    Group group;
    group.variable_indices_ = variable_indices_;
    int64_t all_length = 0;
    size_t offset = 0;

    // It's just for check the sparse or dense
    auto first_varbase = vars_[variable_indices_.front()];
    if (variable_indices_.size() == 1 &&
        is_sparse_gradient_[variable_indices_.front()]) {
      // process the sparse gradient. one sparse, one group
      group.sparse_contents = first_varbase->MutableGradVar();
      group.dtype = first_varbase->DataType();
      group.is_sparse_ = true;
    } else {
      // process the dense gradient.
      for (size_t index = 0; index < variable_indices_.size(); ++index) {
        const auto variable_index = variable_indices_[index];
        const auto &var = vars_[variable_index];
        const auto var_name = var->Name();
        PADDLE_ENFORCE_EQ(
            is_sparse_gradient_[variable_index], false,
            platform::errors::PreconditionNotMet(
                "Tensor `%s` 's GRAD must be LoDTensor.", var_name));

        auto lod_tensor = var->MutableVar()->GetMutable<framework::LoDTensor>();
        PADDLE_ENFORCE_EQ(lod_tensor->IsInitialized(), true,
                          platform::errors::PreconditionNotMet(
                              "Tensor `%s` is not initialized.", var_name));
        auto size = lod_tensor->numel();
        PADDLE_ENFORCE_GT(
            size, 0,
            platform::errors::PreconditionNotMet(
                "The number of tensor `%s`'s elements is 0.", var_name));
        all_length += size;

        group.offset_.push_back(offset);
        group.length_.push_back(size);
        // for concat operator
        group.dense_tensors.push_back(framework::Tensor());
        offset += size;

        // check the dtype and place, it must be same.
        auto dtype = var->DataType();
        auto place = var->Place();
        if (index > 0) {
          PADDLE_ENFORCE_EQ(dtype, group.dtype,
                            platform::errors::PreconditionNotMet(
                                "Tensor `%s` has different dtype.", var_name));
          PADDLE_ENFORCE_EQ(place, place_,
                            platform::errors::PreconditionNotMet(
                                "Tensor `%s` has different place.", var_name));
        } else {
          group.dtype = dtype;
          place_ = place;
        }
      }
      // Alloc the continuous space
      auto tensor = group.dense_contents.GetMutable<framework::LoDTensor>();
      tensor->Resize(framework::make_ddim({all_length}))
          .mutable_data(place_, group.dtype);
    }
    // Debug Message For Reducer
    VLOG(0) << "the groups_[" << group_index << "] basic message:";
    VLOG(0) << "numul: " << all_length << " ;is_sparse: " << group.is_sparse_
            << " ;var number: " << group.variable_indices_.size();
    groups_.emplace_back(std::move(group));
  }
}

void Reducer::SetGradSpace(Group *p_group) {
  const std::vector<size_t> &global_indices = p_group->variable_indices_;
  std::vector<const framework::Tensor *> shape_refer;
  std::vector<framework::Tensor *> outputs;

  for (size_t index = 0; index < global_indices.size(); ++index) {
    const auto &var = vars_[global_indices[index]];  // varbase of var
    auto &grad_var = var->GradVarBase();             // varbase of var grad
    auto grad_tensor =
        grad_var->MutableVar()->GetMutable<framework::LoDTensor>();

    shape_refer.emplace_back(grad_tensor);
    outputs.emplace_back(&(p_group->dense_tensors[index]));
  }

  // split the tensor for fuse grad
  split_functor_(*parallel_ctx_->GetDeviceContext(0),
                 *(p_group->dense_contents.GetMutable<framework::LoDTensor>()),
                 shape_refer, 0, &outputs);
}

void Reducer::PrepareForBackward() {
  VLOG(3) << "start reseting count..";
  next_group_ = 0;
  for (size_t group_index = 0; group_index < groups_.size(); ++group_index) {
    auto &group = groups_[group_index];
    group.pending = group.variable_indices_.size();
  }
}

void Reducer::AddDistHook(VariableWrapper *var_warpper,
                          const VariableIndex &var_index) {
  auto group_index = var_index.group_index;
  auto &group = groups_[group_index];

  if (!group.is_sparse_) {
    // Only dense_contents need memory copy
    MarkVariableReady(var_index, var_warpper);
  }
  if (--group.pending == 0) {
    // can start allreduce
    MarkGroupReady(group_index);
  }

  if (next_group_ == groups_.size()) {
    FinalizeBackward();
  }
}

void Reducer::MarkVariableReady(const VariableIndex &var_index,
                                VariableWrapper *var_warpper) {
  auto group_index = var_index.group_index;
  auto variable_index = var_index.variable_index;
  auto &group = groups_[group_index];
  auto length = group.length_[variable_index];

  auto tensor = var_warpper->MutableVar()->GetMutable<framework::LoDTensor>();
  group.dense_tensors[variable_index].ShareDataWith(*tensor).Resize(
      {static_cast<int64_t>(length)});
}

void Reducer::MarkGroupReady(size_t group_index) {
  if (group_index > next_group_) {
    VLOG(0) << "Maybe it need adjust the order of group";
    return;
  }

  PADDLE_ENFORCE_CUDA_SUCCESS(
      cudaEventRecord(events_[group_index].get(), compute_stream_));
  PADDLE_ENFORCE_CUDA_SUCCESS(
      cudaStreamWaitEvent(comm_stream_, events_[group_index].get(), 0));
  for (; next_group_ < groups_.size() && groups_[next_group_].pending == 0;
       ++next_group_) {
    if (groups_[next_group_].is_sparse_) {
      VLOG(3) << "sparse group [" << next_group_ << "] start allreduce...";
      parallel_ctx_->AllReduceByStream(*groups_[next_group_].sparse_contents,
                                       groups_[next_group_].sparse_contents, 0,
                                       false);
    } else {
      VLOG(3) << "dense group [" << next_group_ << "] start allreduce...";
      // concat the tensor for fuse grad
      concat_functor_(*parallel_ctx_->GetDeviceContext(0),
                      groups_[next_group_].dense_tensors, 0,
                      groups_[next_group_]
                          .dense_contents.GetMutable<framework::LoDTensor>());
      // start allreduce
      parallel_ctx_->AllReduceByStream(groups_[next_group_].dense_contents,
                                       &(groups_[next_group_].dense_contents),
                                       0, false);
      // copy back use comm stream
      SetGradSpace(&groups_[next_group_]);
    }
  }
}

void Reducer::FinalizeBackward() {
  PADDLE_ENFORCE_CUDA_SUCCESS(cudaEventRecord(copy_enent_.get(), comm_stream_));
  PADDLE_ENFORCE_CUDA_SUCCESS(
      cudaStreamWaitEvent(compute_stream_, copy_enent_.get(), 0));
  VLOG(3) << "In the batch, Reducer is finished...";
}

std::vector<std::vector<size_t>> AssignGroupBySize(
    const std::vector<std::shared_ptr<imperative::VarBase>> &vars,
    const std::vector<bool> &is_sparse_gradient,
    const std::vector<size_t> &group_size_limits) {
  PADDLE_ENFORCE_EQ(vars.size(), is_sparse_gradient.size(),
                    platform::errors::PreconditionNotMet(
                        "vars len must be equal to is_sparse_gradient len, but "
                        "[%lu] != [%lu]",
                        vars.size(), is_sparse_gradient.size()));
  // the return vector
  std::vector<std::vector<size_t>> res;

  // Key: the var type
  // Value: should use which index in group_size_limits for group size limit
  std::unordered_map<std::string, size_t> group_limit_index;

  // Key: the var type
  // Value: <the var index in input tensors, total numel in this group>
  std::unordered_map<std::string, std::pair<std::vector<size_t>, size_t>>
      next_group;

  for (size_t i = 0; i < vars.size(); ++i) {
    const auto &var = vars[i];
    if (is_sparse_gradient[i]) {
      // we keep sparse var a single group
      res.push_back({i});
      continue;
    }

    const auto &var_dtype = var->DataType();
    const auto var_dtype_str = framework::DataTypeToString(var_dtype);
    VLOG(3) << "var[" << var->GradVarName() << "] 's type is "
            << var->DataType();
    auto &group_info = next_group[var_dtype_str];
    int64_t var_size = -1;
    if (var->Var().IsType<framework::LoDTensor>()) {
      var_size = var->Var().Get<framework::LoDTensor>().numel();
      VLOG(3) << "dims: " << var->Var().Get<framework::LoDTensor>().dims();
    } else {
      VLOG(3) << "var " << var->Name()
              << " is not tensor or selected_rows, so skip it";
      continue;
    }
    VLOG(3) << "var[" << var->GradVarName() << "] 's size is " << var_size;
    group_info.first.push_back(i);
    group_info.second += framework::SizeOfType(var_dtype) * var_size;

    if (group_limit_index.find(var_dtype_str) == group_limit_index.end()) {
      // means it is the first var of var_dtype
      group_limit_index[var_dtype_str] = 0;
    }
    auto &cur_limit_index = group_limit_index[var_dtype_str];
    if (group_info.second >= group_size_limits[cur_limit_index]) {
      // exceed group capacity and create a new group
      res.emplace_back(std::move(group_info.first));
      group_info = std::pair<std::vector<size_t>, size_t>();
      cur_limit_index =
          (std::min)(cur_limit_index + 1, group_size_limits.size() - 1);
    }
  }

  // add the final groups
  for (auto &e : next_group) {
    auto &group_info = e.second;
    if (!group_info.first.empty()) {
      res.emplace_back(std::move(group_info.first));
    }
  }

  for (const auto &group_index : res) {
    PADDLE_ENFORCE_NE(
        group_index.empty(), true,
        platform::errors::PreconditionNotMet(
            "AssignGroupBySize construct empty group, please check"));
  }
  std::sort(res.begin(), res.end(),
            [](const std::vector<size_t> &x, const std::vector<size_t> &y) {
              return x.front() < y.front();
            });
  return res;
}

}  // namespace imperative
}  // namespace paddle
