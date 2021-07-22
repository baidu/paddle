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

#include <iostream>

#include "paddle/fluid/imperative/layer.h"
#include "paddle/fluid/string/string_helper.h"

#include "paddle/fluid/operators/math/concat_and_split.h"
#include "paddle/fluid/operators/strided_memcpy.h"

#include "paddle/fluid/imperative/parallel_context.h"

namespace paddle {
namespace imperative {

#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL) || \
    defined(PADDLE_WITH_XPU_BKCL)
// div the nranks
void Group::DivNRanks(const platform::DeviceContext &context, int64_t nranks) {
  framework::Tensor *tensor =
      is_sparse_
          ? sparse_contents_->GetMutable<framework::SelectedRows>()
                ->mutable_value()
          : dense_contents_.GetMutable<framework::LoDTensor>();

  if (platform::is_gpu_place(tensor->place())) {
#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
    DivNRanks(tensor, nranks, context);
#endif
  } else if (platform::is_cpu_place(tensor->place())) {
    framework::VisitDataTypeSmall(
        dtype_, DivNRanksForAllReduce<platform::CPUDeviceContext>(
                    tensor, nranks, context));
  } else if (platform::is_xpu_place(tensor->place())) {
#ifdef PADDLE_WITH_XPU_BKCL
// TODO(liuyuhui) support xpu about div nranks in the future
#endif
  }
}

template <typename DeviceContext, typename T>
static void ConcatTensorsForAllReduce(
    const DeviceContext &context,
    const std::vector<framework::Tensor> &dense_tensors_,
    framework::Variable *p_dense_contents) {
  operators::math::ConcatFunctor<DeviceContext, T> concat_functor_;
  concat_functor_(context, dense_tensors_, 0,
                  p_dense_contents->GetMutable<framework::LoDTensor>());
}

template <typename DeviceContext, typename T>
static void SplitTensorsForAllReduce(
    const DeviceContext &context, framework::Variable *p_dense_contents,
    std::vector<framework::Tensor> *p_dense_tensors) {
  auto *in = p_dense_contents->GetMutable<framework::LoDTensor>();
  std::vector<framework::Tensor *> outs;
  std::vector<const framework::Tensor *> shape_refer;

  outs.reserve(p_dense_tensors->size());
  shape_refer.reserve(p_dense_tensors->size());

  for (auto &tensor : *p_dense_tensors) {
    outs.emplace_back(&tensor);
    shape_refer.emplace_back(&tensor);
  }
  // Sometimes direct copies will be faster
  if (p_dense_tensors->size() < 10) {
    operators::StridedMemcpyWithAxis0<T>(context, *in, shape_refer, &outs);
  } else {
    operators::math::SplitFunctor<DeviceContext, T> split_functor_;
    split_functor_(context, *in, shape_refer, 0, &outs);
  }
}

// context is used to select the stream for concat
template <typename DeviceContext>
static void ConcatTensorsWithType(
    const DeviceContext &context,
    const std::vector<framework::Tensor> &dense_tensors_,
    framework::Variable *p_dense_contents,
    framework::proto::VarType::Type type) {
  switch (type) {
    case framework::proto::VarType::FP16:
      ConcatTensorsForAllReduce<DeviceContext, platform::float16>(
          context, dense_tensors_, p_dense_contents);
      break;
    case framework::proto::VarType::FP32:
      ConcatTensorsForAllReduce<DeviceContext, float>(context, dense_tensors_,
                                                      p_dense_contents);
      break;
    case framework::proto::VarType::FP64:
      ConcatTensorsForAllReduce<DeviceContext, double>(context, dense_tensors_,
                                                       p_dense_contents);
      break;
    default:
      PADDLE_THROW(platform::errors::Unimplemented(
          "Data type (%s) is not supported when it concats tensors for "
          "allreduce.",
          framework::DataTypeToString(type)));
  }
}

// context is used to select the stream for split
template <typename DeviceContext>
static void SplitTensorsWithType(
    const DeviceContext &context, framework::Variable *p_dense_contents,
    std::vector<framework::Tensor> *p_dense_tensors,
    framework::proto::VarType::Type type) {
  switch (type) {
    case framework::proto::VarType::FP16:
      SplitTensorsForAllReduce<DeviceContext, platform::float16>(
          context, p_dense_contents, p_dense_tensors);
      break;
    case framework::proto::VarType::FP32:
      SplitTensorsForAllReduce<DeviceContext, float>(context, p_dense_contents,
                                                     p_dense_tensors);
      break;
    case framework::proto::VarType::FP64:
      SplitTensorsForAllReduce<DeviceContext, double>(context, p_dense_contents,
                                                      p_dense_tensors);
      break;
    default:
      PADDLE_THROW(platform::errors::Unimplemented(
          "Data type (%s) is not supported when it splits tensors for "
          "allreduce.",
          framework::DataTypeToString(type)));
  }
}

#ifdef PADDLE_WITH_XPU_BKCL
template <>
void SplitTensorsForAllReduce<platform::XPUDeviceContext, float>(
    const platform::XPUDeviceContext &context,
    framework::Variable *p_dense_contents,
    std::vector<framework::Tensor> *p_dense_tensors) {
  auto *in = p_dense_contents->GetMutable<framework::LoDTensor>();
  std::vector<framework::Tensor *> outs;
  std::vector<const framework::Tensor *> shape_refer;

  outs.reserve(p_dense_tensors->size());
  shape_refer.reserve(p_dense_tensors->size());

  for (auto &tensor : *p_dense_tensors) {
    outs.emplace_back(&tensor);
    shape_refer.emplace_back(&tensor);
  }
  operators::math::SplitFunctor<platform::XPUDeviceContext, float>
      split_functor_;
  split_functor_(context, *in, shape_refer, 0, &outs);
}

// context is used to select the stream for concat
template <>
void ConcatTensorsWithType<platform::XPUDeviceContext>(
    const platform::XPUDeviceContext &context,
    const std::vector<framework::Tensor> &dense_tensors_,
    framework::Variable *p_dense_contents,
    framework::proto::VarType::Type type) {
  switch (type) {
    case framework::proto::VarType::FP32:
      ConcatTensorsForAllReduce<platform::XPUDeviceContext, float>(
          context, dense_tensors_, p_dense_contents);
      break;
    default:
      PADDLE_THROW(platform::errors::Unimplemented(
          "Data type (%s) is not supported when it concats tensors for "
          "allreduce.",
          framework::DataTypeToString(type)));
  }
}

// context is used to select the stream for split
template <>
void SplitTensorsWithType<platform::XPUDeviceContext>(
    const platform::XPUDeviceContext &context,
    framework::Variable *p_dense_contents,
    std::vector<framework::Tensor> *p_dense_tensors,
    framework::proto::VarType::Type type) {
  switch (type) {
    case framework::proto::VarType::FP32:
      SplitTensorsForAllReduce<platform::XPUDeviceContext, float>(
          context, p_dense_contents, p_dense_tensors);
      break;
    default:
      PADDLE_THROW(platform::errors::Unimplemented(
          "Data type (%s) is not supported when it splits tensors for "
          "allreduce.",
          framework::DataTypeToString(type)));
  }
}
#endif

void Group::ConcatTensors(const platform::DeviceContext &context) {
  auto place = context.GetPlace();
  if (platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
    ConcatTensorsWithType(
        static_cast<const platform::CUDADeviceContext &>(context),
        dense_tensors_, &dense_contents_, dtype_);
#else
    PADDLE_THROW(platform::errors::PermissionDenied(
        "Paddle can't concat grad tensors since it's not compiled with NCCL,"
        "Please recompile or reinstall Paddle with NCCL support."));
#endif
  } else if (platform::is_xpu_place(place)) {
#ifdef PADDLE_WITH_XPU_BKCL
    ConcatTensorsWithType(
        static_cast<const platform::XPUDeviceContext &>(context),
        dense_tensors_, &dense_contents_, dtype_);
#else
    PADDLE_THROW(platform::errors::PermissionDenied(
        "Paddle can't concat xpu grads since it's not compiled with BKCL,"
        "Please recompile or reinstall Paddle with BKCL support."));
#endif
  } else if (platform::is_cpu_place(place)) {
    ConcatTensorsWithType(
        static_cast<const platform::CPUDeviceContext &>(context),
        dense_tensors_, &dense_contents_, dtype_);
  } else {
    PADDLE_THROW(platform::errors::Unimplemented(
        "Concat grad tensor not supported on place (%s)", place));
  }
}

void Group::SplitTensors(const platform::DeviceContext &context) {
  auto place = context.GetPlace();
  if (platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
    SplitTensorsWithType(
        static_cast<const platform::CUDADeviceContext &>(context),
        &dense_contents_, &dense_tensors_, dtype_);
#else
    PADDLE_THROW(platform::errors::PermissionDenied(
        "Paddle can't split grad tensor since it's not compiled with NCCL,"
        "Please recompile or reinstall Paddle with NCCL support."));
#endif
  } else if (platform::is_xpu_place(place)) {
#ifdef PADDLE_WITH_XPU_BKCL
    SplitTensorsWithType(
        static_cast<const platform::XPUDeviceContext &>(context),
        &dense_contents_, &dense_tensors_, dtype_);
#else
    PADDLE_THROW(platform::errors::PermissionDenied(
        "Paddle can't split xpu grad since it's not compiled with BKCL,"
        "Please recompile or reinstall Paddle with BKCL support."));
#endif
  } else if (platform::is_cpu_place(place)) {
    SplitTensorsWithType(
        static_cast<const platform::CPUDeviceContext &>(context),
        &dense_contents_, &dense_tensors_, dtype_);
  } else {
    PADDLE_THROW(platform::errors::Unimplemented(
        "Split grad tensor not supported on place (%s)", place));
  }
}

std::ostream &operator<<(std::ostream &out, const Group &group) {
  const auto &vars = group.variable_indices_;
  out << "numel: " << group.all_length_ << " ;is_sparse: " << group.is_sparse_
      << " ;var number: " << vars.size() << "\n";
  auto begin = vars.begin();
  auto end = vars.end();
  out << "[";
  for (int i = 0; begin != end && i < 100; ++i, ++begin) {
    if (i > 0) out << ' ';
    out << *begin;
  }
  if (begin != end) {
    out << " ...";
  }
  out << "]\n";
  return out;
}

Reducer::Reducer(const std::vector<std::shared_ptr<imperative::VarBase>> &vars,
                 const std::vector<std::vector<size_t>> &group_indices,
                 const std::vector<bool> &is_sparse_gradient,
                 std::shared_ptr<imperative::ParallelContext> parallel_ctx,
                 const std::vector<size_t> &group_size_limits,
                 bool find_unused_vars)
    : vars_(vars),
      group_indices_(group_indices),
      is_sparse_gradient_(is_sparse_gradient),
      parallel_ctx_(parallel_ctx),
      group_size_limits_(group_size_limits),
      find_unused_vars_each_step_(find_unused_vars) {
  VLOG(3) << "Start construct the Reducer ...";
  nrings_ = parallel_ctx->GetNRings();
  nranks_ = parallel_ctx->GetNRanks();
#ifdef PADDLE_WITH_XPU_BKCL
  comm_pool_.reset(new ::ThreadPool(1));
  comm_op_count_ = 0;
#endif
  // initialize groups
  InitializeGroups(group_indices);
  for (size_t global_var_index = 0; global_var_index < vars_.size();
       ++global_var_index) {
    auto var = vars_[global_var_index];
    var->GradVarBase()->AddVoidHook(std::make_shared<std::function<void()>>(
        [=]() { this->AddDistHook(global_var_index); }));
    var_index_map_[var->GradVarBase()->SharedVar().get()] = global_var_index;
  }

  // for checking var is ready once
  vars_marked_ready_.resize(vars_.size(), false);

  // Initialize local used vars
  local_used_vars_.resize(vars_.size(), 0);
}

void Reducer::InitializeDenseGroups(
    const std::vector<size_t> &variable_indices_, Group *p_group) {
  int64_t all_length = 0;
  for (size_t index = 0; index < variable_indices_.size(); ++index) {
    const auto variable_index = variable_indices_[index];
    const auto &var = vars_[variable_index];
    const auto &var_name = var->Name();
    PADDLE_ENFORCE_EQ(is_sparse_gradient_[variable_index], false,
                      platform::errors::PreconditionNotMet(
                          "Tensor %s's GRAD must be LoDTensor, but received "
                          "GRAD is SelectedRows",
                          var_name));

    auto lod_tensor = var->MutableVar()->GetMutable<framework::LoDTensor>();
    PADDLE_ENFORCE_EQ(lod_tensor->IsInitialized(), true,
                      platform::errors::PreconditionNotMet(
                          "Tensor %s is not initialized.", var_name));
    const auto size = lod_tensor->numel();
    PADDLE_ENFORCE_GT(
        size, 0, platform::errors::PreconditionNotMet(
                     "The number of tensor %s's elements is 0.", var_name));
    all_length += size;

    p_group->length_.push_back(size);

    // for concat operator
    p_group->dense_tensors_.push_back(framework::Tensor());

    // check the dtype and place, it must be same.
    const auto &dtype = var->DataType();
    const auto &place = var->Place();
    if (index > 0) {
      PADDLE_ENFORCE_EQ(
          dtype, p_group->dtype_,
          platform::errors::PreconditionNotMet(
              "Tensor %s has different dtype. Expected dtype is %s, but actual "
              "dtype is %s",
              var_name, framework::DataTypeToString(p_group->dtype_),
              framework::DataTypeToString(dtype)));
      PADDLE_ENFORCE_EQ(place, place_,
                        platform::errors::PreconditionNotMet(
                            "Tensor %s has different place. Expected place is "
                            "%s, but actual place is %s",
                            var_name, place_, place));
    } else {
      p_group->dtype_ = dtype;
      place_ = place;
    }
  }
  p_group->all_length_ = all_length;
}

// Each parameter will be initialized according to the group information.
// For the sparse parameter, sparse_contents_ in the group directly points
// to the parameter. For dense parameters, first construct an empty Tensor().
// Then specify the actual memory in MarkDenseVarReady.
void Reducer::InitializeGroups(
    const std::vector<std::vector<size_t>> &group_indices) {
  VLOG(3) << "Start initialize groups ..";
  // clear the group
  groups_.clear();
  groups_.reserve(group_indices.size());
  variable_locators_.clear();
  variable_locators_.resize(vars_.size());

  auto group_nums = group_indices.size();
  for (size_t group_index = 0; group_index < group_nums; ++group_index) {
    const auto &variable_indices_ = group_indices[group_index];
    PADDLE_ENFORCE_GT(
        variable_indices_.size(), 0,
        platform::errors::PreconditionNotMet(
            "The number of group[%d]'s elements is 0.", group_index));
    Group group;

    // It's just for check the sparse or dense
    auto first_varbase = vars_[variable_indices_.front()];
    if (variable_indices_.size() == 1 &&
        is_sparse_gradient_[variable_indices_.front()]) {
      // process the sparse gradient. one sparse, one group
      group.dtype_ = first_varbase->DataType();
      group.is_sparse_ = true;
    } else {
      // process the dense gradient.
      InitializeDenseGroups(variable_indices_, &group);
      auto tensor = group.dense_contents_.GetMutable<framework::LoDTensor>();
      tensor->Resize(framework::make_ddim({group.all_length_}))
          .mutable_data(place_, group.dtype_);
    }

    // map variables to this group by VariableLocator
    size_t inside_group_index = 0;
    for (const auto var_index : variable_indices_) {
      variable_locators_[var_index] = VariableLocator{
          .group_index = group_index,
          .inside_group_index = inside_group_index++,
      };
    }
    group.variable_indices_ = std::move(variable_indices_);
    groups_.emplace_back(std::move(group));
    // Debug Message For Reducer
    VLOG(3) << "The Group[" << group_index << "]:" << groups_.back();
  }
}

void Reducer::PrepareDeps(const std::unordered_set<GradOpNode *> &init_nodes) {
  PADDLE_ENFORCE_EQ(
      node_deps_.empty(), true,
      platform::errors::AlreadyExists("Op deps must be initialized here"));

  std::queue<GradOpNode *> q;
  std::unordered_set<GradOpNode *> visited;

  for (auto pos = init_nodes.begin(); pos != init_nodes.end(); pos++) {
    q.push(*pos);
    visited.insert(*pos);
  }

  while (!q.empty()) {
    auto *cur_node = q.front();
    q.pop();

    const auto &grad_pending_nodes = cur_node->GradPendingNodes();
    for (auto &grad_pending_node : grad_pending_nodes) {
      PADDLE_ENFORCE_NOT_NULL(
          grad_pending_node,
          platform::errors::NotFound("Grad pending node should not be null"));
      ++node_deps_[grad_pending_node.get()];
      if (visited.count(grad_pending_node.get()) == 0) {
        visited.insert(grad_pending_node.get());
        q.push(grad_pending_node.get());
      }
    }
  }
}

void Reducer::TraverseBackwardGraph(
    const std::vector<std::shared_ptr<imperative::VarBase>> &outputs) {
  node_deps_.clear();
  std::queue<std::shared_ptr<GradOpNode>> q;
  std::unordered_set<VariableWrapper *> var_visited;
  std::unordered_set<GradOpNode *> init_nodes;

  for (const auto &output : outputs) {
    const auto &grad_node = output->GradVarBase()->GradNode();
    if (grad_node == nullptr || output->OverridedStopGradient()) {
      VLOG(3) << "Skip auto grad since there is no grad op or output is "
                 "stop_gradient=True: "
              << output->Name();
      continue;
    } else {
      init_nodes.insert(grad_node.get());
      var_visited.insert(output->SharedVar().get());
      q.push(grad_node);
    }
  }

  PrepareDeps(init_nodes);
  // Traverse the autograd graph starting at the specified output
  while (!q.empty()) {
    auto cur_node = q.front();
    q.pop();

    for (const auto &cur_op : *cur_node) {
      auto &bwd_outs = cur_op.GetOutsMap();
      for (const auto &pair : bwd_outs) {
        if (!pair.second.IsGrad()) {
          continue;
        }
        for (auto &var : pair.second) {
          if (!var || var->OverridedStopGradient()) {
            continue;
          } else {
            var_visited.insert(var.get());
          }
        }
      }
    }
    for (const auto &grad_pending_node : cur_node->GradPendingNodes()) {
      PADDLE_ENFORCE_NOT_NULL(grad_pending_node,
                              platform::errors::NotFound(
                                  "Grad pending node should not be nullptr"));
      auto iter = node_deps_.find(grad_pending_node.get());
      if (iter == node_deps_.end()) {
        continue;
      }
      if (--(iter->second) == 0) {
        q.push(grad_pending_node);
      }
    }
  }

  for (const auto &it : var_index_map_) {
    if (var_visited.count(it.first) == 0) {
      unused_vars_.push_back(it.second);
      VLOG(3) << "Var[" << it.second << "] [" << it.first->Name()
              << "] is not used";
    }
  }
}

// After each batch is calculated, the counter of each group(group.pending_)
// and allreudce sequence counter(next_group_) will be cleaned up again.
void Reducer::PrepareForBackward(
    const std::vector<std::shared_ptr<imperative::VarBase>> &outputs) {
  VLOG(3) << "after forward, then reset count for backward.";
  next_group_ = 0;
  std::for_each(groups_.begin(), groups_.end(), [](Group &group) {
    group.pending_ = group.variable_indices_.size();
    group.sparse_contents_ = nullptr;
  });

  // reinitialize vars_marked_ready_ for next iteration
  vars_marked_ready_.clear();
  vars_marked_ready_.resize(vars_.size(), false);

  PADDLE_ENFORCE_EQ(
      groups_need_finalize_, false,
      platform::errors::PreconditionNotMet(
          "A serious error has occurred here. Please "
          "set find_unused_parameters=True to traverse backward graph "
          "in each step to prepare reduce in advance. If you have "
          "set, There may be several reasons for this error: "
          "1) Please note that all forward outputs derived from the module "
          "parameters must participate in the calculation of losses and "
          "subsequent gradient calculations. If not, the wrapper will hang, "
          "waiting for autograd to generate gradients for these parameters. "
          "you can use detach or stop_gradient to make the unused parameters "
          "detached from the autograd graph. "
          "2) Used multiple forwards and one backward. You may be able to wrap "
          "multiple forwards in a model."));

  // The first var to trigger the unused parameter
  has_marked_unused_vars_ = false;

  if (find_unused_vars_once_ || find_unused_vars_each_step_) {
    unused_vars_.clear();
    TraverseBackwardGraph(outputs);
    // only check once in first step
    find_unused_vars_once_ = false;
  }

  if (find_unused_vars_each_step_ && unused_vars_.empty()) {
    LOG_FIRST_N(WARNING, 1)
        << "All parameters are involved in the backward pass. "
           "It is recommended to set find_unused_parameters to False "
           "to improve performance. However, if unused parameters "
           "appear in subsequent iterative training, then an error "
           "will occur. Please make it clear that in the subsequent "
           "training, there will be no parameters that are not used "
           "in the backward pass, and then set find_unused_parameters";
  }

  if (unused_vars_.size() == vars_.size()) {
    LOG_FIRST_N(WARNING, 1)
        << "There is no parameter in the device involved "
           "in the backward calculation. If there are "
           "parameters on other devices involved in the "
           "backward, then a serious error will occur here.";
  }
}

// Add hook function to each leaf node. When the gradient of a leaf node is
// generated, if it is the sparse parameter, it will directly execute allreduce,
// if it is the dense parameter, it will execute three steps: 1,
// MarkDenseVarReady. Find the position of the corresponding group
// through var_index, share the gradient memory and the group dense_tensors,
// the group counter is reduced by 1. 2, MarkGroupReady: When the group
// counter is 0, it means that allreduce can be emitted, and
// concat + allreduce + split is emitted in turn according to next_group_.
// 3, FinalizeBackward: after the end, synchronize each stream.
void Reducer::AddDistHook(size_t var_index) {
  PADDLE_ENFORCE_LT(var_index, variable_locators_.size(),
                    platform::errors::OutOfRange(
                        "Out of bounds variable index. it must be less"
                        "than %d, but it is %d",
                        variable_locators_.size(), var_index));

  VLOG(3) << "Var[" << var_index << "] ["
          << vars_[var_index]->GradVarBase()->Name()
          << "] arrived and triggered disthook";

  local_used_vars_[var_index] = 1;

  // rebuild group when find_unused_vars_each_step_ is false
  if (NeedRebuildGroup()) {
    rebuild_vars_.push_back(vars_[var_index]);
    rebuild_var_indices_.push_back(var_index);
  }

  if (!has_marked_unused_vars_) {
    has_marked_unused_vars_ = true;
    for (const auto &unused_index : unused_vars_) {
      MarkVarReady(unused_index, false);
    }
  }

  MarkVarReady(var_index, true);
}

void Reducer::MarkVarReady(const size_t var_index, const bool is_used_var) {
  groups_need_finalize_ = true;

  const auto &var_locator = variable_locators_[var_index];
  const auto group_index = var_locator.group_index;
  auto &group = groups_[group_index];

  // error happened, if the var is ready before.
  if (vars_marked_ready_[var_index]) {
    auto error_info = string::Sprintf(
        "Error happened, when parameter[%d][%s] has been ready before. "
        "Please set find_unused_parameters=True to traverse backward graph "
        "in each step to prepare reduce in advance. If you have set, "
        "there may be several reasons for this error: "
        "1) In multiple reentrant backward phase, some parameters are reused."
        "2) Using model parameters outside of forward function. Please "
        "make sure that model parameters are not shared in concurrent "
        "forward-backward passes.",
        var_index, vars_[var_index]->GradVarBase()->Name());

    PADDLE_ENFORCE_EQ(has_marked_unused_vars_, false,
                      platform::errors::PreconditionNotMet(error_info));

    error_info +=
        "3) Unused parameters retrieval is incorrect. "
        "The return value of forward will be used to retrieve"
        " the unused parameters of the entire model. These "
        "gradients of unused parameters will not be synchronized "
        "between multiple cards. However, if the unused "
        "parameters participate in the backward calculation "
        "again at a later time (e.g. after the forward function, "
        "the loss calculation uses the unused "
        "paramters of the forward and trigger backward), "
        "its gradient will be wrong.";

    PADDLE_ENFORCE_EQ(has_marked_unused_vars_, true,
                      platform::errors::PreconditionNotMet(error_info));
  } else {
    vars_marked_ready_[var_index] = true;
  }

  if (!group.is_sparse_) {
    // process dense group
    const auto inside_group_index = var_locator.inside_group_index;
    const auto length = group.length_[inside_group_index];
    auto &group_tensor = group.dense_tensors_[inside_group_index];

    if (is_used_var) {
      auto var_base = vars_[var_index]->GradVarBase();
      auto tensor = var_base->MutableVar()->GetMutable<framework::LoDTensor>();
      group_tensor.ShareDataWith(*tensor).Resize(
          {static_cast<int64_t>(length)});
    } else {
      // TODO(shenliang03): maybe save the memory
      // by avoiding tensor construction
      if (!group_tensor.IsInitialized()) {
        group_tensor.Resize({static_cast<int64_t>(length)});
        group_tensor.mutable_data(place_, group.dtype_);
      }

#ifdef PADDLE_WITH_XPU_BKCL
      if (platform::is_xpu_place(group_tensor.place())) {
        // TODO(liuyuhui) support XPU set constant
        VLOG(3) << "XPU doesn't support set_constant";
      }
#else
      auto *dev_ctx = platform::DeviceContextPool::Instance().Get(place_);
      if (HasGrad(var_index)) {
        auto var_base = vars_[var_index]->GradVarBase();
        auto tensor =
            var_base->MutableVar()->GetMutable<framework::LoDTensor>();
        TensorCopy(*tensor, place_, *dev_ctx, &group_tensor);
        group_tensor.Resize({static_cast<int64_t>(length)});
      } else {
        group_tensor.Resize({static_cast<int64_t>(length)});
        operators::math::set_constant(*dev_ctx, &group_tensor, 0.0);
      }
#endif
    }
  } else {
    // process sparse group
    PADDLE_ENFORCE_EQ(
        HasGrad(var_index), true,
        platform::errors::PreconditionNotMet(
            "The sparse parameter[%d][%s] should have gradient. "
            "Currently, DataParallel does not support sparse "
            "parameters without generating gradients during training. "
            "For example, if is_sparese=True is used in Embedding, "
            "the current step of this parameter cannot generate gradient "
            "because of stop_gradient/detatch, where error will occur.",
            var_index, vars_[var_index]->Name()));
    auto var_base = vars_[var_index]->GradVarBase();
    // need to check tensor type
    PADDLE_ENFORCE_EQ(
        var_base->Var().IsType<framework::SelectedRows>(), true,
        platform::errors::PreconditionNotMet(
            "The sparse parameter[%d][%s] must have a selectedrows gradient. "
            "Before forward pass, the parameter type is inferred to be "
            "SelectedRows, but after backward pass, its actual type becomes "
            "LodTensor. It is currently not supported by DataParallel. "
            "For example, if sparse embedding is used, and the weight of "
            "embedding is shared with subsequent dense parameters, then "
            "the parameter gradient of the embedding will be converted "
            "to dense parameters.",
            var_index, vars_[var_index]->Name()));

    group.sparse_contents_ = var_base->MutableVar();
  }

  if (--group.pending_ == 0) {
    // can start allreduce
    MarkGroupReady(group_index);
  }

  if (next_group_ == groups_.size()) {
    FinalizeBackward();
  }
}

// TODO(liuyuhui): If BKCL support non-blocking communication, it should be
// fixed as same as multi gpus card trainging.
void Reducer::MarkGroupReady(size_t group_index) {
  PADDLE_ENFORCE_GE(
      group_index, next_group_,
      platform::errors::PreconditionNotMet(
          "The index of the incoming group must be greater "
          "than or equal to the previously synchronized group index, "
          "expect it to greater than or equal to %d, but got %d.",
          next_group_, group_index));

  if (group_index > next_group_) {
    VLOG(3) << "It will adjust the order of group in next batch automatically";
    return;
  }

  for (; next_group_ < groups_.size() && groups_[next_group_].pending_ == 0;
       ++next_group_) {
    auto &group = groups_[next_group_];
    const int run_order = next_group_ % nrings_;

    // For CUDA or XPU, compute_stream --> comm_stream.
    // For CPU, do nothing.
    // NOTE. Because concat uses the comm_stream,
    // so we expose WaitCompute() interface and call
    // it here.
    parallel_ctx_->WaitCompute(run_order);
#ifdef PADDLE_WITH_XPU_BKCL
    {
      std::lock_guard<std::mutex> lock(mutex_);
      comm_op_count_ += 1;  // lock
    }
    // TODO(liuyuhui): Add try catch to deal with exception later,
    // otherwise the main thread will continue to run when an exception is
    // thrown in comm_pool_.
    auto next_group = next_group_;
    comm_pool_->enqueue([this, run_order, next_group, &group] {
      try {
        auto dev_id = BOOST_GET_CONST(platform::XPUPlace, place_).device;
        platform::SetXPUDeviceId(dev_id);
        FusedAllReduceSchedule(run_order, group, next_group);
        std::lock_guard<std::mutex> lock(mutex_);
        comm_op_count_ -= 1;  // lock
      } catch (...) {
        exception_.Catch(std::current_exception());
        std::lock_guard<std::mutex> lock(mutex_);
        comm_op_count_ = 0;  // lock
      }
      { cv_.notify_all(); }
    });
#elif defined(PADDLE_WITH_RCCL) || defined(PADDLE_WITH_NCCL)
    FusedAllReduceSchedule(run_order, group, next_group_);
#else
    PADDLE_THROW(platform::errors::PreconditionNotMet(
        "Not compiled with BKCL or NCCL."));
#endif
  }
}

void Reducer::FusedAllReduceSchedule(const int run_order, Group &group,
                                     const int curr_group_index) {
  // The overall timeline: concat > div_nranks > allreduce > split
  // dev_context is used to select different stream
  const auto &dev_context = *parallel_ctx_->GetDeviceContext(run_order);
  if (group.is_sparse_) {
    VLOG(3) << "sparse group [" << curr_group_index
            << "] start allreduce in ring[" << run_order << "]";
    group.DivNRanks(dev_context, nranks_);
    parallel_ctx_->AllReduceByStream(*group.sparse_contents_,
                                     group.sparse_contents_, run_order, false);
  } else {
    VLOG(3) << "dense group [" << curr_group_index
            << "] start allreduce in ring[" << run_order << "]";
    // Select common commstream to concat tensors
    // group.dense_tensors ---> group.dense_contents_
    group.ConcatTensors(dev_context);

// NOTE(liuyuhui): ConcatTensors use communication stream, but BKCL only support
// default stream for communicating, so there exist some problems in
// synchronization. And need to add a WaitComm there.
// TODO(liuyuhui): If BKCL support non-blocking communication, it should be
// fixed as multi gpus card trainging.
#ifdef PADDLE_WITH_XPU_BKCL
    if (platform::is_xpu_place(group.dense_tensors_[0].place())) {
      parallel_ctx_->WaitComm(run_order);
    }
#endif

    group.DivNRanks(dev_context, nranks_);
    // Start allreduce
    parallel_ctx_->AllReduceByStream(
        group.dense_contents_, &(group.dense_contents_), run_order, false);

    // Select communication stream to split tensors
    // group.dense_contents_ ---> group.dense_tensors
    group.SplitTensors(dev_context);
  }
}

std::vector<std::vector<size_t>> Reducer::RebuildGruops() {
  VLOG(3) << "The order of parameter arrival: "
          << string::join_strings(rebuild_var_indices_, ',');

  PADDLE_ENFORCE_EQ(
      rebuild_vars_.size(), vars_.size(),
      platform::errors::PreconditionNotMet(
          "Rebuild vars's number should be equal to original vars'number, "
          "expect it to be %d, but got %d.",
          vars_.size(), rebuild_vars_.size()));
  std::reverse(rebuild_vars_.begin(), rebuild_vars_.end());
  std::reverse(rebuild_var_indices_.begin(), rebuild_var_indices_.end());
  auto rebuild_group_indices =
      AssignGroupBySize(rebuild_vars_, is_sparse_gradient_, group_size_limits_,
                        rebuild_var_indices_);
  has_rebuilt_group_ = true;
  rebuild_vars_.clear();
  rebuild_var_indices_.clear();
  std::reverse(rebuild_group_indices.begin(), rebuild_group_indices.end());
  return rebuild_group_indices;
}

void Reducer::ProcessUnusedDenseVars() {
  // The calculation stream must be used here to
  // avoid conflicts with communication.
  VLOG(3) << "Local used vars : "
          << string::join_strings(local_used_vars_, ',');
  const auto *dev_ctx = platform::DeviceContextPool::Instance().Get(place_);
  // H2D is to allreduce the local_used_vars_
  auto *global_used_tensor =
      global_used_vars_.GetMutable<framework::LoDTensor>();
  framework::TensorFromVector<int>(local_used_vars_, *dev_ctx,
                                   global_used_tensor);
  parallel_ctx_->AllReduceByStream(global_used_vars_, &global_used_vars_, 0,
                                   true);
  framework::TensorToVector<int>(*global_used_tensor, *dev_ctx,
                                 &local_used_vars_);

  // sync compute stream to get global used var message,
  // but maybe affect speed performance
  parallel_ctx_->SynchronizeCompute();
  VLOG(3) << "Global used vars : "
          << string::join_strings(local_used_vars_, ',');

  for (const auto var_index : unused_vars_) {
    const bool global_unused = (local_used_vars_[var_index] == 0);

    // global used but local unused, set grad
    VLOG(3) << "Var [" << var_index << "] [" << vars_[var_index]->Name()
            << "] global_unused:" << global_unused
            << "  has grad: " << HasGrad(var_index);

    if (!global_unused) {
      VLOG(3) << "Start process unused Var";
      // 1. source var base
      const auto &var_locator = variable_locators_[var_index];
      const auto group_index = var_locator.group_index;
      const auto &group = groups_[group_index];
      const auto inside_group_index = var_locator.inside_group_index;
      const auto &src_tensor = group.dense_tensors_[inside_group_index];
      // sparse no need to check and no support find_unused_parameters
      if (group.is_sparse_) {
        continue;
      }
      // 2. destination var base
      auto dest_var_base = vars_[var_index];
      auto *dest_tensor =
          dest_var_base->MutableVar()->GetMutable<framework::LoDTensor>();
      const auto &dest_dims = dest_tensor->dims();

      // 3. create grad var base or get grad var base
      auto grad_var_base_tmp = dest_var_base->MutableGradVarBase();

      // 4. set grad tensor
      auto *dest_grad_tensor =
          grad_var_base_tmp->MutableVar()->GetMutable<framework::LoDTensor>();
      const auto *dev_ctx = platform::DeviceContextPool::Instance().Get(place_);
      TensorCopy(src_tensor, place_, *dev_ctx, dest_grad_tensor);
      dest_grad_tensor->Resize(dest_dims);
    }
  }
}

bool Reducer::HasGrad(size_t var_index) {
  const auto grad_var = vars_[var_index]->GradVarBase();
  if (!grad_var || !grad_var->Var().IsInitialized()) {
    return false;
  }

  const auto &var = grad_var->Var();
  if (var.IsType<framework::LoDTensor>()) {
    if (var.Get<framework::LoDTensor>().IsInitialized()) {
      return true;
    }
  } else if (var.IsType<framework::SelectedRows>()) {
    if (var.Get<framework::SelectedRows>().value().IsInitialized()) {
      return true;
    }
  } else {
    PADDLE_THROW(platform::errors::PermissionDenied(
        "Only support LoDTensor and SelectedRows for gradient var"));
  }
  return false;
}

void Reducer::FinalizeBackward() {
  groups_need_finalize_ = false;
#ifdef PADDLE_WITH_XPU_BKCL
  {
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait(lock, [&] { return comm_op_count_ == 0; });
  }
  if (exception_.IsCaught()) {
    VLOG(3) << "caught exception " << exception_.Type() << ", rethrow it";
    exception_.ReThrow();
  }
#endif

  // Must prevent compute_stream_ starting until all comm streams have finished
  for (int i = 0; i < nrings_; ++i) {
    parallel_ctx_->WaitComm(i);
  }

  if (NeedRebuildGroup()) {
    VLOG(3) << "Start rebuilding the groups";
    auto rebuild_group_indices = RebuildGruops();
    group_indices_ = std::move(rebuild_group_indices);
    InitializeGroups(group_indices_);
  }

  if (find_unused_vars_each_step_) {
// TODO(liuyuhui) support xpu about Tensorcopy/TensorFromVector/TensorToVector
#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
    ProcessUnusedDenseVars();
#endif
    // Initialize local used vars
    local_used_vars_.clear();
    local_used_vars_.resize(vars_.size(), 0);
    VLOG(3) << "ProcessUnusedDenseVars is finished.";
  }

  VLOG(3) << "In the batch, Reducer is finished.";
}

// According to the size of each parameter, it is allocated to different groups.
// The sparse parameter occupies a group exclusively. The dense parameters of
// the same data type are assigned to the same group. When dividing groups, the
// size of each group will be limited according to each value in
// group_size_limits in turn. When it is not enough, it will be divided
// by the last value of group_size_limits. The limit value is 0, which
// means that the parameter will monopolize the group.
std::vector<std::vector<size_t>> AssignGroupBySize(
    const std::vector<std::shared_ptr<imperative::VarBase>> &vars,
    const std::vector<bool> &is_sparse_gradient,
    const std::vector<size_t> &group_size_limits,
    const std::vector<int64_t> &tensor_indices) {
  PADDLE_ENFORCE_EQ(vars.size(), is_sparse_gradient.size(),
                    platform::errors::PreconditionNotMet(
                        "vars len must be equal to is_sparse_gradient len, but "
                        "[%lu] != [%lu]",
                        vars.size(), is_sparse_gradient.size()));
  auto check_perm = [](const std::vector<int64_t> &x) -> bool {
    size_t len = x.size();
    std::vector<size_t> cnt(len, 0);
    for (size_t i = 0; i < len; ++i) {
      if (x[i] >= static_cast<int64_t>(len) || x[i] < 0 || cnt[x[i]]) {
        return false;
      }
      cnt[x[i]]++;
    }
    return true;
  };
  PADDLE_ENFORCE_EQ(true, check_perm(tensor_indices),
                    platform::errors::PreconditionNotMet(
                        "tensor_indices must be a permutation from 0 to %lu",
                        tensor_indices.size()));
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

    size_t tensor_real_index = i;
    if (!tensor_indices.empty()) {
      tensor_real_index = tensor_indices[i];
    }

    if (is_sparse_gradient[tensor_real_index]) {
      // we keep sparse var a single group
      res.push_back({tensor_real_index});
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
    } else {
      VLOG(3) << "var " << var->Name()
              << " is not tensor or selected_rows, so skip it";
      continue;
    }
    group_info.first.push_back(tensor_real_index);
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
            "AssignGroupBySize construct empty group, please check."));
  }
  if (tensor_indices.empty()) {
    std::sort(res.begin(), res.end(),
              [](const std::vector<size_t> &x, const std::vector<size_t> &y) {
                return x.front() < y.front();
              });
  }
  return res;
}
#endif

}  // namespace imperative
}  // namespace paddle
