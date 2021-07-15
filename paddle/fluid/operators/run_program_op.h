/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

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

#include <algorithm>
#include <iterator>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "paddle/fluid/framework/executor_cache.h"
#include "paddle/fluid/framework/op_desc.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/framework/var_type_traits.h"
#include "paddle/fluid/framework/variable.h"
#ifdef PADDLE_WITH_MKLDNN
#include "paddle/fluid/platform/mkldnn_helper.h"
#endif

DECLARE_bool(use_mkldnn);

namespace paddle {
namespace operators {

using StepScopeVar = std::vector<framework::Scope *>;
using BlockDesc = framework::BlockDesc;
using ProgramDesc = framework::ProgramDesc;

using Variable = framework::Variable;
using LoDTensor = framework::LoDTensor;
using SelectedRows = framework::SelectedRows;

namespace details {

// all input vars should be LoDTensor & is initialized
static void CheckInputVarStatus(const Variable &var,
                                const std::string &var_name) {
  PADDLE_ENFORCE_EQ(
      var.IsType<LoDTensor>(), true,
      platform::errors::InvalidArgument(
          "The input variable %s of "
          "RunProgram(Grad)Op holds "
          "wrong type. Expect type is LoDTensor, but receive type is %s.",
          var_name, platform::demangle(framework::ToTypeName(var.Type()))));
  PADDLE_ENFORCE_EQ(
      var.Get<LoDTensor>().IsInitialized(), true,
      platform::errors::InvalidArgument("The tensor in input variable %s of "
                                        "RunProgram(Grad)Op "
                                        "is not initialized.",
                                        var_name));
}

static void CheckOutputVarStatus(const Variable &src_var,
                                 const Variable &dst_var,
                                 const std::string &var_name) {
  if (dst_var.IsType<LoDTensor>()) {
    PADDLE_ENFORCE_EQ(
        src_var.IsType<LoDTensor>(), true,
        platform::errors::InvalidArgument(
            "The output variable %s get from "
            "RunProgram(Grad)Op's internal scope holds "
            "wrong type. Expect type is LoDTensor, but receive type is %s.",
            var_name,
            platform::demangle(framework::ToTypeName(src_var.Type()))));
    PADDLE_ENFORCE_EQ(src_var.Get<LoDTensor>().IsInitialized(), true,
                      platform::errors::InvalidArgument(
                          "The tensor in output variable %s get from "
                          "RunProgram(Grad)Op's internal "
                          "scope is not initialized.",
                          var_name));
  } else if (dst_var.IsType<SelectedRows>()) {
    PADDLE_ENFORCE_EQ(
        src_var.IsType<SelectedRows>(), true,
        platform::errors::InvalidArgument(
            "The output variable %s get from "
            "RunProgram(Grad)Op's internal scope holds "
            "wrong type. Expect type is SelectedRows, but receive type is %s.",
            var_name,
            platform::demangle(framework::ToTypeName(src_var.Type()))));
    PADDLE_ENFORCE_EQ(src_var.Get<SelectedRows>().value().IsInitialized(), true,
                      platform::errors::InvalidArgument(
                          "The tensor in output variable %s get from "
                          "RunProgram(Grad)Op's "
                          "internal scope is not initialized.",
                          var_name));

  } else {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "The RunProgram(Grad)Op only support output "
        "variable of type LoDTensor or SelectedRows, "
        "but received variable %s's type is %s",
        var_name, platform::demangle(framework::ToTypeName(dst_var.Type()))));
  }
}

static void VariableShare(const Variable &src_var, Variable *dst_var) {
  // The previous check ensures that the variable type can only be LoDTensor or
  // SelectedRows.
  if (src_var.IsType<LoDTensor>()) {
    auto *lod_tensor = dst_var->GetMutable<LoDTensor>();
    lod_tensor->ShareDataWith(src_var.Get<LoDTensor>());
    lod_tensor->set_lod(src_var.Get<LoDTensor>().lod());
  } else if (src_var.IsType<SelectedRows>()) {
    auto *selected_rows = dst_var->GetMutable<SelectedRows>();
    selected_rows->mutable_value()->ShareDataWith(
        src_var.Get<SelectedRows>().value());
    selected_rows->set_rows(src_var.Get<SelectedRows>().rows());
    selected_rows->set_height(src_var.Get<SelectedRows>().height());
  }
}

static void ShareVarsIntoScope(const std::vector<Variable *> &vars,
                               const std::vector<std::string> &var_names,
                               framework::Scope *scope) {
  for (size_t i = 0; i < vars.size(); ++i) {
    if (var_names[i] == "Fake_var") {
      continue;
    }
    auto *var = scope->Var(var_names[i]);
    CheckInputVarStatus(*vars[i], var_names[i]);
    VariableShare(*vars[i], var);
  }
}

static void ShareVarsFromScope(const std::vector<Variable *> &vars,
                               const std::vector<std::string> &var_names,
                               framework::Scope *scope) {
  for (size_t i = 0; i < vars.size(); ++i) {
    if (var_names[i] == framework::kEmptyVarName ||
        var_names[i] == "Fake_var") {
      VLOG(2) << "find variable name is " << var_names[i] << ", skip it!";
      continue;
    }
    // NOTE: Here skip not found var is dangerous, if a bug is caused here,
    // the result is grad calculation error, which will be very hidden!
    auto *var = scope->FindVar(var_names[i]);
    PADDLE_ENFORCE_NOT_NULL(
        var, platform::errors::NotFound("The output variable %s is not in "
                                        "RunProgram(Grad)Op'"
                                        "s internal scope.",
                                        var_names[i]));
    CheckOutputVarStatus(*var, *vars[i], var_names[i]);
    VariableShare(*var, vars[i]);
  }
}

}  // namespace details

template <typename DeviceContext, typename T>
class RunProgramOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    VLOG(2) << "RunProgramOpKernel Compute";
    // Step 1. prepare inputs, outputs, attrs
    auto &input_vars = ctx.MultiInputVar("X");
    auto &param_vars = ctx.MultiInputVar("Params");
    auto output_vars = ctx.MultiOutputVar("Out");
    auto dout_vars = ctx.MultiOutputVar("DOut");

    auto input_var_names = ctx.InputNames("X");
    auto output_var_names = ctx.OutputNames("Out");
    auto dout_var_names = ctx.OutputNames("DOut");

    // current program may not hold parameters
    std::vector<std::string> param_names;
    if (!param_vars.empty()) {
      param_names = ctx.InputNames("Params");
    }

    auto start_op_index = ctx.Attr<int64_t>("start_op_index");
    auto end_op_index = ctx.Attr<int64_t>("end_op_index");
    auto is_test = ctx.Attr<bool>("is_test");
    auto program_id = ctx.Attr<int64_t>("program_id");

    // NOTE(chenweihang): In order not to add new variable type, use vector
    // here. Originally, here can use scope directly.
    auto *out_scope_vec = ctx.Output<StepScopeVar>("OutScope");
    PADDLE_ENFORCE_EQ(
        out_scope_vec->size(), 1,
        platform::errors::InvalidArgument(
            "The OutScope of RunProgramGradOp should only hold one scope."));

    // Step 2. prepare executor and init persistable variables

    // NOTE(Aurelius84): While training some models, forward can be called many
    // times and then apply backpropagation all at once, such as Reinforcement
    // Learning. Tensor data in multi-step training should be saved into single
    // scope separately. Otherwise, the gradients can be miscalculated because
    // always using the Tensor data of the last step in forward.
    framework::Scope *global_inner_scope = out_scope_vec->front();
    VLOG(2) << "The number of sub scopes before forward: "
            << out_scope_vec->front()->kids().size();
    framework::Scope &scope = global_inner_scope->NewScope();

    // share input_vars & parameters into scope
    details::ShareVarsIntoScope(input_vars, input_var_names, &scope);
    details::ShareVarsIntoScope(param_vars, param_names, &scope);

    if (end_op_index > start_op_index) {
      auto *program = ctx.Attr<BlockDesc *>("global_block")->Program();
      auto cache_info = framework::GetExecutorInfoFromCache(
          program, ctx.GetPlace(), start_op_index, end_op_index,
          /*is_grad=*/false, program_id, &scope);
      auto &parallel_executor = cache_info.first;

      // all out_vars are skip_eager_var
      auto &skip_eager_delete_vars =
          framework::ExecutorInfoCache::Instance().SkipEagerDeleteVars(
              program_id, false);
      if (cache_info.second /*is_new_created*/) {
        parallel_executor->SkipMemoryReuse(/*scope_idx=*/0, input_var_names);
        skip_eager_delete_vars.insert(skip_eager_delete_vars.end(),
                                      output_var_names.begin(),
                                      output_var_names.end());
        skip_eager_delete_vars.insert(skip_eager_delete_vars.end(),
                                      dout_var_names.begin(),
                                      dout_var_names.end());
        framework::details::ParseSafeEagerDeletionSkipVars(
            *program, end_op_index, output_var_names, &skip_eager_delete_vars);
      }

      // Step 3. run ops
      parallel_executor->RunWithoutFetch(skip_eager_delete_vars);
    }
    // Step 4. Get Output
    details::ShareVarsFromScope(output_vars, output_var_names, &scope);
    details::ShareVarsFromScope(dout_vars, dout_var_names, &scope);

    // Debug info: scope info when run end
    VLOG(3) << framework::GenScopeTreeDebugInfo(out_scope_vec->front());
    // Step 5. Drop all children scopes while testing.
    if (is_test) {
      out_scope_vec->front()->DropKids();
    }
    VLOG(2) << "The number of sub scopes after forward: "
            << out_scope_vec->front()->kids().size();
#ifdef PADDLE_WITH_MKLDNN
    if (FLAGS_use_mkldnn) DontClearMKLDNNCache(ctx.GetPlace());
#endif
  }
};

template <typename DeviceContext, typename T>
class RunProgramGradOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    VLOG(2) << "RunProgramGradOpKernel Compute";
    // Step 1. prepare inputs and outputs
    auto &output_grad_vars = ctx.MultiInputVar(framework::GradVarName("Out"));
    auto input_grad_vars = ctx.MultiOutputVar(framework::GradVarName("X"));
    auto param_grad_vars = ctx.MultiOutputVar(framework::GradVarName("Params"));

    // if all output vars are set to stop_gradient, grad op no need to executed
    if (input_grad_vars.empty() && param_grad_vars.empty()) return;

    auto output_grad_var_names = ctx.InputNames(framework::GradVarName("Out"));
    // NOTE: after PR22939 [Add double grad] merged, the grad op maker's
    //   SetOutput will set to None if the input var stop_gradient=True,
    //   it will cause an NotFound error when ctx.OutputNames() is called
    std::vector<std::string> input_grad_var_names;
    std::vector<std::string> param_grad_names;
    if (!input_grad_vars.empty()) {
      input_grad_var_names = ctx.OutputNames(framework::GradVarName("X"));
    }
    if (!param_grad_vars.empty()) {
      param_grad_names = ctx.OutputNames(framework::GradVarName("Params"));
    }

    auto *block = ctx.Attr<BlockDesc *>("global_block");
    auto orig_end_op_index = ctx.Attr<int64_t>("end_op_index");
    auto program_id = ctx.Attr<int64_t>("program_id");
    // NOTE: skip `shape` and `fill_constant` op created by
    // fluid.backward.gradients, one forward output will generate one `shape`
    // and `fill_constant`
    int64_t start_op_index = orig_end_op_index + (output_grad_vars.size() * 2);
    int64_t end_op_index = block->OpSize();

    auto *out_scope_vec = ctx.Input<StepScopeVar>("OutScope");
    PADDLE_ENFORCE_EQ(
        out_scope_vec->size(), 1,
        platform::errors::InvalidArgument(
            "The OutScope of RunProgramGradOp should only hold one scope."));

    framework::Scope *global_inner_scope = out_scope_vec->front();
    auto sub_scope_num = global_inner_scope->kids().size();
    VLOG(2) << "The number of sub scopes before backward: " << sub_scope_num;
    PADDLE_ENFORCE_GT(sub_scope_num, 0,
                      platform::errors::InvalidArgument(
                          "The OutScope of RunProgramGradOp should hold at "
                          "least one sub scope."));

    auto &scope = *(global_inner_scope->kids().front());

    if (end_op_index > start_op_index) {
      // Step 2. prepare executor and scope
      auto *program = ctx.Attr<BlockDesc *>("global_block")->Program();
      auto cache_info = framework::GetExecutorInfoFromCache(
          program, ctx.GetPlace(), start_op_index, end_op_index,
          /*is_grad*/ true, program_id, &scope);
      auto &parallel_executor = cache_info.first;

      auto &skip_eager_delete_vars =
          framework::ExecutorInfoCache::Instance().SkipEagerDeleteVars(
              program_id, true);
      if (cache_info.second /*is_new_created*/) {
        parallel_executor->SkipMemoryReuse(/*scope_idx=*/0,
                                           output_grad_var_names);

        skip_eager_delete_vars.insert(skip_eager_delete_vars.end(),
                                      input_grad_var_names.begin(),
                                      input_grad_var_names.end());
        framework::details::AppendSkipDeletionVars(param_grad_names,
                                                   &skip_eager_delete_vars);
      }

      details::ShareVarsIntoScope(output_grad_vars, output_grad_var_names,
                                  &scope);
      // Debug info: scope info when run end
      VLOG(3) << framework::GenScopeTreeDebugInfo(out_scope_vec->front());

      // Step 3. run ops
      parallel_executor->RunWithoutFetch(
          /*skip_eager_delete_vars=*/skip_eager_delete_vars);
    }

    // Step 4. get outputs
    details::ShareVarsFromScope(input_grad_vars, input_grad_var_names, &scope);
    details::ShareVarsFromScope(param_grad_vars, param_grad_names, &scope);

    // Step5. drop current scope
    global_inner_scope->DeleteScope(&scope);
    VLOG(2) << "The number of sub scopes after backward: "
            << global_inner_scope->kids().size();
