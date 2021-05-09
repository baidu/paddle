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

#include "paddle/fluid/framework/op_desc.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/parallel_executor.h"
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
    auto *var = scope->Var(var_names[i]);
    CheckInputVarStatus(*vars[i], var_names[i]);
    VariableShare(*vars[i], var);
  }
}

static void ShareVarsFromScope(const std::vector<Variable *> &vars,
                               const std::vector<std::string> &var_names,
                               framework::Scope *scope) {
  for (size_t i = 0; i < vars.size(); ++i) {
    if (var_names[i] == framework::kEmptyVarName) {
      VLOG(2) << "find variable name is " << framework::kEmptyVarName
              << ", skip it!";
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
    VLOG(3) << "Found " << var_names[i] << " in " << scope << ", it is " << var;
    CheckOutputVarStatus(*var, *vars[i], var_names[i]);
    VLOG(3) << "checkout done";
    VariableShare(*var, vars[i]);
  }
}
/*
 * NOTE(Aurelius84): In ParallelExecutor, memory optimized pass will be applied.
 * To
 * avoid eagerly deleting last alive variables which are necessary in backward
 * program,
 * we firstly parse these variable names as skip_eager_vars. While executing
 * pe.run
 * skip_eager_vars are used to skip memory optimization.
 *
 * Variables satisfying the following rules are considered as skip_eager_var:
 *
 *   1. it is an output var in run_program_op
 *   2. it is an input var used in backward_op
 */
static std::vector<std::string> ParseVarNamesNeededInBackward(
    const ProgramDesc &program, int64_t forward_op_nums,
    const std::vector<std::string> &output_var_names) {
  // step 1: all out_vars are skip_eager_var
  for (auto &var_name : output_var_names) {
    VLOG(3) << "skip output var name: " << var_name;
  }
  std::vector<std::string> skip_eager_vars(output_var_names);
  std::unordered_set<std::string> visited_vars;

  auto all_ops = program.Block(0).AllOps();
  // Between forward and backward op, there are many fill_constant_ops
  size_t backward_op_start_index =
      forward_op_nums + (output_var_names.size() * 2);

  for (auto i = backward_op_start_index; i < all_ops.size(); ++i) {
    framework::OpDesc *op = all_ops[i];
    VLOG(3) << "Parse skip_eager_var with op: " << op->Type();
    for (auto &in_var_name : op->InputArgumentNames()) {
      if (!visited_vars.count(in_var_name)) {
        skip_eager_vars.push_back(in_var_name);
        visited_vars.insert(in_var_name);
      }
    }
  }
  VLOG(3) << "Found skip_eager_vars: " << skip_eager_vars.size();

  return skip_eager_vars;
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

    auto input_var_names = ctx.InputNames("X");
    auto output_var_names = ctx.OutputNames("Out");

    // current program may not hold parameters
    std::vector<std::string> param_names;
    if (!param_vars.empty()) {
      param_names = ctx.InputNames("Params");
    }

    auto start_op_index = ctx.Attr<int64_t>("start_op_index");
    auto end_op_index = ctx.Attr<int64_t>("end_op_index");
    auto is_test = ctx.Attr<bool>("is_test");

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
      framework::BuildStrategy build_strategy;
      framework::ExecutionStrategy execution_strategy;
      execution_strategy.skip_drop_scope_ = true;
      if (platform::is_cpu_place(ctx.GetPlace())) {
        execution_strategy.use_device_ = platform::DeviceType::CPU;
        execution_strategy.num_threads_ = 2;
      } else if (platform::is_gpu_place(ctx.GetPlace())) {
        execution_strategy.use_device_ = platform::DeviceType::CUDA;
        execution_strategy.num_threads_ = 4;
      } else {
        execution_strategy.use_device_ = platform::DeviceType::XPU;
        execution_strategy.num_threads_ = 1;
      }
      auto *program = ctx.Attr<BlockDesc *>("global_block")->Program();
      auto graph = std::make_unique<framework::ir::Graph>(
          *program, start_op_index, end_op_index);

      framework::ParallelExecutor pe(ctx.GetPlace(),      // places
                                     &scope,              // global_scope
                                     execution_strategy,  // exec_strategy
                                     build_strategy,      // build_strategy
                                     graph.get());

      // TODO(Aurelius84): make it only call once
      pe.SkipMemoryReuse(/*scope_idx*/ 0, input_var_names);

      // Step 3. run ops
      VLOG(3) << "start to call pe.Run()...";
      auto skip_eager_vars = details::ParseVarNamesNeededInBackward(
          *program, end_op_index, output_var_names);
      pe.RunWithoutFetch(skip_eager_vars);

      VLOG(3) << "successfully to call pe.Run()...";
    }

    // Step 4. Get Output
    details::ShareVarsFromScope(output_vars, output_var_names, &scope);

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
      framework::BuildStrategy build_strategy;
      framework::ExecutionStrategy execution_strategy;
      execution_strategy.skip_drop_scope_ = true;
      if (platform::is_cpu_place(ctx.GetPlace())) {
        execution_strategy.use_device_ = platform::DeviceType::CPU;
        execution_strategy.num_threads_ = 2;
      } else if (platform::is_gpu_place(ctx.GetPlace())) {
        execution_strategy.use_device_ = platform::DeviceType::CUDA;
        execution_strategy.num_threads_ = 4;
      } else {
        execution_strategy.use_device_ = platform::DeviceType::XPU;
        execution_strategy.num_threads_ = 1;
      }

      auto *program = ctx.Attr<BlockDesc *>("global_block")->Program();
      auto graph = std::make_unique<framework::ir::Graph>(
          *program, start_op_index, end_op_index);

      framework::ParallelExecutor pe(ctx.GetPlace(),      // places
                                     &scope,              // global_scope
                                     execution_strategy,  // exec_strategy
                                     build_strategy,      // build_strategy
                                     graph.get());

      // TODO(Aurelius84): make it only call once
      pe.SkipMemoryReuse(/*scope_idx*/ 0, output_grad_var_names);

      details::ShareVarsIntoScope(output_grad_vars, output_grad_var_names,
                                  &scope);
      // Debug info: scope info when run end
      VLOG(3) << framework::GenScopeTreeDebugInfo(out_scope_vec->front());

      std::vector<std::string> skip_eager_vars(input_grad_var_names);
      skip_eager_vars.insert(skip_eager_vars.end(), param_grad_names.begin(),
                             param_grad_names.end());

      // Step 3. run ops
      // TODO(Aurelius84): Skip fetch_tensor overhead and just run all ops.
      VLOG(3) << "start to call pe.Run()...";
      pe.RunWithoutFetch(/*skip_eager_vars*/ skip_eager_vars);
      VLOG(3) << "successfully to call pe.Run()...";
    }

    // Step 4. get outputs
    details::ShareVarsFromScope(input_grad_vars, input_grad_var_names, &scope);
    details::ShareVarsFromScope(param_grad_vars, param_grad_names, &scope);

    // Step5. drop current scope
    global_inner_scope->DeleteScope(&scope);
    VLOG(2) << "The number of sub scopes after backward: "
            << global_inner_scope->kids().size();
  }
};

}  // namespace operators
}  // namespace paddle
