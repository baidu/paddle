// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/imperative/prepared_operator.h"

#include <sstream>

#include "paddle/fluid/imperative/execution_context.h"
#include "paddle/fluid/imperative/infer_shape_context.h"
#include "paddle/fluid/imperative/infer_var_type_context.h"

namespace paddle {
namespace imperative {

const framework::Tensor* GetTensorFromVar(const framework::Variable& var) {
  if (var.IsType<framework::LoDTensor>()) {
    return &(var.Get<framework::LoDTensor>());
  } else if (var.IsType<framework::SelectedRows>()) {
    return &(var.Get<framework::SelectedRows>().value());
  } else {
    return nullptr;
  }
}

template <typename VarType>
static void PrepareData(const platform::Place& place,
                        const NameVarMap<VarType>& ins,
                        const framework::OperatorWithKernel& op,
                        const framework::OpKernelType& expected_kernel_key) {
  for (const auto& name_pair : ins) {
    for (const auto& var_base : name_pair.second) {
      const auto* tensor = GetTensorFromVar(var_base->Var());
      if (tensor && tensor->IsInitialized()) {
        auto tmp_place = tensor->place();

        // TODO(jiabin): Support transform data layout when we Verify it on more
        // tests
        if (!(tmp_place == place)) {
          auto kernel_type_for_var = op.GetKernelTypeForVar(
              name_pair.first, *tensor, expected_kernel_key);
          if (!NeedTransform(kernel_type_for_var, expected_kernel_key)) {
            continue;
          } else {
            VLOG(3) << "Transform Variable " << var_base->Name() << " from "
                    << kernel_type_for_var << " to " << expected_kernel_key;
            framework::Tensor out;
            TransformData(expected_kernel_key, kernel_type_for_var, *tensor,
                          &out);
            SetTensorToVariable(var_base->Var(), out, var_base->MutableVar());
          }
        }
      }
    }
  }
}

PreparedOp::PreparedOp(const framework::OperatorBase& op,
                       const framework::RuntimeContext& ctx,
                       const framework::OperatorWithKernel::OpKernelFunc& func,
                       platform::DeviceContext* dev_ctx)
    : op_(op), ctx_(ctx), func_(func), dev_ctx_(dev_ctx) {}

template <typename VarType>
PreparedOp PrepareOpImpl(const NameVarMap<VarType>& ins,
                         const NameVarMap<VarType>& outs,
                         const framework::OperatorWithKernel& op,
                         platform::Place place,
                         const framework::AttributeMap& attrs) {
  platform::DeviceContextPool& pool = platform::DeviceContextPool::Instance();
  auto* dev_ctx = pool.Get(place);

  // check if op[type] has kernel registered.
  auto& all_op_kernels = op.AllOpKernels();
  auto kernels_iter = all_op_kernels.find(op.Type());

  PADDLE_ENFORCE_NE(
      kernels_iter, all_op_kernels.end(),
      platform::errors::NotFound(
          "There are no kernels which are registered in the %s operator.",
          op.Type()));

  auto& kernels = kernels_iter->second;

  framework::RuntimeContext ctx({}, {});
  auto expected_kernel_key =
      op.GetExpectedKernelType(DygraphExecutionContext<VarType>(
          op, framework::Scope(), *dev_ctx, ctx, ins, outs, attrs));
  VLOG(3) << "expected_kernel_key:" << expected_kernel_key;

  auto kernel_iter = kernels.find(expected_kernel_key);
  // TODO(jiabin): Add operator.cc's line 1000 part back when we need that case
  PADDLE_ENFORCE_NE(kernel_iter, kernels.end(),
                    platform::errors::NotFound(
                        "Operator %s does not have kernel for %s.", op.Type(),
                        KernelTypeToString(expected_kernel_key)));

  if (!(expected_kernel_key.place_ == place)) {
    dev_ctx = pool.Get(expected_kernel_key.place_);
    place = dev_ctx->GetPlace();
  }

  PrepareData<VarType>(place, ins, op, expected_kernel_key);
  return PreparedOp(op, ctx, kernel_iter->second, dev_ctx);
}

PreparedOp PreparedOp::Prepare(const NameVarMap<VarBase>& ins,
                               const NameVarMap<VarBase>& outs,
                               const framework::OperatorWithKernel& op,
                               const platform::Place& place,
                               const framework::AttributeMap& attrs) {
  return PrepareOpImpl<VarBase>(ins, outs, op, place, attrs);
}

PreparedOp PreparedOp::Prepare(const NameVarMap<VariableWrapper>& ins,
                               const NameVarMap<VariableWrapper>& outs,
                               const framework::OperatorWithKernel& op,
                               const platform::Place& place,
                               const framework::AttributeMap& attrs) {
  return PrepareOpImpl<VariableWrapper>(ins, outs, op, place, attrs);
}

template <typename VarType>
static void PreparedOpRunImpl(
    const framework::OperatorBase& op, const framework::RuntimeContext& ctx,
    const framework::OperatorWithKernel::OpKernelFunc& func,
    platform::DeviceContext* dev_ctx, const NameVarMap<VarType>& ins,
    const NameVarMap<VarType>& outs, const framework::AttributeMap& attrs) {
  // TODO(zjl): remove scope in dygraph
  framework::Scope scope;

  DygraphInferShapeContext<VarType> infer_shape_ctx(&ins, &outs, &attrs,
                                                    op.Type());
  static_cast<const framework::OperatorWithKernel&>(op).InferShape(
      &infer_shape_ctx);

  func(DygraphExecutionContext<VarType>(op, scope, *dev_ctx, ctx, ins, outs,
                                        attrs));
}

void PreparedOp::Run(const NameVarMap<VarBase>& ins,
                     const NameVarMap<VarBase>& outs,
                     const framework::AttributeMap& attrs) {
  PreparedOpRunImpl<VarBase>(op_, ctx_, func_, dev_ctx_, ins, outs, attrs);
}

void PreparedOp::Run(const NameVarMap<VariableWrapper>& ins,
                     const NameVarMap<VariableWrapper>& outs,
                     const framework::AttributeMap& attrs) {
  PreparedOpRunImpl<VariableWrapper>(op_, ctx_, func_, dev_ctx_, ins, outs,
                                     attrs);
}

}  // namespace imperative
}  // namespace paddle
