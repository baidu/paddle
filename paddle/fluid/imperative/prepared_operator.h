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

#pragma once
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "paddle/fluid/framework/data_transform.h"
#include "paddle/fluid/framework/op_kernel_type.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/imperative/execution_context.h"
#include "paddle/fluid/imperative/layer.h"
#include "paddle/fluid/imperative/type_defs.h"

DECLARE_bool(use_mkldnn);

namespace paddle {
namespace framework {
class Tensor;
class Variable;
}  // namespace framework
namespace platform {
class DeviceContext;
}  // namespace platform
}  // namespace paddle

namespace paddle {
namespace imperative {

const framework::Tensor* GetTensorFromVar(const framework::Variable& var);

template <typename VarType>
framework::OpKernelType GetExpectedKernelKey(
    const NameVarMap<VarType>& ins, const NameVarMap<VarType>& outs,
    const framework::OperatorWithKernel& op, const platform::Place& place,
    const framework::AttributeMap& attrs) {
  platform::DeviceContextPool& pool = platform::DeviceContextPool::Instance();
  auto* dev_ctx = pool.Get(place);
  framework::RuntimeContext ctx({}, {});

#ifdef PADDLE_WITH_MKLDNN
  // MKLDNN variant of code reads attributes in some of GetKernelTypeForVar and
  // GetKernelType functions, so we need to copy the attributes there.
  // Const qualifier of Attrs had to be discarded to overwrite it.
  if (FLAGS_use_mkldnn) {
    auto& mutable_op_attrs = const_cast<framework::AttributeMap&>(op.Attrs());
    mutable_op_attrs = attrs;
  }
#endif

  auto expected_kernel_key =
      op.GetExpectedKernelType(DygraphExecutionContext<VarType>(
          op, framework::Scope(), *dev_ctx, ctx, ins, outs, attrs));
  VLOG(3) << "expected_kernel_key:" << expected_kernel_key;

  return expected_kernel_key;
}

template <typename VarType>
NameVarMap<VarType> PrepareData(
    const framework::OperatorWithKernel& op, const NameVarMap<VarType>& ins,
    const framework::OpKernelType& expected_kernel_key) {
  NameVarMap<VarType> tmp_ins(ins);
  for (auto& name_pair : tmp_ins) {
    for (auto& var_base : name_pair.second) {
      const auto* tensor = GetTensorFromVar(var_base->Var());
      if (tensor && tensor->IsInitialized()) {
        auto kernel_type_for_var = op.GetKernelTypeForVar(
            name_pair.first, *tensor, expected_kernel_key);
        if (!NeedTransform(kernel_type_for_var, expected_kernel_key)) {
          continue;
        } else {
          VLOG(3) << "Transform Variable " << var_base->Name() << " from "
                  << kernel_type_for_var << " to " << expected_kernel_key;
          framework::Tensor out;
          auto tmp_var = std::make_shared<VarType>(var_base->Name());
          tmp_var->SetType(var_base->Type());
          TransformData(expected_kernel_key, kernel_type_for_var, *tensor,
                        &out);
          SetTensorToVariable(var_base->Var(), out, tmp_var->MutableVar());
          var_base = tmp_var;
        }
      }
    }
  }
  return tmp_ins;
}

class PreparedOp {
 public:
  PreparedOp(const framework::OperatorBase& op,
             const framework::RuntimeContext& ctx,
             const framework::OperatorWithKernel::OpKernelFunc& func,
             platform::DeviceContext* dev_ctx);

  static PreparedOp Prepare(const framework::OperatorWithKernel& op,
                            const framework::OpKernelType& expected_kernel_key);

  void Run(const NameVarMap<VarBase>& in, const NameVarMap<VarBase>& out,
           const framework::AttributeMap& attrs);

  void Run(const NameVarMap<VariableWrapper>& ins,
           const NameVarMap<VariableWrapper>& outs,
           const framework::AttributeMap& attrs);

 private:
  const framework::OperatorBase& op_;
  const framework::RuntimeContext& ctx_;
  framework::OperatorWithKernel::OpKernelFunc func_;
  platform::DeviceContext* dev_ctx_;
};

}  // namespace imperative
}  // namespace paddle
