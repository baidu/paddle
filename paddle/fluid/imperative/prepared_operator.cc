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

#include "paddle/fluid/framework/data_type_transform.h"
#include "paddle/fluid/imperative/infer_shape_context.h"
uint64_t opbase_run_2 = 0;
uint64_t opbase_run_3 = 0;
// extern uint64_t opbase_run_2;
// extern uint64_t opbase_run_3;

namespace paddle {
namespace imperative {

const std::shared_ptr<VariableWrapper>& GetVariableWrapper(
    const std::shared_ptr<paddle::imperative::VarBase>& var) {
  return var->SharedVar();
}

const std::shared_ptr<VariableWrapper>& GetVariableWrapper(
    const std::shared_ptr<VariableWrapper>& var) {
  return var;
}

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
static void HandleComplexGradToRealGrad(const NameVarMap<VarType>& outs) {
  for (auto& pair : outs) {
    for (auto& var : pair.second) {
      if (var == nullptr) {
        continue;
      }
      if (var->ForwardDataType() ==
          static_cast<framework::proto::VarType::Type>(-1)) {
        VLOG(6) << "Var (" << var->Name()
                << ")'s forward data type is not set.";
        continue;
      }
      if (!framework::IsComplexType(var->DataType()) ||
          framework::IsComplexType(var->ForwardDataType())) {
        continue;
      }
      const auto* tensor = GetTensorFromVar(var->Var());
      if (tensor && tensor->IsInitialized()) {
        VLOG(6) << "Transform " << framework::DataTypeToString(var->DataType())
                << " var `" << var->Name() << "` to "
                << framework::DataTypeToString(var->ForwardDataType())
                << " real var in dynamic graph.";
        framework::Tensor out;
        framework::TransComplexToReal(var->ForwardDataType(), var->DataType(),
                                      *tensor, &out);
        SetTensorToVariable(var->Var(), out, var->MutableVar());
      }
    }
  }
}

PreparedOp::PreparedOp(const framework::OperatorBase& op,
                       const framework::RuntimeContext& ctx,
                       const framework::OpKernelType& kernel_type,
                       const framework::OperatorWithKernel::OpKernelFunc& func,
                       platform::DeviceContext* dev_ctx)
    : op_(op),
      ctx_(ctx),
      kernel_type_(kernel_type),
      func_(func),
      dev_ctx_(dev_ctx) {}

template <typename VarType>
PreparedOp PrepareImpl(const NameVarMap<VarType>& ins,
                       const NameVarMap<VarType>& outs,
                       const framework::OperatorWithKernel& op,
                       const platform::Place& place,
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

  // 1. get expected kernel key
  auto expected_kernel_key =
      op.GetExpectedKernelType(DygraphExecutionContext<VarType>(
          op, framework::Scope(), *dev_ctx, ctx, ins, outs, attrs));
  VLOG(3) << "expected_kernel_key:" << expected_kernel_key;

  // 2. check if op[type] has kernel registered.
  auto& all_op_kernels = op.AllOpKernels();
  auto kernels_iter = all_op_kernels.find(op.Type());
  PADDLE_ENFORCE_NE(
      kernels_iter, all_op_kernels.end(),
      platform::errors::NotFound(
          "There are no kernels which are registered in the %s operator.",
          op.Type()));

  auto& kernels = kernels_iter->second;
  auto kernel_iter = kernels.find(expected_kernel_key);
#ifdef PADDLE_WITH_XPU
  if (kernel_iter == kernels.end() &&
      is_xpu_place(expected_kernel_key.place_)) {
    expected_kernel_key.place_ = platform::CPUPlace();
    kernel_iter = kernels.find(expected_kernel_key);
  }
#endif
  // TODO(jiabin): Add operator.cc's line 1000 part back when we need that case
  PADDLE_ENFORCE_NE(kernel_iter, kernels.end(),
                    platform::errors::NotFound(
                        "Operator %s does not have kernel for %s.", op.Type(),
                        KernelTypeToString(expected_kernel_key)));

  if (!(expected_kernel_key.place_ == place)) {
    dev_ctx = pool.Get(expected_kernel_key.place_);
  }

  return PreparedOp(op, ctx, expected_kernel_key, kernel_iter->second, dev_ctx);
}

PreparedOp PreparedOp::Prepare(const NameVarMap<VarBase>& ins,
                               const NameVarMap<VarBase>& outs,
                               const framework::OperatorWithKernel& op,
                               const platform::Place& place,
                               const framework::AttributeMap& attrs) {
  return PrepareImpl<VarBase>(ins, outs, op, place, attrs);
}

PreparedOp PreparedOp::Prepare(const NameVarMap<VariableWrapper>& ins,
                               const NameVarMap<VariableWrapper>& outs,
                               const framework::OperatorWithKernel& op,
                               const platform::Place& place,
                               const framework::AttributeMap& attrs) {
  return PrepareImpl<VariableWrapper>(ins, outs, op, place, attrs);
}

inline uint64_t GetPosixInUsec2() {
  struct timeval tv;
  gettimeofday(&tv, nullptr);
  return (static_cast<uint64_t>(tv.tv_sec) * 1000000 + tv.tv_usec);
}

template <typename VarType>
static void PreparedOpRunImpl(
    const framework::OperatorBase& op, const framework::RuntimeContext& ctx,
    const framework::OpKernelType& kernel_type,
    const framework::OperatorWithKernel::OpKernelFunc& func,
    platform::DeviceContext* dev_ctx, const NameVarMap<VarType>& ins,
    const NameVarMap<VarType>& outs, const framework::AttributeMap& attrs) {
  // TODO(zjl): remove scope in dygraph
  framework::Scope scope;

  DygraphInferShapeContext<VarType> infer_shape_ctx(&ins, &outs, &attrs,
                                                    op.Type());
  static_cast<const framework::OperatorWithKernel&>(op).InferShape(
      &infer_shape_ctx);

  opbase_run_2 = GetPosixInUsec2();
  func(DygraphExecutionContext<VarType>(op, scope, *dev_ctx, ctx, ins, outs,
                                        attrs));
  opbase_run_3 = GetPosixInUsec2();
  /**
   * [ Why need handle complex gradient to real gradient? ]
   *
   * After the introduction of complex number calculations, Ops that support
   * complex number calculations generally support type promotion, such as
   * x(float32) + y(complex64) = out(complex64), then the type of the grad
   * tensor should be dout(complex64), dx(float32), dy (complex64).
   *
   * But because the dout is complex64, the dx is also complex64 after
   * grad op kernel executed, we need to recognize this situation and
   * convert dx to float32 type. HandleComplexGradToRealGrad does this thing.
   */
  if (framework::IsComplexType(kernel_type.data_type_)) {
    HandleComplexGradToRealGrad<VarType>(outs);
  }
}

void PreparedOp::Run(const NameVarMap<VarBase>& ins,
                     const NameVarMap<VarBase>& outs,
                     const framework::AttributeMap& attrs) {
  PreparedOpRunImpl<VarBase>(op_, ctx_, kernel_type_, func_, dev_ctx_, ins,
                             outs, attrs);
}

void PreparedOp::Run(const NameVarMap<VariableWrapper>& ins,
                     const NameVarMap<VariableWrapper>& outs,
                     const framework::AttributeMap& attrs) {
  PreparedOpRunImpl<VariableWrapper>(op_, ctx_, kernel_type_, func_, dev_ctx_,
                                     ins, outs, attrs);
}

}  // namespace imperative
}  // namespace paddle
