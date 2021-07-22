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

#include <algorithm>
#include <atomic>
#include <memory>
#include <mutex>  // NOLINT
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "glog/logging.h"  // For VLOG
#include "paddle/fluid/framework/attribute.h"
#include "paddle/fluid/framework/block_desc.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/op_info.h"
#include "paddle/fluid/framework/op_kernel_type.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/framework/selected_rows.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/framework/unused_var_check.h"
#include "paddle/fluid/memory/malloc.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/variant.h"
#include "paddle/fluid/framework/operator.h"

namespace paddle {
namespace framework {
class InferShapeContext;
class OpInfo;
class Scope;
class Variable;
}  // namespace framework
}  // namespace paddle

DECLARE_int32(inner_op_parallelism);

namespace paddle {
namespace framework {

/// If a variable is a empty variable, that name will be used.
constexpr char kEmptyVarName[] = "@EMPTY@";

/// If a variable is a temporary variable, that name will be set in Python,
/// but it will be convert to a unique name in scope after OpCreator.
constexpr char kTempVarName[] = "@TEMP@";

/// If a variable's name has a certain suffix, it means that the
/// variable is the gradient of another variable.
/// e.g. Variable "x@GRAD" is the gradient of variable "x".
constexpr char kGradVarSuffix[] = "@GRAD";

constexpr size_t kGradVarSuffixSize = 5U;

/// Variables with this suffix are supposed to be filled up with zeros.
constexpr char kZeroVarSuffix[] = "@ZERO";

/// Variables with this suffix are the new Gradient.
constexpr char kNewGradSuffix[] = "@NEWGRAD@";

/// RuntimeContext is used to relate input/output names of Operator with
/// the corresponding variables in name scope.
/// If an Op has attribute kEnableCacheRuntimeContext, it means that in a same
/// name scope, since the input/output names of this Op do not change in the
/// execution, RuntimeContext could be created only at the first iteration of
/// this Op's execution to save the elapsed time.
constexpr char kEnableCacheRuntimeContext[] = "@ENABLE_CACHE_RUNTIME_CONTEXT@";

/// If an Op has this attribute, all its kernels should calculate output
/// variable's shape in the corresponding Compute() function. And
/// OperatorWithKernel::RunImpl() would skip call this Op's InferShape()
/// function in its runtime for speedup.
/// TODO(luotao): Note that this temporal attribute would be deleted after all
/// ops contain it.
constexpr char kAllKernelsMustComputeRuntimeShape[] =
    "@ALL_KERNELS_MUST_COMPUTE_RUNTIME_SHAPE@";

// define some kernel priority
/* Define multiple kernel type fallback order*/
extern std::vector<std::tuple<platform::Place, LibraryType>> kKernelPriority;

inline std::string GradVarName(const std::string& var_name) {
  std::string result;
  result.reserve(var_name.size() + kGradVarSuffixSize);
  result += var_name;
  result += kGradVarSuffix;
  return result;
}

inline std::string GradOriginalVarName(const std::string& grad_var_name) {
  std::size_t pos = grad_var_name.rfind(kGradVarSuffix);
  if (pos == std::string::npos) {
    return grad_var_name;
  } else {
    return grad_var_name.substr(0, pos);
  }
}

const Tensor* GetLoDTensorOrSelectedRowsValueFromVar(const Variable& var);
Tensor* GetMutableLoDTensorOrSelectedRowsValueFromVar(Variable* var);

class ExecutionContext;
class OperatorBase;

class RuntimeContext {
 public:
  RuntimeContext(const VariableNameMap& innames,
                 const VariableNameMap& outnames, const Scope& scope);

  RuntimeContext(const VariableValueMap& invars,
                 const VariableValueMap& outvars)
      : inputs(invars), outputs(outvars) {}

  VariableValueMap inputs;
  VariableValueMap outputs;
};

/**
 * OperatorBase has the basic elements that Net will call to do computation.
 * Only CreateOperator from OpRegistry will new Operator directly. User
 * should always construct a proto message OpDesc and call
 * OpRegistry::CreateOp(op_desc) to get an Operator instance.
 */
class OperatorBase {
 public:
  OperatorBase(const std::string& type, const VariableNameMap& inputs,
               const VariableNameMap& outputs, const AttributeMap& attrs);

  virtual ~OperatorBase() {}

  /// Executor will call this interface function to Run an op.
  //  The implementation should be written at RunImpl
  void Run(const Scope& scope, const platform::Place& place);

  // FIXME(typhoonzero): this is only used for recv_op to stop event_loop.
  virtual void Stop() {}

  /// if scope is not null, also show dimensions of arguments
  virtual std::string DebugStringEx(const Scope* scope) const;
  std::string DebugString() const { return DebugStringEx(nullptr); }

  virtual bool SupportGPU() const { return false; }
  virtual bool SupportNPU() const { return false; }

  const std::string& Type() const { return type_; }

  bool HasAttr(const std::string& name) const { return attrs_.count(name); }
  template <typename T>
  inline const T& Attr(const std::string& name) const {
    PADDLE_ENFORCE_NE(
        attrs_.find(name), attrs_.end(),
        platform::errors::NotFound("(%s) is not found in AttributeMap.", name));
    return BOOST_GET_CONST(T, attrs_.at(name));
  }
  void SetAttr(const std::string& name, const Attribute& v) {
    PADDLE_ENFORCE_EQ(
        HasAttr(name), true,
        platform::errors::NotFound(
            "The attribute %s is not found in operator %s", name, Type()));

    attrs_[name] = v;
  }
  const AttributeMap& Attrs() const { return attrs_; }

  const VariableNameMap& Inputs() const { return inputs_; }
  const VariableNameMap& Outputs() const { return outputs_; }

  const OpInfo& Info() const {
    PADDLE_ENFORCE_NOT_NULL(
        info_, platform::errors::NotFound(
                   "OpInfo of operator (%s) is not found.", type_));
    return *info_;
  }

  bool HasInputs(const std::string& name) const;
  //! Get a input with argument's name described in `op_proto`
  std::string Input(const std::string& name) const;
  //! Get a input which has multiple variables.
  const std::vector<std::string>& Inputs(const std::string& name) const;
  //! Get all inputs variable names
  std::vector<std::string> InputVars() const;

  bool HasOutputs(const std::string& name) const;
  //! Get a output with argument's name described in `op_proto`
  std::string Output(const std::string& name) const;
  //! Get an output which has multiple variables.
  //! TODO add a vector_view to prevent memory copy.
  const std::vector<std::string>& Outputs(const std::string& name) const;
  //! Get all outputs variable names
  virtual std::vector<std::string> OutputVars(bool has_intermediate) const;

  void SetIsCalledByExecutor(bool x) { run_by_executor_ = x; }

  virtual void RuntimeInferShape(const Scope& scope,
                                 const platform::Place& place,
                                 const RuntimeContext& ctx) const {}

  virtual platform::Place GetExecutionPlace(
      const platform::Place& place) const {
    return place;
  }

 protected:
  std::string type_;
  // NOTE: in case of OpGrad, inputs_ contains:
  // I (Inputs)
  // O (Outputs)
  // OG (Output Gradients)
  VariableNameMap inputs_;

  // NOTE: in case of OpGrad, outputs_ contains
  // IG (Inputs Gradients)
  VariableNameMap outputs_;
  AttributeMap attrs_;

  // OpInfo
  const OpInfo* info_;

  // Whether this operator executes in an Executor.
  bool run_by_executor_{true};

 private:
  void GenerateTemporaryNames();
  void CheckAllInputOutputSet() const;
  virtual void RunImpl(const Scope& scope,
                       const platform::Place& place) const = 0;
};

class ExecutionContext {
 public:
  ExecutionContext(const OperatorBase& op, const Scope& scope,
                   const platform::DeviceContext& device_context,
                   const RuntimeContext& ctx)
      : op_(op), scope_(scope), device_context_(device_context), ctx_(ctx) {}
  virtual ~ExecutionContext() {}

  virtual std::string InputName(const std::string& name) const {
    return op_.Input(name);
  }
  virtual std::vector<std::string> InputNames(const std::string& name) const {
    return op_.Inputs(name);
  }
  virtual std::string OutputName(const std::string& name) const {
    return op_.Output(name);
  }

  virtual std::vector<std::string> OutputNames(const std::string& name) const {
    return op_.Outputs(name);
  }

  virtual bool HasAttr(const std::string& name) const {
    return op_.HasAttr(name);
  }
  virtual const AttributeMap& Attrs() const { return op_.Attrs(); }

  const std::string& Type() const { return op_.Type(); }

  const Scope& scope() const { return scope_; }

  template <typename T>
  inline const T& Attr(const std::string& name) const {
    return BOOST_GET_CONST(T, GetAttr(name));
  }

  virtual const Attribute& GetAttr(const std::string& name) const {
    return op_.Attrs().at(name);
  }

  virtual bool HasInput(const std::string& name) const;

  virtual bool HasOutput(const std::string& name) const;

  virtual size_t InputSize(const std::string& name) const {
    return op_.Inputs(name).size();
  }

  virtual size_t OutputSize(const std::string& name) const {
    return op_.Outputs(name).size();
  }

  virtual const Variable* InputVar(const std::string& name) const;

  virtual Variable* OutputVar(const std::string& name) const;

  virtual const std::vector<Variable*> MultiInputVar(
      const std::string& name) const {
    LogVarUsageIfUnusedVarCheckEnabled(name);

    auto it = ctx_.inputs.find(name);
    if (it == ctx_.inputs.end()) {
      return {};
    }
    return {it->second.begin(), it->second.end()};
  }

  virtual std::vector<Variable*> MultiOutputVar(const std::string& name) const {
    auto it = ctx_.outputs.find(name);
    if (it == ctx_.outputs.end()) {
      return {};
    }
    return it->second;
  }

  virtual std::vector<std::string> InNameList() const {
    std::vector<std::string> vec_temp;
    vec_temp.reserve(ctx_.inputs.size());

    for (auto& input : ctx_.inputs) {
      vec_temp.push_back(input.first);
    }

    return vec_temp;
  }

  template <typename T>
  const T* Input(const std::string& name) const {
    auto* var = InputVar(name);
    return var == nullptr ? nullptr : &var->Get<T>();
  }

  template <typename T>
  T* Output(const std::string& name) const {
    auto var = OutputVar(name);
    return var == nullptr ? nullptr : var->GetMutable<T>();
  }

  template <typename T>
  const std::vector<const T*> MultiInput(const std::string& name) const {
    LogVarUsageIfUnusedVarCheckEnabled(name);

    auto vars = MultiInputVar(name);
    if (vars.size() == 0) {
      return {};
    }
    std::vector<const T*> res;
    res.reserve(vars.size());
    std::transform(vars.begin(), vars.end(), std::back_inserter(res),
                   [&](const Variable* var) -> const T* {
                     return var == nullptr ? nullptr : &var->Get<T>();
                   });
    return res;
  }

  template <typename T>
  std::vector<T*> MultiOutput(const std::string& name) const {
    auto vars = MultiOutputVar(name);

    if (vars.size() == 0) {
      return {};
    }

    std::vector<T*> res;
    res.reserve(vars.size());
    std::transform(vars.begin(), vars.end(), std::back_inserter(res),
                   [&](Variable* var) -> T* {
                     return var == nullptr ? nullptr : var->GetMutable<T>();
                   });

    return res;
  }

  platform::Place GetPlace() const { return device_context_.GetPlace(); }

  template <typename DeviceContextType>
  const DeviceContextType& device_context() const {
    return *reinterpret_cast<const DeviceContextType*>(&device_context_);
  }

  const platform::DeviceContext& device_context() const {
    return device_context_;
  }

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  const inline platform::CUDADeviceContext& cuda_device_context() const {
    PADDLE_ENFORCE_EQ(platform::is_gpu_place(device_context_.GetPlace()), true,
                      platform::errors::PreconditionNotMet(
                          "Current device context place is not GPUPlace."));
    return *reinterpret_cast<const platform::CUDADeviceContext*>(
        &device_context_);
  }
#endif

  template <typename T, typename DevContext>
  Tensor AllocateTmpTensor(const framework::DDim& dim,
                           const DevContext& dev_ctx) const {
    auto tmp_allocation_ptr = memory::Alloc(dev_ctx, product(dim) * sizeof(T));
    auto& deleter = tmp_allocation_ptr.get_deleter();
    auto* allocation_ptr = tmp_allocation_ptr.release();
    auto shared_allocation = std::shared_ptr<memory::allocation::Allocation>(
        allocation_ptr, deleter);

    PADDLE_ENFORCE_GE(
        allocation_ptr->size(), framework::product(dim) * sizeof(T),
        platform::errors::PreconditionNotMet(
            "The data memory size(%d) is less than the tensor needed memory "
            "size(%d).",
            allocation_ptr->size(), framework::product(dim) * sizeof(T)));

    paddle::framework::Tensor temp_tensor(
        framework::ToDataType(std::type_index(typeid(T))));
    temp_tensor.Resize(dim);
    temp_tensor.ResetHolder(std::move(shared_allocation));
    return temp_tensor;
  }

  const RuntimeContext Context() const { return ctx_; }

  std::string DebugString() const { return op_.DebugString(); }
  const OperatorBase& GetOp() const { return op_; }

 private:
  const OperatorBase& op_;
  const Scope& scope_;
  const platform::DeviceContext& device_context_;
  const RuntimeContext& ctx_;
};

template <>
const Tensor* ExecutionContext::Input<Tensor>(const std::string& name) const;

template <>
const std::vector<const Tensor*> ExecutionContext::MultiInput<Tensor>(
    const std::string& name) const;

template <>
Tensor* ExecutionContext::Output<Tensor>(const std::string& name) const;

template <>
std::vector<Tensor*> ExecutionContext::MultiOutput<Tensor>(
    const std::string& name) const;

class OpKernelBase {
 public:
  /**
   * ExecutionContext is the only parameter of Kernel Run function.
   * Run will get input/output variables, state such as momentum and
   * device resource such as CUDA stream, cublas handle, etc. from
   * ExecutionContext. User should construct it before run the Operator.
   */

  virtual void Compute(const ExecutionContext& context) const = 0;

  virtual ~OpKernelBase() = default;
};

template <typename T>
class OpKernel : public OpKernelBase {
 public:
  using ELEMENT_TYPE = T;
};

class OperatorWithKernel : public OperatorBase {
 public:
  using OpKernelFunc = std::function<void(const ExecutionContext&)>;
  using OpKernelMap =
      std::unordered_map<OpKernelType, OpKernelFunc, OpKernelType::Hash>;

  OperatorWithKernel(const std::string& type, const VariableNameMap& inputs,
                     const VariableNameMap& outputs, const AttributeMap& attrs)
      : OperatorBase(type, inputs, outputs, attrs) {}

  static std::unordered_map<std::string /* op_type */, OpKernelMap>&
  AllOpKernels() {
    static std::unordered_map<std::string, OpKernelMap> g_all_op_kernels;
    return g_all_op_kernels;
  }

  bool IsMKLDNNType() const {
    return ((this->kernel_type_) && (this->kernel_type_->data_layout_ ==
                                     framework::DataLayout::kMKLDNN));
  }

  bool SupportGPU() const override {
    auto& op_kernels = OperatorWithKernel::AllOpKernels().at(type_);
    return std::any_of(op_kernels.begin(), op_kernels.end(),
                       [](OpKernelMap::const_reference kern_pair) {
                         return platform::is_gpu_place(kern_pair.first.place_);
                       });
  }
  bool SupportNPU() const override {
    auto& op_kernels = OperatorWithKernel::AllOpKernels().at(type_);
    return std::any_of(op_kernels.begin(), op_kernels.end(),
                       [](OpKernelMap::const_reference kern_pair) {
                         return platform::is_npu_place(kern_pair.first.place_);
                       });
  }
  bool SupportsMKLDNN(proto::VarType::Type data_type) const;

  bool CanMKLDNNBeUsed(const framework::ExecutionContext& ctx,
                       proto::VarType::Type data_type) const;

  virtual void InferShape(InferShapeContext* ctx) const = 0;

  void RuntimeInferShape(const Scope& scope, const platform::Place& place,
                         const RuntimeContext& ctx) const override;

  proto::VarType::Type IndicateVarDataType(const ExecutionContext& ctx,
                                           const std::string& name) const;

  proto::VarType::Type IndicateOrPromoteVarDataTypes(
      const ExecutionContext& ctx, const std::string& name1,
      const std::string& name2) const;

  virtual OpKernelType GetExpectedKernelType(const ExecutionContext& ctx) const;

  // change this to public so that in dygraph mode we can call it to check if we
  // need transform data
  virtual OpKernelType GetKernelTypeForVar(
      const std::string& var_name, const Tensor& tensor,
      const OpKernelType& expected_kernel_type) const;

  platform::Place GetExecutionPlace(
      const platform::Place& platform) const override {
    return kernel_type_->place_;
  }

 private:
  void RunImpl(const Scope& scope, const platform::Place& place) const final;
  void RunImpl(const Scope& scope, const platform::Place& place,
               RuntimeContext* runtime_ctx) const;

  /**
   * Transfer data from scope to a transferred scope. If there is no data need
   * to
   * be tranfered, it returns nullptr.
   *
   * * transfered_inplace_vars is a output vector.
   */
  Scope* PrepareData(const Scope& scope,
                     const OpKernelType& expected_kernel_key,
                     std::vector<std::string>* transfered_inplace_vars,
                     RuntimeContext* ctx) const;

  void TransferInplaceVarsBack(const Scope& scope,
                               const std::vector<std::string>& inplace_vars,
                               const Scope& exec_scope) const;

  void ChooseKernel(const RuntimeContext& ctx, const Scope& scope,
                    const platform::Place& place) const;

  void HandleComplexGradToRealGrad(const Scope& scope,
                                   RuntimeContext* ctx) const;

  /* Inner assist methods */
  // indicate kernel DataType by input data.
  // By default all input data must be same.
  proto::VarType::Type IndicateDataType(const ExecutionContext& ctx) const;
  // used for IndicateDataType
  void ParseInputDataType(const ExecutionContext& ctx, const std::string& name,
                          proto::VarType::Type* type) const;
  // used for IndicateOrPromoteVarDataTypes
  Tensor* GetTensorFormInputSafely(const ExecutionContext& ctx,
                                   const std::string& name) const;

 protected:
  mutable std::unique_ptr<OpKernelType> kernel_type_;
  mutable std::unique_ptr<OpKernelFunc> kernel_func_;
  mutable std::unique_ptr<RuntimeContext> runtime_ctx_;
  mutable const Scope* pre_scope_ = nullptr;
  mutable bool need_prepare_data_ = true;
  mutable bool enable_cache_runtime_context_ = false;
  mutable bool all_kernels_must_compute_runtime_shape_ = false;
  mutable std::mutex cache_update_mutex_;
  mutable bool enable_cache_transfer_scope_ = false;
};

extern bool OpSupportGPU(const std::string& op_type);

/*
class RuntimeInferShapeContext : public InferShapeContext {
 public:
  RuntimeInferShapeContext(const OperatorBase& op, const RuntimeContext& ctx)
      : op_(op), ctx_(ctx) {}
  bool HasInput(const std::string& name) const override {
    // has only one input
    const auto& ins = ctx_.inputs;
    auto it = ins.find(name);
    if (it == ins.end()) {
      return false;
    }
    const auto& in = it->second;
    if (in.size() == 0) return false;
    PADDLE_ENFORCE_EQ(
        in.size(), 1UL,
        platform::errors::InvalidArgument(
            "Input %s should not contain more than one inputs.", name));
    return in[0] != nullptr;
  }
  bool HasOutput(const std::string& name) const override {
    // has only one output
    const auto& outs = ctx_.outputs;
    auto it = outs.find(name);
    if (it == outs.end()) {
      return false;
    }
    const auto& out = it->second;
    if (out.size() == 0) {
      return false;
    }
    PADDLE_ENFORCE_EQ(
        out.size(), 1UL,
        platform::errors::InvalidArgument(
            "Output %s should not contain more than one outputs.", name));
    return out[0] != nullptr;
  }
  bool HasInputs(const std::string& name) const override {
    const auto& ins = ctx_.inputs;
    auto it = ins.find(name);
    if (it == ins.end() || it->second.empty()) {
      return false;
    }
    for (auto& input : it->second) {
      if (input == nullptr) {
        return false;
      }
    }
    return true;
  }
  bool HasOutputs(const std::string& name) const override {
    const auto& outs = ctx_.outputs;
    auto it = outs.find(name);
    if (it == outs.end() || it->second.empty()) {
      return false;
    }
    for (auto& output : it->second) {
      if (output == nullptr) {
        return false;
      }
    }
    return true;
  }
  AttrReader Attrs() const override { return AttrReader(op_.Attrs()); }
  std::vector<std::string> Inputs(const std::string& name) const override {
    return op_.Inputs(name);
  }
  std::vector<std::string> Outputs(const std::string& name) const override {
    return op_.Outputs(name);
  }
  std::string GetInputNameByIdx(size_t idx) const override {
    auto& op_proto =
        paddle::framework::OpInfoMap::Instance().Get(op_.Type()).proto_;
    PADDLE_ENFORCE_LT(idx, op_proto->inputs().size(),
                      platform::errors::OutOfRange(
                          "The index should be less than the size of inputs of "
                          "operator %s, but got index is %d and size is %d",
                          op_.Type(), idx, op_proto->inputs().size()));
    return op_proto->inputs()[idx].name();
  }
  std::string GetOutputNameByIdx(size_t idx) const override {
    auto& op_proto =
        paddle::framework::OpInfoMap::Instance().Get(op_.Type()).proto_;
    PADDLE_ENFORCE_LT(
        idx, op_proto->outputs().size(),
        platform::errors::OutOfRange(
            "The index should be less than the size of outputs of "
            "operator %s, but got index is %d and size is %d",
            op_.Type(), idx, op_proto->outputs().size()));
    return op_proto->outputs()[idx].name();
  }
  void ShareDim(const std::string& in, const std::string& out, size_t i = 0,
                size_t j = 0) override {
    auto in_it = ctx_.inputs.find(in);
    auto out_it = ctx_.outputs.find(out);
    PADDLE_ENFORCE_NE(
        in_it, ctx_.inputs.end(),
        platform::errors::NotFound("Input %s does not exist.", in));
    PADDLE_ENFORCE_NE(
        out_it, ctx_.outputs.end(),
        platform::errors::NotFound("Output %s does not exist.", out));
    PADDLE_ENFORCE_LT(i, in_it->second.size(),
                      platform::errors::InvalidArgument(
                          "The index of input dimension is out of range, "
                          "excepted index less than %zu, but received %zu.",
                          in_it->second.size(), i));
    PADDLE_ENFORCE_LT(j, out_it->second.size(),
                      platform::errors::InvalidArgument(
                          "The index of output dimension is out of range, "
                          "excepted index less than %zu, but received %zu.",
                          out_it->second.size(), j));
    Variable* in_var = in_it->second[i];
    Variable* out_var = out_it->second[j];
    PADDLE_ENFORCE_EQ(
        in_var->Type(), out_var->Type(),
        platform::errors::InvalidArgument(
            "The type of input (%s) and output (%s) are inconsistent.", in,
            out));
    if (in_var->IsType<framework::SelectedRows>()) {
      auto& in_sele_rows = in_var->Get<framework::SelectedRows>();
      auto out_sele_rows = out_var->GetMutable<framework::SelectedRows>();
      out_sele_rows->mutable_value()->Resize(in_sele_rows.value().dims());
      out_sele_rows->set_rows(in_sele_rows.rows());
      out_sele_rows->set_height(in_sele_rows.height());
    } else if (in_var->IsType<framework::LoDTensor>()) {
      auto& in_lod_tensor = in_var->Get<framework::LoDTensor>();
      auto* out_lod_tensor = out_var->GetMutable<framework::LoDTensor>();
      out_lod_tensor->Resize(in_lod_tensor.dims());
    } else {
      PADDLE_THROW(platform::errors::Unimplemented(
          "Currently, the input type of ShareDim only can be LoDTensor "
          "or SelectedRows."));
    }
  }
  void ShareAllLoD(const std::string& in,
                   const std::string& out) const override {
    auto in_it = ctx_.inputs.find(in);
    auto out_it = ctx_.outputs.find(out);
    PADDLE_ENFORCE_NE(in_it, ctx_.inputs.end(),
                      platform::errors::NotFound(
                          "Input [%s] found error in Op [%s]", in, op_.Type()));
    PADDLE_ENFORCE_NE(
        out_it, ctx_.outputs.end(),
        platform::errors::NotFound("Output [%s] found error in Op [%s]", out,
                                   op_.Type()));
    auto& in_var_list = in_it->second;
    auto& out_var_list = out_it->second;
    PADDLE_ENFORCE_EQ(
        in_var_list.size(), out_var_list.size(),
        platform::errors::PreconditionNotMet(
            "Op [%s]: Input var size should be equal with output var size",
            op_.Type()));
    auto& out_var_names = op_.Outputs(out);
    for (size_t i = 0; i < in_var_list.size(); ++i) {
      if (out_var_names[i] == framework::kEmptyVarName) {
        continue;
      }
      Variable* in_var = in_var_list[i];
      if (!in_var->IsType<LoDTensor>()) return;
      Variable* out_var = out_var_list[i];
      PADDLE_ENFORCE_EQ(out_var->IsType<LoDTensor>(), true,
                        platform::errors::PreconditionNotMet(
                            "The %d-th output of Output(%s) must be LoDTensor.",
                            i, out_var_names[i]));
      auto& in_tensor = in_var->Get<LoDTensor>();
      auto* out_tensor = out_var->GetMutable<LoDTensor>();
      out_tensor->set_lod(in_tensor.lod());
#ifdef PADDLE_WITH_MKLDNN
      if (in_tensor.layout() != DataLayout::kMKLDNN)
#endif
        out_tensor->set_layout(in_tensor.layout());
    }
  }
  void ShareLoD(const std::string& in, const std::string& out, size_t i = 0,
                size_t j = 0) const override {
    auto in_it = ctx_.inputs.find(in);
    auto out_it = ctx_.outputs.find(out);
    PADDLE_ENFORCE_NE(
        in_it, ctx_.inputs.end(),
        platform::errors::NotFound("Input %s does not exist.", in));
    PADDLE_ENFORCE_NE(
        out_it, ctx_.outputs.end(),
        platform::errors::NotFound("Output %s does not exist.", out));
    PADDLE_ENFORCE_LT(i, in_it->second.size(),
                      platform::errors::InvalidArgument(
                          "The index of input dimension is out of range, "
                          "excepted index less than %zu, but received %zu.",
                          in_it->second.size(), i));
    PADDLE_ENFORCE_LT(j, out_it->second.size(),
                      platform::errors::InvalidArgument(
                          "The index of output dimension is out of range, "
                          "excepted index less than %zu, but received %zu.",
                          out_it->second.size(), j));
    Variable* in_var = in_it->second.at(i);
    if (!in_var->IsType<LoDTensor>()) return;
    Variable* out_var = out_it->second.at(j);
    PADDLE_ENFORCE_EQ(
        out_var->IsType<LoDTensor>(), true,
        platform::errors::InvalidArgument(
            "The %zu-th output of Output(%s) must be LoDTensor.", j, out));
    auto& in_tensor = in_var->Get<LoDTensor>();
    auto* out_tensor = out_var->GetMutable<LoDTensor>();
    out_tensor->set_lod(in_tensor.lod());
// TODO(dzhwinter) : reuse ShareLoD in most operators.
// Need to call ShareLayout explicitly in sequence related ops.
// Shall we have a better method to shared info between in/out Tensor?
#ifdef PADDLE_WITH_MKLDNN
    // Fix me: ugly workaround below
    // Correct solution:
    //    set_layout() should NOT be called here (i.e. ShareLoD). Instead,
    //    layout of output tensor should be set "manually" in Compute()
    //    of each OPKernel. The reason layout should NOT be shared between
    //    input and output "automatically" (now by InferShape()->ShareLoD())
    //    is that layout transform may occur after InferShape().
    // Workaround:
    //    Skip set_layout() when input layout is kMKLDNN
    //    This is to avoid kMKLDNN is populated wrongly into a non-MKLDNN
    //    OPKernel. In all MKLDNN OPkernel, set_layout(kMKLDNN) should be called
    //    in Compute()
    if (in_tensor.layout() != DataLayout::kMKLDNN)
#endif
      out_tensor->set_layout(in_tensor.layout());
  }
  int32_t GetLoDLevel(const std::string& in, size_t i = 0) const override {
    PADDLE_THROW(platform::errors::PreconditionNotMet(
        "GetLoDLevel is only used in compile time. The calculation of "
        "output's actual lod is different among operators so that should be "
        "set in the runtime kernel."));
  }
  void SetLoDLevel(const std::string& out, int32_t lod_level,
                   size_t j = 0) const override {
    PADDLE_THROW(platform::errors::PreconditionNotMet(
        "SetLoDLevel is only used in compile time. The calculation of "
        "output's actual lod is different among operators so that should be "
        "set in the runtime kernel."));
  }
  bool IsRuntime() const override { return true; }
  // TODO(paddle-dev): Can this be template?
  std::vector<InferShapeVarPtr> GetInputVarPtrs(
      const std::string& name) override {
    const std::vector<Variable*>& vars = InputVars(name);
    std::vector<InferShapeVarPtr> res;
    res.reserve(vars.size());
    res.insert(res.begin(), vars.begin(), vars.end());
    return res;
  }
  std::vector<InferShapeVarPtr> GetOutputVarPtrs(
      const std::string& name) override {
    const std::vector<Variable*>& vars = OutputVars(name);
    std::vector<InferShapeVarPtr> res;
    res.reserve(vars.size());
    res.insert(res.begin(), vars.begin(), vars.end());
    return res;
  }
  DDim GetInputDim(const std::string& name) const override {
    const std::vector<Variable*>& vars = InputVars(name);
    PADDLE_ENFORCE_EQ(
        vars.size(), 1UL,
        platform::errors::InvalidArgument(
            "Input(%s) should hold one element, but now it holds %zu elements.",
            name, vars.size()));
    return this->GetDim(vars[0]);
  }
  std::vector<DDim> GetInputsDim(const std::string& name) const override {
    const std::vector<Variable*>& vars = InputVars(name);
    return GetDims(vars);
  }
  std::vector<proto::VarType::Type> GetInputsVarType(
      const std::string& name) const override {
    return GetVarTypes(InputVars(name));
  }
  std::vector<proto::VarType::Type> GetOutputsVarType(
      const std::string& name) const override {
    return GetVarTypes(OutputVars(name));
  }
  void SetOutputDim(const std::string& name, const DDim& dim) override {
    auto& vars = OutputVars(name);
    PADDLE_ENFORCE_EQ(
        vars.size(), 1UL,
        platform::errors::InvalidArgument("Output(%s) should hold one element, "
                                          "but now it holds %zu elements.",
                                          name, vars.size()));
    SetDim(vars[0], dim);
  }
  void SetOutputsDim(const std::string& name,
                     const std::vector<DDim>& dims) override {
    auto& vars = OutputVars(name);
    SetDims(vars, dims);
  }
 protected:
  DDim GetDim(Variable* var) const {
    PADDLE_ENFORCE_NOT_NULL(
        var, platform::errors::InvalidArgument("Input variable is nullptr."));
    if (var->IsType<LoDTensor>()) {
      return var->Get<LoDTensor>().dims();
    } else if (var->IsType<SelectedRows>()) {
      return var->Get<SelectedRows>().GetCompleteDims();
    } else {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "Only LoDTensor or SelectedRows support 'GetDim', but input "
          "Variable's type is %s.",
          ToTypeName(var->Type())));
    }
  }
  std::vector<DDim> GetDims(const std::vector<Variable*>& vars) const {
    std::vector<DDim> ret;
    ret.reserve(vars.size());
    std::transform(vars.begin(), vars.end(), std::back_inserter(ret),
                   [this](Variable* var) { return this->GetDim(var); });
    return ret;
  }
  std::vector<DDim> GetRepeatedDims(const std::string& name) const override {
    PADDLE_THROW(platform::errors::PreconditionNotMet(
        "GetRepeatedDims method only ban be used in compile time."));
  }
  void SetDim(Variable* var, const DDim& dim) {
    if (var->IsType<LoDTensor>()) {
      var->GetMutable<LoDTensor>()->Resize(dim);
    } else if (var->IsType<SelectedRows>()) {
      var->GetMutable<SelectedRows>()->set_height(dim[0]);
    } else {
      PADDLE_THROW(platform::errors::Unimplemented(
          "Variable type error, expect LoDTensor or SelectedRows, but received "
          "(%s).",
          ToTypeName(var->Type())));
    }
  }
  void SetDims(const std::vector<Variable*>& vars,
               const std::vector<DDim>& dims) {
    size_t length = vars.size();
    PADDLE_ENFORCE_EQ(length, dims.size(),
                      platform::errors::InvalidArgument(
                          "The number of input variables do not match the "
                          "number of input dimensions, the number of variables "
                          "is %zu, the number of dimensions is %zu.",
                          length, dims.size()));
    for (size_t i = 0; i < length; ++i) {
      if (vars[i] == nullptr) {
        continue;
      }
      SetDim(vars[i], dims[i]);
    }
  }
  void SetRepeatedDims(const std::string& name,
                       const std::vector<DDim>& dims) override {
    PADDLE_THROW(platform::errors::PreconditionNotMet(
        "SetRepeatedDims method only can be used in compile time."));
  }
  std::vector<proto::VarType::Type> GetVarTypes(
      const std::vector<Variable*>& vars) const {
    std::vector<proto::VarType::Type> retv;
    retv.resize(vars.size());
    std::transform(vars.begin(), vars.end(), retv.begin(),
                   std::bind(std::mem_fn(&RuntimeInferShapeContext::GetVarType),
                             this, std::placeholders::_1));
    return retv;
  }
  proto::VarType::Type GetVarType(Variable* var) const {
    return ToVarType(var->Type());
  }
 private:
  const std::vector<Variable*>& InputVars(const std::string& name) const {
    auto it = ctx_.inputs.find(name);
    PADDLE_ENFORCE_NE(
        it, ctx_.inputs.end(),
        platform::errors::NotFound(
            "Operator (%s) does not have the input (%s).", op_.Type(), name));
    return it->second;
  }
  const std::vector<Variable*>& OutputVars(const std::string& name) const {
    auto it = ctx_.outputs.find(name);
    PADDLE_ENFORCE_NE(
        it, ctx_.outputs.end(),
        platform::errors::NotFound(
            "Operator (%s) does not have the outputs (%s).", op_.Type(), name));
    return it->second;
  }
  const OperatorBase& op_;
  const RuntimeContext& ctx_;
};
*/

}  // namespace framework
}  // namespace paddle