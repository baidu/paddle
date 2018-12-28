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

#include <gflags/gflags.h>
#include <glog/logging.h>

#include <algorithm>
#include "paddle/fluid/framework/data_transform.h"
#include "paddle/fluid/framework/executor.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/op_proto_maker.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/shape_inference.h"
#include "paddle/fluid/framework/transfer_scope_cache.h"
#include "paddle/fluid/framework/var_type.h"
#include "paddle/fluid/platform/profiler.h"

DECLARE_bool(benchmark);
DEFINE_bool(check_nan_inf, false,
            "Checking whether operator produce NAN/INF or not. It will be "
            "extremely slow so please use this flag wisely.");

namespace paddle {
namespace framework {

std::vector<std::tuple<platform::Place, LibraryType>> kKernelPriority = {
    std::make_tuple(platform::CUDAPlace(0), LibraryType::kCUDNN),
    std::make_tuple(platform::CUDAPlace(0), LibraryType::kPlain),
    std::make_tuple(platform::CPUPlace(), LibraryType::kMKLDNN),
    std::make_tuple(platform::CPUPlace(), LibraryType::kPlain),
};

proto::VarType::Type GetDataTypeOfVar(const Variable* var) {
  if (var->IsType<framework::LoDTensor>()) {
    return var->Get<framework::LoDTensor>().type();
  } else if (var->IsType<framework::SelectedRows>()) {
    return var->Get<framework::SelectedRows>().value().type();
  } else {
    PADDLE_THROW("Var should be LoDTensor or SelectedRows");
  }
}

static DDim GetDims(const Scope& scope, const std::string& name,
                    bool get_actual_dim = false) {
  Variable* var = scope.FindVar(name);
  if (var == nullptr) {
    return DDim({-1});
  }

  if (var->IsType<LoDTensor>()) {
    const LoDTensor& tensor = var->Get<LoDTensor>();
    if (UNLIKELY(!tensor.IsInitialized())) {
      return DDim({-1});
    }
    return tensor.dims();
  } else if (var->IsType<SelectedRows>()) {
    if (get_actual_dim) {
      return var->Get<SelectedRows>().value().dims();
    } else {
      return var->Get<SelectedRows>().GetCompleteDims();
    }
  } else {
    return DDim({-1});
  }
}

static bool VarInited(const Scope& scope, const std::string& name) {
  Variable* var = scope.FindVar(name);
  if (var == nullptr) return false;
  return var->IsInitialized();
}

static std::string GetDtype(const Scope& scope, const std::string& name) {
  Variable* var = scope.FindVar(name);
  if (var == nullptr) {
    return "";
  }

  if (var->IsType<LoDTensor>()) {
    const LoDTensor& tensor = var->Get<LoDTensor>();
    if (UNLIKELY(!tensor.IsInitialized())) {
      return "";
    }
    return DataTypeToString(tensor.type());
  } else if (var->IsType<SelectedRows>()) {
    auto tensor = var->Get<SelectedRows>().value();
    if (UNLIKELY(!tensor.IsInitialized())) {
      return "uninited";
    } else {
      return DataTypeToString(tensor.type());
    }
  } else {
    return "";
  }
}

static int GetRowSize(const Scope& scope, const std::string& name) {
  Variable* var = scope.FindVar(name);
  if (var == nullptr) {
    return -1;
  }

  if (var->IsType<SelectedRows>()) {
    return var->Get<SelectedRows>().rows().size();
  }

  return -1;
}

static LoD GetLoD(const Scope& scope, const std::string& name) {
  Variable* var = scope.FindVar(name);
  auto default_lod = LoD({{}});

  if (var == nullptr) {
    return default_lod;
  }

  if (var->IsType<LoDTensor>()) {
    const LoDTensor& tensor = var->Get<LoDTensor>();
    if (UNLIKELY(!tensor.IsInitialized())) {
      return default_lod;
    }
    return tensor.lod();
  } else {
    return default_lod;
  }
}

RuntimeContext::RuntimeContext(const VariableNameMap& innames,
                               const VariableNameMap& outnames,
                               const Scope& scope) {
  for (auto& var_name_item : innames) {
    std::vector<Variable*>& input_vars = inputs[var_name_item.first];
    input_vars.reserve(var_name_item.second.size());
    for (auto& var_name : var_name_item.second) {
      input_vars.push_back(scope.FindVar(var_name));
    }
  }
  for (auto& var_name_item : outnames) {
    std::vector<Variable*>& output_vars = outputs[var_name_item.first];
    output_vars.reserve(var_name_item.second.size());
    for (auto& var_name : var_name_item.second) {
      output_vars.push_back(scope.FindVar(var_name));
    }
  }
}

void OperatorBase::Run(const Scope& scope, const platform::Place& place) {
  VLOG(4) << place << " " << DebugStringEx(&scope);
  if (platform::is_gpu_place(place)) {
#ifndef PADDLE_WITH_CUDA
    PADDLE_THROW("Cannot run operator on place %s", place);
#else
    auto dev_id = boost::get<platform::CUDAPlace>(place).device;
    platform::SetDeviceId(dev_id);
#endif
  }

  // The profile has a process-wide mutex, results in serious performance issue
  // in concurrency scenerio. Here use an `if` to fix this issue.
  // Please not remove the `if`, ask @Superjomn if there are any concern.
  if (platform::IsProfileEnabled()) {
    platform::DeviceContextPool& pool = platform::DeviceContextPool::Instance();
    platform::RecordEvent record_event(Type(), pool.Get(place));
    RunImpl(scope, place);
  } else {
    RunImpl(scope, place);
  }
  VLOG(3) << place << " " << DebugStringEx(&scope);
}

bool OperatorBase::HasInputs(const std::string& name) const {
  return inputs_.find(name) != inputs_.end();
}

std::string OperatorBase::Input(const std::string& name) const {
  auto& ins = Inputs(name);
  PADDLE_ENFORCE_LE(ins.size(), 1UL,
                    "Operator %s's input %s should contain only one variable.",
                    type_, name);
  return ins.empty() ? kEmptyVarName : ins[0];
}

const std::vector<std::string>& OperatorBase::Inputs(
    const std::string& name) const {
  auto it = inputs_.find(name);
  PADDLE_ENFORCE(it != inputs_.end(), "Operator %s does not have the input %s.",
                 type_, name);
  return it->second;
}

bool OperatorBase::HasOutputs(const std::string& name) const {
  if (outputs_.find(name) != outputs_.end()) {
    return true;
  } else {
    return false;
  }
}

std::string OperatorBase::Output(const std::string& name) const {
  auto& outs = Outputs(name);
  PADDLE_ENFORCE_LE(outs.size(), 1UL,
                    "Operator %s's output %s should contain only one variable.",
                    type_, name);
  return outs.empty() ? kEmptyVarName : outs[0];
}

const std::vector<std::string>& OperatorBase::Outputs(
    const std::string& name) const {
  auto it = outputs_.find(name);
  PADDLE_ENFORCE(it != outputs_.end(),
                 "Operator %s does not have an output called %s.", type_, name);
  return it->second;
}

std::string OperatorBase::DebugStringEx(const Scope* scope) const {
  std::stringstream ss;
  ss << "Op(" << type_ << "), inputs:{";
  for (auto it = inputs_.begin(); it != inputs_.end();) {
    auto& input = *it;
    ss << input.first << "[";
    for (size_t i = 0; i < input.second.size(); ++i) {
      auto var_name = input.second[i];
      ss << var_name;
      if (scope) {
        if (!VarInited(*scope, var_name)) {
          ss << "[uninited]";
        } else {
          int row_size = GetRowSize(*scope, var_name);
          if (row_size >= 0) {
            ss << "[row_size=" << row_size << "]";
          }
          std::string dtype = GetDtype(*scope, var_name);
          ss << ":" << dtype;
          ss << "[" << GetDims(*scope, var_name, true) << "]";
          ss << "(" << GetLoD(*scope, var_name) << ")";
        }
      }
      if (i != input.second.size() - 1) {
        ss << ", ";
      }
    }
    ss << "]";
    ++it;
    if (it != inputs_.end()) {
      ss << ", ";
    }
  }
  ss << "}, outputs:{";
  for (auto it = outputs_.begin(); it != outputs_.end();) {
    auto& output = *it;
    ss << output.first << "[";
    for (size_t i = 0; i < output.second.size(); ++i) {
      auto var_name = output.second[i];
      ss << var_name;
      if (scope) {
        if (!VarInited(*scope, var_name)) {
          ss << "[uninited]";
        } else {
          int row_size = GetRowSize(*scope, output.second[i]);
          if (row_size >= 0) {
            ss << "[row_size=" << row_size << "]";
          }
          std::string dtype = GetDtype(*scope, output.second[i]);
          ss << ":" << dtype;
          ss << "[" << GetDims(*scope, var_name, true) << "]";
          ss << "(" << GetLoD(*scope, var_name) << ")";
        }
      }
      if (i != output.second.size() - 1) {
        ss << ", ";
      }
    }
    ss << "]";
    ++it;
    if (it != outputs_.end()) {
      ss << ", ";
    }
  }
  ss << "}.";
  return ss.str();
}

OperatorBase::OperatorBase(const std::string& type,
                           const VariableNameMap& inputs,
                           const VariableNameMap& outputs,
                           const AttributeMap& attrs)
    : type_(type), inputs_(inputs), outputs_(outputs), attrs_(attrs) {
  GenerateTemporaryNames();
  CheckAllInputOutputSet();
}

std::vector<std::string> OperatorBase::InputVars() const {
  std::vector<std::string> ret_val;
  for (auto& o : inputs_) {
    ret_val.reserve(ret_val.size() + o.second.size());
    ret_val.insert(ret_val.end(), o.second.begin(), o.second.end());
  }
  return ret_val;
}

std::vector<std::string> OperatorBase::OutputVars(bool has_intermediate) const {
  std::vector<std::string> ret_val;
  if (has_intermediate) {
    // push all outputs into ret_val
    for (auto& o : outputs_) {
      ret_val.reserve(ret_val.size() + o.second.size());
      ret_val.insert(ret_val.end(), o.second.begin(), o.second.end());
    }
    return ret_val;
  }
  auto& info = OpInfoMap::Instance().Get(Type());

  // get all OpProto::Var for outputs
  for (auto& o : info.Proto().outputs()) {
    // ignore all intermediate output
    if (o.intermediate()) continue;
    auto out = outputs_.find(o.name());
    if (out != outputs_.end()) {
      ret_val.reserve(ret_val.size() + out->second.size());
      ret_val.insert(ret_val.end(), out->second.begin(), out->second.end());
    }
  }
  return ret_val;
}

void OperatorBase::CheckAllInputOutputSet() const {
  auto& info_map = OpInfoMap::Instance();
  auto* op_info = info_map.GetNullable(Type());
  if (op_info == nullptr || op_info->proto_ == nullptr) return;

  for (auto& in : op_info->Proto().inputs()) {
    if (!in.dispensable()) {
      PADDLE_ENFORCE(inputs_.find(in.name()) != inputs_.end(),
                     "Operator %s's input, %s, is not set", Type(), in.name());
    }
  }

  for (auto& out : op_info->Proto().outputs()) {
    if (!out.dispensable()) {
      PADDLE_ENFORCE(outputs_.find(out.name()) != outputs_.end(),
                     "Operator %s's output, %s, is not set", Type(),
                     out.name());
    }
  }
}

void OperatorBase::GenerateTemporaryNames() {
  static std::atomic<size_t> gUniqId(0UL);
  for (auto& output : outputs_) {
    for (auto& output_name : output.second) {
      if (output_name == kTempVarName) {
        output_name += type_;
        output_name += "@";
        output_name += std::to_string(gUniqId.fetch_add(1));
      }
    }
  }
}

static bool VarIsTensor(const Variable& var) {
  return var.IsType<LoDTensor>() || var.IsType<SelectedRows>();
}

const Tensor* GetLoDTensorOrSelectedRowsValueFromVar(const Variable& var) {
  if (var.IsType<LoDTensor>()) {
    return static_cast<const Tensor*>(&(var.Get<LoDTensor>()));
  } else if (var.IsType<SelectedRows>()) {
    return &(var.Get<SelectedRows>().value());
  } else {
    PADDLE_THROW("Variable type_id %s, expect LoDTensor/SelectedRows.",
                 ToTypeName(var.Type()));
  }
}

Tensor* GetMutableLoDTensorOrSelectedRowsValueFromVar(Variable* var) {
  if (var->IsType<LoDTensor>()) {
    return var->GetMutable<LoDTensor>();
  } else if (var->IsType<SelectedRows>()) {
    return var->GetMutable<SelectedRows>()->mutable_value();
  } else {
    PADDLE_THROW("Variable type_id %s, expect LoDTensor/SelectedRows.",
                 ToTypeName(var->Type()));
  }
}

bool ExecutionContext::HasInput(const std::string& name) const {
  if (!op_.HasInputs(name)) {
    return false;
  }
  auto& ins = Inputs(name);
  size_t length = ins.size();
  if (length == 0) {
    return false;
  }
  PADDLE_ENFORCE_EQ(length, 1UL,
                    "Input %s should not have more than one inputs", name);
  auto arg = ins[0];
  auto* var = arg == kEmptyVarName ? nullptr : scope_.FindVar(arg);
  return var != nullptr;
}

bool ExecutionContext::HasOutput(const std::string& name) const {
  if (!op_.HasOutputs(name)) {
    return false;
  }
  auto& outs = Outputs(name);
  size_t length = outs.size();
  if (length == 0) {
    return false;
  }
  PADDLE_ENFORCE_EQ(length, 1UL,
                    "Output %s should not have more than one inputs", name);
  auto arg = outs[0];
  auto* var = arg == kEmptyVarName ? nullptr : scope_.FindVar(arg);
  return var != nullptr;
}

const Variable* ExecutionContext::InputVar(const std::string& name) const {
  auto it = ctx_.inputs.find(name);
  if (it == ctx_.inputs.end()) return nullptr;

  PADDLE_ENFORCE_LE(it->second.size(), 1UL,
                    "Operator %s's input %s should contain only one variable.",
                    op_.Type(), name);
  return it->second.empty() ? nullptr : it->second[0];
}

const Variable* ExecutionContext::LegacyInputVar(
    const std::string& name) const {
  auto ipt = op_.Input(name);
  return ipt == kEmptyVarName ? nullptr : scope_.FindVar(ipt);
}

Variable* ExecutionContext::OutputVar(const std::string& name) const {
  auto it = ctx_.outputs.find(name);
  if (it == ctx_.outputs.end()) return nullptr;

  PADDLE_ENFORCE_LE(it->second.size(), 1UL,
                    "Operator %s's output %s should contain only one variable.",
                    op_.Type(), name);
  return it->second.empty() ? nullptr : it->second[0];
}

Variable* ExecutionContext::LegacyOutputVar(const std::string& name) const {
  auto opt = op_.Output(name);
  return opt == kEmptyVarName ? nullptr : scope_.FindVar(opt);
}

template <>
const Tensor* ExecutionContext::Input<Tensor>(const std::string& name) const {
  return Input<LoDTensor>(name);
}

template <>
const Tensor* ExecutionContext::LegacyInput<Tensor>(
    const std::string& name) const {
  return LegacyInput<LoDTensor>(name);
}

template <>
const std::vector<const Tensor*> ExecutionContext::MultiInput<Tensor>(
    const std::string& name) const {
  auto it = ctx_.inputs.find(name);
  if (it == ctx_.inputs.end()) {
    return {};
  }
  const std::vector<Variable*>& vars = it->second;
  std::vector<const Tensor*> res;
  res.reserve(vars.size());
  std::transform(vars.begin(), vars.end(), std::back_inserter(res),
                 [&](Variable* var) -> const Tensor* {
                   if (var == nullptr) return nullptr;
                   PADDLE_ENFORCE(
                       var->IsType<LoDTensor>(),
                       "should be LoDTensor, but the received type is %s",
                       ToTypeName(var->Type()));
                   return &(var->Get<LoDTensor>());
                 });
  return res;
}

template <>
const std::vector<const Tensor*> ExecutionContext::LegacyMultiInput<Tensor>(
    const std::string& name) const {
  auto names = op().Inputs(name);
  std::vector<const Tensor*> res;
  res.reserve(names.size());
  std::transform(names.begin(), names.end(), std::back_inserter(res),
                 [&](const std::string& sub_name) -> const Tensor* {
                   auto var = scope_.FindVar(sub_name);
                   if (var == nullptr) return nullptr;
                   PADDLE_ENFORCE(
                       var->IsType<LoDTensor>(),
                       "%s should be LoDTensor, but the received type is %s",
                       sub_name, ToTypeName(var->Type()));
                   return &(var->Get<LoDTensor>());
                 });
  return res;
}

template <>
Tensor* ExecutionContext::Output<Tensor>(const std::string& name) const {
  return Output<LoDTensor>(name);
}

template <>
Tensor* ExecutionContext::LegacyOutput<Tensor>(const std::string& name) const {
  return LegacyOutput<LoDTensor>(name);
}

template <>
std::vector<Tensor*> ExecutionContext::MultiOutput<Tensor>(
    const std::string& name) const {
  auto names = op().Outputs(name);
  std::vector<Tensor*> res;
  res.reserve(names.size());
  std::transform(names.begin(), names.end(), std::back_inserter(res),
                 [&](const std::string& sub_name) -> Tensor* {
                   auto var = scope_.FindVar(sub_name);
                   if (var == nullptr) return nullptr;
                   PADDLE_ENFORCE(
                       var->IsType<LoDTensor>(),
                       "%s should be LoDTensor, but the received type is %s",
                       sub_name, ToTypeName(var->Type()));
                   return var->GetMutable<LoDTensor>();
                 });
  return res;
}

bool OpSupportGPU(const std::string& op_type) {
  auto& all_kernels = OperatorWithKernel::AllOpKernels();
  auto it = all_kernels.find(op_type);
  if (it == all_kernels.end()) {
    // All control operator must support GPU
    return true;
  }
  for (auto& kern_pair : it->second) {
    if (platform::is_gpu_place(kern_pair.first.place_)) {
      return true;
    }
  }
  return false;
}

class RuntimeInferShapeContext : public InferShapeContext {
 public:
  RuntimeInferShapeContext(const OperatorBase& op, const Scope& scope,
                           const RuntimeContext& ctx)
      : op_(op), scope_(scope), ctx_(ctx) {}

  bool HasInput(const std::string& name) const override {
    // has only one input
    const auto& ins = ctx_.inputs;
    auto it = ins.find(name);
    if (it == ins.end()) {
      return false;
    }
    const auto& in = it->second;
    if (in.size() == 0) return false;
    PADDLE_ENFORCE_EQ(in.size(), 1UL,
                      "Input %s should not have more than one inputs", name);
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
    PADDLE_ENFORCE_EQ(out.size(), 1UL,
                      "Output %s should not have more than one outputs", name);
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

  const std::vector<std::string>& Inputs(
      const std::string& name) const override {
    return op_.Inputs(name);
  }

  const std::vector<std::string>& Outputs(
      const std::string& name) const override {
    return op_.Outputs(name);
  }

  void ShareDim(const std::string& in, const std::string& out, size_t i = 0,
                size_t j = 0) override {
    auto in_it = ctx_.inputs.find(in);
    auto out_it = ctx_.outputs.find(out);
    PADDLE_ENFORCE(in_it != ctx_.inputs.end() && in_it->second.size() > i,
                   "Inputs %s should have %llu argument", in, i);
    PADDLE_ENFORCE(out_it != ctx_.outputs.end() && out_it->second.size() > j,
                   "Outputs %s should have %llu argument", out, j);

    Variable* in_var = in_it->second[i];
    Variable* out_var = out_it->second[j];

    PADDLE_ENFORCE(in_var->Type() == out_var->Type(),
                   "The type of %s and %s is not the same.", in, out);

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
      PADDLE_THROW(
          "Currently, the input type of ShareDim only can be LoDTensor "
          "or SelectedRows.");
    }
  }

  void ShareLoD(const std::string& in, const std::string& out, size_t i = 0,
                size_t j = 0) const override {
    auto in_it = ctx_.inputs.find(in);
    auto out_it = ctx_.outputs.find(out);
    PADDLE_ENFORCE(in_it != ctx_.inputs.end() && in_it->second.size() > i,
                   "Inputs %s should have %llu argument", in, i);
    PADDLE_ENFORCE(out_it != ctx_.outputs.end() && out_it->second.size() > j,
                   "Outputs %s should have %llu argument", out, j);

    Variable* in_var = in_it->second.at(i);
    if (!in_var->IsType<LoDTensor>()) return;
    Variable* out_var = out_it->second.at(j);
    PADDLE_ENFORCE(out_var->IsType<LoDTensor>(),
                   "The %d-th output of Output(%s) must be LoDTensor.", j, out);
    auto in_tensor = in_var->Get<LoDTensor>();
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

  void DecreaseLoDLevel(const std::string& in, const std::string& out,
                        size_t i = 0, size_t j = 0) const override {
    PADDLE_THROW("DecreaseLoDLevel is only used in compile time.");
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
    PADDLE_ENFORCE_EQ(vars.size(), 1UL,
                      "Input(%s) should hold one element, but now it holds %d",
                      name, vars.size());
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

  DDim GetOutputDim(const std::string& name) const override {
    const std::vector<Variable*>& vars = OutputVars(name);
    PADDLE_ENFORCE_EQ(vars.size(), 1UL,
                      "Input(%s) should hold one element, but now it holds %d",
                      name, vars.size());
    return this->GetDim(vars[0]);
  }

  std::vector<DDim> GetOutputsDim(const std::string& name) const override {
    const std::vector<Variable*>& vars = OutputVars(name);
    return GetDims(vars);
  }

  void SetOutputDim(const std::string& name, const DDim& dim) override {
    auto& vars = OutputVars(name);
    PADDLE_ENFORCE_EQ(vars.size(), 1UL,
                      "Output(%s) should hold one element, but now it holds %d",
                      name, vars.size());
    SetDim(vars[0], dim);
  }

  void SetOutputsDim(const std::string& name,
                     const std::vector<DDim>& dims) override {
    auto& vars = OutputVars(name);
    SetDims(vars, dims);
  }

 protected:
  DDim GetDim(Variable* var) const {
    PADDLE_ENFORCE_NOT_NULL(var);
    if (var->IsType<LoDTensor>()) {
      return var->Get<LoDTensor>().dims();
    } else if (var->IsType<SelectedRows>()) {
      return var->Get<SelectedRows>().GetCompleteDims();
    } else {
      PADDLE_THROW(
          "Only LoDTensor/SelectedRows support 'GetDim', but Variables "
          "type_id is %s.",
          ToTypeName(var->Type()));
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
    PADDLE_THROW("Only compile time support this method");
  }

  void SetDim(Variable* var, const DDim& dim) {
    if (var->IsType<LoDTensor>()) {
      var->GetMutable<LoDTensor>()->Resize(dim);
    } else if (var->IsType<SelectedRows>()) {
      var->GetMutable<SelectedRows>()->set_height(dim[0]);
    } else {
      PADDLE_THROW("Variable type_id %s, expect LoDTensor/SelectedRows.",
                   ToTypeName(var->Type()));
    }
  }

  void SetDims(const std::vector<Variable*>& vars,
               const std::vector<DDim>& dims) {
    size_t length = vars.size();
    PADDLE_ENFORCE_EQ(length, dims.size());
    for (size_t i = 0; i < length; ++i) {
      if (vars[i] == nullptr) {
        continue;
      }
      SetDim(vars[i], dims[i]);
    }
  }

  void SetRepeatedDims(const std::string& name,
                       const std::vector<DDim>& dims) override {
    PADDLE_THROW("Only compile time support this method");
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
    PADDLE_ENFORCE(it != ctx_.inputs.end(),
                   "Operator %s does not have the input %s.", op_.Type(), name);
    return it->second;
  }

  const std::vector<Variable*>& OutputVars(const std::string& name) const {
    auto it = ctx_.outputs.find(name);
    PADDLE_ENFORCE(it != ctx_.outputs.end(),
                   "Operator %s does not have the outputs %s.", op_.Type(),
                   name);
    return it->second;
  }

  const OperatorBase& op_;
  const Scope& scope_;
  const RuntimeContext& ctx_;
};

static void CheckTensorNANOrInf(const std::string& name,
                                const framework::Tensor& tensor) {
  if (tensor.memory_size() == 0) {
    return;
  }
  if (tensor.type() != proto::VarType::FP32 &&
      tensor.type() != proto::VarType::FP64) {
    return;
  }
  PADDLE_ENFORCE(!framework::TensorContainsInf(tensor),
                 "Tensor %s contains Inf", name);
  PADDLE_ENFORCE(!framework::TensorContainsNAN(tensor),
                 "Tensor %s contains NAN", name);
}

void OperatorWithKernel::RuntimeInferShape(const Scope& scope,
                                           const platform::Place& place,
                                           const RuntimeContext& ctx) const {
  RuntimeInferShapeContext infer_shape_ctx(*this, scope, ctx);
  this->InferShape(&infer_shape_ctx);
}

void OperatorWithKernel::RunImpl(const Scope& scope,
                                 const platform::Place& place) const {
  RuntimeContext ctx(Inputs(), Outputs(), scope);
  platform::DeviceContextPool& pool = platform::DeviceContextPool::Instance();
  auto* dev_ctx = pool.Get(place);

  // check if op[type] has kernel registered.
  auto& all_op_kernels = AllOpKernels();
  auto kernels_iter = all_op_kernels.find(type_);
  if (kernels_iter == all_op_kernels.end()) {
    PADDLE_THROW(
        "There are no kernels which are registered in the %s operator.", type_);
  }

  OpKernelMap& kernels = kernels_iter->second;

  auto expected_kernel_key = this->GetExpectedKernelType(
      ExecutionContext(*this, scope, *dev_ctx, ctx));
  VLOG(3) << "expected_kernel_key:" << expected_kernel_key;

  auto kernel_iter = kernels.find(expected_kernel_key);
#ifdef PADDLE_WITH_MKLDNN
  // workaround for missing MKLDNN kernel when FLAGS_use_mkldnn env var is set
  if (kernel_iter == kernels.end() &&
      expected_kernel_key.library_type_ == LibraryType::kMKLDNN) {
    VLOG(3) << "missing MKLDNN kernel: fallbacking to PLAIN one";
    expected_kernel_key.library_type_ = LibraryType::kPlain;
    expected_kernel_key.data_layout_ = DataLayout::kAnyLayout;
    kernel_iter = kernels.find(expected_kernel_key);
  }
#endif
  if (kernel_iter == kernels.end()) {
    PADDLE_THROW("op %s does not have kernel for %s", type_,
                 KernelTypeToString(expected_kernel_key));
  }

  // do data transformScope &transfer_scope;
  std::vector<std::string> transfered_inplace_vars;
  auto* transfer_scope =
      PrepareData(scope, expected_kernel_key, &transfered_inplace_vars, &ctx);

  // exec scope is the scope that kernel actually executed on.
  const Scope& exec_scope =
      (transfer_scope == nullptr ? scope : *transfer_scope);

  if (!(expected_kernel_key.place_ == dev_ctx->GetPlace())) {
    dev_ctx = pool.Get(expected_kernel_key.place_);
  }

  RuntimeInferShapeContext infer_shape_ctx(*this, exec_scope, ctx);
  this->InferShape(&infer_shape_ctx);
  if (platform::IsProfileEnabled()) {
    std::string ename =
        string::Sprintf("%s%s", Type(), Attr<std::string>("op_namescope"));
    platform::PushEvent(ename, dev_ctx, EstimateFlops(&infer_shape_ctx));

    kernel_iter->second(ExecutionContext(*this, exec_scope, *dev_ctx, ctx));

    platform::PopEvent(ename, dev_ctx);
  } else {
    // TODO(panyx0718): ExecutionContext should only depend on RuntimeContext
    // not Scope. Imperative mode only pass inputs and get outputs.
    kernel_iter->second(ExecutionContext(*this, exec_scope, *dev_ctx, ctx));
  }

  if (!transfered_inplace_vars.empty()) {
    // there is inplace variable has been transfered.
    TransferInplaceVarsBack(scope, transfered_inplace_vars, *transfer_scope);
  }

  /*For profiling/benchmark only*/
  if (FLAGS_benchmark) {
    dev_ctx->Wait();
  }

  if (FLAGS_check_nan_inf) {
    for (auto& vname : OutputVars(true)) {
      auto* var = exec_scope.FindVar(vname);
      if (var == nullptr) continue;
      if (var->IsType<framework::LoDTensor>()) {
        CheckTensorNANOrInf(vname, var->Get<framework::LoDTensor>());
      } else if (var->IsType<framework::SelectedRows>()) {
        CheckTensorNANOrInf(vname, var->Get<framework::SelectedRows>().value());
      }
    }
  }
}

void OperatorWithKernel::TransferInplaceVarsBack(
    const Scope& scope, const std::vector<std::string>& inplace_vars,
    const Scope& transfer_scope) const {
  for (auto& var_name : inplace_vars) {
    VLOG(3) << "share inplace var " + var_name + " back to it's original scope";
    auto* original_tensor =
        GetMutableLoDTensorOrSelectedRowsValueFromVar(scope.FindVar(var_name));
    auto* var = transfer_scope.FindVar(var_name);
    PADDLE_ENFORCE(var != nullptr, "The var[%s] should not be nullptr",
                   var_name);
    auto* transformed_tensor = GetLoDTensorOrSelectedRowsValueFromVar(*var);
    original_tensor->ShareDataWith(*transformed_tensor);
  }
}

Scope* OperatorWithKernel::PrepareData(
    const Scope& scope, const OpKernelType& expected_kernel_key,
    std::vector<std::string>* transfered_inplace_vars,
    RuntimeContext* ctx) const {
  Scope* new_scope = nullptr;
  for (auto& var_name_item : Inputs()) {
    std::vector<Variable*>& input_vars = ctx->inputs[var_name_item.first];

    for (size_t i = 0; i < var_name_item.second.size(); ++i) {
      auto& var_name = var_name_item.second[i];
      auto* var = input_vars[i];

      // Only tensor can be tranfer to another device.
      if (var == nullptr || !VarIsTensor(*var)) {
        continue;
      }

      auto* tensor_in = GetLoDTensorOrSelectedRowsValueFromVar(*var);
      if (!tensor_in->IsInitialized()) {
        continue;
      }

      auto kernel_type_for_var = GetKernelTypeForVar(
          var_name_item.first, *tensor_in, expected_kernel_key);

      if (!NeedTransform(kernel_type_for_var, expected_kernel_key)) {
        continue;
      }

      auto out_var_names = OutputVars(true);
      if (std::find(out_var_names.begin(), out_var_names.end(), var_name) !=
          out_var_names.end()) {
        transfered_inplace_vars->emplace_back(var_name);
      }

      VLOG(3) << "Transform Variable " << var_name << " from "
              << kernel_type_for_var << " to " << expected_kernel_key;

      // In the inference scenerio, the scopes will be reused across the
      // batches, so the `new_scope` here will result in GPU memroy explosion
      // over the  running of operators.
      // We use a thread_local cache to fix that issue, the key in the cache is
      // the combination of the `scope` argument, from_kernel_type,
      // target_kernel_type.
      // Have a discussion with @Superjomn or the inference developers if some
      // changes on this logic for this macro might not tested on the other
      // scenerios.
      // If this op is not called by an Executor or ParallelExecutor, it should
      // called by a NaiveExecutor, the NaiveExecutor will cache the scopes and
      // variables, that behavior a lot different.
      if (!run_by_executor_) {
        new_scope = TryCreateTransferScope(kernel_type_for_var,
                                           expected_kernel_key, &scope);
      }
      if (!new_scope) {
        new_scope = &scope.NewScope();
      }

      auto* trans_var = new_scope->Var(var_name);
      input_vars[i] = trans_var;

      Tensor out;
      TransformData(expected_kernel_key, kernel_type_for_var, *tensor_in, &out);
      SetTensorToVariable(*var, out, trans_var);
    }
  }

  return new_scope;
}

proto::VarType::Type OperatorWithKernel::IndicateDataType(
    const ExecutionContext& ctx) const {
  int data_type = -1;
  for (auto& input : this->inputs_) {
    const std::vector<const Variable*> vars = ctx.MultiInputVar(input.first);
    for (size_t i = 0; i < vars.size(); ++i) {
      const Variable* var = vars[i];
      if (var != nullptr) {
        const Tensor* t = nullptr;
        if (var->IsType<Tensor>()) {
          t = &var->Get<Tensor>();
        } else if (var->IsType<LoDTensor>()) {
          t = &var->Get<LoDTensor>();
        } else if (var->IsType<SelectedRows>()) {
          t = &(var->Get<SelectedRows>().value());
        }
        if (t != nullptr) {
          PADDLE_ENFORCE(t->IsInitialized(), "Input %s(%lu)is not initialized",
                         input.first, i);
          int tmp = static_cast<int>(t->type());
          PADDLE_ENFORCE(
              tmp == data_type || data_type == -1,
              "DataType of Paddle Op %s must be the same. Get (%d) != (%d)",
              Type(), data_type, tmp);
          data_type = tmp;
        }
      }
    }
  }
  PADDLE_ENFORCE(data_type != -1, "DataType should be indicated by input");
  return static_cast<proto::VarType::Type>(data_type);
}

OpKernelType OperatorWithKernel::GetExpectedKernelType(
    const ExecutionContext& ctx) const {
  return OpKernelType(IndicateDataType(ctx), ctx.GetPlace());
}

OpKernelType OperatorWithKernel::GetKernelTypeForVar(
    const std::string& var_name, const Tensor& tensor,
    const OpKernelType& expected_kernel_type) const {
  return OpKernelType(expected_kernel_type.data_type_, tensor.place(),
                      tensor.layout());
}

}  // namespace framework
}  // namespace paddle
