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

#include "paddle/fluid/framework/operator.h"

#include <glog/logging.h>
#include <sstream>
#include <string>

#include "gflags/gflags.h"
#include "paddle/fluid/framework/data_transform.h"
#include "paddle/fluid/framework/data_type_transform.h"
#include "paddle/fluid/framework/details/nan_inf_utils.h"
#include "paddle/fluid/framework/op_call_stack.h"
#include "paddle/fluid/framework/shape_inference.h"
#include "paddle/fluid/framework/transfer_scope_cache.h"
#include "paddle/fluid/framework/unused_var_check.h"
#include "paddle/fluid/framework/var_type.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/profiler.h"

namespace paddle {
namespace framework {
class LoDTensor;
}  // namespace framework
}  // namespace paddle
#ifdef PADDLE_WITH_XPU
#include "paddle/fluid/platform/xpu_info.h"
#endif

#ifdef PADDLE_WITH_MKLDNN
#include "paddle/fluid/platform/mkldnn_helper.h"
#endif

DECLARE_bool(benchmark);
DECLARE_bool(check_nan_inf);
DECLARE_bool(enable_unused_var_check);
DEFINE_int32(inner_op_parallelism, 0, "number of threads for inner op");

namespace paddle {
namespace framework {

std::vector<std::tuple<platform::Place, LibraryType>> kKernelPriority = {
    std::make_tuple(platform::CUDAPlace(0), LibraryType::kCUDNN),
    std::make_tuple(platform::CUDAPlace(0), LibraryType::kPlain),
    std::make_tuple(platform::CPUPlace(), LibraryType::kMKLDNN),
    std::make_tuple(platform::CPUPlace(), LibraryType::kPlain),
};

static DDim GetDimsDebug(const Scope& scope, const std::string& name,
                         bool get_actual_dim = false) {
  Variable* var = scope.FindVar(name);
  if (var == nullptr) {
    return DDim({-1});
  }

  if (var->IsType<LoDTensor>()) {
    const LoDTensor& tensor = var->Get<LoDTensor>();
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

static std::string GetPlace(const Scope& scope, const std::string& name) {
  Variable* var = scope.FindVar(name);
  if (var == nullptr) {
    return "";
  }
  auto to_string = [](const platform::Place& p) {
    std::stringstream sstream;
    sstream << p;
    return sstream.str();
  };

  if (var->IsType<LoDTensor>()) {
    const LoDTensor& tensor = var->Get<LoDTensor>();
    if (UNLIKELY(!tensor.IsInitialized())) {
      return "";
    }
    return to_string(tensor.place());
  } else if (var->IsType<SelectedRows>()) {
    auto tensor = var->Get<SelectedRows>().value();
    if (UNLIKELY(!tensor.IsInitialized())) {
      return "uninited";
    } else {
      return to_string(tensor.place());
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

static LoD GetLoDDebug(const Scope& scope, const std::string& name) {
  Variable* var = scope.FindVar(name);
  auto default_lod = LoD({{}});

  if (var == nullptr) {
    return default_lod;
  }

  if (var->IsType<LoDTensor>()) {
    const LoDTensor& tensor = var->Get<LoDTensor>();
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
  try {
    VLOG(4) << place << " " << DebugStringEx(&scope);
    if (platform::is_gpu_place(place)) {
#if !defined(PADDLE_WITH_CUDA) && !defined(PADDLE_WITH_HIP)
      PADDLE_THROW(platform::errors::Unavailable(
          "Cannot run operator on place %s, please recompile paddle or "
          "reinstall Paddle with CUDA support.",
          place));
#else
      auto dev_id = BOOST_GET_CONST(platform::CUDAPlace, place).device;
      platform::SetDeviceId(dev_id);
#endif
    } else if (platform::is_xpu_place(place)) {
#ifndef PADDLE_WITH_XPU
      PADDLE_THROW(platform::errors::Unavailable(
          "Cannot run operator on place %s, please recompile paddle or "
          "reinstall Paddle with XPU support.",
          place));
#else
      auto dev_id = BOOST_GET_CONST(platform::XPUPlace, place).device;
      platform::SetXPUDeviceId(dev_id);
#endif
    } else if (platform::is_npu_place(place)) {
#ifndef PADDLE_WITH_ASCEND_CL
      PADDLE_THROW(platform::errors::Unavailable(
          "Cannot run operator on place %s, please recompile paddle or "
          "reinstall Paddle with NPU support.",
          place));
#else
      auto dev_id = BOOST_GET_CONST(platform::NPUPlace, place).device;
      platform::SetNPUDeviceId(dev_id);
#endif
    }

    {
      // TODO(wangchaochaohu) : refine code to use only one RecordEvent)
      // in order to record different op type cost time
      // and different op name cost time,we set two event.
      platform::RecordEvent op_type_record_event(Type());
      auto op_name = platform::OpName(outputs_, Type());
      platform::RecordEvent op_name_record_event(
          op_name, platform::EventRole::kUniqueOp);
      RunImpl(scope, place);
    }

    VLOG(3) << GetExecutionPlace(place) << " " << DebugStringEx(&scope);
  } catch (platform::EnforceNotMet& exception) {
    framework::InsertCallStackInfo(Type(), Attrs(), &exception);
    throw std::move(exception);
  } catch (platform::EOFException&) {
    std::rethrow_exception(std::current_exception());
  } catch (std::exception& ex) {
    LOG(WARNING) << Type() << " raises an exception "
                 << platform::demangle(typeid(ex).name()) << ", " << ex.what();
    std::rethrow_exception(std::current_exception());
  } catch (...) {
    LOG(WARNING) << Type() << " raises an unknown exception";
    std::rethrow_exception(std::current_exception());
  }
}

bool OperatorBase::HasInputs(const std::string& name) const {
  return inputs_.find(name) != inputs_.end();
}

std::string OperatorBase::Input(const std::string& name) const {
  auto& ins = Inputs(name);
  PADDLE_ENFORCE_LE(
      ins.size(), 1UL,
      platform::errors::InvalidArgument(
          "Operator %s's input %s should contain only one variable.", type_,
          name));
  return ins.empty() ? kEmptyVarName : ins[0];
}

const std::vector<std::string>& OperatorBase::Inputs(
    const std::string& name) const {
  auto it = inputs_.find(name);
  PADDLE_ENFORCE_NE(
      it, inputs_.end(),
      platform::errors::NotFound("Operator %s does not have the input %s.",
                                 type_, name));
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
  PADDLE_ENFORCE_LE(
      outs.size(), 1UL,
      platform::errors::InvalidArgument(
          "Operator %s's output %s should contain only one variable.", type_,
          name));
  return outs.empty() ? kEmptyVarName : outs[0];
}

const std::vector<std::string>& OperatorBase::Outputs(
    const std::string& name) const {
  auto it = outputs_.find(name);
  PADDLE_ENFORCE_NE(
      it, outputs_.end(),
      platform::errors::NotFound(
          "Operator %s does not have an output called %s.", type_, name));
  return it->second;
}

std::string OperatorBase::DebugStringEx(const Scope* scope) const {
  std::stringstream ss;
  ss << "Op(" << type_ << "), inputs:{";

  const std::unordered_set<std::string>* no_need_buffer_vars = nullptr;
  if (info_ && info_->NoNeedBufferVarsInferer()) {
    no_need_buffer_vars =
        &(Info().NoNeedBufferVarsInferer()(Inputs(), Outputs(), Attrs()));
    if (no_need_buffer_vars->empty()) no_need_buffer_vars = nullptr;
  }

  for (auto it = inputs_.begin(); it != inputs_.end();) {
    auto& input = *it;
    bool is_no_need_buffer_var =
        (no_need_buffer_vars && no_need_buffer_vars->count(input.first) > 0);
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
          std::string dtype = is_no_need_buffer_var
                                  ? "unknown_dtype"
                                  : GetDtype(*scope, var_name);
          ss << ":" << dtype;
          ss << "[" << GetDimsDebug(*scope, var_name, true) << "]";
          ss << "(" << GetLoDDebug(*scope, var_name) << ")";
          ss << "(" << GetPlace(*scope, var_name) << ")";
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
          ss << "[" << GetDimsDebug(*scope, var_name, true) << "]";
          ss << "(" << GetLoDDebug(*scope, var_name) << ")";
          ss << "(" << GetPlace(*scope, var_name) << ")";
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
    : type_(type),
      inputs_(inputs),
      outputs_(outputs),
      attrs_(attrs),
      // NOTE(zjl): why op_info may be nullptr?
      info_(OpInfoMap::Instance().GetNullable(type)) {
  // In dygraph mode, all the OperatorBase will be constructed by function:
  // framework::OpRegistry::CreateOp(type, {}, {}, {}, false).
  // Inputs, outputs and attrs will be set to empty map
  // to improve the execution efficiency of dygraph.
  if (inputs_.size() > 0 || outputs_.size() > 0) {
    GenerateTemporaryNames();
    CheckAllInputOutputSet();
  }
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
  auto& info = Info();

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
  if (info_ == nullptr || info_->proto_ == nullptr) return;

  for (auto& in : info_->Proto().inputs()) {
    if (!in.dispensable()) {
      PADDLE_ENFORCE_NE(
          inputs_.find(in.name()), inputs_.end(),
          platform::errors::NotFound("Operator %s's input (%s) is not set.",
                                     Type(), in.name()));
    }
  }

  for (auto& out : info_->Proto().outputs()) {
    if (!out.dispensable()) {
      PADDLE_ENFORCE_NE(
          outputs_.find(out.name()), outputs_.end(),
          platform::errors::NotFound("Operator %s's output (%s) is not set.",
                                     Type(), out.name()));
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
    PADDLE_THROW(platform::errors::InvalidArgument(
        "Variable type is %s, expect LoDTensor or SelectedRows.",
        ToTypeName(var.Type())));
  }
}

Tensor* GetMutableLoDTensorOrSelectedRowsValueFromVar(Variable* var) {
  if (var->IsType<LoDTensor>()) {
    return var->GetMutable<LoDTensor>();
  } else if (var->IsType<SelectedRows>()) {
    return var->GetMutable<SelectedRows>()->mutable_value();
  } else {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "Variable type is %s, expect LoDTensor or SelectedRows.",
        ToTypeName(var->Type())));
  }
}

bool ExecutionContext::HasInput(const std::string& name) const {
  auto* var = InputVar(name);
  return var != nullptr;
}

bool ExecutionContext::HasOutput(const std::string& name) const {
  auto* var = OutputVar(name);
  return var != nullptr;
}

const Variable* ExecutionContext::InputVar(const std::string& name) const {
  LogVarUsageIfUnusedVarCheckEnabled(name);

  auto it = ctx_.inputs.find(name);
  if (it == ctx_.inputs.end()) return nullptr;

  PADDLE_ENFORCE_LE(
      it->second.size(), 1UL,
      platform::errors::InvalidArgument(
          "Operator %s's input %s should contain only one variable.",
          op_.Type(), name));
  return it->second.empty() ? nullptr : it->second[0];
}

Variable* ExecutionContext::OutputVar(const std::string& name) const {
  auto it = ctx_.outputs.find(name);
  if (it == ctx_.outputs.end()) return nullptr;

  PADDLE_ENFORCE_LE(
      it->second.size(), 1UL,
      platform::errors::InvalidArgument(
          "Operator %s's output %s should contain only one variable.",
          op_.Type(), name));
  return it->second.empty() ? nullptr : it->second[0];
}

template <>
const Tensor* ExecutionContext::Input<Tensor>(const std::string& name) const {
  return Input<LoDTensor>(name);
}

template <>
const std::vector<const Tensor*> ExecutionContext::MultiInput<Tensor>(
    const std::string& name) const {
  LogVarUsageIfUnusedVarCheckEnabled(name);

  auto vars = MultiInputVar(name);
  if (vars.size() == 0) {
    return {};
  }
  std::vector<const Tensor*> res;
  res.reserve(vars.size());
  std::transform(vars.begin(), vars.end(), std::back_inserter(res),
                 [&](const Variable* var) -> const Tensor* {
                   if (var == nullptr) return nullptr;
                   PADDLE_ENFORCE_EQ(var->IsType<LoDTensor>(), true,
                                     platform::errors::InvalidArgument(
                                         "Input variable should be LoDTensor, "
                                         "but the received type is %s.",
                                         ToTypeName(var->Type())));
                   return &(var->Get<LoDTensor>());
                 });
  return res;
}

template <>
Tensor* ExecutionContext::Output<Tensor>(const std::string& name) const {
  return Output<LoDTensor>(name);
}

template <>
std::vector<Tensor*> ExecutionContext::MultiOutput<Tensor>(
    const std::string& name) const {
  auto vars = MultiOutputVar(name);

  if (vars.size() == 0) {
    return {};
  }
  std::vector<Tensor*> res;
  res.reserve(vars.size());
  std::transform(vars.begin(), vars.end(), std::back_inserter(res),
                 [&](Variable* var) -> Tensor* {
                   return var == nullptr ? nullptr
                                         : var->GetMutable<LoDTensor>();
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

static void CheckTensorNANOrInf(const std::string& op_type,
                                const std::string& name,
                                const framework::Tensor& tensor) {
  if (tensor.memory_size() == 0) {
    return;
  }
  if (tensor.type() != proto::VarType::FP32 &&
      tensor.type() != proto::VarType::FP64) {
    return;
  }
  PADDLE_ENFORCE_NE(
      framework::TensorContainsInf(tensor), true,
      platform::errors::Fatal("Operator %s output Tensor %s contains Inf.",
                              op_type, name));
  PADDLE_ENFORCE_NE(
      framework::TensorContainsNAN(tensor), true,
      platform::errors::Fatal("Operator %s output Tensor %s contains NAN.",
                              op_type, name));
}

bool OperatorWithKernel::SupportsMKLDNN(
    const proto::VarType::Type data_type) const {
  auto& op_kernels = OperatorWithKernel::AllOpKernels().at(type_);
  return std::any_of(op_kernels.begin(), op_kernels.end(),
                     [data_type](OpKernelMap::const_reference kern_pair) {
                       return platform::is_cpu_place(kern_pair.first.place_) &&
                              kern_pair.first.library_type_ ==
                                  LibraryType::kMKLDNN &&
                              kern_pair.first.data_type_ == data_type;
                     });
}

bool OperatorWithKernel::CanMKLDNNBeUsed(const framework::ExecutionContext& ctx,
                                         proto::VarType::Type data_type) const {
  bool use_mkldnn_ctx =
      ctx.Attr<bool>("use_mkldnn") && platform::is_cpu_place(ctx.GetPlace());
  return use_mkldnn_ctx && this->SupportsMKLDNN(data_type);
}

void OperatorWithKernel::RuntimeInferShape(const Scope& scope,
                                           const platform::Place& place,
                                           const RuntimeContext& ctx) const {
  RuntimeInferShapeContext infer_shape_ctx(*this, ctx);
  this->InferShape(&infer_shape_ctx);
}

void OperatorWithKernel::RunImpl(const Scope& scope,
                                 const platform::Place& place) const {
  // To reduce the elapsed time of HasAttr, we use bool variable to record the
  // result of HasAttr.
  if (!enable_cache_runtime_context_ && HasAttr(kEnableCacheRuntimeContext))
    enable_cache_runtime_context_ = true;
  if (!all_kernels_must_compute_runtime_shape_ &&
      HasAttr(kAllKernelsMustComputeRuntimeShape))
    all_kernels_must_compute_runtime_shape_ = true;
  const Scope* cur_scope = &scope;
  if (!enable_cache_runtime_context_) {
    RuntimeContext ctx(Inputs(), Outputs(), scope);
    RunImpl(scope, place, &ctx);
    pre_scope_ = cur_scope;
  } else {
    if (runtime_ctx_.get() == nullptr || pre_scope_ != cur_scope) {
      std::lock_guard<std::mutex> lock(cache_update_mutex_);
      if (runtime_ctx_.get() == nullptr || pre_scope_ != cur_scope) {
        runtime_ctx_.reset(new RuntimeContext(Inputs(), Outputs(), scope));
        pre_scope_ = cur_scope;
      }
    }
    RunImpl(scope, place, runtime_ctx_.get());
  }
}

void OperatorWithKernel::RunImpl(const Scope& scope,
                                 const platform::Place& place,
                                 RuntimeContext* runtime_ctx) const {
  platform::DeviceContextPool& pool = platform::DeviceContextPool::Instance();
  auto* dev_ctx = pool.Get(place);

  if (kernel_type_.get() == nullptr || kernel_func_.get() == nullptr) {
    ChooseKernel(*runtime_ctx, scope, place);
  }

  // do data transformScope &transfer_scope;
  std::vector<std::string> transfered_inplace_vars;
  Scope* transfer_scope = nullptr;
  {
    platform::RecordEvent record_event("prepare_data",
                                       platform::EventRole::kInnerOp);
    if (need_prepare_data_) {
      transfer_scope = PrepareData(scope, *kernel_type_,
                                   &transfered_inplace_vars, runtime_ctx);
    }
  }
  // exec scope is the scope that kernel actually executed on.
  const Scope& exec_scope =
      (transfer_scope == nullptr ? scope : *transfer_scope);

  if (!(kernel_type_->place_ == dev_ctx->GetPlace())) {
    dev_ctx = pool.Get(kernel_type_->place_);
  }

  if (!all_kernels_must_compute_runtime_shape_) {
    platform::RecordEvent record_event("infer_shape",
                                       platform::EventRole::kInnerOp);
    RuntimeInferShapeContext infer_shape_ctx(*this, *runtime_ctx);
    this->InferShape(&infer_shape_ctx);
  }

  if (FLAGS_enable_unused_var_check) {
    GetThreadLocalUsedVarNameSet()->clear();
  }

  // TODO(panyx0718): ExecutionContext should only depend on RuntimeContext
  // not Scope. Imperative mode only pass inputs and get outputs.
  {
    platform::RecordEvent record_event("compute",
                                       platform::EventRole::kInnerOp);
    (*kernel_func_)(
        ExecutionContext(*this, exec_scope, *dev_ctx, *runtime_ctx));
#ifdef PADDLE_WITH_XPU
    if (std::getenv("FLAGS_opt_xpu_mem") != nullptr &&
        platform::is_xpu_place(kernel_type_->place_)) {
      dev_ctx->Wait();
    }
#endif
  }

  if (!transfered_inplace_vars.empty()) {
    // there is inplace variable has been transferred.
    TransferInplaceVarsBack(scope, transfered_inplace_vars, *transfer_scope);
  }

  // See [ Why need handle complex gradient to real gradient? ]
  // Only handle the case where the current kernel data type is complex
  if (framework::IsComplexType(kernel_type_->data_type_)) {
    HandleComplexGradToRealGrad(scope, runtime_ctx);
  }

  if (FLAGS_enable_unused_var_check) {
    // skip op that uses mkldnn because it has different memory reuse strategy.
    // use attr here because some GradMakers (like ActivationGradOpMaker) add
    // input when use_mkldnn=true;
    if (!(HasAttr("use_mkldnn") && Attr<bool>("use_mkldnn"))) {
      CheckUnusedVar(*this, scope);
    }
  }

  /*For profiling/benchmark only*/
  if (FLAGS_benchmark) {
    dev_ctx->Wait();
#if defined(PADDLE_WITH_CUDA)
    PADDLE_ENFORCE_CUDA_SUCCESS(cudaGetLastError());
    VLOG(4) << "Operator(" << Type() << "): context wait and get last error";
#endif
#if defined(PADDLE_WITH_HIP)
    PADDLE_ENFORCE_CUDA_SUCCESS(hipGetLastError());
    VLOG(4) << "Operator(" << Type() << "): context wait and get last error";
#endif
  }

  if (FLAGS_check_nan_inf) {
    framework::details::CheckOpHasNanOrInf(*this, exec_scope, place);
  }

  // To solve issue #15032, have a discussion with @Luotao for cpu inference,
  // do not cache transfer scope, hence in this case delete transfer scope
  // after run to avoid memory leak
  if (transfer_scope && !run_by_executor_ && !enable_cache_transfer_scope_) {
    scope.DeleteScope(transfer_scope);
  }
}

void OperatorWithKernel::ChooseKernel(const RuntimeContext& ctx,
                                      const Scope& scope,
                                      const platform::Place& place) const {
  platform::DeviceContextPool& pool = platform::DeviceContextPool::Instance();
  auto* dev_ctx = pool.Get(place);

  // check if op[type] has kernel registered.
  auto& all_op_kernels = AllOpKernels();
  auto kernels_iter = all_op_kernels.find(type_);
  PADDLE_ENFORCE_NE(
      kernels_iter, all_op_kernels.end(),
      platform::errors::Unavailable(
          "There are no kernels which are registered in the %s operator.",
          type_));

  OpKernelMap& kernels = kernels_iter->second;

  auto expected_kernel_key = this->GetExpectedKernelType(
      ExecutionContext(*this, scope, *dev_ctx, ctx));
  if (HasAttr("op_device")) {
    if (Attr<std::string>("op_device") == "cpu") {
      expected_kernel_key.place_ = platform::CPUPlace();
    } else if (Attr<std::string>("op_device").find("gpu") !=
               std::string::npos) {
      auto device = Attr<std::string>("op_device");
      size_t pos = device.find(':');
      if (pos != std::string::npos) {
        device = device.substr(0, pos);
        LOG_FIRST_N(WARNING, 1)
            << "Device index is only supported under pipeline parallelism, "
            << "so it will be ignored.";
      }
      // when the Op that only has CPUKernel is assigned to GPU, the CPUKernel
      // will be executed and a warning will be given at the same time.
      if (SupportGPU()) {
        expected_kernel_key.place_ = dev_ctx->GetPlace();
      } else if (SupportNPU()) {
        expected_kernel_key.place_ = dev_ctx->GetPlace();
      } else {
        expected_kernel_key.place_ = platform::CPUPlace();
        LOG_FIRST_N(WARNING, 1)
            << "Op(" << type_
            << ") has no CUDA implementation. It will be assigned to CPUPlace.";
      }
    }
  }
  VLOG(3) << "op type:" << type_
          << ", expected_kernel_key:" << expected_kernel_key;

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
#ifdef PADDLE_WITH_XPU
  if (kernel_iter == kernels.end() &&
      is_xpu_place(expected_kernel_key.place_)) {
    VLOG(3) << "missing XPU kernel: " << type_
            << ", expected_kernel_key:" << expected_kernel_key
            << ", fallbacking to CPU one!";
    expected_kernel_key.place_ = platform::CPUPlace();
    kernel_iter = kernels.find(expected_kernel_key);
  }
#endif
#ifdef PADDLE_WITH_ASCEND_CL
  if (kernel_iter == kernels.end() &&
      is_npu_place(expected_kernel_key.place_)) {
    VLOG(3) << "missing NPU kernel: " << type_
            << ", expected_kernel_key:" << expected_kernel_key
            << ", fallbacking to CPU one!";
    expected_kernel_key.place_ = platform::CPUPlace();
    kernel_iter = kernels.find(expected_kernel_key);
  }
#endif
  PADDLE_ENFORCE_NE(kernel_iter, kernels.end(),
                    platform::errors::NotFound(
                        "Operator (%s) does not have kernel for %s.", type_,
                        KernelTypeToString(expected_kernel_key)));

  std::lock_guard<std::mutex> lock(cache_update_mutex_);
  if (kernel_type_.get() == nullptr || kernel_func_.get() == nullptr) {
    kernel_type_.reset(new OpKernelType(expected_kernel_key));
    kernel_func_.reset(new OpKernelFunc(kernel_iter->second));
  }
}

void OperatorWithKernel::TransferInplaceVarsBack(
    const Scope& scope, const std::vector<std::string>& inplace_vars,
    const Scope& transfer_scope) const {
  for (auto& var_name : inplace_vars) {
    VLOG(3) << "share inplace var " + var_name + " back to it's original scope";
    auto* origin_var = scope.FindVar(var_name);
    PADDLE_ENFORCE_NOT_NULL(origin_var,
                            platform::errors::InvalidArgument(
                                "The variable[%s] is nullptr.", var_name));
    auto* original_tensor =
        GetMutableLoDTensorOrSelectedRowsValueFromVar(origin_var);
    auto* var = transfer_scope.FindVar(var_name);
    PADDLE_ENFORCE_NOT_NULL(var, platform::errors::InvalidArgument(
                                     "The variable[%s] is nullptr.", var_name));
    auto* transformed_tensor = GetLoDTensorOrSelectedRowsValueFromVar(*var);
    auto original_dims = original_tensor->dims();
    original_tensor->ShareDataWith(*transformed_tensor);
    // In order to solve the problem that the output latitude of NPU reshape
    // operator is not changed when inplace.
    if (type_ != "reshape2" && type_ != "reshape2_grad") {
      original_tensor->Resize(original_dims);
    }
  }
}

void OperatorWithKernel::HandleComplexGradToRealGrad(
    const Scope& scope, RuntimeContext* ctx) const {
  for (auto& var_name_item : Outputs()) {
    std::vector<Variable*>& output_vars = ctx->outputs[var_name_item.first];
    for (size_t i = 0; i < var_name_item.second.size(); ++i) {
      // 1. find grad_var & check whether is complex tensor
      auto var_name = var_name_item.second[i];
      auto orig_var_name = GradOriginalVarName(var_name);
      // only focus on gradient var
      if (var_name == orig_var_name) {
        continue;
      }
      auto* grad_var = output_vars[i];
      // skip nullptr var
      if (grad_var == nullptr) {
        continue;
      }
      // don't process LoDTensorArray temporarily,
      // add support if necessary for complex number calculations in the future
      if (!VarIsTensor(*grad_var)) {
        continue;
      }
      auto* grad_tensor =
          GetMutableLoDTensorOrSelectedRowsValueFromVar(grad_var);
      // skip nullptr tensor
      if (grad_tensor == nullptr || !grad_tensor->IsInitialized()) {
        continue;
      }
      // only focus on complex dtype now
      auto src_type = grad_tensor->type();
      if (!IsComplexType(src_type)) {
        continue;
      }

      // 2. find forward var & check whether need to cast
      auto* var = scope.FindVar(orig_var_name);
      // if forward var not exists, do nothing
      if (var == nullptr) {
        continue;
      }
      if (!VarIsTensor(*var)) {
        continue;
      }
      const auto* tensor = GetLoDTensorOrSelectedRowsValueFromVar(*var);
      PADDLE_ENFORCE_NOT_NULL(
          tensor,
          platform::errors::Unavailable(
              "Forward tensor is nullptr when handle complex data to real."));
      // only need record type, the allocation may have been released
      auto dst_type = tensor->saved_type();
      // only focus on real dtype and need casting
      if (IsComplexType(dst_type)) {
        continue;
      }

      // 3. cast complex grad to real grad
      VLOG(6) << "Transform " << framework::DataTypeToString(src_type)
              << " var `" << var_name << "` to "
              << framework::DataTypeToString(dst_type)
              << " real var in static graph.";
      Tensor out;
      TransComplexToReal(dst_type, src_type, *grad_tensor, &out);
      SetTensorToVariable(*grad_var, out, grad_var);
    }
  }
}

Scope* OperatorWithKernel::PrepareData(
    const Scope& scope, const OpKernelType& expected_kernel_key,
    std::vector<std::string>* transfered_inplace_vars,
    RuntimeContext* ctx) const {
  Scope* new_scope = nullptr;

  const std::unordered_set<std::string>* no_buffer_ins = nullptr;
  if (info_) {
    auto& no_buffer_inferer = info_->NoNeedBufferVarsInferer();
    // Some op may not register NoNeedBufferVarsInferer
    if (no_buffer_inferer) {
      no_buffer_ins = &(no_buffer_inferer(Inputs(), Outputs(), Attrs()));
      if (no_buffer_ins->empty()) no_buffer_ins = nullptr;
    }
  }

  for (auto& var_name_item : Inputs()) {
    bool should_skip_input =
        no_buffer_ins && no_buffer_ins->count(var_name_item.first) > 0;

    std::vector<Variable*>& input_vars = ctx->inputs[var_name_item.first];

    for (size_t i = 0; i < var_name_item.second.size(); ++i) {
      auto& var_name = var_name_item.second[i];
      auto* var = input_vars[i];

      // Only tensor can be tranfer to another device.
      if (var == nullptr || !VarIsTensor(*var)) {
        continue;
      }

      auto* tensor_in = GetLoDTensorOrSelectedRowsValueFromVar(*var);

      // When no_buffer_ins then checking of Tensor::holder_ is
      // not a thread safe. And for infershape scenario checks
      // to be omitted are not really needed
      if (should_skip_input == true) {
#ifdef PADDLE_WITH_MKLDNN
        // Var without buffer may be needed
        // for some situation like InferShape().
        // In this situation We cannot skip Var analysis, as
        // MKL-DNN shape of Var may differ from kNHWC Var
        // In such situation corressponding resized Var
        // has to be created and registered
        if ((tensor_in->layout() == DataLayout::kMKLDNN) &&
            (var->IsType<LoDTensor>() == true) &&
            (expected_kernel_key.data_layout_ != DataLayout::kMKLDNN) &&
            (paddle::platform::MKLDNNDeviceContext::tls()
                 .get_cur_paddle_data_layout() == DataLayout::kNHWC)) {
          // Mixed execution : MKL-DNN and GPU is not supported!
          if (!new_scope) {
            new_scope = &scope.NewScope();
          }
          auto* trans_var = new_scope->Var(var_name);
          input_vars[i] = trans_var;
          auto out = trans_var->GetMutable<LoDTensor>();
          out->Resize(tensor_in->dims());
          platform::MatchShapeToLayout(out, tensor_in->layout(),
                                       DataLayout::kNHWC);
          VLOG(7) << "Created reshaped dummy input based on MKL-DNN Tensor , "
                     "but kNHWC layout"
                  << var_name_item.first << " in Operator " << type_;
        } else {
          VLOG(7) << "Skip scanning input " << var_name_item.first
                  << " in Operator " << type_;
        }
#endif
        continue;
      }

      if (!tensor_in->IsInitialized()) {
        continue;
      }

      auto kernel_type_for_var = GetKernelTypeForVar(
          var_name_item.first, *tensor_in, expected_kernel_key);

      if (!NeedTransform(kernel_type_for_var, expected_kernel_key)) {
        continue;
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
      //
      // To solve issue #15032, have a discussion with @Luotao for cpu
      // inference, for all cpu kernels cases without GPU participation, here
      // not do transfer scope caching, and cpu inference performance is not
      // impacted by test.
      enable_cache_transfer_scope_ = false;
      if (!run_by_executor_ &&
          (platform::is_gpu_place(kernel_type_for_var.place_) ||
           platform::is_gpu_place(expected_kernel_key.place_))) {
        new_scope = TryCreateTransferScope(kernel_type_for_var,
                                           expected_kernel_key, &scope);
        enable_cache_transfer_scope_ = true;
      }
      if (!new_scope) {
        new_scope = &scope.NewScope();
      }
      // For inference, if a gpu model has an op which could only run on CPU,
      // each result of different input will be the same with the first one.
      // The reason is that if a gpu tensor is the input of a cpu kernel,
      // we will create a new cpu tensor in new scope.
      // However, if enable_cache_runtime_context_, we get the cpu tensor each
      // time, not the gpu tensor. Thus, we set pre_scope_ = nullptr
      // to trigger `new RuntimeContext()` in RunImpl().
      if (enable_cache_runtime_context_) {
        pre_scope_ = nullptr;
      }

      // Create new var with the same name in transfer scopes
      auto* trans_var = new_scope->Var(var_name);
      input_vars[i] = trans_var;

      // Find if inplace exists between input and output
      // If inplace exists, set the new created var to inplaced output, and
      // record its name in transfered_inplace_vars.
      for (auto& pair : Outputs()) {
        for (size_t j = 0; j < pair.second.size(); ++j) {
          if (pair.second[j] == var_name) {
            VLOG(4) << "Found inplace between input(" << var_name_item.first
                    << ") and output(" << pair.first
                    << "), the variable name is " << var_name;
            ctx->outputs[pair.first][j] = trans_var;
            transfered_inplace_vars->emplace_back(var_name);
          }
        }
      }

      // Do transfer
      Tensor out;
      TransformData(expected_kernel_key, kernel_type_for_var, *tensor_in, &out);
      SetTensorToVariable(*var, out, trans_var);
    }
  }

  // If pre_scope = &scope, it means that scope is cached and the op is not in
  // while block. If new_scope = nullptr, it means that for each input of this
  // Op, there is no need to do PrepareData. So PrepareData could be skipped at
  // the rest iterations to save the elapsed time.
  // We do not support skipping PrepareData in while block, because the Op's
  // input may be changed by subsequent Ops, which may cause an error.

  // For inference, ops that behind conditional branch aren't supported well,
  // so disable prepare optimization conservatively.
  bool force_prepare_data = HasAttr("inference_force_prepare_data") &&
                            Attr<bool>("inference_force_prepare_data");
  if (pre_scope_ == &scope && new_scope == nullptr && !force_prepare_data) {
    need_prepare_data_ = false;
  }

  return new_scope;
}

void OperatorWithKernel::ParseInputDataType(
    const ExecutionContext& ctx, const std::string& name,
    proto::VarType::Type* data_type) const {
  proto::VarType::Type default_data_type =
      static_cast<proto::VarType::Type>(-1);
  const std::vector<Variable*> vars = ctx.MultiInputVar(name);
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
      } else if (var->IsType<LoDTensorArray>()) {
        auto t_arr = &var->Get<LoDTensorArray>();
        for (size_t j = 0; j < t_arr->size(); j++) {
          if (t_arr->at(j).IsInitialized()) {
            t = &(t_arr->at(j));
          }
        }
      }
      if (t != nullptr) {
        PADDLE_ENFORCE_EQ(
            t->IsInitialized(), true,
            platform::errors::InvalidArgument(
                "The Tensor in the %s Op's Input Variable %s(%s) is "
                "not initialized.",
                Type(), name, ctx.InputNames(name).at(i)));
        proto::VarType::Type tmp = t->type();
        PADDLE_ENFORCE(
            tmp == *data_type || *data_type == default_data_type,
            platform::errors::InvalidArgument(
                "The DataType of %s Op's duplicable Variable %s must be "
                "consistent. The current variable type is (%s), but the "
                "previous variable type is (%s).",
                Type(), name, DataTypeToString(tmp),
                DataTypeToString(*data_type)));
        *data_type = tmp;
      }
    }
  }
}

proto::VarType::Type OperatorWithKernel::IndicateDataType(
    const ExecutionContext& ctx) const {
  proto::VarType::Type dafault_data_type =
      static_cast<proto::VarType::Type>(-1);
  proto::VarType::Type data_type = dafault_data_type;
  for (auto& input : ctx.InNameList()) {
    ParseInputDataType(ctx, input, &data_type);
  }
  PADDLE_ENFORCE_NE(
      data_type, dafault_data_type,
      platform::errors::NotFound(
          "DataType should be indicated by input Variable at %s.", Type()));
  return data_type;
}

proto::VarType::Type OperatorWithKernel::IndicateVarDataType(
    const ExecutionContext& ctx, const std::string& name) const {
  proto::VarType::Type dafault_data_type =
      static_cast<proto::VarType::Type>(-1);
  proto::VarType::Type data_type = dafault_data_type;
  ParseInputDataType(ctx, name, &data_type);
  PADDLE_ENFORCE_NE(
      data_type, dafault_data_type,
      platform::errors::InvalidArgument(
          "The Input Variable(%s) of (%s) Operator used to determine kernel "
          "data type is empty or not LoDTensor or SelectedRows or "
          "LoDTensorArray.",
          name, Type()));
  return data_type;
}

Tensor* OperatorWithKernel::GetTensorFormInputSafely(
    const ExecutionContext& ctx, const std::string& name) const {
  // 1. get variable and check
  // NOTE: only supports signal input var now
  // NOTE: using const_cast is because we don't have method
  // can get single mutable var, and here will not change
  // the var's data, only use some attribute
  Variable* var = const_cast<Variable*>(ctx.InputVar(name));
  PADDLE_ENFORCE_NOT_NULL(
      var,
      platform::errors::NotFound(
          "The variable %s is not found when promote complex types.", name));
  // 2. get tensor and check
  Tensor* t = nullptr;
  if (var->IsType<Tensor>()) {
    t = var->GetMutable<Tensor>();
  } else if (var->IsType<LoDTensor>()) {
    t = var->GetMutable<LoDTensor>();
  } else if (var->IsType<SelectedRows>()) {
    t = var->GetMutable<SelectedRows>()->mutable_value();
  } else {
    PADDLE_THROW(platform::errors::Unimplemented(
        "Unsupported input variable type in complex type promotion."));
  }
  PADDLE_ENFORCE_NOT_NULL(
      t,
      platform::errors::InvalidArgument(
          "The Tensor of variable %s is nullptr when promote complex types."));
  PADDLE_ENFORCE_EQ(t->IsInitialized(), true,
                    platform::errors::InvalidArgument(
                        "The Tensor in the %s Op's Input Variable %s(%s) is "
                        "not initialized.",
                        Type(), name, ctx.InputName(name)));
  return t;
}

/** NOTE(chenweihang): For safety reasons, we now only
 * perform type promotes for binary operations with
 * complex type inputs, which is used to support the
 * paddle quantum function.
 * In other cases, the first input data type is used as
 * the kernel data type.
 */
proto::VarType::Type OperatorWithKernel::IndicateOrPromoteVarDataTypes(
    const ExecutionContext& ctx, const std::string& name1,
    const std::string& name2) const {
  // 1. Get tensor
  auto* tensor_a = GetTensorFormInputSafely(ctx, name1);
  auto* tensor_b = GetTensorFormInputSafely(ctx, name2);

  // 2. Get two input types
  auto type_a = tensor_a->type();
  auto type_b = tensor_b->type();

  // 3. Get first input type or promote complex types
  auto target_type = PromoteTypesIfComplexExists(type_a, type_b);

  return target_type;
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
