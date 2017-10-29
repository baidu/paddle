/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/framework/op_desc.h"
#include <functional>
#include <mutex>
#include <unordered_map>
#include "glog/logging.h"
#include "paddle/framework/block_desc.h"
#include "paddle/framework/operator.h"
#include "paddle/framework/program_desc.h"
#include "paddle/framework/shape_inference.h"

namespace paddle {
namespace framework {

class OpDescBind;
class BlockDescBind;
class CompileTimeInferShapeContext : public InferShapeContext {
 public:
  CompileTimeInferShapeContext(const OpDescBind &op,
                               const BlockDescBind &block);

  bool HasInput(const std::string &name) const override;

  bool HasOutput(const std::string &name) const override;

  bool HasInputs(const std::string &name) const override;

  bool HasOutputs(const std::string &name) const override;

  DDim GetInputDim(const std::string &name) const override;

  void SetOutputDim(const std::string &name, const DDim &dim) override;

  AttrReader Attrs() const override;

  const std::vector<std::string> &Inputs(
      const std::string &name) const override;

  const std::vector<std::string> &Outputs(
      const std::string &name) const override;

 private:
  DDim GetDim(const std::string &name) const override;

  void SetDim(const std::string &name, const DDim &dim) override;

  const OpDescBind &op_;
  const BlockDescBind &block_;
};

OpDescBind::OpDescBind(const std::string &type, const VariableNameMap &inputs,
                       const VariableNameMap &outputs,
                       const AttributeMap &attrs) {
  desc_.set_type(type);
  inputs_ = inputs;
  outputs_ = outputs;
  attrs_ = attrs;
  need_update_ = true;
}

OpDescBind::OpDescBind(const OpDesc &desc, ProgramDescBind *prog)
    : desc_(desc), need_update_(false) {
  // restore inputs_
  int input_size = desc_.inputs_size();
  for (int i = 0; i < input_size; ++i) {
    const OpDesc::Var &var = desc_.inputs(i);
    std::vector<std::string> &args = inputs_[var.parameter()];
    int argu_size = var.arguments_size();
    args.reserve(argu_size);
    for (int j = 0; j < argu_size; ++j) {
      args.push_back(var.arguments(j));
    }
  }
  // restore outputs_
  int output_size = desc_.outputs_size();
  for (int i = 0; i < output_size; ++i) {
    const OpDesc::Var &var = desc_.outputs(i);
    std::vector<std::string> &args = outputs_[var.parameter()];
    int argu_size = var.arguments_size();
    args.reserve(argu_size);
    for (int j = 0; j < argu_size; ++j) {
      args.push_back(var.arguments(j));
    }
  }
  // restore attrs_
  for (const OpDesc::Attr &attr : desc_.attrs()) {
    std::string attr_name = attr.name();
    if (attr.type() != AttrType::BLOCK) {
      attrs_[attr_name] = GetAttrValue(attr);
    } else {
      auto bid = attr.block_idx();
      attrs_[attr_name] = prog->MutableBlock(bid);
    }
  }
}

OpDesc *OpDescBind::Proto() {
  Flush();
  return &desc_;
}

const std::vector<std::string> &OpDescBind::Input(
    const std::string &name) const {
  auto it = inputs_.find(name);
  PADDLE_ENFORCE(it != inputs_.end(), "Input %s cannot be found in Op %s", name,
                 Type());
  return it->second;
}

std::vector<std::string> OpDescBind::InputArgumentNames() const {
  std::vector<std::string> retv;
  for (auto &ipt : this->inputs_) {
    retv.insert(retv.end(), ipt.second.begin(), ipt.second.end());
  }
  return retv;
}

void OpDescBind::SetInput(const std::string &param_name,
                          const std::vector<std::string> &args) {
  need_update_ = true;
  inputs_[param_name] = args;
}

const std::vector<std::string> &OpDescBind::Output(
    const std::string &name) const {
  auto it = outputs_.find(name);
  PADDLE_ENFORCE(it != outputs_.end(), "Output %s cannot be found in Op %s",
                 name, Type());
  return it->second;
}

std::vector<std::string> OpDescBind::OutputArgumentNames() const {
  std::vector<std::string> retv;
  for (auto &ipt : this->outputs_) {
    retv.insert(retv.end(), ipt.second.begin(), ipt.second.end());
  }
  return retv;
}

void OpDescBind::SetOutput(const std::string &param_name,
                           const std::vector<std::string> &args) {
  need_update_ = true;
  this->outputs_[param_name] = args;
}

AttrType OpDescBind::GetAttrType(const std::string &name) const {
  auto it = attrs_.find(name);
  PADDLE_ENFORCE(it != attrs_.end(), "Attribute %s is not found", name);
  return static_cast<AttrType>(it->second.which() - 1);
}

std::vector<std::string> OpDescBind::AttrNames() const {
  std::vector<std::string> retv;
  retv.reserve(attrs_.size());
  for (auto &attr : attrs_) {
    retv.push_back(attr.first);
  }
  return retv;
}

void OpDescBind::SetAttr(const std::string &name, const Attribute &v) {
  this->attrs_[name] = v;
  need_update_ = true;
}

void OpDescBind::SetBlockAttr(const std::string &name, BlockDescBind &block) {
  this->attrs_[name] = &block;
  need_update_ = true;
}

void OpDescBind::SetAttrMap(
    const std::unordered_map<std::string, Attribute> &attr_map) {
  attrs_ = attr_map;
  need_update_ = true;
}

Attribute OpDescBind::GetAttr(const std::string &name) const {
  auto it = attrs_.find(name);
  PADDLE_ENFORCE(it != attrs_.end(), "Attribute %s is not found", name);
  return it->second;
}

int OpDescBind::GetBlockAttr(const std::string &name) const {
  auto it = attrs_.find(name);
  PADDLE_ENFORCE(it != attrs_.end(), "Attribute %s is not found", name);
  return boost::get<BlockDescBind *>(it->second)->ID();
}

const std::unordered_map<std::string, Attribute> &OpDescBind::GetAttrMap()
    const {
  return attrs_;
}

void OpDescBind::Rename(const std::string &old_name,
                        const std::string &new_name) {
  for (auto &input : inputs_) {
    std::replace(input.second.begin(), input.second.end(), old_name, new_name);
  }
  for (auto &output : outputs_) {
    std::replace(output.second.begin(), output.second.end(), old_name,
                 new_name);
  }
  need_update_ = true;
}

struct SetAttrDescVisitor : public boost::static_visitor<void> {
  explicit SetAttrDescVisitor(OpDesc::Attr *attr) : attr_(attr) {}
  mutable OpDesc::Attr *attr_;
  void operator()(int v) const { attr_->set_i(v); }
  void operator()(float v) const { attr_->set_f(v); }
  void operator()(const std::string &v) const { attr_->set_s(v); }
  void operator()(bool b) const { attr_->set_b(b); }

  void operator()(const std::vector<int> &v) const {
    VectorToRepeated(v, attr_->mutable_ints());
  }
  void operator()(const std::vector<float> &v) const {
    VectorToRepeated(v, attr_->mutable_floats());
  }
  void operator()(const std::vector<std::string> &v) const {
    VectorToRepeated(v, attr_->mutable_strings());
  }
  void operator()(const std::vector<bool> &v) const {
    VectorToRepeated(v, attr_->mutable_bools());
  }
  void operator()(BlockDesc *desc) const { attr_->set_block_idx(desc->idx()); }
  void operator()(boost::blank) const { PADDLE_THROW("Unexpected branch"); }
};

void OpDescBind::Flush() {
  if (need_update_) {
    this->desc_.mutable_inputs()->Clear();
    for (auto &ipt : inputs_) {
      auto *input = desc_.add_inputs();
      input->set_parameter(ipt.first);
      VectorToRepeated(ipt.second, input->mutable_arguments());
    }

    this->desc_.mutable_outputs()->Clear();
    for (auto &opt : outputs_) {
      auto *output = desc_.add_outputs();
      output->set_parameter(opt.first);
      VectorToRepeated(opt.second, output->mutable_arguments());
    }

    this->desc_.mutable_attrs()->Clear();
    for (auto &attr : attrs_) {
      auto *attr_desc = desc_.add_attrs();
      attr_desc->set_name(attr.first);
      attr_desc->set_type(
          static_cast<framework::AttrType>(attr.second.which() - 1));
      SetAttrDescVisitor visitor(attr_desc);
      boost::apply_visitor(visitor, attr.second);
    }

    need_update_ = false;
  }
}

static std::once_flag init_infer_shape_funcs;

static void InitInferShapeFuncs() {
  std::call_once(init_infer_shape_funcs, [] {
    auto &map = OpInfoMap::Instance();
    auto &info_map = *map.mutable_map();

    for (auto &kern_pair : OperatorWithKernel::AllOpKernels()) {
      auto op_type = kern_pair.first;
      auto &op_info = info_map.at(op_type);
      auto op =
          static_cast<OperatorWithKernel *>(op_info.Creator()("", {}, {}, {}));
      if (op_info.infer_shape_) {  // infer_shape has been registered.
        continue;
      }
      op_info.infer_shape_ = [op](InferShapeContext *ctx) {
        op->InferShape(ctx);
      };
    }
  });
}

void OpDescBind::CheckAttrs() {
  PADDLE_ENFORCE(!Type().empty(),
                 "CheckAttr() can not be called before type is setted.");
  auto *checker = OpInfoMap::Instance().Get(Type()).Checker();
  if (checker == nullptr) {
    // checker is not configured. That operator could be generated by Paddle,
    // not by users.
    return;
  }
  checker->Check(attrs_);
}

void OpDescBind::InferShape(const BlockDescBind &block) const {
  VLOG(3) << "CompileTime infer shape on " << Type();
  InitInferShapeFuncs();
  auto &infer_shape = OpInfoMap::Instance().Get(this->Type()).infer_shape_;
  PADDLE_ENFORCE(static_cast<bool>(infer_shape),
                 "%s's infer_shape has not been registered", this->Type());
  CompileTimeInferShapeContext ctx(*this, block);
  infer_shape(&ctx);
}

void OpDescBind::InferVarType(BlockDescBind *block) const {
  auto &info = OpInfoMap::Instance().Get(this->Type());
  if (info.infer_var_type_) {
    info.infer_var_type_(*this, block);
  } else {
    // all output type is LoDTensor by default
    for (auto &out_pair : this->outputs_) {
      for (auto &out_var_name : out_pair.second) {
        block->Var(out_var_name)->SetType(VarDesc::LOD_TENSOR);
      }
    }
  }
}

CompileTimeInferShapeContext::CompileTimeInferShapeContext(
    const OpDescBind &op, const BlockDescBind &block)
    : op_(op), block_(block) {}

bool CompileTimeInferShapeContext::HasInput(const std::string &name) const {
  const std::vector<std::string> &input_names = op_.Input(name);
  auto length = input_names.size();
  if (length == 0) {
    return false;
  }
  PADDLE_ENFORCE_EQ(length, 1UL,
                    "Input(%s) should have only one value, "
                    "but it have %d now",
                    name, length);
  return block_.HasVarRecursive(input_names[0]);
}

bool CompileTimeInferShapeContext::HasOutput(const std::string &name) const {
  const std::vector<std::string> &output_names = op_.Output(name);
  auto length = output_names.size();
  if (length == 0) {
    return false;
  }
  PADDLE_ENFORCE_EQ(length, 1UL,
                    "Output(%s) should have only one value, "
                    "but it have %d now",
                    name, length);
  return block_.HasVarRecursive(output_names[0]);
}

bool CompileTimeInferShapeContext::HasInputs(const std::string &name) const {
  const std::vector<std::string> &input_names = op_.Input(name);
  if (input_names.empty()) {
    return false;
  }
  for (auto &input : input_names) {
    if (!block_.HasVarRecursive(input)) return false;
  }
  return true;
}

bool CompileTimeInferShapeContext::HasOutputs(const std::string &name) const {
  const std::vector<std::string> &output_names = op_.Output(name);
  if (output_names.empty()) {
    return false;
  }
  for (auto &output : output_names) {
    if (!block_.HasVarRecursive(output)) return false;
  }
  return true;
}

DDim CompileTimeInferShapeContext::GetInputDim(const std::string &name) const {
  std::vector<DDim> ddims = GetInputsDim(name);
  auto length = ddims.size();
  PADDLE_ENFORCE_EQ(length, 1UL,
                    "Input(%s) should have 1 value, "
                    "but it has %d now",
                    name, length);
  return ddims[0];
}

void CompileTimeInferShapeContext::SetOutputDim(const std::string &name,
                                                const DDim &dim) {
  SetOutputsDim(name, {dim});
}

AttrReader CompileTimeInferShapeContext::Attrs() const {
  return AttrReader(op_.GetAttrMap());
}

const std::vector<std::string> &CompileTimeInferShapeContext::Inputs(
    const std::string &name) const {
  return op_.Input(name);
}

const std::vector<std::string> &CompileTimeInferShapeContext::Outputs(
    const std::string &name) const {
  return op_.Output(name);
}

DDim CompileTimeInferShapeContext::GetDim(const std::string &name) const {
  auto var = block_.FindVarRecursive(name);
  PADDLE_ENFORCE(var != nullptr, "Cannot find variable %s", name);
  return framework::make_ddim(var->Shape());
}

void CompileTimeInferShapeContext::SetDim(const std::string &name,
                                          const DDim &dim) {
  block_.FindVarRecursive(name)->SetShape(framework::vectorize(dim));
}

}  // namespace framework
}  // namespace paddle
