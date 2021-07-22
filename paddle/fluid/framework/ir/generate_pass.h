// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
#include "paddle/fluid/framework/ir/graph_pattern_detector.h"
#include "paddle/fluid/framework/ir/pass.h"
#include "paddle/fluid/framework/pass_desc.pb.h"

namespace paddle {
namespace framework {
namespace ir {

// Generate a substitute pass from protobuf.
class GeneratePass : public Pass {
 public:
  // from serialized string
  explicit GeneratePass(const std::string& binary_str);
  // from PassDesc/MultiPassDesc
  explicit GeneratePass(const proto::PassDesc& pass_desc);
  explicit GeneratePass(const proto::MultiPassDesc& multi_pass_desc)
      : multi_pass_desc_(multi_pass_desc) {}
  virtual ~GeneratePass() {}

 protected:
  void ApplyImpl(Graph* graph) const override;

 private:
  GeneratePass() = delete;
  // for debug
  void ExportDotString() const {}
  // Verify pass desc
  void VerifyDesc() const {}
  // Verify graph
  void VerifyGraph() const {}
  // InitDetector
  void InitPattern(PDPattern* pattern, const proto::PassDesc& pass_desc) const;
  // Substitute
  GraphPatternDetector::handle_t Substitute(
      const PDPattern& pattern, const proto::PassDesc& pass_desc) const;

  proto::MultiPassDesc multi_pass_desc_;
};

#define PATTERN_OP(op) pattern_##op
#define PATTERN_OP_VAR(op, var) pattern_##op##_##var

#define ADD_PATTERN_OP(desc, op)                \
  auto PATTERN_OP(op) = desc->add_pattern_op(); \
  PATTERN_OP(op)->set_type(#op)

#define PATTERN_OP_ADD_INPUT(op, var)                         \
  auto PATTERN_OP_VAR(op, var) = PATTERN_OP(op)->add_input(); \
  PATTERN_OP_VAR(op, var)->set_name(#var)

#define PATTERN_OP_ADD_INPUT_FROM(op, var, from_op, from_var) \
  PATTERN_OP_ADD_INPUT(op, var);                              \
  PATTERN_OP_VAR(op, var)->set_from_op_type(#from_op);        \
  PATTERN_OP_VAR(op, var)->set_from_op_var(#from_var)

#define PATTERN_OP_ADD_OUTPUT(op, var)                         \
  auto PATTERN_OP_VAR(op, var) = PATTERN_OP(op)->add_output(); \
  PATTERN_OP_VAR(op, var)->set_name(#var)

#define ALGEBRA_OP(op) algebra_##op
#define ALGEBRA_OP_VAR(op, var) algebra_##op##_##var

#define ADD_ALGEBRA_OP(desc, op)                \
  auto ALGEBRA_OP(op) = desc->add_algebra_op(); \
  ALGEBRA_OP(op)->set_type(#op)

#define ALGEBRA_OP_ADD_INPUT(op, var)                         \
  auto ALGEBRA_OP_VAR(op, var) = ALGEBRA_OP(op)->add_input(); \
  ALGEBRA_OP_VAR(op, var)->set_name(#var)

#define ALGEBRA_OP_ADD_INPUT_FROM(op, var, from_op, from_var) \
  ALGEBRA_OP_ADD_INPUT(op, var);                              \
  ALGEBRA_OP_VAR(op, var)->set_from_op_type(#from_op);        \
  ALGEBRA_OP_VAR(op, var)->set_from_op_var(#from_var)

#define ALGEBRA_OP_ADD_OUTPUT(op, var)                         \
  auto ALGEBRA_OP_VAR(op, var) = ALGEBRA_OP(op)->add_output(); \
  ALGEBRA_OP_VAR(op, var)->set_name(#var)

#define ALGEBRA_OP_ADD_OUTPUT_FROM(op, var, from_op, from_var) \
  ALGEBRA_OP_ADD_OUTPUT(op, var);                              \
  ALGEBRA_OP_VAR(op, var)->set_from_op_type(#from_op);         \
  ALGEBRA_OP_VAR(op, var)->set_from_op_var(#from_var)

}  // namespace ir
}  // namespace framework
}  // namespace paddle
