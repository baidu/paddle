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

//
// Created by Jiabin on 2019-08-19.
//

#include <paddle/fluid/framework/op_registry.h>
#include <memory>
#include <string>
#include <vector>
#include "gtest/gtest.h"
#include "paddle/fluid/framework/op_info.h"
#include "paddle/fluid/imperative/prepared_operator.h"
#include "paddle/fluid/imperative/type_defs.h"

namespace imperative = paddle::imperative;
namespace platform = paddle::platform;
namespace framework = paddle::framework;

namespace paddle {
namespace imperative {

static framework::VariableNameMap CreateVarNameMap(
    const framework::OpInfo& op_info, const std::string& op_type,
    const NameVarBaseMap& varbase_map, bool is_input) {
  if (op_info.proto_ == nullptr) {
    return {};
  }

  framework::VariableNameMap result;

  for (auto& var :
       is_input ? op_info.Proto().inputs() : op_info.Proto().outputs()) {
    auto it = varbase_map.find(var.name());
    if (it == varbase_map.end()) {
      PADDLE_ENFORCE_EQ(
          var.dispensable(), true,
          platform::errors::NotFound("Variable %s is not dispensable and "
                                     "there are no such var in inputs",
                                     var.name()));
      result[var.name()] = {};
    } else {
      auto& var_vector = it->second;
      std::vector<std::string> args;
      args.reserve(var_vector.size());
      for (auto& var_base : var_vector) {
        args.emplace_back(var_base->Name());
      }
      result[var.name()] = std::move(args);
    }
  }
  return result;
}

using vb_vector = std::vector<std::shared_ptr<imperative::VarBase>>;

using var_pair = std::pair<std::string, vb_vector>;

TEST(test_prepare_op, test_prepare_op) {
  std::shared_ptr<imperative::VarBase> vin(
      new imperative::VarBase(false, "vin"));
  std::shared_ptr<imperative::VarBase> vout(
      new imperative::VarBase(false, "vout"));
  framework::OpDesc desc;
  platform::CPUPlace place;
  vin->MutableVar()->GetMutable<framework::LoDTensor>()->mutable_data<float>(
      place);
  var_pair x_pair = var_pair("X", vb_vector(1, vin));
  var_pair out_pair = var_pair("Out", vb_vector(1, vout));
  imperative::NameVarBaseMap ins = {x_pair};
  imperative::NameVarBaseMap outs = {out_pair};
  framework::AttributeMap split_attr_map;
  const auto& info = framework::OpInfoMap::Instance().Get("split");
  if (info.Checker()) info.Checker()->Check(&split_attr_map);
  framework::VariableNameMap var_in_map =
      CreateVarNameMap(info, "split", ins, true);
  framework::VariableNameMap var_out_map =
      CreateVarNameMap(info, "split", outs, false);
  auto op = framework::OpRegistry::CreateOp("split", var_in_map, var_out_map,
                                            split_attr_map);
  ASSERT_NO_FATAL_FAILURE(PreparedOp preparedOp = PreparedOp::Prepare(
                              ins, outs,
                              dynamic_cast<framework::OperatorWithKernel&>(*op),
                              place, split_attr_map, {}));
}

const framework::Tensor* GetTensorFromVar(const framework::Variable& var);

TEST(test_prepare_op, test_get_tensor_from_var) {
  std::shared_ptr<imperative::VarBase> vout_error(
      new imperative::VarBase(false, "vout_error"));
  vout_error->MutableVar()->GetMutable<framework::SelectedRows>();
  auto* ts = GetTensorFromVar(*vout_error->MutableVar());
  ASSERT_TRUE(ts != nullptr);
}

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
TEST(test_prepare_op, test_prepare_data) {
  std::shared_ptr<imperative::VarBase> vin(
      new imperative::VarBase(false, "vin"));
  std::shared_ptr<imperative::VarBase> vout(
      new imperative::VarBase(false, "vout"));

  framework::OpDesc desc;
  platform::CPUPlace cpu_place;
  platform::CUDAPlace gpu_place(0);
  std::vector<float> src_data(10, 2.0);
  std::vector<int64_t> dims = {2, 5};

  // prepare an cpu only input
  auto* vin_tensor = vin->MutableVar()->GetMutable<framework::LoDTensor>();
  vin_tensor->Resize(framework::make_ddim(dims));
  auto* vin_mutable_tensor = vin_tensor->mutable_data<float>(cpu_place);
  paddle::memory::Copy(cpu_place, vin_mutable_tensor, cpu_place,
                       src_data.data(), sizeof(float) * src_data.size());

  var_pair x_pair = var_pair("X", vb_vector(1, vin));
  var_pair out_pair = var_pair("Out", vb_vector(1, vout));
  imperative::NameVarBaseMap ins = {x_pair};
  imperative::NameVarBaseMap outs = {out_pair};
  const std::string op_type = "relu";
  framework::AttributeMap attr_map;
  const auto& info = framework::OpInfoMap::Instance().Get(op_type);
  if (info.Checker()) info.Checker()->Check(&attr_map);
  framework::VariableNameMap var_in_map =
      CreateVarNameMap(info, op_type, ins, true);
  framework::VariableNameMap var_out_map =
      CreateVarNameMap(info, op_type, outs, false);
  auto op = framework::OpRegistry::CreateOp(op_type, var_in_map, var_out_map,
                                            attr_map);

  // test if it can be transformed to GPU place
  auto prepared_op = PreparedOp::Prepare(
      ins, outs, dynamic_cast<framework::OperatorWithKernel&>(*op), gpu_place,
      attr_map, {});
  PrepareData<imperative::VarBase>(
      dynamic_cast<framework::OperatorWithKernel&>(*op), ins,
      prepared_op.kernel_type());
  for (const auto& name_pair : ins) {
    for (const auto& vb : name_pair.second) {
      ASSERT_TRUE(platform::is_same_place(
          vb->Var().Get<framework::LoDTensor>().place(), gpu_place));
    }
  }
}
#endif

void TestPrepareDataSamePlace(framework::AttributeMap attr_map) {
  std::shared_ptr<imperative::VarBase> vin(
      new imperative::VarBase(false, "vin"));
  std::shared_ptr<imperative::VarBase> vout(
      new imperative::VarBase(false, "vout"));

  framework::OpDesc desc;
  platform::CPUPlace cpu_place;
  std::vector<float> src_data(10, 2.0);
  std::vector<int64_t> dims = {2, 5};

  // prepare an cpu only input
  auto* vin_tensor = vin->MutableVar()->GetMutable<framework::LoDTensor>();
  vin_tensor->Resize(framework::make_ddim(dims));
  auto* vin_mutable_tensor = vin_tensor->mutable_data<float>(cpu_place);
  paddle::memory::Copy(cpu_place, vin_mutable_tensor, cpu_place,
                       src_data.data(), sizeof(float) * src_data.size());

  var_pair x_pair = var_pair("X", vb_vector(1, vin));
  var_pair out_pair = var_pair("Out", vb_vector(1, vout));
  imperative::NameVarBaseMap ins = {x_pair};
  imperative::NameVarBaseMap outs = {out_pair};
  const std::string op_type = "relu";
  const auto& info = framework::OpInfoMap::Instance().Get(op_type);
  if (info.Checker()) info.Checker()->Check(&attr_map);
  framework::VariableNameMap var_in_map =
      CreateVarNameMap(info, op_type, ins, true);
  framework::VariableNameMap var_out_map =
      CreateVarNameMap(info, op_type, outs, false);

  auto op = framework::OpRegistry::CreateOp(op_type, var_in_map, var_out_map,
                                            attr_map);

  // test if it never transferred on GPU place
  auto prepared_op = PreparedOp::Prepare(
      ins, outs, dynamic_cast<framework::OperatorWithKernel&>(*op), cpu_place,
      attr_map, {});
  PrepareData<imperative::VarBase>(
      dynamic_cast<framework::OperatorWithKernel&>(*op), ins,
      prepared_op.kernel_type());
  for (const auto& name_pair : ins) {
    for (const auto& vb : name_pair.second) {
      ASSERT_TRUE(platform::is_same_place(
          vb->Var().Get<framework::LoDTensor>().place(), cpu_place));
    }
  }
}

TEST(test_prepare_op, test_prepare_data_same_place) {
  TestPrepareDataSamePlace({});
}

#ifdef PADDLE_WITH_MKLDNN
TEST(test_prepare_op, test_prepare_data_cpu_mkldnn) {
  TestPrepareDataSamePlace({{"use_mkldnn", true}});
}
#endif
}  // namespace imperative
}  // namespace paddle

USE_OP(split);
USE_OP(relu);
#ifdef PADDLE_WITH_MKLDNN
USE_OP_DEVICE_KERNEL(relu, MKLDNN);
#endif
