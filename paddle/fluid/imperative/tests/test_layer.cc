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
// Created by Jiabin on 2019-08-16.
//

#include <paddle/fluid/framework/op_registry.h>

#include <memory>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "paddle/fluid/imperative/execution_context.h"
#include "paddle/fluid/imperative/infer_shape_context.h"
#include "paddle/fluid/imperative/infer_var_type_context.h"
#include "paddle/fluid/imperative/layer.h"

namespace imperative = paddle::imperative;
namespace platform = paddle::platform;
namespace framework = paddle::framework;

namespace paddle {
namespace imperative {

using vb_vector = std::vector<std::shared_ptr<imperative::VarBase>>;

using var_pair = std::pair<std::string, vb_vector>;

template <typename VarType>
class TestRuntimeInferVarTypeContext
    : public RuntimeInferVarTypeContext<VarType> {
 public:
  TestRuntimeInferVarTypeContext(const NameVarMap<VarType>& inputs,
                                 const NameVarMap<VarType>& outputs,
                                 const framework::AttributeMap& attrs_map)
      : RuntimeInferVarTypeContext<VarType>(inputs, outputs, attrs_map) {}

  bool HasVar(const std::string& name) const {
    return RuntimeInferVarTypeContext<VarType>::HasVar(name);
  }

  const std::vector<std::string>& InputVars(const std::string& name) const {
    return RuntimeInferVarTypeContext<VarType>::InputVars(name);
  }

  const std::vector<std::string>& OutputVars(const std::string& name) const {
    return RuntimeInferVarTypeContext<VarType>::OutputVars(name);
  }

  framework::proto::VarType::Type GetVarType(const std::string& name) const {
    return RuntimeInferVarTypeContext<VarType>::GetVarType(name);
  }

  void SetVarType(const std::string& name,
                  framework::proto::VarType::Type type) {
    RuntimeInferVarTypeContext<VarType>::SetVarType(name, type);
  }

  framework::proto::VarType::Type GetVarDataType(
      const std::string& name) const {
    return RuntimeInferVarTypeContext<VarType>::GetVarDataType(name);
  }

  void SetVarDataType(const std::string& name,
                      framework::proto::VarType::Type type) {
    RuntimeInferVarTypeContext<VarType>::SetVarDataType(name, type);
  }

  std::vector<framework::proto::VarType::Type> GetVarDataTypes(
      const std::string& name) const {
    return RuntimeInferVarTypeContext<VarType>::GetVarDataTypes(name);
  }

  void SetVarDataTypes(
      const std::string& name,
      const std::vector<framework::proto::VarType::Type>& multiple_data_type) {
    RuntimeInferVarTypeContext<VarType>::SetVarDataTypes(name,
                                                         multiple_data_type);
  }

  std::vector<int64_t> GetVarShape(const std::string& name) const {
    return RuntimeInferVarTypeContext<VarType>::GetVarShape(name);
  }

  void SetVarShape(const std::string& name, const std::vector<int64_t>& dims) {
    RuntimeInferVarTypeContext<VarType>::SetVarShape(name, dims);
  }

  int32_t GetVarLoDLevel(const std::string& name) const {
    return RuntimeInferVarTypeContext<VarType>::GetVarLoDLevel(name);
  }

  void SetVarLoDLevel(const std::string& name, int32_t lod_level) {
    RuntimeInferVarTypeContext<VarType>::SetVarLoDLevel(name, lod_level);
  }
};

TEST(test_layer, test_runtime_context) {
  std::shared_ptr<imperative::VarBase> vin(
      new imperative::VarBase(false, "vin"));
  std::shared_ptr<imperative::VarBase> vin_b(
      new imperative::VarBase(false, "vin_b"));
  std::shared_ptr<imperative::VarBase> vout(
      new imperative::VarBase(false, "vout"));
  std::shared_ptr<imperative::VarBase> vout_b(
      new imperative::VarBase(false, "vout_b"));
  var_pair in_pair = var_pair("X", {vin, vin_b});
  var_pair out_pair = var_pair("Out", {vout, vout_b});
  imperative::NameVarBaseMap ins = {in_pair};
  imperative::NameVarBaseMap outs = {out_pair};
  framework::AttributeMap attrs;

  auto* ctx =
      new imperative::TestRuntimeInferVarTypeContext<imperative::VarBase>(
          ins, outs, attrs);

  ASSERT_TRUE(ctx->HasInput("X"));
  ASSERT_TRUE(ctx->HasOutput("Out"));

  ASSERT_EQ(2u, ctx->InputSize("X"));
  ASSERT_EQ("vin", ctx->InputVarName("X", 0));

  ASSERT_TRUE(ctx->InputTypeAnyOf("X", framework::proto::VarType::LOD_TENSOR));
  ASSERT_TRUE(ctx->InputTypeAllOf("X", framework::proto::VarType::LOD_TENSOR));

  ASSERT_EQ(framework::proto::VarType::LOD_TENSOR, ctx->GetInputType("X"));
  ASSERT_EQ(framework::proto::VarType::FP32, ctx->GetInputDataType("X"));

  ctx->SyncTypeAndDataType("X", "Out");

  ASSERT_EQ(framework::proto::VarType::FP32, vout->DataType());

  ASSERT_EQ(framework::proto::VarType::LOD_TENSOR, ctx->GetOutputType("Out"));

  ctx->SetOutputType("Out", framework::proto::VarType::SELECTED_ROWS,
                     framework::ALL_ELEMENTS);
  ctx->SetOutputType("Out", framework::proto::VarType::LOD_TENSOR_ARRAY);
  ASSERT_EQ(framework::proto::VarType::LOD_TENSOR_ARRAY, vout->Type());
  ASSERT_EQ(framework::proto::VarType::SELECTED_ROWS, vout_b->Type());

  ctx->SetOutputDataType("Out", framework::proto::VarType::FP64,
                         framework::ALL_ELEMENTS);
  ctx->SetOutputDataType("Out", framework::proto::VarType::INT8);

  ASSERT_EQ(framework::proto::VarType::INT8, vout->DataType());
  ASSERT_EQ(framework::proto::VarType::FP64, vout_b->DataType());

  // no throw, but do nothing
  ASSERT_NO_THROW(
      ctx->InsertVar("vout", framework::proto::VarType::LOD_TENSOR));
  ASSERT_EQ(framework::proto::VarType::LOD_TENSOR_ARRAY, vout->Type());

  ASSERT_ANY_THROW(ctx->HasVar("vin"));
  ASSERT_ANY_THROW(ctx->InputVars("X"));
  ASSERT_ANY_THROW(ctx->OutputVars("Out"));
  ASSERT_ANY_THROW(ctx->GetVarType("vin"));
  ASSERT_ANY_THROW(
      ctx->SetVarType("vin", framework::proto::VarType::LOD_TENSOR));
  ASSERT_ANY_THROW(ctx->GetVarDataType("vin"));
  ASSERT_ANY_THROW(
      ctx->SetVarDataType("vout", framework::proto::VarType::FP32));

  ASSERT_ANY_THROW(ctx->GetVarDataTypes("vin"));
  std::vector<framework::proto::VarType::Type> NullType;
  ASSERT_ANY_THROW(ctx->SetVarDataTypes("vin", NullType));
  ASSERT_ANY_THROW(ctx->GetVarShape("vin"));
  ASSERT_ANY_THROW(ctx->SetVarShape("vin", {}));
  ASSERT_ANY_THROW(ctx->GetVarLoDLevel("vin"));
  ASSERT_ANY_THROW(ctx->SetVarLoDLevel("vin", 2));

  ASSERT_TRUE(ctx->IsDygraph());
}

std::string LayerDebugString(const std::string& op_type,
                             const NameVarBaseMap& ins,
                             const NameVarBaseMap& outs);

TEST(test_layer, test_debug_string) {
  platform::CPUPlace place;
  std::shared_ptr<imperative::VarBase> vin(
      new imperative::VarBase(false, "vin"));
  var_pair in_pair = var_pair("X", vb_vector(1, vin));

  auto test_func = [&](std::shared_ptr<imperative::VarBase>& vout) {
    var_pair out_pair = var_pair("Out", vb_vector(1, vout));
    imperative::NameVarBaseMap ins = {in_pair};
    imperative::NameVarBaseMap outs = {out_pair};
    return LayerDebugString("test_op", ins, outs);
  };

  // 1. test null
  std::shared_ptr<imperative::VarBase> null_out(nullptr);
  std::string res_null = test_func(null_out);
  ASSERT_TRUE(res_null.find("NULL") != std::string::npos);

  // 2. test uninit var
  std::shared_ptr<imperative::VarBase> un_init_out(
      new imperative::VarBase(false, "un_init_out"));
  std::string res_un_init = test_func(un_init_out);
  ASSERT_TRUE(res_un_init.find("NOT_INITED_VAR") != std::string::npos);

  // 3. test unresolved type
  std::shared_ptr<imperative::VarBase> ut_out(
      new imperative::VarBase(false, "ut_out"));
  ut_out->MutableVar()->GetMutable<framework::LoDTensorArray>();
  std::string res_ut = test_func(ut_out);
  ASSERT_TRUE(res_ut.find("UNRESOLVED_TYPE") != std::string::npos);

  // 4. test uninit lod tensor
  std::shared_ptr<imperative::VarBase> lod_tensor(
      new imperative::VarBase(false, "lod_tensor"));
  auto tensor_l = lod_tensor->MutableVar()->GetMutable<framework::LoDTensor>();
  std::string res_ui_lod_t = test_func(lod_tensor);
  ASSERT_TRUE(res_ui_lod_t.find("NOT_INITED") != std::string::npos);

  // 5. test init lod tensor
  tensor_l->mutable_data<float>(place);
  std::string res_lod_t = test_func(lod_tensor);
  ASSERT_TRUE(res_lod_t.find("LoDTensor") != std::string::npos);

  // 6. test uninit selected rows
  std::shared_ptr<imperative::VarBase> selected_rows(
      new imperative::VarBase(false, "selected_rows"));
  auto tensor_sr = selected_rows->MutableVar()
                       ->GetMutable<framework::SelectedRows>()
                       ->mutable_value();
  std::string res_ui_sr = test_func(selected_rows);
  ASSERT_TRUE(res_ui_sr.find("NOT_INITED") != std::string::npos);

  // 7. test init selected rows
  tensor_sr->mutable_data<float>(place);
  std::string res_sr = test_func(selected_rows);
  ASSERT_TRUE(res_sr.find("SelectedRows") != std::string::npos);
}

static std::shared_ptr<imperative::GradOpNode> CreateGradNode(
    size_t id, const std::string& type, const imperative::NameVarBaseMap& ins,
    const imperative::NameVarBaseMap& outs,
    const framework::AttributeMap& attrs, const platform::Place& place) {
  auto node = std::make_shared<imperative::GradOpNode>();
  auto* op = &(node->emplace_back());
  op->SetId(id);
  op->SetPlace(place);
  op->SetType(type);
  op->SetAttrMap(attrs);
  for (auto& pair : ins) {
    std::vector<std::shared_ptr<VariableWrapper>> vars;
    for (auto& var : pair.second) {
      vars.emplace_back(var->SharedVar());
    }
    op->SetInput(pair.first, vars, false);
  }

  for (auto& pair : outs) {
    std::vector<std::shared_ptr<VariableWrapper>> vars;
    for (auto& var : pair.second) {
      vars.emplace_back(var->SharedVar());
    }
    op->SetOutput(pair.first, vars, false);
  }

  return node;
}

TEST(test_layer, test_clear_backward_info) {
  std::shared_ptr<imperative::VarBase> vin(
      new imperative::VarBase(false, "vin"));
  std::shared_ptr<imperative::VarBase> vout(
      new imperative::VarBase(false, "vout"));
  framework::OpDesc desc;
  platform::CPUPlace place;
  var_pair x_pair = var_pair("X", vb_vector(1, vin));
  var_pair y_pair = var_pair("Y", vb_vector(1, vin));
  var_pair out_pair = var_pair("Out", vb_vector(1, vout));
  imperative::NameVarBaseMap ins = {x_pair, y_pair};
  imperative::NameVarBaseMap outs = {out_pair};
  framework::AttributeMap concat_att_map;
  concat_att_map["axis"] = 1;

  auto node = CreateGradNode(0, "mul", ins, outs, concat_att_map, place);
  auto pending_node =
      CreateGradNode(0, "mul", ins, outs, concat_att_map, place);
  node->InsertGradPendingNode(pending_node);

  ASSERT_EQ(node->size(), 1UL);
  auto* op = &(node->back());

  ASSERT_GT(op->GetInsMap().size(), 0UL);
  ASSERT_GT(op->GetOutsMap().size(), 0UL);

  op->ClearBackwardTrace();

  ASSERT_EQ(op->GetInsMap().size(), 0UL);
  ASSERT_EQ(op->GetOutsMap().size(), 0UL);
}

TEST(test_layer, test_varbase_basic) {
  platform::CPUPlace place;
  std::shared_ptr<imperative::VarBase> vin(
      new imperative::VarBase(false, "vin"));
  vin->MutableVar()->GetMutable<framework::LoDTensor>()->mutable_data<float>(
      place);
  std::shared_ptr<imperative::VarBase> vout(vin->NewVarBase(place, false));
  ASSERT_EQ(vout->Name(), "vin0");

  std::shared_ptr<imperative::VarBase> vin_with_grad(
      new imperative::VarBase(true, "vin"));
  ASSERT_ANY_THROW(vin->MutableGradVar());
  ASSERT_NO_THROW(ASSERT_TRUE(dynamic_cast<framework::Variable*>(
                                  vin_with_grad->MutableGradVar()) != 0));
  ASSERT_TRUE(
      dynamic_cast<framework::Variable*>(vin_with_grad->MutableGradVar()) != 0);
  vin_with_grad->SetOverridedStopGradient(false);
  ASSERT_FALSE(vin_with_grad->OverridedStopGradient());
  ASSERT_NO_FATAL_FAILURE(vin_with_grad->SetPersistable(true));
  ASSERT_FALSE(vin_with_grad->OverridedStopGradient());
  ASSERT_NO_FATAL_FAILURE(vin_with_grad->SetName("new_name"));
  ASSERT_EQ(vin_with_grad->Name(), "new_name");
}
// TODO(jiabin): Add more ut here for layer

TEST(test_layer, test_dygraph_execution_context) {
  std::shared_ptr<imperative::VarBase> vin(
      new imperative::VarBase(false, "vin"));
  std::shared_ptr<imperative::VarBase> vout(
      new imperative::VarBase(false, "vout"));
  framework::OpDesc desc;
  platform::CPUPlace place;
  var_pair x_pair = var_pair("X", vb_vector(1, vin));
  var_pair y_pair = var_pair("Y", vb_vector(1, vin));
  var_pair out_pair = var_pair("Out", vb_vector(1, vout));
  imperative::NameVarBaseMap ins = {x_pair, y_pair};
  imperative::NameVarBaseMap outs = {out_pair};

  framework::AttributeMap concat_att_map;
  concat_att_map["axis"] = 1;

  auto op = framework::OpRegistry::CreateOp("mul", {}, {}, {}, false);
  paddle::platform::CPUPlace cpu_place;

  paddle::platform::DeviceContextPool& pool =
      paddle::platform::DeviceContextPool::Instance();
  auto* dev_ctx = pool.Get(cpu_place);
  paddle::framework::RuntimeContext ctx({}, {});
  framework::Scope scope;

  DygraphExecutionContext<imperative::VarBase> dy_exe_context(
      *(op.get()), scope, *dev_ctx, ctx, ins, outs, concat_att_map);

  ASSERT_EQ(dy_exe_context.InputSize("X"), 1u);
  ASSERT_EQ(dy_exe_context.InputName("X"), "vin");
  ASSERT_EQ(dy_exe_context.HasAttr("axis"), true);
  auto attr_map = dy_exe_context.Attrs();
  ASSERT_EQ(BOOST_GET(int, attr_map["axis"]), 1);
  ASSERT_EQ(dy_exe_context.OutputSize("Out"), 1u);
  ASSERT_EQ(dy_exe_context.HasOutput("Out"), true);
}

TEST(test_layer, test_dygraph_infershape_context) {
  std::shared_ptr<imperative::VarBase> vin(
      new imperative::VarBase(false, "vin"));
  std::shared_ptr<imperative::VarBase> vout(
      new imperative::VarBase(false, "vout"));
  framework::OpDesc desc;
  platform::CPUPlace place;
  var_pair x_pair = var_pair("X", vb_vector(1, vin));
  var_pair y_pair = var_pair("Y", vb_vector(1, vin));
  var_pair out_pair = var_pair("Out", vb_vector(1, vout));
  imperative::NameVarBaseMap ins = {x_pair, y_pair};
  imperative::NameVarBaseMap outs = {out_pair};

  framework::AttributeMap concat_att_map;
  concat_att_map["axis"] = 1;

  DygraphInferShapeContext<imperative::VarBase> infer_shape_ctx(
      &ins, &outs, &concat_att_map, "dummy");

  bool have_x = infer_shape_ctx.HasOutputs("Out");
  ASSERT_EQ(have_x, true);
  bool have_z = infer_shape_ctx.HasOutputs("Z");
  ASSERT_EQ(have_z, false);
}

TEST(test_layer, test_inner_op_not_inited) {
  OpBase op;
  std::string kUnknown = "unknown";
  ASSERT_EQ(op.Type(), kUnknown);
  ASSERT_THROW(op.Info(), platform::EnforceNotMet);
  ASSERT_THROW(op.InnerOp(), platform::EnforceNotMet);
  ASSERT_THROW(op.CheckAttrs(), platform::EnforceNotMet);
}

}  // namespace imperative
}  // namespace paddle

USE_OP(mul);
