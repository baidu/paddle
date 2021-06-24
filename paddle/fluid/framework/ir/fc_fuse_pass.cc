// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/ir/fc_fuse_pass.h"
#include <string>

#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace framework {
namespace ir {

FCFusePass::FCFusePass() {
  AddOpCompat(OpCompat("mul"))
      .AddInput("X")
      .IsTensor()
      .End()
      .AddInput("Y")
      .IsTensor()
      .End()
      .AddOutput("Out")
      .IsTensor()
      .End()
      .AddAttr("x_num_col_dims")
      .IsNumGE(1)
      .End()
      .AddAttr("y_num_col_dims")
      .IsNumEQ(1)
      .End();

  AddOpCompat(OpCompat("elementwise_add"))
      .AddInput("X")
      .IsTensor()
      .End()
      .AddInput("Y")
      .IsTensor()
      .End()
      .AddOutput("Out")
      .IsTensor()
      .End()
      .AddAttr("axis")
      .IsNumGE(1)
      .End();

  AddOpCompat(OpCompat("relu"))
      .AddInput("X")
      .IsTensor()
      .End()
      .AddOutput("Out")
      .IsTensor()
      .End();

  AddOpCompat(OpCompat("fc"))
      .AddInput("Input")
      .IsTensor()
      .End()
      .AddInput("W")
      .IsTensor()
      .End()
      .AddInput("Bias")
      .IsTensor()
      .End()
      .AddOutput("Out")
      .IsTensor()
      .End()
      .AddAttr("in_num_col_dims")
      .IsNumGE(1)
      .End()
      .AddAttr("activation_type")
      .IsStringIn({"relu", ""})
      .End();
}

void FCFusePass::ApplyImpl(ir::Graph* graph) const {
  PADDLE_ENFORCE_NOT_NULL(
      graph, platform::errors::InvalidArgument("Graph cannot be nullptr."));
  FusePassBase::Init("fc_fuse", graph);

  int found_fc_count = 0;
  for (bool with_relu : {true, false}) {
    found_fc_count += ApplyFCPattern(graph, with_relu);
  }

  AddStatis(found_fc_count);
}

int FCFusePass::ApplyFCPattern(Graph* graph, bool with_relu) const {
  GraphPatternDetector gpd;
  auto* x = gpd.mutable_pattern()
                ->NewNode("fc_fuse/x")
                ->AsInput()
                ->assert_is_op_input("mul", "X");
  patterns::FC fc_pattern(gpd.mutable_pattern(), "fc_fuse");
  fc_pattern(x, true /*with bias*/, with_relu);

  int found_fc_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    if (subgraph.count(x) <= 0) {
      LOG(WARNING) << "The subgraph is empty.";
      return;
    }
    if (!IsCompat(subgraph, g)) {
      LOG(WARNING) << "Pass in op compat failed.";
      return;
    }

    VLOG(4) << "handle FC fuse";
    GET_IR_NODE_FROM_SUBGRAPH(w, w, fc_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(bias, bias, fc_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(elementwise_add_out, elementwise_add_out,
                              fc_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(mul, mul, fc_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(elementwise_add, elementwise_add, fc_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(mul_out, mul_out, fc_pattern);
    Node* relu = nullptr;
    Node* relu_out = nullptr;
    if (with_relu) {
      GET_IR_NODE_FROM_SUBGRAPH(tmp_relu, relu, fc_pattern);
      GET_IR_NODE_FROM_SUBGRAPH(tmp_relu_out, relu_out, fc_pattern);
      relu = tmp_relu;
      relu_out = tmp_relu_out;
    }

    // Create an FC Node.
    OpDesc desc;
    desc.SetType("fc");

    // Set inputs of fc
    desc.SetInput("Input", {subgraph.at(x)->Name()});
    desc.SetInput("W", {w->Name()});
    desc.SetInput("Bias", {bias->Name()});

    // Set output of fc
    std::string fc_out_name =
        with_relu ? relu_out->Name() : elementwise_add_out->Name();
    desc.SetOutput("Out", std::vector<std::string>({fc_out_name}));

    // Set attrs of fc
    desc.SetAttr("in_num_col_dims", mul->Op()->GetAttr("x_num_col_dims"));
    std::string activation_type = with_relu ? "relu" : "";
    desc.SetAttr("activation_type", activation_type);

    // This is to add padding for dimension 128 on concern of MKL performance
    bool use_gpu = Has("use_gpu") ? Get<bool>("use_gpu") : false;
    bool use_fc_padding =
        Has("use_fc_padding") ? Get<bool>("use_fc_padding") : true;
    const std::string& w_name = patterns::UniqueKey(w->Name());
    VarDesc w_key(w_name);
    w_key.SetPersistable(true);
    auto* w_node = g->CreateVarNode(&w_key);
    if (!use_gpu && use_fc_padding) {
      auto* scope = param_scope();
      auto* weight = scope->FindVar(w->Name())->GetMutable<LoDTensor>();
      auto* weight_data = weight->data<float>();
      auto weight_dims = weight->dims();
      int weight_num = product(weight_dims);
      int w_h = weight_dims[0];
      int w_w = weight_dims[1];
      if (w_h % 128 == 0 && w_w % 128 == 0) {
        auto* w_var = scope->Var(w_name);
        auto* w_tensor = w_var->GetMutable<framework::LoDTensor>();

        auto* weight_data_tmp = new float[weight_num];
        for (int i = 0; i < w_h; i++) {
          memcpy(weight_data_tmp + i * w_w, weight_data + i * w_w,
                 w_w * sizeof(float));
        }
        w_tensor->Resize(DDim{weight_dims[0] + 4, weight_dims[1] + 4});
        auto* weight_data_new =
            w_tensor->mutable_data<float>(platform::CPUPlace());
        for (int i = 0; i < w_h; i++) {
          memcpy(weight_data_new + i * (w_w + 4), weight_data_tmp + i * w_w,
                 w_w * sizeof(float));
        }
        delete[] weight_data_tmp;
        desc.SetInput("W", {w_name});
        desc.SetAttr("padding_weights", true);
        desc.Flush();
      }
    }

    // For anakin subgraph int8
    // When in anakin subgraph int8 mode, the pattern like "fake_quant + mul +
    // fake_dequant" can be detected by the quant_dequant_fuse_pass. This pass
    // will add "input_scale", "weight_scale" which are extracted from
    // fake_quant op and fake_dequant op to mul op, and then delete the
    // fake_quant op and fake_dequant op in the graph. If the mul op has the
    // scale info, we should add those to the fused fc.
    auto* mul_op_desc = mul->Op();
    if (mul_op_desc->HasAttr("enable_int8")) {
      desc.SetAttr("enable_int8", mul_op_desc->GetAttr("enable_int8"));
      desc.SetAttr("Input_scale", mul_op_desc->GetAttr("X_scale"));
      desc.SetAttr("weight_scale", mul_op_desc->GetAttr("weight_scale"));
      if (mul_op_desc->HasAttr("out_scale"))
        desc.SetAttr("out_scale", mul_op_desc->GetAttr("out_scale"));
      auto elementwise_desc = elementwise_add->Op();
      if (elementwise_desc->HasAttr("out_scale"))
        desc.SetAttr("out_scale", elementwise_desc->GetAttr("out_scale"));
    }

    auto* elementwise_add_op_desc = elementwise_add->Op();
    // if we can find out_threshold in elementwise_add, then set it as the
    // out_thrshold of fc
    auto out_threshold_attr =
        elementwise_add_op_desc->GetNullableAttr("out_threshold");
    if (out_threshold_attr.which()) {
      VLOG(4) << "setting out_threshold: "
              << BOOST_GET_CONST(float, out_threshold_attr);
      desc.SetAttr("out_threshold", out_threshold_attr);
    }
    desc.Flush();

    if (!IsCompat(desc)) {
      LOG(WARNING) << "Fc fuse pass in out fc op compat failed.";
      return;
    }

    auto fc_node = g->CreateOpNode(&desc);  // OpDesc will be copied.
    if (with_relu) {
      GraphSafeRemoveNodes(
          graph, {mul, elementwise_add, mul_out, elementwise_add_out, relu});
    } else {
      GraphSafeRemoveNodes(graph, {mul, elementwise_add, mul_out});
    }

    IR_NODE_LINK_TO(subgraph.at(x), fc_node);
    if (desc.GetAttrIfExists<bool>("padding_weights")) {
      IR_NODE_LINK_TO(w_node, fc_node);
    } else {
      GraphSafeRemoveNodes(g, {w_node});
      IR_NODE_LINK_TO(w, fc_node);
    }
    IR_NODE_LINK_TO(bias, fc_node);
    if (with_relu) {
      IR_NODE_LINK_TO(fc_node, relu_out);
    } else {
      IR_NODE_LINK_TO(fc_node, elementwise_add_out);
    }

    found_fc_count++;
  };
  gpd(graph, handler);
  return found_fc_count;
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(fc_fuse_pass, paddle::framework::ir::FCFusePass)
    .RequirePassAttr("use_gpu");
REGISTER_PASS_CAPABILITY(fc_fuse_pass)
    .AddCombination(
        paddle::framework::compatible::OpVersionComparatorCombination()
            .EQ("mul", 0)
            .LE("elementwise_add", 1)
            .EQ("relu", 0)
            .EQ("fc", 0));
