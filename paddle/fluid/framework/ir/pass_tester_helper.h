/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

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

#include <memory>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>
#include "paddle/fluid/framework/ir/graph.h"
#include "paddle/fluid/framework/op_proto_maker.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/program_desc.h"

namespace paddle {
namespace framework {
namespace ir {

struct Layers {
 public:
  const ProgramDesc& main_program() { return program_; }

  VarDesc* data(std::string name, std::vector<int64_t> shape = {},
                bool is_persistable = false,
                proto::VarType::Type data_type = proto::VarType::FP32) {
    return lod_tensor(name, shape, is_persistable, data_type);
  }

  VarDesc* conv2d(VarDesc* input, VarDesc* filter, VarDesc* bias,
                  int groups = 1, std::vector<int> strides = {1, 1},
                  std::vector<int> paddings = {0, 0},
                  std::string padding_algorithm = "EXPLICIT",
                  std::vector<int> dilations = {1, 1},
                  std::string data_format = "NCHW", bool use_cudnn = false) {
    VarDesc* out = lod_tensor(unique_name());
    OpDesc* op = program_.MutableBlock(0)->AppendOp();
    op->SetType("conv2d");
    op->SetInput("Input", {input->Name()});
    op->SetInput("Filter", {filter->Name()});
    op->SetInput("Bias", {bias->Name()});
    op->SetOutput("Output", {out->Name()});
    op->SetAttr("use_cudnn", use_cudnn);
    op->SetAttr("groups", groups);
    op->SetAttr("strides", strides);
    op->SetAttr("paddings", paddings);
    op->SetAttr("padding_algorithm", padding_algorithm);
    op->SetAttr("dilations", dilations);
    op->SetAttr("data_format", data_format);
    op->SetAttr(OpProtoAndCheckerMaker::OpRoleAttrName(),
                static_cast<int>(OpRole::kForward));
    return out;
  }

  VarDesc* conv2d_transpose(VarDesc* input, VarDesc* filter, VarDesc* bias,
                            int groups = 1, std::vector<int> strides = {1, 1},
                            std::vector<int> paddings = {0, 0},
                            std::string padding_algorithm = "EXPLICIT",
                            std::vector<int> dilations = {1, 1},
                            std::string data_format = "NCHW") {
    VarDesc* out = lod_tensor(unique_name());
    OpDesc* op = program_.MutableBlock(0)->AppendOp();
    op->SetType("conv2d_transpose");
    op->SetInput("Input", {input->Name()});
    op->SetInput("Filter", {filter->Name()});
    op->SetInput("Bias", {bias->Name()});
    op->SetOutput("Output", {out->Name()});
    op->SetAttr("groups", groups);
    op->SetAttr("strides", strides);
    op->SetAttr("paddings", paddings);
    op->SetAttr("padding_algorithm", padding_algorithm);
    op->SetAttr("dilations", dilations);
    op->SetAttr("data_format", data_format);
    op->SetAttr(OpProtoAndCheckerMaker::OpRoleAttrName(),
                static_cast<int>(OpRole::kForward));
    return out;
  }

  VarDesc* depthwise_conv2d(VarDesc* input, VarDesc* filter, VarDesc* bias,
                            bool use_cudnn) {
    VarDesc* out = lod_tensor(unique_name());
    OpDesc* op = program_.MutableBlock(0)->AppendOp();
    op->SetType("depthwise_conv2d");
    op->SetInput("Input", {input->Name()});
    op->SetInput("Filter", {filter->Name()});
    op->SetInput("Bias", {bias->Name()});
    op->SetOutput("Out", {out->Name()});
    op->SetAttr("use_cudnn", use_cudnn);
    op->SetAttr(OpProtoAndCheckerMaker::OpRoleAttrName(),
                static_cast<int>(OpRole::kForward));
    return out;
  }

  VarDesc* pool2d(VarDesc* x, bool use_cudnn,
                  const AttributeMap* attrs = nullptr) {
    VarDesc* out = lod_tensor(unique_name());
    OpDesc* op = program_.MutableBlock(0)->AppendOp();
    op->SetType("pool2d");
    op->SetInput("X", {x->Name()});
    op->SetOutput("Out", {out->Name()});
    op->SetAttr("use_cudnn", use_cudnn);
    if (attrs) {
      for (auto& iter : *attrs) {
        op->SetAttr(iter.first, iter.second);
      }
    }
    op->SetAttr(OpProtoAndCheckerMaker::OpRoleAttrName(),
                static_cast<int>(OpRole::kForward));
    return out;
  }

  VarDesc* unsqueeze2(VarDesc* x, const std::vector<int> axes) {
    VarDesc* out = lod_tensor(unique_name());
    OpDesc* op = program_.MutableBlock(0)->AppendOp();
    op->SetType("unsqueeze2");
    op->SetInput("X", {x->Name()});
    op->SetOutput("Out", {out->Name()});
    op->SetAttr("axes", axes);
    return out;
  }

  VarDesc* relu(VarDesc* x, VarDesc* out = nullptr) {
    return unary_op("relu", x, out);
  }

  VarDesc* sigmoid(VarDesc* x, VarDesc* out = nullptr) {
    return unary_op("sigmoid", x, out);
  }

  VarDesc* tanh(VarDesc* x, VarDesc* out = nullptr) {
    return unary_op("tanh", x, out);
  }

  VarDesc* fc(VarDesc* input, VarDesc* w, VarDesc* bias,
              int in_num_col_dims = 1, std::string activation_type = "") {
    VarDesc* out = lod_tensor(unique_name());
    OpDesc* op = program_.MutableBlock(0)->AppendOp();
    op->SetType("fc");
    op->SetInput("Input", {input->Name()});
    op->SetInput("W", {w->Name()});
    op->SetInput("Bias", {bias->Name()});
    op->SetOutput("Out", {out->Name()});
    op->SetAttr("in_num_col_dims", in_num_col_dims);
    op->SetAttr("activation_type", activation_type);
    op->SetAttr(OpProtoAndCheckerMaker::OpRoleAttrName(),
                static_cast<int>(OpRole::kForward));
    return out;
  }

  void lstm(VarDesc* input, VarDesc* w, VarDesc* bias, VarDesc* cell,
            VarDesc* batch_gate, VarDesc* hidden, VarDesc* batch_cell_pre_act,
            VarDesc* h0 = nullptr, VarDesc* c0 = nullptr,
            bool use_peepholes = true, bool is_reverse = false,
            std::string gate_activation = "sigmoid",
            std::string cell_activation = "tanh",
            std::string candidate_activation = "tanh") {
    OpDesc* op = program_.MutableBlock(0)->AppendOp();
    op->SetType("lstm");
    op->SetInput("Input", {input->Name()});
    op->SetInput("Weight", {w->Name()});
    op->SetInput("Bias", {bias->Name()});
    if (h0) {
      op->SetInput("H0", {h0->Name()});
    }
    if (c0) {
      op->SetInput("C0", {c0->Name()});
    }
    op->SetOutput("Hidden", {hidden->Name()});
    op->SetOutput("Cell", {cell->Name()});
    op->SetOutput("BatchGate", {batch_gate->Name()});
    op->SetOutput("BatchCellPreAct", {batch_cell_pre_act->Name()});
    op->SetAttr("use_peepholes", use_peepholes);
    op->SetAttr("is_reverse", is_reverse);
    op->SetAttr("gate_activation", gate_activation);
    op->SetAttr("cell_activation", cell_activation);
    op->SetAttr("candidate_activation", candidate_activation);
    op->SetAttr(OpProtoAndCheckerMaker::OpRoleAttrName(),
                static_cast<int>(OpRole::kForward));
  }

  void gru(VarDesc* input, VarDesc* w, VarDesc* bias, VarDesc* batch_gate,
           VarDesc* batch_reset_hidden_prev, VarDesc* batch_hidden,
           VarDesc* hidden, VarDesc* h0 = nullptr, bool origin_mode = false,
           bool is_reverse = false, std::string activation = "tanh",
           std::string gate_activation = "sigmoid") {
    OpDesc* op = program_.MutableBlock(0)->AppendOp();
    op->SetType("gru");
    op->SetInput("Input", {input->Name()});
    op->SetInput("Weight", {w->Name()});
    op->SetInput("Bias", {bias->Name()});
    if (h0) {
      op->SetInput("H0", {h0->Name()});
    }
    op->SetOutput("BatchGate", {batch_gate->Name()});
    op->SetOutput("BatchResetHiddenPrev", {batch_reset_hidden_prev->Name()});
    op->SetOutput("BatchHidden", {batch_hidden->Name()});
    op->SetOutput("Hidden", {hidden->Name()});
    op->SetAttr("origin_mode", origin_mode);
    op->SetAttr("is_reverse", is_reverse);
    op->SetAttr("activation", activation);
    op->SetAttr("gate_activation", gate_activation);
    op->SetAttr(OpProtoAndCheckerMaker::OpRoleAttrName(),
                static_cast<int>(OpRole::kForward));
  }

  VarDesc* mul(VarDesc* x, VarDesc* y, VarDesc* out = nullptr,
               int x_num_col_dims = 1, int y_num_col_dims = 1,
               bool use_mkldnn = false) {
    AttributeMap attrs;
    attrs["x_num_col_dims"] = x_num_col_dims;
    attrs["y_num_col_dims"] = y_num_col_dims;
    attrs["use_mkldnn"] = use_mkldnn;
    return binary_op("mul", x, y, out, &attrs);
  }

  VarDesc* elementwise_add(VarDesc* x, VarDesc* y, VarDesc* out = nullptr,
                           int axis = -1, bool use_mkldnn = false) {
    AttributeMap attrs;
    attrs["axis"] = axis;
    attrs["use_mkldnn"] = use_mkldnn;
    return binary_op("elementwise_add", x, y, out, &attrs);
  }

  VarDesc* elementwise_mul(VarDesc* x, VarDesc* y, VarDesc* out = nullptr,
                           const AttributeMap* attrs = nullptr) {
    return binary_op("elementwise_mul", x, y, out, attrs);
  }

  VarDesc* dropout(VarDesc* x, float dropout_prob,
                   std::string dropout_implementation) {
    VarDesc* out = lod_tensor(unique_name());
    OpDesc* op = program_.MutableBlock(0)->AppendOp();
    op->SetType("dropout");
    op->SetInput("X", {x->Name()});
    op->SetOutput("Out", {out->Name()});
    op->SetAttr("is_test", true);
    op->SetAttr("dropout_prob", dropout_prob);
    op->SetAttr("dropout_implementation", dropout_implementation);
    op->SetAttr(OpProtoAndCheckerMaker::OpRoleAttrName(),
                static_cast<int>(OpRole::kForward));
    return out;
  }

  VarDesc* concat(std::vector<VarDesc*> inputs, int axis = -1) {
    VarDesc* out = lod_tensor(unique_name());
    OpDesc* op = program_.MutableBlock(0)->AppendOp();
    op->SetType("concat");
    std::vector<std::string> input_names(inputs.size());
    for (size_t i = 0; i < inputs.size(); ++i) {
      input_names[i] = inputs[i]->Name();
    }
    op->SetInput("X", input_names);
    op->SetOutput("Out", {out->Name()});
    op->SetAttr("axis", axis);
    op->SetAttr(OpProtoAndCheckerMaker::OpRoleAttrName(),
                static_cast<int>(OpRole::kForward));
    return out;
  }

  std::vector<VarDesc*> layer_norm(VarDesc* x, VarDesc* scale = nullptr,
                                   VarDesc* bias = nullptr) {
    VarDesc* y = lod_tensor(unique_name());
    VarDesc* mean = lod_tensor(unique_name());
    VarDesc* variance = lod_tensor(unique_name());
    OpDesc* op = program_.MutableBlock(0)->AppendOp();
    op->SetType("layer_norm");
    op->SetInput("X", {x->Name()});
    if (scale) {
      op->SetInput("Scale", {scale->Name()});
    }
    if (bias) {
      op->SetInput("Bias", {bias->Name()});
    }
    op->SetOutput("Y", {y->Name()});
    op->SetOutput("Mean", {mean->Name()});
    op->SetOutput("Variance", {variance->Name()});
    op->SetAttr("epsilon", static_cast<float>(1E-05));
    op->SetAttr("begin_norm_axis", static_cast<int>(1));
    op->SetAttr(OpProtoAndCheckerMaker::OpRoleAttrName(),
                static_cast<int>(OpRole::kForward));
    std::vector<VarDesc*> outs = {y, mean, variance};
    return outs;
  }

  VarDesc* matmul(VarDesc* x, VarDesc* y, VarDesc* alpha = nullptr) {
    VarDesc* out = lod_tensor(unique_name());
    OpDesc* op = program_.MutableBlock(0)->AppendOp();
    op->SetType("matmul");
    op->SetInput("X", {x->Name()});
    op->SetInput("Y", {y->Name()});
    op->SetOutput("Out", {out->Name()});
    return out;
  }

  VarDesc* transpose2(VarDesc* x, std::vector<int> axis,
                      bool with_xshape = false) {
    VarDesc* out = lod_tensor(unique_name());
    OpDesc* op = program_.MutableBlock(0)->AppendOp();
    op->SetType("transpose2");
    op->SetInput("X", {x->Name()});
    op->SetAttr("axis", axis);
    op->SetOutput("Out", {out->Name()});
    if (with_xshape) {
      VarDesc* xshape = lod_tensor(unique_name());
      op->SetOutput("XShape", {xshape->Name()});
    }
    return out;
  }

  VarDesc* reshape2(VarDesc* x, std::vector<int> shape,
                    bool with_xshape = false) {
    VarDesc* out = lod_tensor(unique_name());
    OpDesc* op = program_.MutableBlock(0)->AppendOp();
    op->SetType("reshape2");
    op->SetInput("X", {x->Name()});
    op->SetAttr("shape", shape);
    op->SetOutput("Out", {out->Name()});
    if (with_xshape) {
      VarDesc* xshape = lod_tensor(unique_name());
      op->SetOutput("XShape", {xshape->Name()});
    }
    return out;
  }

  VarDesc* softmax(VarDesc* x, int axis) {
    VarDesc* out = lod_tensor(unique_name());
    OpDesc* op = program_.MutableBlock(0)->AppendOp();
    op->SetType("softmax");
    op->SetInput("X", {x->Name()});
    op->SetAttr("axis", axis);
    op->SetOutput("Out", {out->Name()});
    return out;
  }

  VarDesc* scale(VarDesc* x, float scale, float bias, bool bias_after) {
    VarDesc* out = lod_tensor(unique_name());
    OpDesc* op = program_.MutableBlock(0)->AppendOp();
    op->SetType("scale");
    op->SetInput("X", {x->Name()});
    op->SetAttr("scale", scale);
    op->SetAttr("bias", bias);
    op->SetAttr("bias_after_scale", bias_after);
    op->SetOutput("Out", {out->Name()});
    return out;
  }

  std::vector<VarDesc*> batch_norm(VarDesc* x, VarDesc* scale, VarDesc* bias,
                                   VarDesc* mean, VarDesc* variance) {
    VarDesc* y = lod_tensor(unique_name());
    VarDesc* mean_out = lod_tensor(unique_name());
    VarDesc* variance_out = lod_tensor(unique_name());
    VarDesc* saved_mean = lod_tensor(unique_name());
    VarDesc* saved_variance = lod_tensor(unique_name());
    OpDesc* op = program_.MutableBlock(0)->AppendOp();
    op->SetType("batch_norm");
    op->SetInput("X", {x->Name()});
    op->SetInput("Scale", {scale->Name()});
    op->SetInput("Bias", {bias->Name()});
    op->SetInput("Mean", {mean->Name()});
    op->SetInput("Variance", {variance->Name()});
    op->SetOutput("Y", {y->Name()});
    op->SetOutput("MeanOut", {mean_out->Name()});
    op->SetOutput("VarianceOut", {variance_out->Name()});
    op->SetOutput("SavedMean", {saved_mean->Name()});
    op->SetOutput("SavedVariance", {saved_variance->Name()});
    op->SetAttr("epsilon", static_cast<float>(1e-5));
    op->SetAttr(OpProtoAndCheckerMaker::OpRoleAttrName(),
                static_cast<int>(OpRole::kForward));
    std::vector<VarDesc*> outs = {y, mean_out, variance_out, saved_mean,
                                  saved_variance};
    return outs;
  }

  VarDesc* embedding(VarDesc* x, VarDesc* weights) {
    VarDesc* out = lod_tensor(unique_name());
    OpDesc* op = program_.MutableBlock(0)->AppendOp();
    op->SetType("lookup_table");
    op->SetInput("Ids", {x->Name()});
    op->SetInput("W", {weights->Name()});
    op->SetOutput("Out", {out->Name()});
    return out;
  }

  void backward(std::vector<VarDesc*> targets) {
    // This function is designed to simulate the structure of training program,
    //  but is constructed differently as the actual program.
    BlockDesc* block = program_.MutableBlock(0);
    std::vector<OpDesc*> forward_ops = block->AllOps();
    for (auto* var : targets) {
      OpDesc* none_op = block->AppendOp();
      none_op->SetType("none");
      none_op->SetInput("X", {var->Name()});
      VarDesc* grad_var =
          lod_tensor(GradVarName(var->Name()), var->GetShape(), false);
      none_op->SetOutput("Out", {grad_var->Name()});
    }
    for (int i = forward_ops.size() - 1; i >= 0; --i) {
      OpDesc* op = forward_ops[i];
      OpDesc* grad_op = block->AppendOp();
      grad_op->SetType(op->Type() + "_grad");
      // All op's inputs are grad_op's input.
      for (auto name : op->InputNames()) {
        grad_op->SetInput(name, op->Input(name));
      }
      // All op's outputs are grad_op's input.
      for (auto name : op->OutputNames()) {
        grad_op->SetInput(name, op->Output(name));
      }
      // All op's outputs grad are grad_op's input.
      for (auto name : op->OutputNames()) {
        std::vector<std::string> grad_var_names;
        for (auto var_name : op->Output(name)) {
          VarDesc* var = block->FindVar(var_name);
          VarDesc* grad_var =
              lod_tensor(GradVarName(var_name), var->GetShape(), false);
          grad_var_names.push_back(grad_var->Name());
        }
        grad_op->SetInput(GradVarName(name), grad_var_names);
      }
      // All op's inputs grad are grad_op's output.
      for (auto name : op->InputNames()) {
        std::vector<std::string> grad_var_names;
        for (auto var_name : op->Input(name)) {
          VarDesc* var = block->FindVar(var_name);
          VarDesc* grad_var =
              lod_tensor(GradVarName(var_name), var->GetShape(), false);
          grad_var_names.push_back(grad_var->Name());
        }
        grad_op->SetOutput(GradVarName(name), grad_var_names);
      }
      // TODO(liuyiqun): attrs
    }
  }

 private:
  VarDesc* lod_tensor(std::string name, std::vector<int64_t> shape = {},
                      bool is_persistable = false,
                      proto::VarType::Type data_type = proto::VarType::FP32) {
    auto* var = program_.MutableBlock(0)->Var(name);
    var->SetType(proto::VarType::LOD_TENSOR);
    var->SetDataType(data_type);
    var->SetShape(shape);
    var->SetPersistable(is_persistable);
    return var;
  }

  VarDesc* unary_op(std::string type, VarDesc* x, VarDesc* out = nullptr) {
    if (!out) {
      out = lod_tensor(unique_name());
    }
    OpDesc* op = program_.MutableBlock(0)->AppendOp();
    op->SetType(type);
    op->SetInput("X", {x->Name()});
    op->SetOutput("Out", {out->Name()});
    op->SetAttr(OpProtoAndCheckerMaker::OpRoleAttrName(),
                static_cast<int>(OpRole::kForward));
    return out;
  }

  VarDesc* binary_op(std::string type, VarDesc* x, VarDesc* y,
                     VarDesc* out = nullptr,
                     const AttributeMap* attrs = nullptr) {
    if (!out) {
      out = lod_tensor(unique_name());
    }
    OpDesc* op = program_.MutableBlock(0)->AppendOp();
    op->SetType(type);
    op->SetInput("X", {x->Name()});
    op->SetInput("Y", {y->Name()});
    op->SetOutput("Out", {out->Name()});
    if (attrs) {
      for (auto& iter : *attrs) {
        op->SetAttr(iter.first, iter.second);
      }
    }
    op->SetAttr(OpProtoAndCheckerMaker::OpRoleAttrName(),
                static_cast<int>(OpRole::kForward));
    return out;
  }

  std::string unique_name() { return "tmp_" + std::to_string(idx_++); }

 private:
  ProgramDesc program_;
  int idx_{0};
};

static std::string DebugString(OpDesc* op) {
  std::ostringstream os;
  os << "Op(" << op->Type() << "), inputs:{";
  bool is_first = true;
  for (auto& name : op->InputNames()) {
    if (!is_first) {
      os << ", ";
    }
    os << name << "[";
    bool is_first_var_name = true;
    for (auto& var_name : op->Input(name)) {
      if (!is_first_var_name) {
        os << ", ";
      }
      os << var_name;
      is_first_var_name = false;
    }
    os << "]";
    is_first = false;
  }

  os << "}, outputs:{";
  is_first = true;
  for (auto& name : op->OutputNames()) {
    if (!is_first) {
      os << ", ";
    }
    os << name << "[";
    bool is_first_var_name = true;
    for (auto& var_name : op->Output(name)) {
      if (!is_first_var_name) {
        os << ", ";
      }
      os << var_name;
      is_first_var_name = false;
    }
    os << "]";
    is_first = false;
  }
  os << "}";
  return os.str();
}

static std::string DebugString(const Node* node) {
  std::ostringstream os;
  if (node->IsOp() && node->Op()) {
    OpDesc* op = node->Op();
    os << "Node(" << DebugString(op) << "), inputs:{";
    bool is_first = true;
    for (auto* in : node->inputs) {
      if (!is_first) {
        os << ", ";
      }
      os << in->Name();
      is_first = false;
    }
    os << "}, outputs:{";
    is_first = true;
    for (auto* out : node->outputs) {
      if (!is_first) {
        os << ", ";
      }
      os << out->Name();
      is_first = false;
    }
    os << "}.";
  } else {
    os << "Node(" << node->Name();
    if (node->IsVar() && node->Var()) {
      os << "{";
      bool is_first = true;
      for (auto dim : node->Var()->GetShape()) {
        if (!is_first) {
          os << "x";
        }
        os << dim;
        is_first = false;
      }
      os << "}";
    }
    os << "), inputs:{";
    bool is_first = true;
    for (auto* in : node->inputs) {
      if (!is_first) {
        os << ", ";
      }
      if (in->IsOp() && in->Op()) {
        os << in->Op()->Type();
      }
      is_first = false;
    }
    os << "}, outputs:{";
    is_first = true;
    for (auto* out : node->outputs) {
      if (!is_first) {
        os << ", ";
      }
      if (out->IsOp() && out->Op()) {
        os << out->Op()->Type();
      }
      is_first = false;
    }
    os << "}";
  }
  return os.str();
}

static std::string DebugString(const std::vector<Node*>& nodes) {
  std::ostringstream os;
  for (auto* node : nodes) {
    if (node->IsOp() && node->Op()) {
      os << "  ";
    } else if ((node->IsVar() && node->Var()) || node->IsCtrlVar()) {
      os << "    ";
    }
    os << DebugString(node) << "\n";
  }
  return os.str();
}

static std::string DebugString(const std::unordered_set<Node*>& nodes) {
  std::vector<Node*> vec;
  for (auto* node : nodes) {
    vec.push_back(node);
  }
  return DebugString(vec);
}

static std::string DebugString(Graph* graph) {
  std::ostringstream os;
  os << "Graph: {\n" << DebugString(graph->Nodes()) << "}\n";
  return os.str();
}

static std::string DebugString(const std::unique_ptr<Graph>& graph) {
  return DebugString(graph.get());
}

static std::vector<ir::Node*> GetOpNodes(const std::unique_ptr<Graph>& graph,
                                         std::string op_type) {
  std::vector<ir::Node*> rc;
  for (auto* node : graph->Nodes()) {
    if (node->IsOp() && node->Op() && node->Op()->Type() == op_type) {
      rc.push_back(node);
    }
  }
  return rc;
}

static int GetNumOpNodes(const std::unique_ptr<Graph>& graph,
                         std::string op_type) {
  int num_nodes = 0;
  for (auto* node : graph->Nodes()) {
    if (node->IsOp() && node->Op() && node->Op()->Type() == op_type) {
      num_nodes++;
    }
  }
  return num_nodes;
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle
