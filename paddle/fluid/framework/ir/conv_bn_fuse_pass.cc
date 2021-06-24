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

#include "paddle/fluid/framework/ir/conv_bn_fuse_pass.h"

#include <string>

#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace framework {
class LoDTensor;
class Scope;
}  // namespace framework
}  // namespace paddle

namespace paddle {
namespace framework {
namespace ir {

#define GET_CONV_BN_NODES(pattern_name)                                      \
  /* OPERATORS */                                                            \
  GET_IR_NODE_FROM_SUBGRAPH(conv, conv, pattern_name);                       \
  GET_IR_NODE_FROM_SUBGRAPH(batch_norm, batch_norm, pattern_name);           \
  /* CONV inputs */                                                          \
  GET_IR_NODE_FROM_SUBGRAPH(conv_weight, conv_weight, pattern_name);         \
  /* CONV outputs */                                                         \
  GET_IR_NODE_FROM_SUBGRAPH(conv_out, conv_out, pattern_name);               \
  /* BN inputs */                                                            \
  GET_IR_NODE_FROM_SUBGRAPH(bn_scale, bn_scale, pattern_name);               \
  GET_IR_NODE_FROM_SUBGRAPH(bn_bias, bn_bias, pattern_name);                 \
  GET_IR_NODE_FROM_SUBGRAPH(bn_mean, bn_mean, pattern_name);                 \
  GET_IR_NODE_FROM_SUBGRAPH(bn_variance, bn_variance, pattern_name);         \
  /* BN outputs */                                                           \
  GET_IR_NODE_FROM_SUBGRAPH(bn_out, bn_out, pattern_name); /* Out */         \
  GET_IR_NODE_FROM_SUBGRAPH(bn_mean_out, bn_mean_out, pattern_name);         \
  GET_IR_NODE_FROM_SUBGRAPH(bn_variance_out, bn_variance_out, pattern_name); \
  GET_IR_NODE_FROM_SUBGRAPH(bn_saved_mean, bn_saved_mean, pattern_name);     \
  GET_IR_NODE_FROM_SUBGRAPH(bn_saved_variance, bn_saved_variance, pattern_name)

void recompute_bias_and_weights(const Scope* scope,
                                ir::Node* conv_weight,            //
                                const ir::Node& bn_scale,         //
                                const LoDTensor& bn_bias_tensor,  //
                                const ir::Node& bn_mean,          //
                                const ir::Node& bn_variance,      //
                                LoDTensor* eltwise_y_in_tensor,   //
                                float epsilon, const std::string& conv_type) {
  using EigenVectorArrayMap =
      Eigen::Map<Eigen::Array<float, Eigen::Dynamic, 1>>;
  using ConstEigenVectorArrayMap =
      Eigen::Map<const Eigen::Array<float, Eigen::Dynamic, 1>>;
  using EigenMatrixArrayMap = Eigen::Map<
      Eigen::Array<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;

  // Re-compute bias of conv2d from BN
  PADDLE_ENFORCE_EQ(
      eltwise_y_in_tensor->dims(), bn_bias_tensor.dims(),
      platform::errors::InvalidArgument("Tensor elementwise y(%d) and batch "
                                        "norm bias(%d) must have same dims.",
                                        eltwise_y_in_tensor->dims().size(),
                                        bn_bias_tensor.dims().size()));

  auto* scale_tensor = scope->FindVar(bn_scale.Name())->GetMutable<LoDTensor>();
  auto* variance_tensor =
      scope->FindVar(bn_variance.Name())->GetMutable<LoDTensor>();
  auto* mean_tensor = scope->FindVar(bn_mean.Name())->GetMutable<LoDTensor>();

  ConstEigenVectorArrayMap scale_array(scale_tensor->data<float>(),
                                       scale_tensor->numel(), 1);
  EigenVectorArrayMap variance_array(
      variance_tensor->mutable_data<float>(platform::CPUPlace()),
      variance_tensor->numel(), 1);
  ConstEigenVectorArrayMap mean_array(mean_tensor->data<float>(),
                                      mean_tensor->numel(), 1);
  ConstEigenVectorArrayMap bn_bias_array(bn_bias_tensor.data<float>(),
                                         bn_bias_tensor.numel(), 1);

  // variance will not be used anymore, so make it std_array and then tmp_array
  variance_array += epsilon;
  variance_array = variance_array.sqrt();
  variance_array = scale_array / variance_array;
  for (int i = 0; i < variance_tensor->numel(); i++) {
    PADDLE_ENFORCE_EQ(std::isfinite(variance_array[i]), true,
                      platform::errors::InvalidArgument(
                          "The inverse of Fused batch norm variance "
                          "should be finite. Found nonfinite values! "
                          "Please check %s ",
                          bn_variance.Name()));
  }
  EigenVectorArrayMap eltwise_y_in_array(
      eltwise_y_in_tensor->mutable_data<float>(platform::CPUPlace()),
      eltwise_y_in_tensor->numel(), 1);

  eltwise_y_in_array =
      ((eltwise_y_in_array - mean_array) * variance_array) + bn_bias_array;
  for (int i = 0; i < eltwise_y_in_tensor->numel(); i++) {
    PADDLE_ENFORCE_EQ(std::isfinite(eltwise_y_in_array[i]), true,
                      platform::errors::InvalidArgument(
                          "Fused batch norm bias should be "
                          "finite. Found nonfinite values! "
                          "Please check %s and related variables.",
                          bn_variance.Name()));
  }

  // Re-compute weight of conv2d from BN
  auto* weights = scope->FindVar(conv_weight->Name())->GetMutable<LoDTensor>();
  auto weights_shape = weights->dims();
  auto weights_data = weights->mutable_data<float>(platform::CPUPlace());

  // ConvTranspose weights are in IOHW format
  if (conv_type == "conv2d_transpose") {
    int kernel_size = weights_shape[2] * weights_shape[3];
    for (int i = 0; i < weights->numel();) {
      for (int j = 0; j < weights_shape[1]; ++j) {
        for (int k = 0; k < kernel_size; ++k, ++i) {
          weights_data[i] *= variance_array[j];
        }
      }
    }
  } else {
    auto weights_shape_2d = flatten_to_2d(weights_shape, 1);

    EigenMatrixArrayMap weights_array_2d(weights_data, weights_shape_2d[0],
                                         weights_shape_2d[1]);

    weights_array_2d.colwise() *= variance_array;
  }
}

ConvBNFusePass::ConvBNFusePass() {
  AddOpCompat(OpCompat("conv2d"))
      .AddInput("Input")
      .IsTensor()
      .End()
      .AddInput("Filter")
      .IsTensor()
      .End()
      .AddInput("Bias")
      .IsOptional()
      .End()
      .AddInput("ResidualData")
      .IsOptional()
      .End()
      .AddOutput("Output")
      .IsTensor()
      .End()
      .AddAttr("strides")
      .End()
      .AddAttr("paddings")
      .End()
      .AddAttr("padding_algorithm")
      .IsOptional()
      .IsStringIn({"EXPLICIT", "SAME", "VALID"})
      .End()
      .AddAttr("groups")
      .IsNumGE(1)
      .End()
      .AddAttr("dilations")
      .End()
      .AddAttr("data_format")
      .IsStringIn({"NCHW", "NHWC", "AnyLayout"})
      .End();

  AddOpCompat(OpCompat("batch_norm"))
      .AddInput("X")
      .IsTensor()
      .End()
      .AddInput("Scale")
      .IsTensor()
      .End()
      .AddInput("Bias")
      .IsTensor()
      .End()
      .AddInput("Mean")
      .IsTensor()
      .End()
      .AddInput("Variance")
      .IsTensor()
      .End()
      .AddOutput("MeanOut")
      .IsTensor()
      .End()
      .AddOutput("VarianceOut")
      .IsTensor()
      .End()
      .AddOutput("SavedMean")
      .IsTensor()
      .End()
      .AddOutput("SavedVariance")
      .IsTensor()
      .End()
      .AddOutput("Y")
      .IsTensor()
      .End()
      .AddAttr("epsilon")
      .IsNumLE(0.001f)
      .IsNumGE(0.0f)
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
      .IsNumEQ(1)
      .End();
}

void ConvBNFusePass::ApplyImpl(ir::Graph* graph) const {
  PADDLE_ENFORCE_NOT_NULL(
      graph, platform::errors::InvalidArgument("Graph cannot be nullptr."));
  FusePassBase::Init(name_scope_, graph);

  auto* scope = param_scope();
  PADDLE_ENFORCE_NOT_NULL(
      scope, platform::errors::InvalidArgument("Scope cannot be nullptr."));

  GraphPatternDetector gpd;
  auto* conv_input =
      gpd.mutable_pattern()
          ->NewNode(patterns::PDNodeName(name_scope_, "conv_input"))
          ->AsInput()
          ->assert_is_op_input(conv_type(), "Input");
  patterns::ConvBN conv_bn_pattern(gpd.mutable_pattern(), name_scope_);
  conv_bn_pattern(conv_input, conv_type(), false /*with_eltwise_add*/);

  int found_conv_bn_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    if (!IsCompat(subgraph, g)) {
      LOG(WARNING) << "Pass in op compat failed.";
      return;
    }
    VLOG(4) << "handle " + conv_type() + "BN fuse";
    // conv, batch_norm,
    // conv_weight, conv_out,
    // bn_scale, bn_bias, bn_mean, bn_variance,
    // bn_out, bn_mean_out, bn_variance_out, bn_saved_mean,
    // bn_saved_variance
    GET_CONV_BN_NODES(conv_bn_pattern);

    // check if fuse can be done and if MKL-DNN should be used
    FuseOptions fuse_option = FindFuseOption(*conv, *batch_norm);
    if (fuse_option == DO_NOT_FUSE) {
      VLOG(3) << "do not perform " + conv_type() + " bn fuse";
      return;
    }

    // Get batch norm bias
    auto* bn_bias_tensor =
        scope->FindVar(bn_bias->Name())->GetMutable<LoDTensor>();

    // Create eltwise_y (conv bias) variable
    VarDesc eltwise_y_in_desc(
        patterns::PDNodeName("fuse_conv_bn", conv_type() + "_eltwise_y_in"));
    eltwise_y_in_desc.SetShape(framework::vectorize(bn_bias_tensor->dims()));
    eltwise_y_in_desc.SetDataType(bn_bias_tensor->type());
    eltwise_y_in_desc.SetLoDLevel(bn_bias->Var()->GetLoDLevel());
    eltwise_y_in_desc.SetPersistable(true);
    auto* eltwise_y_in_node = g->CreateVarNode(&eltwise_y_in_desc);
    auto* eltwise_y_in_tensor =
        scope->Var(eltwise_y_in_node->Name())->GetMutable<LoDTensor>();

    // Initialize eltwise_y
    eltwise_y_in_tensor->Resize(bn_bias_tensor->dims());
    std::fill_n(eltwise_y_in_tensor->mutable_data<float>(platform::CPUPlace()),
                eltwise_y_in_tensor->numel(), 0.0f);

    // update weights and biases
    float epsilon =
        BOOST_GET_CONST(float, batch_norm->Op()->GetAttr("epsilon"));
    recompute_bias_and_weights(scope, conv_weight, *bn_scale, *bn_bias_tensor,
                               *bn_mean, *bn_variance, eltwise_y_in_tensor,
                               epsilon, conv_type());

    // with MKL-DNN fuse conv+bn into conv with bias
    // without MKL-DNN fuse conv+bn into conv+elementwise_add
    if (fuse_option == FUSE_MKLDNN) {
      auto input_names = conv->Op()->InputNames();
      bool has_bias = std::find(input_names.begin(), input_names.end(),
                                "Bias") != input_names.end();
      if (has_bias && conv->Op()->Input("Bias").size() > 0) {
        // reuse existing conv bias node
        auto conv_bias_names = conv->Op()->Input("Bias");
        PADDLE_ENFORCE_EQ(
            conv_bias_names.size(), 1UL,
            platform::errors::InvalidArgument("Find input var Bais error."));
        auto* conv_bias_var = scope->FindVar(conv_bias_names[0]);
        auto* conv_bias_tensor = conv_bias_var->GetMutable<LoDTensor>();
        PADDLE_ENFORCE_EQ(
            conv_bias_tensor->dims(), eltwise_y_in_tensor->dims(),
            platform::errors::InvalidArgument(
                "Tensor convolution bias(%d) and elementwise y(%d) "
                "must have same dims.",
                conv_bias_tensor->dims().size(),
                eltwise_y_in_tensor->dims().size()));

        auto eigen_conv_bias = EigenVector<float>::From(*conv_bias_tensor);
        eigen_conv_bias += EigenVector<float>::From(*eltwise_y_in_tensor);
      } else {
        // add new conv_bias node
        conv->Op()->SetInput(
            "Bias", std::vector<std::string>({eltwise_y_in_node->Name()}));
        IR_NODE_LINK_TO(eltwise_y_in_node, conv);
      }
      conv->Op()->SetOutput("Output",
                            std::vector<std::string>({bn_out->Name()}));
      if (!IsCompat(*conv->Op())) {
        LOG(WARNING) << "conv_bn fuse pass in out conv op compat failed.";
        return;
      }
      GraphSafeRemoveNodes(
          graph,
          {conv_out, bn_scale, bn_bias, bn_mean, bn_variance, batch_norm,
           bn_mean_out, bn_variance_out, bn_saved_mean, bn_saved_variance});

      IR_NODE_LINK_TO(conv, bn_out);
      found_conv_bn_count++;
    } else {  // fuse_option == FUSE_NATIVE
      // create an elementwise add node.
      OpDesc desc;
      desc.SetInput("X", std::vector<std::string>({conv_out->Name()}));
      desc.SetInput("Y", std::vector<std::string>({eltwise_y_in_node->Name()}));
      desc.SetOutput("Out", std::vector<std::string>({bn_out->Name()}));
      desc.SetType("elementwise_add");
      desc.SetAttr("axis", 1);
      if (!IsCompat(desc)) {
        LOG(WARNING)
            << "conv_bn fuse pass in out elementwise_add op compat failed.";
        return;
      }
      auto eltwise_op = g->CreateOpNode(&desc);  // OpDesc will be copied.

      GraphSafeRemoveNodes(graph, {bn_scale, bn_bias, bn_mean, bn_variance,
                                   batch_norm, bn_mean_out, bn_variance_out,
                                   bn_saved_mean, bn_saved_variance});

      IR_NODE_LINK_TO(conv_out, eltwise_op);
      IR_NODE_LINK_TO(eltwise_y_in_node, eltwise_op);
      IR_NODE_LINK_TO(eltwise_op, bn_out);
      found_conv_bn_count++;
    }
  };

  gpd(graph, handler);

  AddStatis(found_conv_bn_count);
}

ConvEltwiseAddBNFusePass::ConvEltwiseAddBNFusePass() {
  AddOpCompat(OpCompat("conv2d"))
      .AddInput("Input")
      .IsTensor()
      .End()
      .AddInput("Filter")
      .IsTensor()
      .End()
      .AddInput("Bias")
      .IsOptional()
      .End()
      .AddInput("ResidualData")
      .IsOptional()
      .End()
      .AddOutput("Output")
      .IsTensor()
      .End()
      .AddAttr("strides")
      .End()
      .AddAttr("paddings")
      .End()
      .AddAttr("padding_algorithm")
      .IsStringIn({"EXPLICIT", "SAME", "VALID"})
      .IsOptional()
      .End()
      .AddAttr("groups")
      .IsNumGE(1)
      .End()
      .AddAttr("dilations")
      .End()
      .AddAttr("data_format")
      .IsStringIn({"NCHW", "NHWC", "AnyLayout"})
      .End();

  AddOpCompat(OpCompat("batch_norm"))
      .AddInput("X")
      .IsTensor()
      .End()
      .AddInput("Scale")
      .IsTensor()
      .End()
      .AddInput("Bias")
      .IsTensor()
      .End()
      .AddInput("Mean")
      .IsTensor()
      .End()
      .AddInput("Variance")
      .IsTensor()
      .End()
      .AddOutput("MeanOut")
      .IsTensor()
      .End()
      .AddOutput("VarianceOut")
      .IsTensor()
      .End()
      .AddOutput("SavedMean")
      .IsTensor()
      .End()
      .AddOutput("SavedVariance")
      .IsTensor()
      .End()
      .AddOutput("Y")
      .IsTensor()
      .End()
      .AddAttr("epsilon")
      .IsNumLE(0.001f)
      .IsNumGE(0.0f)
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
      .IsNumEQ(1)
      .End();
}

void ConvEltwiseAddBNFusePass::ApplyImpl(ir::Graph* graph) const {
  PADDLE_ENFORCE_NOT_NULL(
      graph, platform::errors::InvalidArgument("Graph cannot be nullptr."));
  FusePassBase::Init(name_scope_, graph);

  auto* scope = param_scope();
  PADDLE_ENFORCE_NOT_NULL(
      scope, platform::errors::InvalidArgument("Scope cannot be nullptr."));

  GraphPatternDetector gpd;
  auto* conv_input =
      gpd.mutable_pattern()
          ->NewNode(patterns::PDNodeName(name_scope_, "conv_input"))
          ->AsInput()
          ->assert_is_op_input(conv_type(), "Input");
  patterns::ConvBN conv_bn_pattern(gpd.mutable_pattern(), name_scope_);
  conv_bn_pattern(conv_input, conv_type(), true /*with_eltwise_add*/);

  int found_conv_bn_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    if (!IsCompat(subgraph, g)) {
      LOG(WARNING) << "Pass in op compat failed.";
      return;
    }
    VLOG(4) << "handle " + conv_type() + "BN fuse";
    // conv, batch_norm,
    // conv_weight, conv_out,
    // bn_scale, bn_bias, bn_mean, bn_variance,
    // bn_out, bn_mean_out, bn_variance_out, bn_saved_mean,bn_saved_variance
    GET_CONV_BN_NODES(conv_bn_pattern);
    // OPERATORS
    GET_IR_NODE_FROM_SUBGRAPH(eltwise, eltwise, conv_bn_pattern);
    // BIAS inputs
    GET_IR_NODE_FROM_SUBGRAPH(eltwise_y_in, eltwise_y_in, conv_bn_pattern);
    // BIAS outputs
    GET_IR_NODE_FROM_SUBGRAPH(eltwise_out, eltwise_out, conv_bn_pattern);

    // Get eltwise_y (conv bias) variable
    auto* eltwise_y_in_tensor =
        scope->FindVar(eltwise_y_in->Name())->GetMutable<LoDTensor>();

    // Get batch norm bias
    auto* bn_bias_tensor =
        scope->FindVar(bn_bias->Name())->GetMutable<LoDTensor>();

    // update weights and biases
    float epsilon =
        BOOST_GET_CONST(float, batch_norm->Op()->GetAttr("epsilon"));

    // if bias is an input to other ops as well then we cannot overwrite it
    // so we create separate elementwise Y in nodes
    if (eltwise_y_in->outputs.size() > 1) {
      // Make a copy of eltwise Y input tensor
      // Create eltwise_y (conv bias) variable
      VarDesc eltwise_y_in_desc(patterns::PDNodeName(
          name_scope_, "eltwise_y_in" + std::to_string(found_conv_bn_count)));
      eltwise_y_in_desc.SetShape(
          framework::vectorize(eltwise_y_in_tensor->dims()));
      eltwise_y_in_desc.SetDataType(eltwise_y_in_tensor->type());
      eltwise_y_in_desc.SetLoDLevel(eltwise_y_in->Var()->GetLoDLevel());
      eltwise_y_in_desc.SetPersistable(true);
      auto* eltwise_y_in_node = g->CreateVarNode(&eltwise_y_in_desc);
      auto* eltwise_y_in_tensor_ex =
          scope->Var(eltwise_y_in_node->Name())->GetMutable<LoDTensor>();

      // Initialize eltwise_y
      TensorCopy(*eltwise_y_in_tensor, platform::CPUPlace(),
                 eltwise_y_in_tensor_ex);

      recompute_bias_and_weights(scope, conv_weight, *bn_scale, *bn_bias_tensor,
                                 *bn_mean, *bn_variance, eltwise_y_in_tensor_ex,
                                 epsilon, conv_type());
      // Set new var
      eltwise->Op()->RenameInput(eltwise_y_in->Name(),
                                 eltwise_y_in_node->Name());
      // Link new bias node to eltwise
      IR_NODE_LINK_TO(eltwise_y_in_node, eltwise);
      // unlink original bias from eltwise_op
      eltwise_y_in->outputs.erase(
          std::remove_if(eltwise_y_in->outputs.begin(),
                         eltwise_y_in->outputs.end(),
                         [&](Node*& n) {
                           return n->id() == eltwise->id() ? true : false;
                         }),
          eltwise_y_in->outputs.end());
    } else {
      recompute_bias_and_weights(scope, conv_weight, *bn_scale, *bn_bias_tensor,
                                 *bn_mean, *bn_variance, eltwise_y_in_tensor,
                                 epsilon, conv_type());
    }

    // Update the elementwise_add node
    eltwise->Op()->SetAttr("axis", 1);
    eltwise->Op()->SetOutput("Out", std::vector<std::string>({bn_out->Name()}));
    if (!IsCompat(*eltwise->Op())) {
      LOG(WARNING)
          << "conv_eltwise_bn fuse pass in out eltwise op compat failed.";
      return;
    }
    GraphSafeRemoveNodes(
        graph,
        {bn_scale, bn_bias, bn_mean, bn_variance, batch_norm, bn_mean_out,
         bn_variance_out, bn_saved_mean, bn_saved_variance, eltwise_out});

    IR_NODE_LINK_TO(eltwise, bn_out);

    found_conv_bn_count++;
  };

  gpd(graph, handler);

  AddStatis(found_conv_bn_count);
}

ConvTransposeBNFusePass::ConvTransposeBNFusePass() {
  AddOpCompat(OpCompat("conv2d_transpose"))
      .AddInput("Input")
      .IsTensor()
      .End()
      .AddInput("Filter")
      .IsTensor()
      .End()
      .AddInput("Bias")
      .IsOptional()
      .End()
      .AddOutput("Output")
      .IsTensor()
      .End()
      .AddAttr("strides")
      .End()
      .AddAttr("paddings")
      .End()
      .AddAttr("padding_algorithm")
      .IsStringIn({"EXPLICIT", "SAME", "VALID"})
      .IsOptional()
      .End()
      .AddAttr("groups")
      .IsNumGE(1)
      .End()
      .AddAttr("dilations")
      .End()
      .AddAttr("data_format")
      .IsStringIn({"NCHW", "NHWC", "AnyLayout"})
      .End();
}

ConvTransposeEltwiseAddBNFusePass::ConvTransposeEltwiseAddBNFusePass() {
  AddOpCompat(OpCompat("conv2d_transpose"))
      .AddInput("Input")
      .IsTensor()
      .End()
      .AddInput("Filter")
      .IsTensor()
      .End()
      .AddInput("Bias")
      .IsOptional()
      .End()
      .AddOutput("Output")
      .IsTensor()
      .End()
      .AddAttr("strides")
      .End()
      .AddAttr("paddings")
      .End()
      .AddAttr("padding_algorithm")
      .IsStringIn({"EXPLICIT", "SAME", "VALID"})
      .IsOptional()
      .End()
      .AddAttr("groups")
      .IsNumGE(1)
      .End()
      .AddAttr("dilations")
      .End()
      .AddAttr("data_format")
      .IsStringIn({"NCHW", "NHWC", "AnyLayout"})
      .End();
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(conv_bn_fuse_pass, paddle::framework::ir::ConvBNFusePass);
REGISTER_PASS(conv_eltwiseadd_bn_fuse_pass,
              paddle::framework::ir::ConvEltwiseAddBNFusePass);
REGISTER_PASS(conv_transpose_bn_fuse_pass,
              paddle::framework::ir::ConvTransposeBNFusePass);
REGISTER_PASS(conv_transpose_eltwiseadd_bn_fuse_pass,
              paddle::framework::ir::ConvTransposeEltwiseAddBNFusePass);
REGISTER_PASS(depthwise_conv_bn_fuse_pass,
              paddle::framework::ir::DepthwiseConvBNFusePass);
REGISTER_PASS(depthwise_conv_eltwiseadd_bn_fuse_pass,
              paddle::framework::ir::DepthwiseConvEltwiseAddBNFusePass);
REGISTER_PASS_CAPABILITY(conv_bn_fuse_pass)
    .AddCombination(
        paddle::framework::compatible::OpVersionComparatorCombination()
            .LE("conv2d", 1)
            .EQ("batch_norm", 0));
REGISTER_PASS_CAPABILITY(conv_eltwiseadd_bn_fuse_pass)
    .AddCombination(
        paddle::framework::compatible::OpVersionComparatorCombination()
            .LE("conv2d", 1)
            .LE("elementwise_add", 1)
            .EQ("batch_norm", 0));
