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

#pragma once

#ifdef PADDLE_WITH_TESTING
#include <gtest/gtest_prod.h>
#endif

#include <map>
#include <memory>
#include <numeric>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "paddle/fluid/framework/ir/graph.h"
#include "paddle/fluid/framework/ir/node.h"
#include "paddle/fluid/inference/analysis/dot.h"

namespace paddle {
namespace framework {
namespace ir {
class Graph;
class Node;
}  // namespace ir
}  // namespace framework
}  // namespace paddle

namespace paddle {
namespace framework {
namespace ir {
class PDPattern;

// Some basic terminologies:
//   - PDPattern: a pattern defined as a data flow graph.
//   - PDNode: the node in the pattern, each PDNode represents an `ir::Node`
//     that meets some conditions defined in `PDNode.teller`.
//   - A pattern is defined with PDNodes with edges.

// Pattern detector node. This node helps to build a pattern.
struct PDNode {
  // tell whether an ir::Node* is a candidation for a PDNode.
  using teller_t = std::function<bool(Node*)>;
  enum class Type { kOp, kVar };
  enum class Role {
    kUnknown,      // No role,
    kInput,        // an input and will be retained,
    kOutput,       // an output and will be retained,
    kIntermediate  // will be removed after handler.
  };

  // this link to others
  PDNode& LinksTo(const std::vector<PDNode*>& others);
  PDNode& LinksFrom(const std::vector<PDNode*>& others);

  bool Tell(Node* node) const {
    if (teller_) return teller_(node);

    for (auto& asrt : asserts_) {
      if (!asrt(node)) return false;
    }
    return true;
  }

  bool IsOp() const { return type_ == Type::kOp; }
  bool IsVar() const { return type_ == Type::kVar; }

  const std::string& name() const { return name_; }

  PDNode& operator=(const PDNode&) = delete;
  PDNode(const PDNode&) = delete;

  // Mark this node is an Input of a subgraph and will be retained.
  PDNode* AsInput() {
    role_ = Role::kInput;
    return this;
  }
  // Mark this node is an Output of a subgraph and will be retained.
  PDNode* AsOutput() {
    role_ = Role::kOutput;
    return this;
  }
  // Mark this node will be removed, so all the links should be inside a matched
  // sub-graph.
  PDNode* AsIntermediate() {
    role_ = Role::kIntermediate;
    return this;
  }

  bool IsIntermediate() const { return role_ == Role::kIntermediate; }
  bool IsInput() const { return role_ == Role::kInput; }
  bool IsOutput() const { return role_ == Role::kOutput; }

  // Assertions, helper functions to simplify the pattern definition.
  PDNode* assert_is_op();
  PDNode* assert_is_op(const std::string& op_type);
  PDNode* assert_is_var();
  PDNode* assert_var_dtype(proto::VarType::Type dtype);
  PDNode* assert_is_not_ctrl_var();
  PDNode* assert_var_not_persistable();
  PDNode* assert_is_persistable_var();
  PDNode* assert_is_op_output(const std::string& op_type);
  PDNode* assert_is_op_output(const std::string& op_type,
                              const std::string& argument);
  PDNode* assert_is_op_input(const std::string& op_type);
  PDNode* assert_is_op_input(const std::string& op_type,
                             const std::string& argument);
  PDNode* assert_is_op_nth_input(const std::string& op_type,
                                 const std::string& argument, int nth);
  PDNode* assert_is_not_op_input(const std::string& argument);
  PDNode* assert_is_op_nth_output(const std::string& op_type,
                                  const std::string& argument, int nth);
  PDNode* assert_is_only_input_of_op(const std::string& op_type);
  PDNode* assert_is_only_output_of_op(const std::string& op_type);
  PDNode* assert_op_has_n_inputs(const std::string& op_type, size_t n);
  PDNode* assert_op_has_n_outputs(const std::string& op_type, size_t n);
  PDNode* assert_more(teller_t&& teller);

  PDNode* assert_is_ops_output(const std::unordered_set<std::string>& op_types);
  PDNode* assert_is_ops(const std::unordered_set<std::string>& op_types);
  PDNode* assert_is_ops_output(const std::unordered_set<std::string>& op_types,
                               const std::string& argument);
  PDNode* assert_is_ops_nth_input(
      const std::unordered_set<std::string>& op_types,
      const std::string& argument, int nth);
  PDNode* assert_is_ops_input(const std::unordered_set<std::string>& op_types);
  PDNode* assert_is_ops_input(const std::unordered_set<std::string>& op_types,
                              const std::string& argument);
  PDNode* assert_is_ops_nth_output(
      const std::unordered_set<std::string>& op_types,
      const std::string& argument, int nth);

  PDNode* assert_is_only_input_of_ops(
      const std::unordered_set<std::string>& op_types);
  PDNode* assert_is_only_output_of_ops(
      const std::unordered_set<std::string>& op_types);

  PDNode* assert_has_n_inputs(size_t n);
  PDNode* assert_has_n_outputs(size_t n);

  template <typename T>
  PDNode* assert_op_attr(const std::string& attr_name, const T& attr) {
    asserts_.emplace_back([=](Node* x) {
      return x && x->IsOp() && x->Op()->HasAttr(attr_name) &&
             BOOST_GET_CONST(T, x->Op()->GetAttr(attr_name)) == attr;
    });
    return this;
  }

 private:
  PDNode(PDPattern* pattern, const std::string& name = "",
         Type type = Type::kVar)
      : pattern_(pattern), name_(name), type_(type) {}
  PDNode(teller_t&& teller, PDPattern* pattern, const std::string& name = "",
         Type type = Type::kVar)
      : teller_(std::move(teller)),
        pattern_(pattern),
        name_(name),
        type_(type) {
    PADDLE_ENFORCE_NOT_NULL(
        teller_,
        platform::errors::NotFound("invalid teller is set, teller is null"));
  }

  PDNode(PDNode&& other) = default;

  friend class PDPattern;

  // Will removed latter.
  teller_t teller_;
  std::vector<teller_t> asserts_;
  PDPattern* pattern_;
  std::string name_;
  Type type_;
  Role role_{Role::kUnknown};
};

/*
 * A pattern in a graph, which defined with PDNode and edges. Most graph
 * patterns can be divided into PDNodes and link relations between them.
 *
 * For example, the FC fusion need to filter the MUL and ELEMENTWISE_ADD
 * operators from the computation graph, the MUL's output should have only one
 * consumer which is the ELEMENTWISE_ADD.
 * This pattern can be defined as with the following pseudo codes
 *
 *     // Create two operator PDNodes.
 *     MUL = PDPattern.NewNode().assert_is_op("mul");
 *     ELE = PDPattern.NewNode().assert_is_op("elementwise_add");
 *     // Create the variable PDNodes.
 *     MUL_out = PDPattern.NewNode().assert_is_op_output("mul") \
 *                                  .assert_is_op_input("elementwise_add") \
 *                                  .AsIntermediate();
 *     // Add relations.
 *     MUL->LinksTo({MUL_out});
 *     MUL_out->LinksTo({ELE});
 *
 * One can add more specific asserts for PDNodes or edges, both the Operator
 * and Variable Nodes can be ruled in PDNode.assert_more(...).
 *
 * PDPattern can record the general patterns, such as the pattern represents
 *   - Op in CPU -> Op in GPU -> Op in CPU, to findout the IO abnormal place.
 *   - Ops whose inputs and outputs share the same variables
 */
class PDPattern {
 public:
  using edge_t = std::pair<PDNode*, PDNode*>;

  void AddEdge(PDNode* a, PDNode* b);

  PDNode* NewNode(PDNode::teller_t&& teller, const std::string& name = NewID());
  PDNode* NewNode(const std::string& name = NewID());
  PDNode* NewNode(const std::string& prefix, const std::string& name) {
    return NewNode(prefix + "/" + name);
  }
  PDNode* RetrieveNode(const std::string& id) const;

  const std::vector<std::unique_ptr<PDNode>>& nodes() const { return nodes_; }
  const std::vector<edge_t>& edges() const { return edges_; }

  std::string DotString() const;

 private:
#ifdef PADDLE_WITH_TESTING
  FRIEND_TEST(PDPattern, AddEdge);
  FRIEND_TEST(PDPattern, NewNode);
#endif

  static std::string NewID() { return "pdnode-" + std::to_string(id_++); }

  std::vector<std::unique_ptr<PDNode>> nodes_;
  std::vector<edge_t> edges_;
  std::map<std::string, PDNode*> node_map_;
  static size_t id_;
};

/*
 * GraphPatternDetector helps to detect the specific patterns in the graph.
 * Input a pattern, output a list of the matched subgraphs/nodes.
 * This helper can be used to support fuse(conv+batchnorm => batchnorm e.g.).
 *
 * The algorithm has three phases:
 *   1. Mark the nodes that match the defined PDNodes in a PDPattern,
 *   2. Extend a PDNode to subgraphs by deducing the connection relation defined
 *      in PAPattern(the edges),
 *   3. Get the filtered subgraphs and treat them with a pre-defined handler.
 *
 * Usage:
 *    // Create a detector
 *    GraphPatternDetector detector;
 *    // Define the detector's pattern, by adding PDNode and define the edges.
 *    auto* node0 = detector.mutable_pattern().AddNode(...)
 *    auto* node1 = detector.mutable_pattern().AddNode(...)
 *    node0->teller = some lambda.
 *    node1->teller = some lambda.
 *    detector.mutable_pattern().AddEdge(node0, node1);
 *    // Create an handler, to define the behavior of treating the filtered
 *    // subgraphs that comply with the patterns.
 *    GraphPatternDetector::handle_t handler = some labmda
 *    // Execute the detector.
 *    detector(&graph, handler);
 */
class GraphPatternDetector {
 public:
  using subgraph_t = std::map<PDNode*, Node*>;

  // Operate on the detected pattern.
  using handle_t =
      std::function<void(const subgraph_t& /*hitted pattern*/, Graph*)>;

  void operator()(Graph* graph, handle_t handler);

  const PDPattern& pattern() const { return pattern_; }
  PDPattern* mutable_pattern() { return &pattern_; }

 private:
  // Mark the nodes that fits the pattern.
  bool MarkPDNodesInGraph(const ir::Graph& graph);

  // Detect all the pattern and output the hit records.
  std::vector<subgraph_t> DetectPatterns();

  // Remove duplicate patterns.
  void UniquePatterns(std::vector<subgraph_t>* subgraphs);

  // Sort subgraphs, sort subgraphs by the specified node so that
  // the removed forward and backward subgraphs are corresponding
  // when two subgraphs are overlapped. Note: this function is
  // currently only used for bn_add_act, refer to PR28196 for details.
  void SortSubgraphs(std::vector<subgraph_t>* subgraphs);

  // Remove overlapped match subgraphs, when overlapped, keep the previous one.
  // The intermediate PDNodes will be removed, so can't shared by multiple
  // patterns.
  void RemoveOverlappedMatch(std::vector<subgraph_t>* subgraphs);

  // Validate whether the intermediate nodes are linked by external nodes.
  void ValidateByNodeRole(std::vector<subgraph_t>* subgraphs);

#ifdef PADDLE_WITH_TESTING
  FRIEND_TEST(GraphPatternDetecter, MarkPDNodesInGraph);
  FRIEND_TEST(GraphPatternDetecter, DetectPatterns);
#endif

 private:
  using hit_rcd_t =
      std::pair<Node* /*node in graph*/, PDNode* /*node in pattern*/>;
  PDPattern pattern_;
  std::map<const PDNode*, std::set<Node*>> pdnodes2nodes_;
};

// some helper methods.

// Tell if a var links to an Op
bool VarLinksToOp(Node* node, const std::string& op_type);

// Tell if an op links to a var
bool VarLinksFromOp(Node* node, const std::string& op_type);

// Check whether a var node is a op node's nth input.
bool IsNthInput(Node* var, Node* op, const std::string& argument, size_t nth);

// Check whether the op node has input of given name.
bool HasInput(Node* op, const std::string& argument);

// Check whether the op node has output of given name.
bool HasOutput(Node* op, const std::string& argument);

// Tell whether a var node is a op node's nth output.
bool IsNthOutput(Node* var, Node* op, const std::string& argument, size_t nth);

// Graph safely remove some nodes, will automatically clean up the edges.
void GraphSafeRemoveNodes(Graph* graph,
                          const std::unordered_set<const Node*>& nodes);

// Some pre-defined patterns those can be reused in multiple passes.
// The related Fluid Layer or Op should be one pattern here for better re-usage
// across different fusion.
namespace patterns {

struct KeyCounter {
  static KeyCounter& Instance() {
    static KeyCounter x;
    return x;
  }

  int IncCounter(const std::string& key) { return dic_[key]++; }

 private:
  std::unordered_map<std::string, size_t> dic_;
};

// Generate a unique PDNode's name with name_scope and id.
// The format is {name_scope}/{repr}/{id}/{name}
static std::string PDNodeName(const std::string& name_scope,
                              const std::string& repr, size_t id,
                              const std::string& name) {
  return string::Sprintf("%s/%s/%d/%s", name_scope, repr, id, name);
}
// Generate a unique PDNode's name.
// The format is {name_scope}/{repr}/{id}
static std::string PDNodeName(const std::string& name_scope,
                              const std::string& repr) {
  return string::Sprintf("%s/%s/%d", name_scope, repr,
                         KeyCounter::Instance().IncCounter(repr));
}
// Generate a unique key. It can be used for a universally unique temporary
// name.
// The format is {repr}/{id}
static std::string UniqueKey(const std::string& repr) {
  return string::Sprintf("%s/%d", repr,
                         KeyCounter::Instance().IncCounter(repr));
}

// Declare a PDNode in a pattern, will create two methods:
// std::string xxx_repr(); return this PDNode's string id.
// PDNode* xxx_n(); return the corresponding PDNode.
#define PATTERN_DECL_NODE(name__)                        \
  std::string name__##_repr() const {                    \
    return PDNodeName(name_scope_, repr_, id_, #name__); \
  }                                                      \
  PDNode* name__##_n() const { return pattern->RetrieveNode(name__##_repr()); }

// Get an ir::Node* from the matched subgraph.
// var: variable.
// arg: the argument declared by PATTERN_DECL_NODE in a pattern definition.
// pat: the pattern object.
#define GET_IR_NODE_FROM_SUBGRAPH(var, arg, pat)                               \
  PADDLE_ENFORCE_NE(subgraph.count(pat.arg##_n()), 0UL,                        \
                    platform::errors::NotFound("Node not found for PDNode %s", \
                                               pat.arg##_repr()));             \
  Node* var = subgraph.at(pat.arg##_n());                                      \
  PADDLE_ENFORCE_NOT_NULL(                                                     \
      var, platform::errors::NotFound("node %s not exists in the sub-graph",   \
                                      #arg));

// The base class of all the patterns.
struct PatternBase {
  PatternBase(PDPattern* pattern, const std::string& name_scope,
              const std::string& repr)
      : pattern(pattern),
        name_scope_(name_scope),
        repr_(repr),
        id_(KeyCounter::Instance().IncCounter(repr)) {}

  PDPattern* pattern;

 protected:
  std::string name_scope_;
  std::string repr_;
  size_t id_;
};

// Conv with batch norm
// op: conv + (elementwise_add +) batch_norm
// named nodes:
// conv_weight, conv_out, conv,
// bn_x, bn_scale, bn_bias, bn_mean,  bn_variance,
// bn_batch_norm, bn_y, bn_mean_out, bn_variance_out,
// bn_saved_mean, bn_saved_variance
struct ConvBN : public PatternBase {
  ConvBN(PDPattern* pattern, const std::string& name_scope)
      : PatternBase(pattern, name_scope, "conv_bn") {}

  PDNode* operator()(PDNode* conv_input, const std::string& conv_type,
                     bool with_eltwise_add);

  // declare operator node's name
  PATTERN_DECL_NODE(conv);
  PATTERN_DECL_NODE(batch_norm);
  PATTERN_DECL_NODE(eltwise);  // ELEMENTWISE_ADD
  // CONV inputs
  PATTERN_DECL_NODE(conv_weight);  // Filter
  // CONV outputs
  PATTERN_DECL_NODE(conv_out);  // tmp
  // ELTWISE inputs
  PATTERN_DECL_NODE(eltwise_y_in);
  // ELTWISE outputs
  PATTERN_DECL_NODE(eltwise_out);  // tmp
  // BN inputs
  PATTERN_DECL_NODE(bn_scale);
  PATTERN_DECL_NODE(bn_bias);
  PATTERN_DECL_NODE(bn_mean);
  PATTERN_DECL_NODE(bn_variance);
  // BN outputs
  PATTERN_DECL_NODE(bn_out);  // Out
  PATTERN_DECL_NODE(bn_mean_out);
  PATTERN_DECL_NODE(bn_variance_out);
  PATTERN_DECL_NODE(bn_saved_mean);
  PATTERN_DECL_NODE(bn_saved_variance);
};

// Conv with Activation
// op: conv + activation
// named nodes:
// conv_input, conv_weight,
// conv_out, conv,
// activation_out, activation
struct ConvActivation : public PatternBase {
  ConvActivation(PDPattern* pattern, const std::string& name_scope)
      : PatternBase(pattern, name_scope, "conv_activation") {}

  PDNode* operator()(PDNode* conv_input, std::string conv_type = "conv2d",
                     std::string activation_type = "relu");

  // declare operator node's name
  PATTERN_DECL_NODE(conv);
  PATTERN_DECL_NODE(activation);
  // declare variable node's name
  PATTERN_DECL_NODE(conv_weight);
  PATTERN_DECL_NODE(conv_out);
  PATTERN_DECL_NODE(activation_out);
};

// SEQCONV with Elementwise_Add ReLU
// op: seqconv + elementwise_add + relu
// named nodes:
// seqconv_input, seqconv_weight,
// seqconv_out, seqconv,
// elementwise_add_bias, elementwise_add_out, elementwise_add
// relu_out, relu
struct SeqConvEltAddRelu : public PatternBase {
  SeqConvEltAddRelu(PDPattern* pattern, const std::string& name_scope)
      : PatternBase(pattern, name_scope, "seqconv_eltadd_relu") {}

  PDNode* operator()(PDNode* seqconv_input);

  // declare operator node's name
  PATTERN_DECL_NODE(seqconv);
  PATTERN_DECL_NODE(eltadd);
  PATTERN_DECL_NODE(relu);
  // declare variable node's name
  PATTERN_DECL_NODE(seqconv_weight);
  PATTERN_DECL_NODE(seqconv_out);
  PATTERN_DECL_NODE(eltadd_bias);
  PATTERN_DECL_NODE(eltadd_out);
  PATTERN_DECL_NODE(relu_out);
};

// FC with bias
// op: mul + elementwise_add
// named nodes:
// mul, elementwise_add
// w, mul_out, bias, fc_out
struct FC : public PatternBase {
  FC(PDPattern* pattern, const std::string& name_scope)
      : PatternBase(pattern, name_scope, "fc") {}

  PDNode* operator()(PDNode* x, bool with_bias, bool with_relu);

  // declare operator node's name
  PATTERN_DECL_NODE(fc);
  PATTERN_DECL_NODE(mul);
  PATTERN_DECL_NODE(elementwise_add);
  PATTERN_DECL_NODE(relu);
  // declare variable node's name
  PATTERN_DECL_NODE(w);
  PATTERN_DECL_NODE(mul_out);  // (x,w) -> mul_out
  PATTERN_DECL_NODE(bias);
  PATTERN_DECL_NODE(elementwise_add_out);
  PATTERN_DECL_NODE(relu_out);
};

// MKL-DNN's FC with bias
// op: fc
// named node:
// fc
// w, bias, output
struct FCMKLDNN : public PatternBase {
  FCMKLDNN(PDPattern* pattern, const std::string& name_scope)
      : PatternBase(pattern, name_scope, "fc_mkldnn") {}

  PDNode* operator()(PDNode* x, bool with_bias);

  // declare operator node's name
  PATTERN_DECL_NODE(fc);
  // declare variable node's name
  PATTERN_DECL_NODE(input);
  PATTERN_DECL_NODE(weights);
  PATTERN_DECL_NODE(bias);
  PATTERN_DECL_NODE(output);
};

//
// \brief   Pattern looking for fc and a directly following activation
// operator.
//
// \note    Currently only gelu and tanh are supported as an activation
// function.
//          Formula: act(fc(x))
//          Op: fc + act
struct FCActOneDNN : public PatternBase {
  FCActOneDNN(PDPattern* pattern, const std::string& name_scope)
      : PatternBase(pattern, name_scope, "fc_act_onednn") {}

  PDNode* operator()(const std::string& act_type);

  // declare operator node's name
  PATTERN_DECL_NODE(fc);
  PATTERN_DECL_NODE(act);
  PATTERN_DECL_NODE(fc_out);
  PATTERN_DECL_NODE(act_out);
};

// Embedding
struct Embedding : public PatternBase {
  Embedding(PDPattern* pattern, const std::string& name_scope)
      : PatternBase(pattern, name_scope, "embedding") {}

  PDNode* operator()(PDNode* x);

  // declare operator node's name
  PATTERN_DECL_NODE(lookup_table);
  // Inputs
  //
  PATTERN_DECL_NODE(Ids);
  PATTERN_DECL_NODE(W);  // embeddings
  // Outputs
  PATTERN_DECL_NODE(Out);
};

struct LSTM : public PatternBase {
  LSTM(PDPattern* pattern, const std::string& name_scope)
      : PatternBase(pattern, name_scope, "lstm") {}

  PDNode* operator()(PDNode* x);

  // Operators
  PATTERN_DECL_NODE(lstm);

  // Inputs
  PATTERN_DECL_NODE(Input);
  PATTERN_DECL_NODE(H0);
  PATTERN_DECL_NODE(C0);
  PATTERN_DECL_NODE(Weight);
  PATTERN_DECL_NODE(Bias);

  // Outputs
  PATTERN_DECL_NODE(Hidden);
  PATTERN_DECL_NODE(Cell);
  PATTERN_DECL_NODE(BatchGate);
  PATTERN_DECL_NODE(BatchCellPreAct);
};

struct GRU : public PatternBase {
  GRU(PDPattern* pattern, const std::string& name_scope)
      : PatternBase(pattern, name_scope, "gru") {}

  PDNode* operator()(PDNode* x);

  // Operators
  PATTERN_DECL_NODE(gru);

  // Inputs
  PATTERN_DECL_NODE(Bias);
  PATTERN_DECL_NODE(Weight);

  // Outputs
  PATTERN_DECL_NODE(BatchGate);
  PATTERN_DECL_NODE(BatchResetHiddenPrev);
  PATTERN_DECL_NODE(BatchHidden);
  PATTERN_DECL_NODE(Hidden);
};

// The following pattern is used to fuse batch_norm and act
// formula: act(bn(x))
// op: batch_norm + act
struct BatchNormAct : public PatternBase {
  BatchNormAct(PDPattern* pattern, const std::string& name_scope)
      : PatternBase(pattern, name_scope, "bn_act") {}

  PDNode* operator()(PDNode* x, std::unordered_set<std::string> acts);

  // declare operator node's name
  PATTERN_DECL_NODE(batch_norm);
  PATTERN_DECL_NODE(act);
  // declare variable node's name
  // BN inputs
  PATTERN_DECL_NODE(bn_scale);
  PATTERN_DECL_NODE(bn_bias);
  PATTERN_DECL_NODE(bn_variance);
  PATTERN_DECL_NODE(bn_mean);
  // BN outputs
  PATTERN_DECL_NODE(bn_mean_out);
  PATTERN_DECL_NODE(bn_variance_out);
  PATTERN_DECL_NODE(bn_saved_variance);
  PATTERN_DECL_NODE(bn_saved_mean);
  PATTERN_DECL_NODE(bn_reserve_space);
  PATTERN_DECL_NODE(bn_out);
  // ACT output
  PATTERN_DECL_NODE(act_out);
};

// the backward of act(bn(x))
// op: batch_norm_grad + act_grad
struct BatchNormActGrad : public PatternBase {
  BatchNormActGrad(PDPattern* pattern, const std::string& name_scope)
      : PatternBase(pattern, name_scope, "bn_act_grad") {}

  // act_grad: in["Out", "Out@GRAD"], out["X@GRAD"]
  // bn_grad: in["X", "Y@GRAD", "Scale", "Bias", "SavedMean", "SavedVariance",
  // "ReserveSpace"],
  // out["X@GRAD", "Scale@GRAD", "Bias@GRAD"]
  PDNode* operator()(PDNode* x, std::unordered_set<std::string> act_grad_types);

  // declare operator node's name
  PATTERN_DECL_NODE(act_grad);
  PATTERN_DECL_NODE(batch_norm_grad);
  // declare variable node's name
  PATTERN_DECL_NODE(act_out);
  PATTERN_DECL_NODE(d_itermediate_out);
  PATTERN_DECL_NODE(bn_x);
  PATTERN_DECL_NODE(bn_scale);
  PATTERN_DECL_NODE(bn_bias);
  PATTERN_DECL_NODE(bn_saved_mean);
  PATTERN_DECL_NODE(bn_saved_variance);
  PATTERN_DECL_NODE(bn_reserve_space);
  PATTERN_DECL_NODE(d_bn_x);
  PATTERN_DECL_NODE(d_bn_scale);
  PATTERN_DECL_NODE(d_bn_bias);
};

//
// \brief   Pattern looking for batch_norm and a directly following activation
// operator.
//
// \note    Currently only ReLU is supported as an activation function.
//          Formula: act(bn(x))
//          Op: batch_norm + act
struct BatchNormActOneDNN : public PatternBase {
  BatchNormActOneDNN(PDPattern* pattern, const std::string& name_scope)
      : PatternBase(pattern, name_scope, "bn_act_onednn") {}

  PDNode* operator()(const std::string& act_type);

  // declare operator node's name
  PATTERN_DECL_NODE(bn_in);
  PATTERN_DECL_NODE(batch_norm);
  PATTERN_DECL_NODE(act);
  PATTERN_DECL_NODE(bn_out);
  PATTERN_DECL_NODE(act_out);
};

// The following pattern is used to fuse batch_norm, elewise_add, and act
// formula: act(bn(x) + z)
// op: batch_norm + elewise_add + act
struct BatchNormAddAct : public PatternBase {
  BatchNormAddAct(PDPattern* pattern, const std::string& name_scope)
      : PatternBase(pattern, name_scope, "bn_add_act") {}

  PDNode* operator()(PDNode* x, std::unordered_set<std::string> acts);

  // declare operator node's name
  PATTERN_DECL_NODE(batch_norm);
  PATTERN_DECL_NODE(elewise_add);
  PATTERN_DECL_NODE(act);
  // declare variable node's name
  // BN inputs
  PATTERN_DECL_NODE(bn_scale);
  PATTERN_DECL_NODE(bn_bias);
  // BN outputs
  PATTERN_DECL_NODE(bn_mean_out);
  PATTERN_DECL_NODE(bn_variance_out);
  PATTERN_DECL_NODE(bn_saved_variance);
  PATTERN_DECL_NODE(bn_saved_mean);
  PATTERN_DECL_NODE(bn_reserve_space);
  PATTERN_DECL_NODE(bn_out);
  // Elewise_Add input
  PATTERN_DECL_NODE(elewise_add_in);
  // Elewise_Add output
  PATTERN_DECL_NODE(elewise_add_out);
  // ACT output
  PATTERN_DECL_NODE(act_out);
};

// the backward of act(bn(x) + z)
// op: batch_norm_grad + elewise_add_grad + act_grad
struct BatchNormAddActGrad : public PatternBase {
  BatchNormAddActGrad(PDPattern* pattern, const std::string& name_scope)
      : PatternBase(pattern, name_scope, "bn_add_act_grad") {}

  // act_grad: in["Out", "Out@GRAD"], out["X@GRAD"]
  // elewise_add_grad: in["Out@GRAD"], out["X@GRAD", "Y@GRAD"]
  // bn_grad: in["X", "Z", "Y@GRAD", "Scale", "Bias", "SavedMean",
  // "SavedVariance",
  // "ReserveSpace"],
  // out["X@GRAD", "Z@GRAD", "Scale@GRAD", "Bias@GRAD"]
  PDNode* operator()(PDNode* x, std::unordered_set<std::string> act_grad_types);

  // declare operator node's name
  PATTERN_DECL_NODE(act_grad);
  PATTERN_DECL_NODE(elewise_add_grad);
  PATTERN_DECL_NODE(batch_norm_grad);
  // declare variable node's name
  PATTERN_DECL_NODE(act_out);
  PATTERN_DECL_NODE(d_act_x);
  PATTERN_DECL_NODE(d_elewise_add_in);
  PATTERN_DECL_NODE(d_bn_out);
  PATTERN_DECL_NODE(bn_x);
  PATTERN_DECL_NODE(bn_scale);
  PATTERN_DECL_NODE(bn_bias);
  PATTERN_DECL_NODE(bn_saved_mean);
  PATTERN_DECL_NODE(bn_saved_variance);
  PATTERN_DECL_NODE(bn_reserve_space);
  PATTERN_DECL_NODE(d_bn_x);
  PATTERN_DECL_NODE(d_bn_scale);
  PATTERN_DECL_NODE(d_bn_bias);
};

// The following patterns are used to fuse elewise_add and act
// formula: act(ele_add(x, y))
// op: elementwise_add + act
// named nodes: elementwise_add, act
//              ele_x, ele_y, elewise_add_out, act_out
struct ElewiseAddAct : public PatternBase {
  ElewiseAddAct(PDPattern* pattern, const std::string& name_scope)
      : PatternBase(pattern, name_scope, "elewise_add_act") {}

  PDNode* operator()(PDNode* x, std::unordered_set<std::string> acts);

  // declare operator node's name
  PATTERN_DECL_NODE(ele_add);
  PATTERN_DECL_NODE(act);
  // declare variable node's name
  PATTERN_DECL_NODE(elewise_add_out);
  PATTERN_DECL_NODE(ele_y);
  PATTERN_DECL_NODE(act_out);
};

// formula: ele_add(x, act(y))
// op: elementwise_add + act
// named nodes: elementwise_add, act
//              act_in, act_out, ele_x, elewise_add_out
struct ActElewiseAdd : public PatternBase {
  ActElewiseAdd(PDPattern* pattern, const std::string& name_scope)
      : PatternBase(pattern, name_scope, "act_elewise_add") {}

  PDNode* operator()(PDNode* x, std::unordered_set<std::string> acts);

  // declare operator node's name
  PATTERN_DECL_NODE(act);
  PATTERN_DECL_NODE(ele_add);
  // declare variable node's name
  PATTERN_DECL_NODE(act_out);
  PATTERN_DECL_NODE(ele_x);
  PATTERN_DECL_NODE(elewise_add_out);
};

// the backward of act(ele_add(x, y))
// the act is inplace.
// op: elementwise_add_grad + act_grad
// named nodes: elementwise_add_grad, act_grad
//              act_out, act_out_g, ele_y, d_itermediate_out, d_ele_x, d_ele_y
struct ElewiseAddActInplaceGrad : public PatternBase {
  ElewiseAddActInplaceGrad(PDPattern* pattern, const std::string& name_scope)
      : PatternBase(pattern, name_scope, "elewise_add_act_grad1") {}

  // act_grad: in["Out", "Out@GRAD"], out["X@GRAD"]
  // ele_add_grad: in["Y", "Out@GRAD"], out["X@GRAD", "Y@GRAD"]
  PDNode* operator()(PDNode* x, std::unordered_set<std::string> acts);

  // declare operator node's name
  PATTERN_DECL_NODE(act_grad);
  PATTERN_DECL_NODE(ele_add_grad);
  // declare variable node's name
  PATTERN_DECL_NODE(act_out);
  PATTERN_DECL_NODE(d_itermediate_out);
  PATTERN_DECL_NODE(d_ele_x);
  PATTERN_DECL_NODE(d_ele_y);
  PATTERN_DECL_NODE(ele_y);
};

// Conv with Elementwise_add as bias
// op: conv + elementwise_add
// named nodes:
// conv_input, conv_weight,
// conv_out, conv,
// eltwise_bias, eltwise_out,
// elementwise_add
struct ConvBias : public PatternBase {
  ConvBias(PDPattern* pattern, const std::string& name_scope)
      : PatternBase(pattern, name_scope, "conv_bias") {}
  PDNode* operator()(PDNode* conv_input, std::string conv_type = "conv2d");
  // declare operator node's name
  PATTERN_DECL_NODE(conv);
  PATTERN_DECL_NODE(eltwise);
  // declare variable node's name
  PATTERN_DECL_NODE(conv_weight);
  PATTERN_DECL_NODE(conv_out);
  PATTERN_DECL_NODE(eltwise_bias);
  PATTERN_DECL_NODE(eltwise_out);
};

// Convolution op
// Forward pass for convolution.
// conv_input, conv_bias and conv_filter are inputs.
// conv_output is a result of the operator.
// residual_data is data used by skip connection.
// If residual connection fusion is on, the formula is:
// conv_output = conv_op(conv_filter, conv_input, conv_bias)
//             + conv_residual_data
// If the fusion is off, conv_residual_data is not added.
struct Conv : public PatternBase {
  Conv(PDPattern* pattern, const std::string& name_scope)
      : PatternBase(pattern, name_scope, "convolution") {}

  PDNode* operator()();

  PATTERN_DECL_NODE(conv_op);
  PATTERN_DECL_NODE(conv_input);
  PATTERN_DECL_NODE(conv_filter);
  PATTERN_DECL_NODE(conv_residual_data);
  PATTERN_DECL_NODE(conv_output);
};

// Convolution op with residual data
struct ConvResidual : public PatternBase {
  ConvResidual(PDPattern* pattern, const std::string& name_scope)
      : PatternBase(pattern, name_scope, "conv_residual") {}

  PDNode* operator()(bool with_residual_data);

  PATTERN_DECL_NODE(conv_op);
  PATTERN_DECL_NODE(conv_input);
  PATTERN_DECL_NODE(conv_filter);
  PATTERN_DECL_NODE(conv_residual_data);
  PATTERN_DECL_NODE(conv_output);
};

// Pool op
// Forward pass for pooling.
// pool_input is the input.
// pool_output is a result of the operator.
struct Pool : public PatternBase {
  Pool(PDPattern* pattern, const std::string& name_scope)
      : PatternBase(pattern, name_scope, "pooling") {}

  PDNode* operator()();

  PATTERN_DECL_NODE(pool_op);
  PATTERN_DECL_NODE(pool_input);
  PATTERN_DECL_NODE(pool_output);
};

// ElementwiseAdd used in residual connections.
// y_var is used and convolution output.
// The operator is removed, when residual
// connection fusion is on.
struct ElementwiseAdd : public PatternBase {
  ElementwiseAdd(PDPattern* pattern, const std::string& name_scope)
      : PatternBase(pattern, name_scope, "elementwise_add") {}

  PDNode* operator()(PDNode* x_var, PDNode* y_var);

  PATTERN_DECL_NODE(elementwise_add_op);
  PATTERN_DECL_NODE(elementwise_add_x);
  PATTERN_DECL_NODE(elementwise_add_y);
  PATTERN_DECL_NODE(elementwise_add_out);
};

// Transpose op
// Forward pass for transpose.
// transpose_out is a result of the operator.
struct Transpose : public PatternBase {
  Transpose(PDPattern* pattern, const std::string& name_scope)
      : PatternBase(pattern, name_scope, "transpose2") {}

  PDNode* operator()();
  PATTERN_DECL_NODE(prev_op);
  PATTERN_DECL_NODE(transpose_in);
  PATTERN_DECL_NODE(transpose_op);
  PATTERN_DECL_NODE(transpose_out);
  PATTERN_DECL_NODE(next_op);
};

// Reshape op
// Forward pass for reshape.
// reshape_out is a result of the operator.
struct Reshape : public PatternBase {
  Reshape(PDPattern* pattern, const std::string& name_scope)
      : PatternBase(pattern, name_scope, "reshape2") {}

  PDNode* operator()();
  PATTERN_DECL_NODE(prev_op);
  PATTERN_DECL_NODE(reshape_in);
  PATTERN_DECL_NODE(reshape_op);
  PATTERN_DECL_NODE(reshape_out);
  PATTERN_DECL_NODE(next_op);
};

// Matmul op
// Forward pass for matmul.
struct Matmul : public PatternBase {
  Matmul(PDPattern* pattern, const std::string& name_scope)
      : PatternBase(pattern, name_scope, "matmul") {}

  PDNode* operator()();
  PATTERN_DECL_NODE(matmul_in_x);
  PATTERN_DECL_NODE(matmul_in_y);
  PATTERN_DECL_NODE(matmul_op);
  PATTERN_DECL_NODE(matmul_out);
};

// Squeeze2 + Matmul
// Forward pass.
struct Squeeze2Matmul : public PatternBase {
  Squeeze2Matmul(PDPattern* pattern, const std::string& name_scope)
      : PatternBase(pattern, name_scope, "squeeze2_matmul") {}

  PDNode* operator()();
  PATTERN_DECL_NODE(squeeze2_in_x);
  PATTERN_DECL_NODE(squeeze2_op);
  PATTERN_DECL_NODE(matmul_in_x);
  PATTERN_DECL_NODE(matmul_in_y);
  PATTERN_DECL_NODE(matmul_op);
  PATTERN_DECL_NODE(matmul_out);
};

// Reshape2 + Matmul
// Forward pass.
struct Reshape2Matmul : public PatternBase {
  Reshape2Matmul(PDPattern* pattern, const std::string& name_scope)
      : PatternBase(pattern, name_scope, "reshape2_matmul") {}

  PDNode* operator()();
  PATTERN_DECL_NODE(reshape2_in_x);
  PATTERN_DECL_NODE(reshape2_op);
  PATTERN_DECL_NODE(matmul_in_x);
  PATTERN_DECL_NODE(matmul_in_y);
  PATTERN_DECL_NODE(matmul_op);
  PATTERN_DECL_NODE(matmul_out);
};

// Forward pass for two input ops and matmul op.
// matmul_out is a result of the operator.
struct MatmulWithInputOps : public PatternBase {
  MatmulWithInputOps(PDPattern* pattern, const std::string& name_scope)
      : PatternBase(pattern, name_scope, "matmul_with_input_ops") {}

  PDNode* operator()();
  PATTERN_DECL_NODE(prev_op_x);
  PATTERN_DECL_NODE(prev_op_y);
  PATTERN_DECL_NODE(matmul_in_x);
  PATTERN_DECL_NODE(matmul_in_y);
  PATTERN_DECL_NODE(matmul_op);
  PATTERN_DECL_NODE(matmul_out);
};

// Flatten2 + Matmul
// Forward pass.
struct Flatten2Matmul : public PatternBase {
  Flatten2Matmul(PDPattern* pattern, const std::string& name_scope)
      : PatternBase(pattern, name_scope, "flatten2_matmul") {}

  PDNode* operator()();
  PATTERN_DECL_NODE(flatten2_in_x);
  PATTERN_DECL_NODE(flatten2_op);
  PATTERN_DECL_NODE(matmul_in_x);
  PATTERN_DECL_NODE(matmul_in_y);
  PATTERN_DECL_NODE(matmul_op);
  PATTERN_DECL_NODE(matmul_out);
};

// Concat op
// Forward pass for concat.
// concat_out is a result of the operator.
struct Concat : public PatternBase {
  Concat(PDPattern* pattern, const std::string& name_scope)
      : PatternBase(pattern, name_scope, "concat") {}

  PDNode* operator()();

  PATTERN_DECL_NODE(concat_op);
  PATTERN_DECL_NODE(concat_out);
};

// Concat + ReLU
// named nodes:
// concat_op, concat_out, relu_op, relu_out
struct ConcatReLU : public PatternBase {
  ConcatReLU(PDPattern* pattern, const std::string& name_scope)
      : PatternBase(pattern, name_scope, "concat_relu") {}

  PDNode* operator()();

  PATTERN_DECL_NODE(concat_op);
  PATTERN_DECL_NODE(concat_out);
  PATTERN_DECL_NODE(relu_op);
  PATTERN_DECL_NODE(relu_out);
};

// Conv + Concat + ReLU
// named nodes:
// conv_op, conv_out
// concat_op, concat_out, relu_op, relu_out
struct ConvConcatReLU : public PatternBase {
  ConvConcatReLU(PDPattern* pattern, const std::string& name_scope)
      : PatternBase(pattern, name_scope, "conv_concat_relu") {}

  PDNode* operator()();

  PATTERN_DECL_NODE(conv_op);
  PATTERN_DECL_NODE(conv_out);
  PATTERN_DECL_NODE(concat_op);
  PATTERN_DECL_NODE(concat_out);
  PATTERN_DECL_NODE(relu_op);
  PATTERN_DECL_NODE(relu_out);
};

// Op + Requant
// named nodes:
// any_op, any_out
// requant_op, requant_out
struct OpRequant : public PatternBase {
  OpRequant(PDPattern* pattern, const std::string& name_scope)
      : PatternBase(pattern, name_scope, "op_requant") {}

  PDNode* operator()();

  PATTERN_DECL_NODE(any_op);
  PATTERN_DECL_NODE(requant_in);
  PATTERN_DECL_NODE(requant_op);
  PATTERN_DECL_NODE(requant_out);
};

// Requant + Op
// named nodes:
// requant_in, requant_op,
// requant_out, any_op
struct RequantOp : public PatternBase {
  RequantOp(PDPattern* pattern, const std::string& name_scope)
      : PatternBase(pattern, name_scope, "requant_op") {}

  PDNode* operator()();

  PATTERN_DECL_NODE(any_op);
  PATTERN_DECL_NODE(requant_in);
  PATTERN_DECL_NODE(requant_op);
  PATTERN_DECL_NODE(requant_out);
};

// Op + Dequant
// named nodes:
// any_op, dequant_in
// dequant_op, dequant_out
struct OpDequant : public PatternBase {
  OpDequant(PDPattern* pattern, const std::string& name_scope)
      : PatternBase(pattern, name_scope, "op_dequant") {}

  PDNode* operator()();

  PATTERN_DECL_NODE(any_op);
  PATTERN_DECL_NODE(dequant_in);
  PATTERN_DECL_NODE(dequant_op);
  PATTERN_DECL_NODE(dequant_out);
};

// Dequantize + Scale
struct DequantScale : public PatternBase {
  DequantScale(PDPattern* pattern, const std::string& name_scope)
      : PatternBase(pattern, name_scope, "dequant_scale") {}

  PDNode* operator()();

  PATTERN_DECL_NODE(dequant_op);
  PATTERN_DECL_NODE(dequant_out);
  PATTERN_DECL_NODE(scale_op);
  PATTERN_DECL_NODE(scale_out);
};

// Scale + Quantize
struct ScaleQuant : public PatternBase {
  ScaleQuant(PDPattern* pattern, const std::string& name_scope)
      : PatternBase(pattern, name_scope, "scale_quant") {}

  PDNode* operator()();

  PATTERN_DECL_NODE(scale_in);
  PATTERN_DECL_NODE(scale_op);
  PATTERN_DECL_NODE(quant_in);
  PATTERN_DECL_NODE(quant_op);
};

// Quantize + Conv2d
struct QuantConv : public PatternBase {
  QuantConv(PDPattern* pattern, const std::string& name_scope)
      : PatternBase(pattern, name_scope, "quant_conv") {}

  PDNode* operator()();

  PATTERN_DECL_NODE(quant_in);
  PATTERN_DECL_NODE(quant_op);
  PATTERN_DECL_NODE(conv_in);
  PATTERN_DECL_NODE(conv_op);
};

// Scale + Matmul
struct ScaleMatmul : public PatternBase {
  ScaleMatmul(PDPattern* pattern, const std::string& name_scope)
      : PatternBase(pattern, name_scope, "scale_matmul") {}

  PDNode* operator()();
  PATTERN_DECL_NODE(scale_in);
  PATTERN_DECL_NODE(scale_op);
  PATTERN_DECL_NODE(scale_out);
  PATTERN_DECL_NODE(matmul_op);
};

// PriorBox operator
// operator: prior_box_op
// inputs: prior_box_input, prior_box_image
// outputs: prior_box_boxes, prior_box_variances
struct PriorBox : public PatternBase {
  PriorBox(PDPattern* pattern, const std::string& name_scope)
      : PatternBase(pattern, name_scope, "PriorBox") {}

  PDNode* operator()();

  PATTERN_DECL_NODE(prior_box_op);
  PATTERN_DECL_NODE(prior_box_input);
  PATTERN_DECL_NODE(prior_box_image);
  PATTERN_DECL_NODE(prior_box_boxes);
  PATTERN_DECL_NODE(prior_box_variances);
};

// Conv + ElementwiseAdd + an activation
// This pattern can futher fuse the conv related ops after the conv+bn fusion.
struct ConvElementwiseaddAct : public PatternBase {
  ConvElementwiseaddAct(PDPattern* pattern, const std::string& name_scope)
      : PatternBase(pattern, name_scope, "conv_elementwiseadd_act") {}

  PDNode* operator()(PDNode* conv_in);

  PATTERN_DECL_NODE(conv_op);
  PATTERN_DECL_NODE(conv_out);
  PATTERN_DECL_NODE(conv_filter);

  PATTERN_DECL_NODE(elementwise_add_op);
  PATTERN_DECL_NODE(elementwise_add_in_y);  // input
  PATTERN_DECL_NODE(elementwise_add_out);

  PATTERN_DECL_NODE(act_op);
  PATTERN_DECL_NODE(act_out);
};

// Conv + ElementwiseAdd + ElementwiseAdd + Activation
struct ConvElementwiseadd2Act : public PatternBase {
  ConvElementwiseadd2Act(PDPattern* pattern, const std::string& name_scope)
      : PatternBase(pattern, name_scope,
                    "conv_elementwiseadd2_elementwiseadd_act") {}

  PDNode* operator()(PDNode* conv_in);

  PATTERN_DECL_NODE(conv_op);
  PATTERN_DECL_NODE(conv_filter);
  PATTERN_DECL_NODE(conv_out);

  PATTERN_DECL_NODE(elementwise_add_op);
  PATTERN_DECL_NODE(elementwise_add_in_y);  // input
  PATTERN_DECL_NODE(elementwise_add_out);

  PATTERN_DECL_NODE(elementwise_add_op_1);
  PATTERN_DECL_NODE(elementwise_add_in_y_1);  // input
  PATTERN_DECL_NODE(elementwise_add_out_1);

  PATTERN_DECL_NODE(act_op);
  PATTERN_DECL_NODE(act_out);
};

// Conv + ElementwiseAdd
// This pattern should be used after ConvElementwiseadd2Act or
// ConvElementwiseadd pass
struct ConvElementwiseadd : public PatternBase {
  ConvElementwiseadd(PDPattern* pattern, const std::string& name_scope)
      : PatternBase(pattern, name_scope, "conv_elementwiseadd") {}

  PDNode* operator()(PDNode* conv_in);

  PATTERN_DECL_NODE(conv_op);
  PATTERN_DECL_NODE(conv_out);
  PATTERN_DECL_NODE(conv_filter);

  PATTERN_DECL_NODE(elementwise_add_op);
  PATTERN_DECL_NODE(elementwise_add_in_y);
  PATTERN_DECL_NODE(elementwise_add_out);
};

// Conv with affine_channel
// op: conv + (elementwise_add +) affine_channel
// named nodes:
// conv_weight, conv_out, conv,
// ac_x, ac_scale, ac_bias
// affine_channel, ac_out
struct ConvAffineChannel : public PatternBase {
  ConvAffineChannel(PDPattern* pattern, const std::string& name_scope)
      : PatternBase(pattern, name_scope, "conv_affine_channel") {}

  PDNode* operator()(PDNode* conv_input, bool with_eltwise_add);

  // declare operator node's name
  PATTERN_DECL_NODE(conv);
  PATTERN_DECL_NODE(affine_channel);
  PATTERN_DECL_NODE(eltwise);  // ELEMENTWISE_ADD
  // CONV inputs
  PATTERN_DECL_NODE(conv_weight);  // Filter
  // CONV outputs
  PATTERN_DECL_NODE(conv_out);  // tmp
  // ELTWISE inputs
  PATTERN_DECL_NODE(eltwise_y_in);
  // ELTWISE outputs
  PATTERN_DECL_NODE(eltwise_out);  // tmp

  // AC(Affine_Channel) inputs
  PATTERN_DECL_NODE(ac_scale);
  PATTERN_DECL_NODE(ac_bias);
  // AC outputs
  PATTERN_DECL_NODE(ac_out);  // Out
};

// Dequantize + Quantize + anyOP
// This pattern is used for squashing the dequantize-quantize pairs.
struct DequantQuantAny : public PatternBase {
  DequantQuantAny(PDPattern* pattern, const std::string& name_scope)
      : PatternBase(pattern, name_scope, "dequant_quant_any") {}
  PDNode* operator()();

  PATTERN_DECL_NODE(dequant_in);
  PATTERN_DECL_NODE(dequant_op);
  PATTERN_DECL_NODE(dequant_out);
  PATTERN_DECL_NODE(quant_op);
  PATTERN_DECL_NODE(quant_out);
  PATTERN_DECL_NODE(next_op);
};

// Dequantize + anyOP
// This quantize is used for getting number of ops the Dequantize's
// output is an input to.
struct DequantAny : public PatternBase {
  DequantAny(PDPattern* pattern, const std::string& name_scope)
      : PatternBase(pattern, name_scope, "dequant_any") {}
  PDNode* operator()();

  PATTERN_DECL_NODE(dequant_op);
  PATTERN_DECL_NODE(dequant_out);
  PATTERN_DECL_NODE(next_op);
};

// anyOp + more then one quantize op
// This pattern is used for squashing multiple quantize with the same scale.
struct MultipleQuantize : public PatternBase {
  MultipleQuantize(PDPattern* pattern, const std::string& name_scope)
      : PatternBase(pattern, name_scope, "multiple_quantize") {}
  PDNode* operator()();

  PATTERN_DECL_NODE(prev_out);
};

struct QuantizePlacement : public PatternBase {
  QuantizePlacement(PDPattern* pattern, const std::string& name_scope)
      : PatternBase(pattern, name_scope, "quantize_placement") {}
  PDNode* operator()(
      const std::unordered_set<std::string>& quantize_enabled_op_types);

  PATTERN_DECL_NODE(op);
};

struct Bfloat16Placement : public PatternBase {
  Bfloat16Placement(PDPattern* pattern, const std::string& name_scope)
      : PatternBase(pattern, name_scope, "bfloat16_placement") {}
  PDNode* operator()(
      const std::unordered_set<std::string>& bfloat16_enabled_op_types);

  PATTERN_DECL_NODE(op);
};

struct OrphanedBfloat16 : public PatternBase {
  OrphanedBfloat16(PDPattern* pattern, const std::string& name_scope)
      : PatternBase(pattern, name_scope, "orphaned_bfloat16") {}
  PDNode* operator()();

  PATTERN_DECL_NODE(prev_op);
  PATTERN_DECL_NODE(prev_out);
  PATTERN_DECL_NODE(op);
  PATTERN_DECL_NODE(op_out);
  PATTERN_DECL_NODE(next_op);
};

struct LastBfloat16Ops : public PatternBase {
  LastBfloat16Ops(PDPattern* pattern, const std::string& name_scope)
      : PatternBase(pattern, name_scope, "last_bfloat16_ops") {}
  PDNode* operator()();

  PATTERN_DECL_NODE(op);
  PATTERN_DECL_NODE(op_out);
};

struct FirstBfloat16Ops : public PatternBase {
  FirstBfloat16Ops(PDPattern* pattern, const std::string& name_scope)
      : PatternBase(pattern, name_scope, "first_bfloat16_ops") {}
  PDNode* operator()();

  PATTERN_DECL_NODE(op_in);
  PATTERN_DECL_NODE(op);
};

struct DuplicatedInputs : public PatternBase {
  DuplicatedInputs(PDPattern* pattern, const std::string& name_scope)
      : PatternBase(pattern, name_scope, "many_inputs_op") {}

  PDNode* operator()();

  PATTERN_DECL_NODE(op);
};

// Pattern used for enforcing inplace computation for in-place computation
// supporting DNNL ops. softmax, batch_norm and layer_norm
struct MKLDNNInPlace : public PatternBase {
  MKLDNNInPlace(PDPattern* pattern, const std::string& name_scope)
      : PatternBase(pattern, name_scope, "mkldnn_inplace") {}
  PDNode* operator()();

  // MKL-DNN's in-place ops: BatchNorm, Softmax, Elementwise_add
  PATTERN_DECL_NODE(inplace_to_be_op);
  PATTERN_DECL_NODE(inplace_to_be_op_in);
  PATTERN_DECL_NODE(inplace_to_be_op_out);
  PATTERN_DECL_NODE(next_op);
  PATTERN_DECL_NODE(next_op_out);
};

struct TransposeFlattenConcat : public PatternBase {
  TransposeFlattenConcat(PDPattern* pattern, const std::string& name_scope)
      : PatternBase(pattern, name_scope, "transpose_flatten_concat") {}

  PDNode* operator()(std::vector<PDNode*> conv_inputs, int times);

  std::string GetNodeName(const std::string& op_type) {
    return PDNodeName(name_scope_, repr_, id_, op_type);
  }

  PDNode* GetPDNode(const std::string& op_type) {
    return pattern->RetrieveNode(GetNodeName(op_type));
  }
};

struct DeleteQuantOpFuse : public PatternBase {
  DeleteQuantOpFuse(PDPattern* pattern, const std::string& name_scope)
      : PatternBase(pattern, name_scope, "delete_quant_fuse") {}

  void operator()(PDNode* input_act_node, const std::string& quant_type);

  std::string GetNodeName(const std::string& op_type) {
    return PDNodeName(name_scope_, repr_, id_, op_type);
  }

  PDNode* GetPDNode(const std::string& op_type) {
    return pattern->RetrieveNode(GetNodeName(op_type));
  }
};

struct DequantOpFuse : public PatternBase {
  DequantOpFuse(PDPattern* pattern, const std::string& name_scope)
      : PatternBase(pattern, name_scope, "dequant_fuse") {}

  void operator()(PDNode* quant_op_input, const std::string& quantized_op_type,
                  const std::string& dequant_type,
                  const std::string& weight_name);

  std::string GetNodeName(const std::string& op_type) {
    return PDNodeName(name_scope_, repr_, id_, op_type);
  }

  PDNode* GetPDNode(const std::string& op_type) {
    return pattern->RetrieveNode(GetNodeName(op_type));
  }
};

struct ShuffleChannelPattern : public PatternBase {
  ShuffleChannelPattern(PDPattern* pattern, const std::string& name_scope)
      : PatternBase(pattern, name_scope, "shufflechannel_pattern") {}

  void operator()(PDNode* reshape1_in);

  PATTERN_DECL_NODE(reshape1_op);
  PATTERN_DECL_NODE(reshape1_out);

  PATTERN_DECL_NODE(transpose_op);
  PATTERN_DECL_NODE(transpose_out);
  PATTERN_DECL_NODE(reshape2_op);
  PATTERN_DECL_NODE(reshape2_out);
};

struct DeleteDropoutOpPattern : public PatternBase {
  DeleteDropoutOpPattern(PDPattern* pattern, const std::string& name_scope)
      : PatternBase(pattern, name_scope, "delete_dropout_op_pattern") {}

  void operator()();

  PATTERN_DECL_NODE(any_op_out);
  PATTERN_DECL_NODE(dropout_op);
  PATTERN_DECL_NODE(dropout_op_out);
  PATTERN_DECL_NODE(dropout_op_outmask);
  PATTERN_DECL_NODE(any_op2);
};

struct DeleteQuantDequantOpPattern : public PatternBase {
  DeleteQuantDequantOpPattern(PDPattern* pattern, const std::string& name_scope)
      : PatternBase(pattern, name_scope, "delete_quantdequant_op_pattern") {}

  void operator()();

  PATTERN_DECL_NODE(any_op_out);
  PATTERN_DECL_NODE(quant_dequant_op_inscale);
  PATTERN_DECL_NODE(quant_dequant_op);
  PATTERN_DECL_NODE(quant_dequant_op_outscale);
  PATTERN_DECL_NODE(quant_dequant_op_out);
  PATTERN_DECL_NODE(any_op2);
};

struct DeleteQuantDequantFilterOpPattern : public PatternBase {
  DeleteQuantDequantFilterOpPattern(PDPattern* pattern,
                                    const std::string& name_scope)
      : PatternBase(pattern, name_scope,
                    "delete_quantdequant_filter_op_pattern") {}

  void operator()();

  PATTERN_DECL_NODE(quant_dequant_op_x);
  PATTERN_DECL_NODE(quant_dequant_op);
  PATTERN_DECL_NODE(quant_dequant_op_outscale);
  PATTERN_DECL_NODE(quant_dequant_op_out);
  PATTERN_DECL_NODE(any_op2);
};

// Reshape + Transpose + Matmul
// named nodes:
// reshape_op, reshape_out, reshape_xshape,
// transpose_op, transpose_out, transpose_xshape,
// matmul_op, matmul_out
struct ReshapeTransposeMatmulPattern : public PatternBase {
  ReshapeTransposeMatmulPattern(PDPattern* pattern,
                                const std::string& name_scope)
      : PatternBase(pattern, name_scope, "reshape_transpose_matmul") {}

  PDNode* operator()(bool with_reshape_xshape, bool with_transpose_xshape);

  PATTERN_DECL_NODE(reshape_in);
  PATTERN_DECL_NODE(reshape_op);
  PATTERN_DECL_NODE(reshape_out);
  PATTERN_DECL_NODE(reshape_xshape);
  PATTERN_DECL_NODE(transpose_op);
  PATTERN_DECL_NODE(transpose_out);
  PATTERN_DECL_NODE(transpose_xshape);
  PATTERN_DECL_NODE(matmul_op);
  PATTERN_DECL_NODE(matmul_out);
};

// Matmul + Transpose + Reshape
struct MatmulTransposeReshapePattern : public PatternBase {
  MatmulTransposeReshapePattern(PDPattern* pattern,
                                const std::string& name_scope)
      : PatternBase(pattern, name_scope, "matmul_transpose_reshape") {}

  PDNode* operator()();

  PATTERN_DECL_NODE(matmul_op);
  PATTERN_DECL_NODE(matmul_out);
  PATTERN_DECL_NODE(transpose_op);
  PATTERN_DECL_NODE(transpose_out);
  PATTERN_DECL_NODE(transpose_out_xshape);
  PATTERN_DECL_NODE(reshape_op);
  PATTERN_DECL_NODE(reshape_out);
  PATTERN_DECL_NODE(reshape_out_xshape);
};

// fusion_gru op
// Forward pass for fusion_gru.
// fusion_gru out is a result of the operator.
struct FusionGru : public PatternBase {
  FusionGru(PDPattern* pattern, const std::string& name_scope)
      : PatternBase(pattern, name_scope, "fusion_gru") {}

  PDNode* operator()();
  PATTERN_DECL_NODE(op);
  PATTERN_DECL_NODE(x);
  PATTERN_DECL_NODE(weight_h);
  PATTERN_DECL_NODE(weight_x);
  PATTERN_DECL_NODE(out);
};

// fusion_lstm op
// Forward pass for fusion_lstm.
// fusion_lstm out is a result of the operator.
struct FusionLSTM : public PatternBase {
  FusionLSTM(PDPattern* pattern, const std::string& name_scope)
      : PatternBase(pattern, name_scope, "fusion_lstm") {}
  // TODO(lidanqing): Is it enough to detect fusion_lstm with these things
  PDNode* operator()();

  // declare op
  PATTERN_DECL_NODE(op);

  // declate inputs
  PATTERN_DECL_NODE(x);
  PATTERN_DECL_NODE(weight_h);
  PATTERN_DECL_NODE(weight_x);

  // decalre outputs
  PATTERN_DECL_NODE(hidden);
  PATTERN_DECL_NODE(cell);
};

// two concatenated fusion_gru ops
// Forward pass for fusion of two concatenated fusion_gru ops.
// concat_out is a result of the operator().
struct TwoFusionGruConcat : public PatternBase {
  TwoFusionGruConcat(PDPattern* pattern, const std::string& name_scope)
      : PatternBase(pattern, name_scope, "bi_fusion_gru") {}

  PDNode* operator()();
  PATTERN_DECL_NODE(x);
  PATTERN_DECL_NODE(gru1);
  PATTERN_DECL_NODE(gru2);
  PATTERN_DECL_NODE(wh1);
  PATTERN_DECL_NODE(wh2);
  PATTERN_DECL_NODE(wx1);
  PATTERN_DECL_NODE(wx2);
  PATTERN_DECL_NODE(b1);
  PATTERN_DECL_NODE(b2);
  PATTERN_DECL_NODE(h1);
  PATTERN_DECL_NODE(h2);
  PATTERN_DECL_NODE(concat);
  PATTERN_DECL_NODE(out);
};

// two subsequent bi_fusion_gru ops
// Forward pass for fusion of two subsequent fusion_gru ops.
// Hidden of the last fusion_gru op is a result of the operator().
struct MultiGruSeq : public PatternBase {
  MultiGruSeq(PDPattern* pattern, const std::string& name_scope)
      : PatternBase(pattern, name_scope, "multi_gru_seq") {}

  PDNode* operator()();
  PATTERN_DECL_NODE(x);
  PATTERN_DECL_NODE(gru1);
  PATTERN_DECL_NODE(wx11);
  PATTERN_DECL_NODE(wx12);
  PATTERN_DECL_NODE(wh11);
  PATTERN_DECL_NODE(wh12);
  PATTERN_DECL_NODE(b11);
  PATTERN_DECL_NODE(b12);
  PATTERN_DECL_NODE(h1);
  PATTERN_DECL_NODE(gru2);
  PATTERN_DECL_NODE(wx21);
  PATTERN_DECL_NODE(wx22);
  PATTERN_DECL_NODE(wh21);
  PATTERN_DECL_NODE(wh22);
  PATTERN_DECL_NODE(b21);
  PATTERN_DECL_NODE(b22);
  PATTERN_DECL_NODE(h2);
};

// multi_gru op
// Quantization pass for multi_gru op.
// Hidden of the multi_gru op is a result of the operator().
struct MultiGru : public PatternBase {
  MultiGru(PDPattern* pattern, const std::string& name_scope)
      : PatternBase(pattern, name_scope, "multi_gru") {}

  PDNode* operator()();
  PATTERN_DECL_NODE(x);
  PATTERN_DECL_NODE(gru);
  PATTERN_DECL_NODE(wx);
  PATTERN_DECL_NODE(wh);
  PATTERN_DECL_NODE(h);
};

//
// \brief   Pattern looking for subgraph representing layer normalization
//          operation.
//
struct LayerNorm : public PatternBase {
  LayerNorm(PDPattern* pattern, const std::string& name_scope)
      : PatternBase(pattern, name_scope, "layer_norm") {}

  PDNode* operator()();

  PATTERN_DECL_NODE(x);
  PATTERN_DECL_NODE(x_mean);
  PATTERN_DECL_NODE(x_mean_out);
  PATTERN_DECL_NODE(x_sub_mean);
  PATTERN_DECL_NODE(x_sub_mean_out);
  PATTERN_DECL_NODE(sqr_pow);
  PATTERN_DECL_NODE(x_sub_mean_sqr);
  PATTERN_DECL_NODE(x_sub_mean_sqr_out);
  PATTERN_DECL_NODE(std_dev);
  PATTERN_DECL_NODE(std_dev_out);
  PATTERN_DECL_NODE(eps);
  PATTERN_DECL_NODE(std_dev_eps);
  PATTERN_DECL_NODE(std_dev_eps_out);
  PATTERN_DECL_NODE(std_dev_eps_sqrt);
  PATTERN_DECL_NODE(std_dev_eps_sqrt_out);
  PATTERN_DECL_NODE(division);
  PATTERN_DECL_NODE(division_out);
  PATTERN_DECL_NODE(gamma);
  PATTERN_DECL_NODE(scale);
  PATTERN_DECL_NODE(scale_out);
  PATTERN_DECL_NODE(beta);
  PATTERN_DECL_NODE(shift);
  PATTERN_DECL_NODE(shift_out);
};

}  // namespace patterns

// Link two ir::Nodes from each other.
#define IR_NODE_LINK_TO(a, b) \
  a->outputs.push_back(b);    \
  b->inputs.push_back(a);

// Set the out_var as the output of the op
#define IR_OP_VAR_LINK(op, out_var) \
  op->outputs.push_back(out_var);   \
  out_var->inputs.clear();          \
  out_var->inputs.push_back(op);

}  // namespace ir
}  // namespace framework
}  // namespace paddle
