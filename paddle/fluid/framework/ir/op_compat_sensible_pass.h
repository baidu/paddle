#include <vector>
#include "paddle/fluid/framework/ir/graph.h"
#include "paddle/fluid/framework/ir/graph_pattern_detector.h"
#include "paddle/fluid/framework/ir/pass.h"

namespace paddle {
namespace framework {
namespace ir {

class AttrCompat {
 public:
  AttrCompat(const std::string& attr_name) : attr_name_(attr_name) {}

  // @{ String-related methods
  //! Assert the attribute is an string in the `candidates` domain.
  AttrCompat& IsStringIn(const std::set<std::string>& candidates);
  //! Assert the attribute is a string and match a custom judging function.
  AttrCompat& IsStringMatch(std::function<bool(const std::string&)>);
  // @}

  //! Assert the attribute is an integer in the `candidates` domain.
  AttrCompat& IsIntIn(const std::set<std::string>& candidates);

  // @{ Number-releated methods
  //! Assert the attribute is a number and > `v`.
  template <typename T>
  AttrCompat& IsNumGT(T v);
  //! Assert the attribute is a number and >= `v`.
  template <typename T>
  AttrCompat& IsNumGE(T v);
  //! Assert the attribute is a number and < `v`.
  template <typename T>
  AttrCompat& IsNumLT(T v);
  //! Assert the attribute is a number and <= `v`.
  template <typename T>
  AttrCompat& IsNumLE(T v);
  //! Assert the attribute is a number and == `v`.
  template <typename T>
  AttrCompat& IsNumEQ(T v);
  //! Assert the attribute is a number and matches a customized judging
  //! function.
  template <typename T>
  AttrCompat& IsNumMatch(bool (*)(T v));
  // @}

  //! Assert the attribute is a boolean value equals `v`.
  AttrCompat& IsBoolEQ(bool v);

  //! Tell whether this attribute is left as default value.
  AttrCompat& IsLeftDefault();

 private:
  std::string attr_name_;
};

class InputOrOutputCompat {
 public:
  InputOrOutputCompat(bool is_input, const std::string& name)
      : is_input_(is_input), name_(name) {}

  InputOrOutputCompat& IsTensor();
  InputOrOutputCompat& IsTensorList();
  InputOrOutputCompat& IsOptional();
  InputOrOutputCompat& IsExists();

 private:
  bool is_input_{};
  std::string name_;
};

/**
 * OpCompat is a helper class to help define the compatible Op definition.
 *
 * Usage:
 *   OpCompat compat("FC");
 *   compat.AddAttr("in_num_col_dims").IsNumLE(1).End()
 *         .AddAttr("activation_type").IsStringIn({"tanh", "sigmoid"}).End()
 *         .AddInput("Input").IsTensor().IsExists().End()
 *         .AddInput("W").IsTensor().IsExists().End()
 *         .AddInput("Bias").IsTensor().IsOptional().End()
 *         .AddOutput("Out").IsTensor().IsExists().End()
 *
 * All the inference-aware Op defition is as above, all the other attributes not
 * contained in the definition should be set default value or it would be judged
 * incompatible.
 */
class OpCompat {
 public:
  AttrCompat& AddAttr(const std::string& attr_name);
  InputOrOutputCompat& AddInput();
  InputOrOutputCompat& AddOutput();

  //! Jump back to retrieve OpCompat instance.
  OpCompat& End();

  //! Judge whether an OpDesc match the defined Op compatibility.
  bool Judge(const OpDesc& op_desc);

 private:
  OpCompat(const std::string& op_name) : op_name_(op_name) {}
  friend OpCompat&& CreateOpCompat(const std::string& op_type);

 private:
  std::vector<AttrCompat> attr_compats_;
  std::vector<InputOrOutputCompat> io_compats_;

  std::string op_name_;
};

OpCompat&& CreateOpCompat(const std::string& op_type);

/**
 * OpCompatSensiblePass is a base class for all the passes thouse is sensitive
 * to Op update.
 * There are two methods to help tell the compability of an Op
 *   bool IsCompat(const GraphPatternDetector::subgraph_t& subgraph, Graph* g);
 *   bool IsCompat(const OpDesc& op_desc);
 *
 * One can register the related Op compabilities using
 *   void AddOpCompat(OpCompat&& judger);
 *
 * Most of the Passes are used for fusing ops, so we define a method for such
 * scenerios.
 *   void AccessSubgraph(const GraphPatternDetector::subgraph_t& subgraph,
 Graph* g);
 * It will check the Op compatibility automatically.
 * For other scenirios, one should call `IsCompat` by himself.
 *
 * A FC fuse pass example:
 * class FcFusePass : public OpCompatSensiblePass {
 *  public:
 *   FcFusePass() {
 *     // define Mul op compatiblity.
 *     AddOpCompat(CreateOpCompat("Mul")
 *        .AddInput("Input").IsTensor().IsExists().End()
 *        .AddAttr("in_num_col_dims").IsNumGE(1).End());
 *     AddOpCompat(CreateOpCompat("Add") ...);
 *     // There are multiple activation implemention.
 *     AddOpCompat(CreateOpCompat("Tanh") ...);
 *     AddOpCompat(CreateOpCompat("Sigmoid") ...);
 *   }
 *
 *   // override the subgraph access method
 *   virtual bool AccessSubgraphImpl(
 *   const GraphPatternDetector::subgraph_t& subgraph,
 *         Graph* g) override { ... }
 *
 *   // Call the AccessSubgraph method in main procedure of this Pass.
 * };
 */
class OpCompatSensiblePass : public Pass {
 public:
  //! Access the subgraph and pattern.
  void AccessSubgraph(const GraphPatternDetector::subgraph_t& subgraph,
                      Graph* g) {
    if (IsCompat(subgraph, g)) {
      AccessSubgraphImpl(subgraph, g);
    }
  }

 protected:
  /**
   * Developer should push the compatibility `teller` for each kind of Op in the
   * subgraph.
   * NOTE One should add all the related op compatiblity in the construct so
   * that all the following methods are valid.
   */
  void AddOpCompat(OpCompat&& judger);

  //! Modify the subgraph.
  virtual bool AccessSubgraphImpl(
      const GraphPatternDetector::subgraph_t& subgraph, Graph* g) const {}

  //! Tell the Op compability of a subgraph.
  bool IsCompat(const GraphPatternDetector::subgraph_t& subgraph,
                Graph* g) const {
    CHECK(!op_compat_judgers_.empty())
        << "At least one OpCompat instance should be added in the "
           "OpCompatSensiblePass.";
    // Check the all the ops in the subgraph are contained in the
    // op_compat.
    for (auto [pd_node, node] : subgraph) {
      if (!pd_node->IsOp()) continue;
      auto op_type = node->Op()->Name();
      if (pd_node->IsOp() && !op_compat_judgers_.count(op_type)) {
        return false;
      }

      auto& judger = *op_compat_judgers_.at(op_type);
      if (!judger.Judge(*node->Op())) {
        return false;
      }
    }

    return true;
  }

  //! Tell the op compatibility of a single Op.
  bool IsCompat(const OpDesc& op_desc) const {
    if (!op_compat_judgers_.count(op_desc.Name())) return false;

    return op_compat_judgers_.at(op_desc.Name())->Judge(op_desc);
  }

 private:
  std::map<std::string, std::unique_ptr<OpCompat>> op_compat_judgers_;
};

}  // namespace ir
}  // namespace framework
}  // namespace paddle
