/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/framework/ir/is_test_pass.h"

#include "glog/logging.h"

namespace paddle {
namespace framework {
namespace ir {

class Graph;

void IsTestPass::ApplyImpl(ir::Graph* graph) const {
  VLOG(3) << "Sets is_test attrbiute to true and if it is missing, inserts it "
             "for activations and pooling.";
  auto op_list = {"pool2d",      "sigmoid",      "logsigmoid",
                  "softshrink",  "exp",          "brelu",
                  "pow",         "leaky_relu",   "stanh",
                  "relu",        "tanh",         "tanh_shrink",
                  "sqrt",        "abs",          "ceil",
                  "elu",         "floor",        "cos",
                  "sin",         "round",        "reciprocal",
                  "hard_shrink", "hard_sigmoid", "relu6",
                  "soft_relu",   "swish",        "thresholded_relu",
                  "log",         "square",       "softplus",
                  "softsign",    "silu",         "mish"};
  for (const Node* n : graph->Nodes()) {
    if (n->IsOp()) {
      auto* op = n->Op();
      if (op->HasAttr("is_test") || op->HasProtoAttr("is_test")) {
        op->SetAttr("is_test", true);
      } else if (std::find(begin(op_list), end(op_list), op->Type()) !=
                 end(op_list)) {
        op->MutableAttrMap()->insert(
            std::pair<std::string, Attribute>("is_test", true));
      }
    }
  }
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(is_test_pass, paddle::framework::ir::IsTestPass);
