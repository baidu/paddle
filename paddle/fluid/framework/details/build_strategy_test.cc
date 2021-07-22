//   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "gtest/gtest.h"
#include "paddle/fluid/framework/details/build_strategy.h"
#include "paddle/fluid/framework/ir/graph.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/platform/device_context.h"

DECLARE_bool(convert_all_blocks);

namespace paddle {
namespace framework {

class NOP : public OperatorBase {
 public:
  NOP(const std::string &type, const VariableNameMap &inputs,
      const VariableNameMap &outputs, const AttributeMap &attrs)
      : OperatorBase(type, inputs, outputs, attrs) {}

 private:
  void RunImpl(const Scope &scope,
               const platform::Place &place) const override {}
};

class SumOpMaker : public OpProtoAndCheckerMaker {
 public:
  void Make() {
    AddInput("X", "").AsDuplicable();
    AddOutput("Out", "").AsDuplicable();
    AddComment("");
  }
};

class SumOpVarTypeInference : public VarTypeInference {
 public:
  void operator()(InferVarTypeContext *ctx) const override {
    auto default_var_type = proto::VarType::SELECTED_ROWS;

    if (ctx->InputTypeAnyOf("X", proto::VarType::LOD_TENSOR)) {
      default_var_type = proto::VarType::LOD_TENSOR;
    }

    ctx->SetOutputType("Out", default_var_type);
  }
};

class DummyOpMaker : public OpProtoAndCheckerMaker {
 public:
  void Make() {
    AddInput("X", "").AsDuplicable();
    AddOutput("Out", "").AsDuplicable();
    AddComment("");
  }
};

class DummyOpVarTypeInference : public VarTypeInference {
 public:
  void operator()(framework::InferVarTypeContext *ctx) const override {}
};

}  // namespace framework
}  // namespace paddle

REGISTER_OPERATOR(sum, paddle::framework::NOP, paddle::framework::SumOpMaker,
                  paddle::framework::SumOpVarTypeInference);
REGISTER_OPERATOR(dummy, paddle::framework::NOP, paddle::framework::SumOpMaker,
                  paddle::framework::SumOpVarTypeInference);
REGISTER_OPERATOR(sum_without_infer_var_type, paddle::framework::NOP,
                  paddle::framework::SumOpMaker);

namespace paddle {
namespace framework {
namespace details {

static std::vector<platform::Place> CreatePlaces(size_t num, bool use_cuda) {
  std::vector<platform::Place> result;
  result.reserve(num);
  for (size_t i = 0; i < num; ++i) {
    if (use_cuda) {
      result.emplace_back(platform::CUDAPlace(i));
    } else {
      result.emplace_back(platform::CPUPlace());
    }
  }
  return result;
}

void BuildStrategyApply(BuildStrategy *build_strategy, ir::Graph *graph) {
  std::string loss_name = "";
  Scope scope;
  std::vector<Scope *> scopes = {&scope};

  bool use_cuda = false;
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  use_cuda = true;
#endif
  auto places = CreatePlaces(1, use_cuda);
  auto device = platform::Place2DeviceType(places[0]);

#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
  platform::NCCLCommunicator ctxs;
#elif defined(PADDLE_WITH_XPU) && defined(PADDLE_WITH_XPU_BKCL)
  platform::BKCLCommunicator ctxs;
#endif

  build_strategy->Apply(graph, places, loss_name, scopes, 1,
#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
                        device, &ctxs);
#elif defined(PADDLE_WITH_XPU) && defined(PADDLE_WITH_XPU_BKCL)
                        device, &ctxs);
#else
                        device);
#endif
}

std::unique_ptr<ir::Graph> CreateGraph() {
  ProgramDesc prog;
  auto *op = prog.MutableBlock(0)->AppendOp();
  op->SetType("sum");
  op->SetInput("X", {"a"});
  op->SetOutput("Out", {"b"});
  op->SetAttr("op_role", 1);

  op = prog.MutableBlock(0)->AppendOp();
  op->SetType("dummy");
  op->SetInput("X", {"c"});
  op->SetOutput("Out", {"b"});
  op->SetAttr("op_role", 1);

  prog.MutableBlock(0)->Var("a")->SetType(proto::VarType::LOD_TENSOR);
  prog.MutableBlock(0)->Var("b")->SetType(proto::VarType::LOD_TENSOR);
  prog.MutableBlock(0)->Var("c")->SetType(proto::VarType::LOD_TENSOR);

  std::unique_ptr<ir::Graph> g(new ir::Graph(prog));
  return g;
}

std::unique_ptr<ir::Graph> CreateMultiGraph() {
  ProgramDesc prog;
  prog.AppendBlock(prog.Block(0));
  prog.AppendBlock(prog.Block(0));

  // Set contents in block_0.
  auto *op = prog.MutableBlock(0)->AppendOp();
  op->SetType("sum");
  op->SetInput("X", {"test_a", "test_b", "test_c"});
  op->SetOutput("Out", {"test_out"});
  op->SetAttr("op_role", 1);

  prog.MutableBlock(0)->Var("test_a")->SetType(proto::VarType::SELECTED_ROWS);
  prog.MutableBlock(0)->Var("test_b")->SetType(proto::VarType::SELECTED_ROWS);
  prog.MutableBlock(0)->Var("test_c")->SetType(proto::VarType::SELECTED_ROWS);
  prog.MutableBlock(0)->Var("test_out");
  op->InferVarType(prog.MutableBlock(0));

  prog.MutableBlock(0)->Var("test_b")->SetType(proto::VarType::LOD_TENSOR);
  op->InferVarType(prog.MutableBlock(0));

  // Set contents in block_1.
  op = prog.MutableBlock(1)->AppendOp();
  op->SetType("sum");
  op->SetInput("X", {"a"});
  op->SetOutput("Out", {"b"});
  op->SetAttr("op_role", 1);

  op = prog.MutableBlock(1)->AppendOp();
  op->SetType("dummy");
  op->SetInput("X", {"c"});
  op->SetOutput("Out", {"a"});
  op->SetAttr("op_role", 1);

  prog.MutableBlock(1)->Var("a")->SetType(proto::VarType::LOD_TENSOR);
  prog.MutableBlock(1)->Var("b")->SetType(proto::VarType::LOD_TENSOR);
  prog.MutableBlock(1)->Var("c")->SetType(proto::VarType::LOD_TENSOR);

  // Set contents in block_2.
  op = prog.MutableBlock(2)->AppendOp();
  op->SetType("sum");
  op->SetInput("X", {"a"});
  op->SetOutput("Out", {"b"});
  op->SetAttr("op_role", 1);

  op = prog.MutableBlock(2)->AppendOp();
  op->SetType("dummy");
  op->SetInput("X", {"c"});
  op->SetOutput("Out", {"b"});
  op->SetAttr("op_role", 1);

  prog.MutableBlock(2)->Var("a")->SetType(proto::VarType::LOD_TENSOR);
  prog.MutableBlock(2)->Var("b")->SetType(proto::VarType::LOD_TENSOR);
  prog.MutableBlock(2)->Var("c")->SetType(proto::VarType::LOD_TENSOR);

  std::unique_ptr<ir::Graph> g(new ir::Graph(prog));
  return g;
}

TEST(BuildStrategy, Basic) {
  BuildStrategy build_strategy;

  ProgramDesc prog;
  ir::Graph graph(prog);

  BuildStrategyApply(&build_strategy, &graph);
}

TEST(BuildStrategy, TestSingleGraph) {
  BuildStrategy build_strategy;
  auto graph = CreateGraph();

  BuildStrategyApply(&build_strategy, graph.get());
}

TEST(BuildStrategy, TestMultiGraph) {
  // Set FLAGS_convert_all_blocks to true to make sure this test works.
  bool flag_temp = FLAGS_convert_all_blocks;
  FLAGS_convert_all_blocks = true;

  BuildStrategy build_strategy;

  auto graph = CreateMultiGraph();

  BuildStrategyApply(&build_strategy, graph.get());

  // Recover FLAGS_convert_all_blocks.
  FLAGS_convert_all_blocks = flag_temp;
}

}  // namespace details
}  // namespace framework
}  // namespace paddle
