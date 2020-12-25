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

#include "gtest/gtest.h"
#include "paddle/fluid/framework/details/multi_devices_helper.h"
#include "paddle/fluid/framework/ir/graph.h"
#include "paddle/fluid/framework/ir/graph_helper.h"
#include "paddle/fluid/framework/ir/memory_optimize_pass/memory_optimization_var_info.h"
#include "paddle/fluid/framework/ir/memory_optimize_pass/reference_count_pass_helper.h"
#include "paddle/fluid/framework/parallel_executor.h"
#include "paddle/fluid/framework/program_desc.h"

USE_OP(scale);
USE_OP(elementwise_mul);
USE_OP(elementwise_add);
USE_OP(elementwise_add_grad);

DECLARE_double(eager_delete_tensor_gb);

namespace paddle {
namespace framework {
namespace p = paddle::platform;

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

static void NewVar(BlockDesc *block, const std::string &name,
                   const std::vector<int64_t> &shape) {
  auto *var_desc = block->Var(name);
  var_desc->SetShape(shape);
}

static void AppendOp(BlockDesc *block, const std::string &type,
                     VariableNameMap inputs, VariableNameMap outputs,
                     AttributeMap attrs) {
  auto &op_info = OpInfoMap::Instance().Get(type);
  if (op_info.Checker()) {
    op_info.Checker()->Check(&attrs);
  }

  auto *op = block->AppendOp();
  op->SetType(type);
  for (auto &pair : inputs) {
    op->SetInput(pair.first, pair.second);
  }

  for (auto &pair : outputs) {
    op->SetOutput(pair.first, pair.second);
    for (auto &var_name : pair.second) {
      if (!block->FindVarRecursive(var_name)) {
        NewVar(block, var_name, {});
      }
    }
  }

  op->SetAttrMap(attrs);
  op->InferVarType(block);
  op->InferShape(*block);
}

class ReferenceCountPassTestHelper {
 public:
  ReferenceCountPassTestHelper(const ProgramDesc &program, bool use_cuda)
      : graph_(program) {
    details::BuildStrategy build_strategy;
    build_strategy.enable_inplace_ = false;
    build_strategy.memory_optimize_ = false;
    FLAGS_eager_delete_tensor_gb = -1;

    details::ExecutionStrategy exec_strategy;
    exec_strategy.use_device_ = use_cuda ? p::kCUDA : p::kCPU;

    executor_.reset(new ParallelExecutor(CreatePlaces(1, use_cuda), {}, "",
                                         &scope_, {}, exec_strategy,
                                         build_strategy, &graph_));

    auto ref_cnt_pass =
        ir::PassRegistry::Instance().Get("reference_count_pass");
    ref_cnt_pass->SetNotOwned(ir::kMemOptVarInfoMapList, &mem_opt_var_infos_);
    ref_cnt_pass->SetNotOwned(ir::kLastLiveOpsOfVars, &last_live_ops_of_vars_);
    ref_cnt_pass->Apply(&const_cast<ir::Graph &>(executor_->Graph()));
  }

  bool IsLastLivedOps(const std::string &name,
                      std::vector<std::string> ops) const {
    std::sort(ops.begin(), ops.end());
    return LastLivedOpTypes(name) == ops;
  }

  std::vector<OperatorBase *> LastLivedOps(const std::string &name) const {
    auto &ops = last_live_ops_of_vars_[0].at(name).ops();
    std::vector<OperatorBase *> ret;
    for (auto *op : ops) {
      ret.emplace_back(op->GetOp());
    }
    return ret;
  }

 private:
  std::vector<std::string> LastLivedOpTypes(const std::string &name) const {
    auto iter = last_live_ops_of_vars_[0].find(name);
    std::vector<std::string> ret;
    if (iter != last_live_ops_of_vars_[0].end()) {
      for (auto *op : iter->second.ops()) {
        ret.emplace_back(op->GetOp()->Type());
      }
    }
    std::sort(ret.begin(), ret.end());
    return ret;
  }

 private:
  ir::Graph graph_;
  Scope scope_;
  std::unique_ptr<ParallelExecutor> executor_;

  ir::MemOptVarInfoMapList mem_opt_var_infos_;
  std::vector<ir::LastLiveOpsOfVars> last_live_ops_of_vars_;
};

TEST(test_reference_count_pass, test_no_need_buffer_var_shrink) {
  ProgramDesc program;
  auto *block = program.MutableBlock(0);
  std::vector<int64_t> shape{{3, 4, 5}};

  /**
   * The network is:
   *
   * x0 = fluid.layer.data(...)
   * x1 = scale(x0, scale=1)
   * x2 = scale(x1, scale=2)
   * x3 = elementwise_mul(x1, x2)
   * scale(x3, out=x1, scale=3) # produce a new version of x1
   * x4, x5 = elementwise_add_grad(dout=x3, x=x2, y=x1)
   * x6 = elementwise_mul(x4, x5)
   * x7 = elementwise_add(x5, x5)
   */
  std::string x0 = "x0";
  std::string x1 = "x1";
  std::string x2 = "x2";
  std::string x3 = "x3";
  std::string x4 = "x4";
  std::string x5 = "x5";
  std::string x6 = "x6";
  std::string x7 = "x7";

  NewVar(block, x0, shape);
  AppendOp(block, "scale", {{"X", {x0}}}, {{"Out", {x1}}}, {{"scale", 1.0f}});
  AppendOp(block, "scale", {{"X", {x1}}}, {{"Out", {x2}}}, {{"scale", 2.0f}});
  AppendOp(block, "elementwise_mul", {{"X", {x1}}, {"Y", {x2}}},
           {{"Out", {x3}}}, {});
  AppendOp(block, "scale", {{"X", {x3}}}, {{"Out", {x1}}}, {{"scale", 3.0f}});
  AppendOp(block, "elementwise_add_grad",
           {{GradVarName("Out"), {x3}}, {"X", {x2}}, {"Y", {x1}}},
           {{GradVarName("X"), {x4}}, {GradVarName("Y"), {x5}}}, {});
  AppendOp(block, "elementwise_mul", {{"X", {x4}}, {"Y", {x5}}},
           {{"Out", {x6}}}, {});
  AppendOp(block, "elementwise_add", {{"X", {x5}}, {"Y", {x5}}},
           {{"Out", {x7}}}, {});

  std::vector<bool> use_cuda_list{false};
#ifdef PADDLE_WITH_CUDA
  use_cuda_list.push_back(true);
#endif
  for (auto use_cuda : use_cuda_list) {
    ReferenceCountPassTestHelper helper(program, use_cuda);
    ASSERT_TRUE(helper.IsLastLivedOps(x0, {"scale"}));
    ASSERT_EQ(
        BOOST_GET_CONST(float, helper.LastLivedOps(x0)[0]->Attrs().at("scale")),
        1.0f);

    ASSERT_TRUE(helper.IsLastLivedOps(x1, {"scale"}));
    ASSERT_EQ(
        BOOST_GET_CONST(float, helper.LastLivedOps(x1)[0]->Attrs().at("scale")),
        3.0f);

    ASSERT_TRUE(helper.IsLastLivedOps(x2, {"elementwise_mul"}));
    ASSERT_TRUE(helper.IsLastLivedOps(x3, {"elementwise_add_grad"}));

    ASSERT_TRUE(helper.IsLastLivedOps(x4, {"elementwise_mul"}));
    ASSERT_TRUE(
        helper.IsLastLivedOps(x5, {"elementwise_mul", "elementwise_add"}));

    ASSERT_TRUE(helper.IsLastLivedOps(x6, {"elementwise_mul"}));
    ASSERT_TRUE(helper.IsLastLivedOps(x7, {"elementwise_add"}));
  }
}

}  // namespace framework
}  // namespace paddle
