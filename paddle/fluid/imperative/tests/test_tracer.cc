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
#include <set>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "paddle/fluid/imperative/basic_engine.h"
#include "paddle/fluid/imperative/tracer.h"
#include "paddle/fluid/memory/memcpy.h"

namespace imperative = paddle::imperative;
namespace platform = paddle::platform;
namespace framework = paddle::framework;

namespace paddle {
namespace imperative {

using vb_vector = std::vector<std::shared_ptr<imperative::VarBase>>;

using var_pair = std::pair<std::string, vb_vector>;

TEST(test_tracer, test_trace_op) {
  // Doing an mul
  imperative::Tracer tracer;
  std::shared_ptr<imperative::VarBase> x_in(
      new imperative::VarBase(true, "x_in"));
  std::shared_ptr<imperative::VarBase> y_in(
      new imperative::VarBase(true, "y_in"));
  std::shared_ptr<imperative::VarBase> vout(
      new imperative::VarBase(true, "vout"));
  platform::CPUPlace place;
  std::vector<float> src_data(10, 2.0);
  std::vector<int64_t> dims1 = {2, 5};
  std::vector<int64_t> dims2 = {5, 2};

  auto* x_in_tensor = x_in->MutableVar()->GetMutable<framework::LoDTensor>();
  auto* y_in_tensor = y_in->MutableVar()->GetMutable<framework::LoDTensor>();
  x_in_tensor->Resize(framework::make_ddim(dims1));
  auto* mutable_x = x_in_tensor->mutable_data<float>(place);
  paddle::memory::Copy(place, mutable_x, place, src_data.data(),
                       sizeof(float) * src_data.size());
  y_in_tensor->Resize(framework::make_ddim(dims2));
  auto* mutable_y = y_in_tensor->mutable_data<float>(place);
  paddle::memory::Copy(place, mutable_y, place, src_data.data(),
                       sizeof(float) * src_data.size());

  var_pair x_pair = var_pair("X", vb_vector(1, x_in));
  var_pair y_pair = var_pair("Y", vb_vector(1, y_in));
  var_pair out_pair = var_pair("Out", vb_vector(1, vout));
  imperative::NameVarBaseMap ins = {x_pair, y_pair};
  imperative::NameVarBaseMap outs = {out_pair};
  framework::AttributeMap mul_attr_map;
  mul_attr_map["use_mkldnn"] = false;
  tracer.TraceOp("mul", ins, outs, mul_attr_map, place, true);
  const auto& out_tensor = vout->Var().Get<framework::LoDTensor>();
  for (int i = 0; i < vout->Var().Get<framework::LoDTensor>().numel(); i++) {
    ASSERT_EQ(out_tensor.data<float>()[i], 20.0);
  }
}

TEST(test_tracer, test_trace_op_with_backward) {
  // Doing an mul
  imperative::Tracer tracer;
  std::shared_ptr<imperative::VarBase> x_in(
      new imperative::VarBase(true, "x_in"));
  std::shared_ptr<imperative::VarBase> y_in(
      new imperative::VarBase(true, "y_in"));
  std::shared_ptr<imperative::VarBase> vout(
      new imperative::VarBase(true, "vout"));
  platform::CPUPlace place;
  std::vector<float> src_data(10, 2.0);
  std::vector<int64_t> dims1 = {2, 5};
  std::vector<int64_t> dims2 = {5, 2};

  auto* x_in_tensor = x_in->MutableVar()->GetMutable<framework::LoDTensor>();
  auto* y_in_tensor = y_in->MutableVar()->GetMutable<framework::LoDTensor>();
  x_in_tensor->Resize(framework::make_ddim(dims1));
  auto* mutable_x = x_in_tensor->mutable_data<float>(place);
  paddle::memory::Copy(place, mutable_x, place, src_data.data(),
                       sizeof(float) * src_data.size());
  y_in_tensor->Resize(framework::make_ddim(dims2));
  auto* mutable_y = y_in_tensor->mutable_data<float>(place);
  paddle::memory::Copy(place, mutable_y, place, src_data.data(),
                       sizeof(float) * src_data.size());

  var_pair x_pair = var_pair("X", vb_vector(1, x_in));
  var_pair y_pair = var_pair("Y", vb_vector(1, y_in));
  var_pair out_pair = var_pair("Out", vb_vector(1, vout));
  imperative::NameVarBaseMap ins = {x_pair, y_pair};
  imperative::NameVarBaseMap outs = {out_pair};
  framework::AttributeMap mul_attr_map;
  mul_attr_map["use_mkldnn"] = false;
  tracer.TraceOp("mul", ins, outs, mul_attr_map, place, true);
  const auto& out_tensor = vout->Var().Get<framework::LoDTensor>();
  for (int i = 0; i < vout->Var().Get<framework::LoDTensor>().numel(); i++) {
    ASSERT_EQ(out_tensor.data<float>()[i], 20.0);
  }
}

TEST(test_tracer, test_track_backward_output) {
  // Doing an mul
  imperative::Tracer tracer;
  std::shared_ptr<imperative::VarBase> x_in(
      new imperative::VarBase(true, "x_in"));
  std::shared_ptr<imperative::VarBase> y_in(
      new imperative::VarBase(true, "y_in"));
  x_in->SetOverridedStopGradient(false);
  std::shared_ptr<imperative::VarBase> vout(
      new imperative::VarBase(true, "vout"));
  platform::CPUPlace place;
  std::vector<float> src_data(10, 2.0);
  std::vector<int64_t> dims1 = {2, 5};
  std::vector<int64_t> dims2 = {5, 2};

  auto* x_in_tensor = x_in->MutableVar()->GetMutable<framework::LoDTensor>();
  auto* y_in_tensor = y_in->MutableVar()->GetMutable<framework::LoDTensor>();
  x_in_tensor->Resize(framework::make_ddim(dims1));
  auto* mutable_x = x_in_tensor->mutable_data<float>(place);
  paddle::memory::Copy(place, mutable_x, place, src_data.data(),
                       sizeof(float) * src_data.size());
  y_in_tensor->Resize(framework::make_ddim(dims2));
  auto* mutable_y = y_in_tensor->mutable_data<float>(place);
  paddle::memory::Copy(place, mutable_y, place, src_data.data(),
                       sizeof(float) * src_data.size());

  var_pair x_pair = var_pair("X", vb_vector(1, x_in));
  var_pair y_pair = var_pair("Y", vb_vector(1, y_in));
  var_pair out_pair = var_pair("Out", vb_vector(1, vout));
  imperative::NameVarBaseMap ins = {x_pair, y_pair};
  imperative::NameVarBaseMap outs = {out_pair};
  framework::AttributeMap mul_attr_map;
  mul_attr_map["use_mkldnn"] = false;
  tracer.TraceOp("mul", ins, outs, mul_attr_map, place, true);
  ASSERT_EQ(x_in->GradVarBase()->GradOpNum(), 0UL);
  ASSERT_EQ(y_in->GradVarBase()->GradOpNum(), 0UL);
  ASSERT_EQ(vout->GradVarBase()->GradOpNum(), 1UL);
}

TEST(test_tracer, test_track_backward_input) {
  // Doing an mul
  imperative::Tracer tracer;
  std::shared_ptr<imperative::VarBase> x_in(
      new imperative::VarBase(true, "x_in"));
  std::shared_ptr<imperative::VarBase> y_in(
      new imperative::VarBase(true, "y_in"));
  std::shared_ptr<imperative::VarBase> vout(
      new imperative::VarBase(true, "vout"));
  platform::CPUPlace place;
  x_in->SetOverridedStopGradient(false);
  std::vector<float> src_data(10, 2.0);
  std::vector<int64_t> dims1 = {2, 5};
  std::vector<int64_t> dims2 = {5, 2};

  auto* x_in_tensor = x_in->MutableVar()->GetMutable<framework::LoDTensor>();
  auto* y_in_tensor = y_in->MutableVar()->GetMutable<framework::LoDTensor>();
  x_in_tensor->Resize(framework::make_ddim(dims1));
  auto* mutable_x = x_in_tensor->mutable_data<float>(place);
  paddle::memory::Copy(place, mutable_x, place, src_data.data(),
                       sizeof(float) * src_data.size());
  y_in_tensor->Resize(framework::make_ddim(dims2));
  auto* mutable_y = y_in_tensor->mutable_data<float>(place);
  paddle::memory::Copy(place, mutable_y, place, src_data.data(),
                       sizeof(float) * src_data.size());

  var_pair x_pair = var_pair("X", vb_vector(1, x_in));
  var_pair y_pair = var_pair("Y", vb_vector(1, y_in));
  var_pair out_pair = var_pair("Out", vb_vector(1, vout));
  imperative::NameVarBaseMap ins = {x_pair, y_pair};
  imperative::NameVarBaseMap outs = {out_pair};
  framework::AttributeMap mul_attr_map;
  mul_attr_map["use_mkldnn"] = false;
  tracer.TraceOp("mul", ins, outs, mul_attr_map, place, true);

  ASSERT_EQ(x_in->GradVarBase()->GradOpNum(), 0UL);
  ASSERT_EQ(y_in->GradVarBase()->GradOpNum(), 0UL);
  ASSERT_EQ(vout->GradVarBase()->GradOpNum(), 1UL);
}
#if defined(PADDLE_WITH_CUDA)
TEST(test_tracer, test_trace_op_with_multi_device_inputs) {
  // Doing an mul
  imperative::Tracer tracer;
  std::shared_ptr<imperative::VarBase> x_in(
      new imperative::VarBase(true, "x_in"));
  x_in->SetOverridedStopGradient(false);  // force to run backward
  std::shared_ptr<imperative::VarBase> y_in(
      new imperative::VarBase(true, "y_in"));
  y_in->SetOverridedStopGradient(false);
  std::shared_ptr<imperative::VarBase> vout(
      new imperative::VarBase(true, "vout"));
  platform::CPUPlace place;
  platform::CUDAPlace gpu_place(0);
  std::vector<float> src_data(10, 2.0);
  std::vector<int64_t> dims1 = {2, 5};
  std::vector<int64_t> dims2 = {2, 5};

  auto* x_in_tensor = x_in->MutableVar()->GetMutable<framework::LoDTensor>();
  auto* y_in_tensor = y_in->MutableVar()->GetMutable<framework::LoDTensor>();
  x_in_tensor->Resize(framework::make_ddim(dims1));
  auto* mutable_x = x_in_tensor->mutable_data<float>(place);
  paddle::memory::Copy(place, mutable_x, place, src_data.data(),
                       sizeof(float) * src_data.size());
  y_in_tensor->Resize(framework::make_ddim(dims2));
  auto* mutable_y = y_in_tensor->mutable_data<float>(gpu_place);
  paddle::memory::Copy(gpu_place, mutable_y, place, src_data.data(),
                       sizeof(float) * src_data.size(), 0);
  var_pair x_pair = var_pair("X", vb_vector(1, x_in));
  var_pair y_pair = var_pair("Y", vb_vector(1, y_in));
  var_pair out_pair = var_pair("Out", vb_vector(1, vout));
  imperative::NameVarBaseMap ins = {x_pair, y_pair};
  imperative::NameVarBaseMap outs = {out_pair};
  framework::AttributeMap mul_attr_map;
  mul_attr_map["use_mkldnn"] = false;
  tracer.TraceOp("elementwise_add", ins, outs, mul_attr_map, gpu_place, true);

  // run reduce sum
  std::shared_ptr<imperative::VarBase> reduce_sum_out(
      new imperative::VarBase(true, "reduce_sum_out"));
  var_pair reduce_sum_in_pair = var_pair("X", vb_vector(1, vout));
  var_pair reduce_sum_out_pair = var_pair("Out", vb_vector(1, reduce_sum_out));
  imperative::NameVarBaseMap reduce_in = {reduce_sum_in_pair};
  imperative::NameVarBaseMap reduce_out = {reduce_sum_out_pair};
  framework::AttributeMap reduce_attr_map;
  tracer.TraceOp("reduce_sum", reduce_in, reduce_out, reduce_attr_map,
                 gpu_place, true);
  imperative::BasicEngine engine;
  engine.Init(reduce_sum_out.get());
  engine.Execute();

  framework::LoDTensor rlt;
  framework::TensorCopySync(vout->Var().Get<framework::LoDTensor>(), place,
                            &rlt);
  for (int i = 0; i < rlt.numel(); i++) {
    ASSERT_EQ(rlt.data<float>()[i], 4.0);
  }

  framework::LoDTensor out_grad;
  framework::TensorCopySync(vout->GradVar().Get<framework::LoDTensor>(), place,
                            &out_grad);
  for (int i = 0; i < out_grad.numel(); ++i) {
    ASSERT_EQ(out_grad.data<float>()[i], 1.0);
  }

  framework::LoDTensor x_grad;
  framework::TensorCopySync(x_in->GradVar().Get<framework::LoDTensor>(), place,
                            &x_grad);

  for (int i = 0; i < x_grad.numel(); ++i) {
    ASSERT_EQ(x_grad.data<float>()[i], 1.0);
  }

  framework::LoDTensor y_grad;
  framework::TensorCopySync(y_in->GradVar().Get<framework::LoDTensor>(), place,
                            &y_grad);

  for (int i = 0; i < y_grad.numel(); ++i) {
    ASSERT_EQ(y_grad.data<float>()[i], 1.0);
  }
}

#endif

TEST(test_tracer, test_unique_name_generator) {
  // generate two unique names
  imperative::Tracer tracer;
  auto fc_1 = tracer.GenerateUniqueName("fc");
  auto fc_2 = tracer.GenerateUniqueName("fc");
  ASSERT_STREQ("fc_0", fc_1.c_str());
  ASSERT_STREQ("fc_1", fc_2.c_str());
  // use `eager_tmp` as key if not specify it.
  auto tmp_var_2 = tracer.GenerateUniqueName();
  ASSERT_STREQ("dygraph_tmp_2", tmp_var_2.c_str());
  auto tmp_var_3 = tracer.GenerateUniqueName("dygraph_tmp");
  ASSERT_STREQ("dygraph_tmp_3", tmp_var_3.c_str());
}

TEST(test_tracer, test_current_tracer) {
  // use current_tracer
  auto tracer = std::make_shared<imperative::Tracer>();
  imperative::SetCurrentTracer(tracer);
  auto current_tracer = imperative::GetCurrentTracer();
  ASSERT_EQ(current_tracer, tracer);
}

TEST(test_tracer, test_expected_place) {
  // default expected place is CPUPlace
  imperative::Tracer tracer;
  ASSERT_EQ(platform::is_cpu_place(tracer.ExpectedPlace()), true);
  // set to CUDAPlace
  platform::CUDAPlace gpu_place(0);
  tracer.SetExpectedPlace(gpu_place);
  ASSERT_EQ(platform::is_gpu_place(tracer.ExpectedPlace()), true);
}

TEST(test_tracer, test_var_without_grad_var) {
  // Doing an mul
  imperative::Tracer tracer;
  std::shared_ptr<imperative::VarBase> x_in(
      new imperative::VarBase(true, "x_in"));
  x_in->ClearGradVarBase();
  std::shared_ptr<imperative::VarBase> y_in(
      new imperative::VarBase(true, "y_in"));
  std::shared_ptr<imperative::VarBase> vout(
      new imperative::VarBase(true, "vout"));
  x_in->SetOverridedStopGradient(false);
  y_in->SetOverridedStopGradient(false);
  platform::CPUPlace place;
  std::vector<float> src_data(10, 2.0);
  std::vector<int64_t> dims1 = {2, 5};
  std::vector<int64_t> dims2 = {5, 2};

  auto* x_in_tensor = x_in->MutableVar()->GetMutable<framework::LoDTensor>();
  auto* y_in_tensor = y_in->MutableVar()->GetMutable<framework::LoDTensor>();
  x_in_tensor->Resize(framework::make_ddim(dims1));
  auto* mutable_x = x_in_tensor->mutable_data<float>(place);
  paddle::memory::Copy(place, mutable_x, place, src_data.data(),
                       sizeof(float) * src_data.size());
  y_in_tensor->Resize(framework::make_ddim(dims2));
  auto* mutable_y = y_in_tensor->mutable_data<float>(place);
  paddle::memory::Copy(place, mutable_y, place, src_data.data(),
                       sizeof(float) * src_data.size());

  var_pair x_pair = var_pair("X", vb_vector(1, x_in));
  var_pair y_pair = var_pair("Y", vb_vector(1, y_in));
  var_pair out_pair = var_pair("Out", vb_vector(1, vout));
  imperative::NameVarBaseMap ins = {x_pair, y_pair};
  imperative::NameVarBaseMap outs = {out_pair};
  framework::AttributeMap mul_attr_map;
  mul_attr_map["use_mkldnn"] = false;
  tracer.TraceOp("mul", ins, outs, mul_attr_map, place, true);

  const auto& out_tensor = vout->Var().Get<framework::LoDTensor>();
  for (int i = 0; i < vout->Var().Get<framework::LoDTensor>().numel(); i++) {
    ASSERT_EQ(out_tensor.data<float>()[i], 20.0);
  }

  ASSERT_EQ(x_in->GradVarBase()->GradOpNum(), 0UL);
  ASSERT_EQ(y_in->GradVarBase()->GradOpNum(), 0UL);
  ASSERT_EQ(vout->GradVarBase()->GradOpNum(), 1UL);

  imperative::BasicEngine engine;
  engine.Init(vout.get());
  engine.Execute();

  // check the grad
  framework::LoDTensor x_grad;
  framework::TensorCopySync(x_in->GradVar().Get<framework::LoDTensor>(), place,
                            &x_grad);

  for (int i = 0; i < x_grad.numel(); ++i) {
    ASSERT_EQ(x_grad.data<float>()[i], 4.0);
  }

  framework::LoDTensor y_grad;
  framework::TensorCopySync(y_in->GradVar().Get<framework::LoDTensor>(), place,
                            &y_grad);

  for (int i = 0; i < y_grad.numel(); ++i) {
    ASSERT_EQ(y_grad.data<float>()[i], 4.0);
  }
}

template <typename T>
using WeakPtrSet =
    std::set<std::weak_ptr<T>, std::owner_less<std::weak_ptr<T>>>;

static void TestVarOpDestructionMain(const platform::Place& place,
                                     int64_t tensor_size = 10,
                                     size_t loop_num = 10) {
  WeakPtrSet<VariableWrapper> var_wrappers;
  WeakPtrSet<VarBase> var_bases;
  WeakPtrSet<GradOpNode> op_bases;

  Tracer tracer;

  {
    auto x = std::make_shared<VarBase>("x");
    auto y = std::make_shared<VarBase>("y");

    x->MutableVar()
        ->GetMutable<framework::LoDTensor>()
        ->Resize({tensor_size, tensor_size})
        .mutable_data<float>(place);

    y->MutableVar()
        ->GetMutable<framework::LoDTensor>()
        ->Resize({tensor_size, tensor_size})
        .mutable_data<float>(place);

    x->SetOverridedStopGradient(false);
    y->SetOverridedStopGradient(true);

    for (size_t i = 0; i < loop_num; ++i) {
      size_t var_wrapper_num = var_wrappers.size();
      size_t var_base_num = var_bases.size();
      size_t op_base_num = op_bases.size();

      auto z = std::make_shared<VarBase>("z_" + std::to_string(i));
      tracer.TraceOp("mul", NameVarBaseMap{{"X", {x}}, {"Y", {y}}},
                     NameVarBaseMap{{"Out", {z}}}, framework::AttributeMap{},
                     place, true);

      ASSERT_EQ(z->GradOpNum(), 0UL);
      ASSERT_EQ(z->GradVarBase()->GradOpNum(), 1UL);
      auto new_op = z->GradVarBase()->GradNode();

      ASSERT_EQ(x->GradOpNum(), 0UL);
      ASSERT_EQ(y->GradOpNum(), 0UL);

      std::unordered_set<std::shared_ptr<GradOpNode>> expected_pending_ops;
      if (i == 0) {
        ASSERT_EQ(x->GradVarBase()->GradOpNum(), 0UL);
        ASSERT_EQ(y->GradVarBase()->GradOpNum(), 0UL);
      } else {
        ASSERT_EQ(x->GradVarBase()->GradOpNum(), 1UL);
        ASSERT_EQ(y->GradVarBase()->GradOpNum(), 0UL);

        if (x->GradVarBase()->GradNode()) {
          expected_pending_ops.emplace(x->GradVarBase()->GradNode());
        }

        if (y->GradVarBase()->GradNode()) {
          expected_pending_ops.emplace(y->GradVarBase()->GradNode());
        }

        std::unordered_set<std::shared_ptr<GradOpNode>> actual_pending_ops;
        for (auto& op : new_op->GradPendingNodes()) {
          actual_pending_ops.emplace(op);
        }

        ASSERT_TRUE(expected_pending_ops == actual_pending_ops);
        ASSERT_EQ(expected_pending_ops.count(new_op), 0UL);
      }

      var_wrappers.emplace(x->SharedVar());
      var_wrappers.emplace(x->GradVarBase()->SharedVar());
      var_wrappers.emplace(y->SharedVar());
      var_wrappers.emplace(y->GradVarBase()->SharedVar());
      var_wrappers.emplace(z->SharedVar());
      var_wrappers.emplace(z->GradVarBase()->SharedVar());

      var_bases.emplace(x);
      var_bases.emplace(x->GradVarBase());
      var_bases.emplace(y);
      var_bases.emplace(y->GradVarBase());
      var_bases.emplace(z);
      var_bases.emplace(z->GradVarBase());

      for (auto& op : expected_pending_ops) {
        op_bases.emplace(op);
      }

      if (i == 0) {
        ASSERT_EQ(var_wrapper_num, 0UL);
        ASSERT_EQ(var_base_num, 0UL);
        ASSERT_EQ(op_base_num, 0UL);
        ASSERT_EQ(var_wrappers.size(), 6UL);
        ASSERT_EQ(var_bases.size(), 6UL);
        ASSERT_EQ(op_bases.size(), 0UL);
      } else {
        ASSERT_EQ(var_wrappers.size(), var_wrapper_num + 2);
        ASSERT_EQ(var_bases.size(), var_base_num + 2);
        ASSERT_EQ(op_bases.size(), op_base_num + 1);
      }

      x = z;  // recurrent usage
    }
  }

  for (auto& var : var_wrappers) {
    ASSERT_TRUE(var.expired());
  }

  for (auto& var : var_bases) {
    ASSERT_TRUE(var.expired());
  }

  for (auto& op : op_bases) {
    ASSERT_TRUE(op.expired());
  }
}

TEST(test_tracer, test_var_op_destruction) {
  TestVarOpDestructionMain(platform::CPUPlace());
#ifdef PADDLE_WITH_CUDA
  TestVarOpDestructionMain(platform::CUDAPlace(0));
#endif
}

}  // namespace imperative
}  // namespace paddle

USE_OP(mul);
USE_OP(mul_grad);
USE_OP(reduce_sum);
USE_OP(reduce_sum_grad);
USE_OP(elementwise_add);
