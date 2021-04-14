/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

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

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include "paddle/fluid/framework/executor.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/var_type.h"

namespace paddle {
namespace operators {

class ConditionalOp : public framework::OperatorBase {
 public:
  ConditionalOp(const std::string &type,
                const framework::VariableNameMap &inputs,
                const framework::VariableNameMap &outputs,
                const framework::AttributeMap &attrs)
      : OperatorBase(type, inputs, outputs, attrs) {}

  static const char kInputs[];
  static const char kOutputs[];
  static const char kCondition[];
  static const char kScope[];
  static const char kSkipEagerDeletionVars[];

 protected:
  std::vector<const framework::LoDTensor *> InputTensors(
      const framework::Scope &scope, const std::string &in_name) const {
    std::vector<const framework::LoDTensor *> retv;
    auto xs = Inputs(in_name);
    retv.resize(xs.size(), nullptr);
    std::transform(
        xs.begin(), xs.end(), retv.begin(),
        [&scope](const std::string &var_name) -> const framework::LoDTensor * {
          auto *var = scope.FindVar(var_name);
          PADDLE_ENFORCE_NOT_NULL(
              var, platform::errors::InvalidArgument("Cannot find variable %s",
                                                     var_name));
          return &var->Get<framework::LoDTensor>();
        });
    return retv;
  }

  bool ScalarCondition(
      const std::vector<const framework::LoDTensor *> &ips) const {
    PADDLE_ENFORCE_EQ(
        ips.size() == 1UL && ips[0]->IsInitialized(), true,
        platform::errors::InvalidArgument(
            "condition should have one initialized input as condition"));

    PADDLE_ENFORCE_EQ(ips[0]->type() == framework::proto::VarType::BOOL &&
                          ips[0]->numel() == 1,
                      true, platform::errors::InvalidArgument(
                                "condition input's data type should be bool, "
                                "numel should be 1, actual numel is %d",
                                ips[0]->numel()));
    bool res = false;
    if (platform::is_gpu_place(ips[0]->place())) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      framework::LoDTensor cpu_tensor;
      framework::TensorCopy(*ips[0], platform::CPUPlace(), &cpu_tensor);
      platform::DeviceContextPool::Instance().Get(ips[0]->place())->Wait();
      res = cpu_tensor.data<bool>()[0];
#endif
    }
    if (platform::is_npu_place(ips[0]->place())) {
#ifdef PADDLE_WITH_ASCEND_CL
      framework::LoDTensor cpu_tensor;
      framework::TensorCopy(*ips[0], platform::CPUPlace(), &cpu_tensor);
      platform::DeviceContextPool::Instance().Get(ips[0]->place())->Wait();
      res = cpu_tensor.data<bool>()[0];
#endif
    } else {
      res = ips[0]->data<bool>()[0];
    }
    return res;
  }
};

class ConditionalBlockOpProtoMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput(ConditionalOp::kCondition,
             "The conditional variable of this operator. If Cond is empty, the "
             "whole sub-block will not be executed.")
        .AsDuplicable();
    AddInput(ConditionalOp::kInputs, "The input variables of the sub-block.")
        .AsDuplicable();
    AddOutput(ConditionalOp::kOutputs, "The output variables of the sub-block.")
        .AsDuplicable();
    AddOutput(ConditionalOp::kScope,
              "(std::vector<Scope*>) The step scope of conditional block. To "
              "unify the conditional block, rnn and while op, the type of "
              "scope is std::vector<Scope*>");
    AddAttr<framework::BlockDesc *>(
        "sub_block", "The step block of conditional block operator");
    AddAttr<bool>("is_scalar_condition",
                  "The conditional variable (Cond) is used as scalar "
                  "condition.")
        .SetDefault(false);
    AddAttr<std::vector<std::string>>(ConditionalOp::kSkipEagerDeletionVars,
                                      "Vars that would not be deleted when "
                                      "garbage collection strategy enables")
        .SetDefault(std::vector<std::string>());
    AddComment(R"DOC(Conditional block operator

If `is_scalar_condition` is True, the conditional variable (Cond) is a scalar,
run the operators in sub-block if Cond is True.

If `is_scalar_condition` is False, the conditional variable (Cond) is a vector or
tensor, run the operators in sub-block if all of input variables are not empty.


)DOC");
  }
};

}  // namespace operators
}  // namespace paddle
