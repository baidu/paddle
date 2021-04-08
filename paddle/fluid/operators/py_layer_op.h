// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include <algorithm>
#include <functional>
#include <utility>
#include <vector>
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/python_headers.h"
#include "paddle/fluid/operators/py_layer_context/py_context.h"

namespace paddle {
namespace operators {
using CtxPtr = std::shared_ptr<imperative::PyLayerContext>;

class PyLayerOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
  void InferShape(framework::InferShapeContext* ctx) const override {}

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    auto data_type = OperatorWithKernel::IndicateVarDataType(ctx, "X");
    return framework::OpKernelType(data_type, ctx.device_context());
  }

 public:
  CtxPtr& GetMutablePyLayerContext() { return py_context; }
  const CtxPtr& GetMutablePyLayerContext() const { return py_context; }

 private:
  CtxPtr py_context;
};

template <typename T>
class PyLayerGradOpMaker {};
template <>
class PyLayerGradOpMaker<paddle::framework::OpDesc>
    : public framework::SingleGradOpMaker<paddle::framework::OpDesc> {
 public:
  using framework::SingleGradOpMaker<
      paddle::framework::OpDesc>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<paddle::framework::OpDesc> grad_op) const override {}
};

template <>
class PyLayerGradOpMaker<paddle::imperative::OpBase>
    : public framework::SingleGradOpMaker<paddle::imperative::OpBase> {
 public:
  using framework::SingleGradOpMaker<
      paddle::imperative::OpBase>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<paddle::imperative::OpBase> grad_op) const override;

 public:
  CtxPtr& GetMutablePyLayerContext() { return py_context; }

 private:
  CtxPtr py_context;
};

}  // namespace operators
}  // namespace paddle
