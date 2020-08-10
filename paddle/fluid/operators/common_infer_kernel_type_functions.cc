/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/common_infer_kernel_type_functions.h"

#include <algorithm>
#include <string>
#include <vector>

#include "paddle/fluid/framework/data_type.h"
#ifdef PADDLE_WITH_MKLDNN
#include "paddle/fluid/platform/mkldnn_helper.h"
#endif

// This file almostly contains all the infershape functions that are used in
// operators.

namespace paddle {
namespace operators {
namespace details {

using Tensor = paddle::framework::Tensor;
using LoDTensor = paddle::framework::LoDTensor;
using LoDTensorArray = paddle::framework::LoDTensorArray;
using SelectedRows = paddle::framework::SelectedRows;
using ExecutionContext = paddle::framework::ExecutionContext;
using Scope = paddle::framework::Scope;
using Variable = paddle::framework::Variable;
using Place = paddle::platform::Place;
using DType = paddle::framework::proto::VarType::Type;

void ParseInputDataType(const ExecutionContext& ctx, const std::string& name,
                        DType* data_type) {
  DType default_data_type = static_cast<DType>(-1);
  const std::vector<Variable*> vars = ctx.MultiInputVar(name);
  for (size_t i = 0; i < vars.size(); ++i) {
    const Variable* var = vars[i];
    if (var != nullptr) {
      const Tensor* t = nullptr;
      if (var->IsType<Tensor>()) {
        t = &var->Get<Tensor>();
      } else if (var->IsType<LoDTensor>()) {
        t = &var->Get<LoDTensor>();
      } else if (var->IsType<SelectedRows>()) {
        t = &(var->Get<SelectedRows>().value());
      } else if (var->IsType<LoDTensorArray>()) {
        auto t_arr = var->Get<LoDTensorArray>();
        for (size_t j = 0; j < t_arr.size(); j++) {
          if (t_arr[j].IsInitialized()) {
            t = &(t_arr[j]);
          }
        }
      }
      if (t != nullptr) {
        PADDLE_ENFORCE_EQ(
            t->IsInitialized(), true,
            platform::errors::InvalidArgument(
                "The Tensor in the %s Op's Input Variable %s(%s) is "
                "not initialized.",
                ctx.Type(), name, ctx.InputNames(name).at(i)));
        DType tmp = t->type();
        PADDLE_ENFORCE(
            tmp == *data_type || *data_type == default_data_type,
            platform::errors::InvalidArgument(
                "The DataType of %s Op's duplicable Variable %s must be "
                "consistent. The current variable type is (%s), but the "
                "previous variable type is (%s).",
                ctx.Type(), name, paddle::framework::DataTypeToString(tmp),
                paddle::framework::DataTypeToString(*data_type)));
        *data_type = tmp;
      }
    }
  }
}

DType IndicateVarDataType(const ExecutionContext& ctx,
                          const std::string& name) {
  DType dafault_data_type = static_cast<DType>(-1);
  DType data_type = dafault_data_type;
  ParseInputDataType(ctx, name, &data_type);
  PADDLE_ENFORCE_NE(
      data_type, dafault_data_type,
      "The Input Variable(%s) of %s Op used to determine kernel data type "
      "is empty or not LoDTensor or SelectedRows or LoDTensorArray.",
      name, ctx.Type());
  return data_type;
}
}  // namespace details

// infer by input(0)
framework::OpKernelType UnaryOpInferKernelType(
    const framework::ExecutionContext& ctx) {
  framework::LibraryType library = framework::LibraryType::kPlain;
  framework::DataLayout layout = framework::DataLayout::kAnyLayout;
  auto x_name = ctx.GetInputNameByIdx(0);
// FIXME(liuwei1031) temporarily disable the code to unblock users
// TODO(liuwei1031) figure out the reason behind
// https://github.com/PaddlePaddle/Paddle/issues/16096
// and re-enable this in the future
// #ifdef PADDLE_WITH_CUDA
//   auto it1 = oper.Attrs().find("use_cudnn");
//   if (it1 != oper.Attrs().end() && platform::CanCUDNNBeUsed(ctx)) {
//     library = framework::LibraryType::kCUDNN;
//   }
// #endif
#ifdef PADDLE_WITH_MKLDNN
  if (library == framework::LibraryType::kPlain &&
      platform::CanMKLDNNBeUsed(ctx)) {
    library = framework::LibraryType::kMKLDNN;
    layout = framework::DataLayout::kMKLDNN;
  }
#endif
  return framework::OpKernelType(details::IndicateVarDataType(ctx, x_name),
                                 ctx.GetPlace(), layout, library);
}

// broadcast input(0) and input(1) -> output(0)
framework::OpKernelType BinaryOpInferKernelType(
    const framework::ExecutionContext& ctx) {
  auto x_name = ctx.GetInputNameByIdx(0);
  auto input_data_type = details::IndicateVarDataType(ctx, x_name);

#ifdef PADDLE_WITH_MKLDNN
  if (platform::CanMKLDNNBeUsed(ctx)) {
    return framework::OpKernelType(input_data_type, ctx.GetPlace(),
                                   framework::DataLayout::kMKLDNN,
                                   framework::LibraryType::kMKLDNN);
  }
#endif
  return framework::OpKernelType(input_data_type, ctx.GetPlace());
}

}  // namespace operators
}  // namespace paddle
