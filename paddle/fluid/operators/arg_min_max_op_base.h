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

#pragma once
#include <string>
#include <type_traits>
#include <vector>
#include "paddle/fluid/framework/ddim.h"
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/string/printf.h"

namespace paddle {
namespace operators {

enum ArgMinMaxType { kArgMin, kArgMax };

template <typename DeviceContext, typename T, typename Tout, int64_t Rank,
          ArgMinMaxType argMinMaxValue>
struct ArgMinMaxFunctor {};

#define DECLARE_ARG_MIN_MAX_FUNCTOR(eigen_op_type, enum_argminmax_value)      \
  template <typename DeviceContext, typename T, typename Tout, int64_t Rank>  \
  struct ArgMinMaxFunctor<DeviceContext, T, Tout, Rank,                       \
                          enum_argminmax_value> {                             \
    void operator()(const DeviceContext& ctx, const framework::LoDTensor& in, \
                    framework::LoDTensor* out, framework::DDim x_dims,        \
                    int64_t axis, bool keepdims) {                            \
      auto in_eigen = framework::EigenTensor<T, Rank>::From(in, x_dims);      \
      if (keepdims) {                                                         \
        auto out_eigen = framework::EigenTensor<Tout, Rank>::From(*out);      \
        out_eigen.device(*(ctx.eigen_device())) =                             \
            in_eigen.eigen_op_type(axis).template cast<Tout>();               \
      } else {                                                                \
        auto out_eigen = framework::EigenTensor<Tout, Rank - 1>::From(*out);  \
        out_eigen.device(*(ctx.eigen_device())) =                             \
            in_eigen.eigen_op_type(axis).template cast<Tout>();               \
      }                                                                       \
    }                                                                         \
  }

DECLARE_ARG_MIN_MAX_FUNCTOR(argmin, ArgMinMaxType::kArgMin);
DECLARE_ARG_MIN_MAX_FUNCTOR(argmax, ArgMinMaxType::kArgMax);

template <typename DeviceContext, typename T, ArgMinMaxType EnumArgMinMaxValue>
struct VisitDataArgMinMaxFunctor {
  const framework::ExecutionContext& ctx;

  explicit VisitDataArgMinMaxFunctor(const framework::ExecutionContext& ctx)
      : ctx(ctx) {}
  template <typename Tout>
  void apply() const {
    auto& x = *(ctx.Input<framework::LoDTensor>("X"));
    auto& out = *(ctx.Output<framework::LoDTensor>("Out"));
    out.template mutable_data<Tout>(ctx.GetPlace());
    auto axis = ctx.Attr<int64_t>("axis");
    auto keepdims = ctx.Attr<bool>("keepdims");
    const bool& flatten = ctx.Attr<bool>("flatten");
    // paddle do not have the scalar tensor, just return the shape [1] tensor
    if (flatten) {
      keepdims = true;
    }

    // if flatten, will construct the new dims for the cacluate
    framework::DDim x_dims;
    if (flatten) {
      x_dims = framework::make_ddim({x.numel()});
      // if flatten, the axis just as 0
      axis = 0;
    } else {
      x_dims = x.dims();
      if (axis < 0) axis += x_dims.size();
    }
    auto& dev_ctx = ctx.template device_context<DeviceContext>();

#define CALL_ARG_MINMAX_FUNCTOR(rank)                                \
  ArgMinMaxFunctor<DeviceContext, T, Tout, rank, EnumArgMinMaxValue> \
      functor##rank;                                                 \
  functor##rank(dev_ctx, x, &out, x_dims, axis, keepdims)

    switch (x_dims.size()) {
      case 1:
        CALL_ARG_MINMAX_FUNCTOR(1);
        break;
      case 2:
        CALL_ARG_MINMAX_FUNCTOR(2);
        break;
      case 3:
        CALL_ARG_MINMAX_FUNCTOR(3);
        break;
      case 4:
        CALL_ARG_MINMAX_FUNCTOR(4);
        break;
      case 5:
        CALL_ARG_MINMAX_FUNCTOR(5);
        break;
      case 6:
        CALL_ARG_MINMAX_FUNCTOR(6);
        break;
      default:
        PADDLE_THROW(
            "%s operator doesn't supports tensors whose ranks are greater "
            "than 6.",
            (EnumArgMinMaxValue == kArgMin ? "argmin" : "argmax"));
        break;
#undef CALL_ARG_MINMAX_FUNCTOR
    }
  }
};

template <typename DeviceContext, typename T, ArgMinMaxType EnumArgMinMaxValue>
class ArgMinMaxKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto& dtype = ctx.Attr<int>("dtype");
    if (dtype < 0) {
      framework::VisitDataTypeInt(
          static_cast<framework::proto::VarType::Type>(
              framework::proto::VarType::INT64),
          VisitDataArgMinMaxFunctor<DeviceContext, T, EnumArgMinMaxValue>(ctx));
      return;
    }
    framework::VisitDataTypeInt(
        static_cast<framework::proto::VarType::Type>(dtype),
        VisitDataArgMinMaxFunctor<DeviceContext, T, EnumArgMinMaxValue>(ctx));
  }
};

template <typename DeviceContext, typename T>
using ArgMinKernel = ArgMinMaxKernel<DeviceContext, T, ArgMinMaxType::kArgMin>;

template <typename DeviceContext, typename T>
using ArgMaxKernel = ArgMinMaxKernel<DeviceContext, T, ArgMinMaxType::kArgMax>;

class ArgMinMaxOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "arg_min_max");
    OP_INOUT_CHECK(ctx->HasOutput("Out"), "Output", "Out", "arg_min_max");
    const auto& x_dims = ctx->GetInputDim("X");
    int64_t axis = ctx->Attrs().Get<int64_t>("axis");
    bool keepdims = ctx->Attrs().Get<bool>("keepdims");
    const bool& flatten = ctx->Attrs().Get<bool>("flatten");
    auto x_rank = x_dims.size();

    PADDLE_ENFORCE_GE(axis, -x_dims.size(),
                      platform::errors::InvalidArgument(
                          "'axis'(%d) must be greater than or equal to"
                          " -Rank(X)(%d).",
                          axis, -x_dims.size()));
    PADDLE_ENFORCE_LT(
        axis, x_dims.size(),
        platform::errors::InvalidArgument(
            "'axis'(%d) must be less than Rank(X)(%d).", axis, x_dims.size()));

    if (axis < 0) axis += x_rank;

    // check the element num of sorted axis is not be larger than INT_MAX
    if (ctx->IsRuntime()) {
      const int& dtype = ctx->Attrs().Get<int>("dtype");
      if (dtype == framework::proto::VarType::INT32) {
        int64_t all_element_num = 0;
        if (flatten) {
          all_element_num = framework::product(x_dims);
        } else {
          all_element_num = x_dims[axis];
        }
        PADDLE_ENFORCE_LE(
            all_element_num, INT_MAX,
            "The element num of the argmin/argmax input at axis is "
            "%d, is larger than int32 maximum value:%d, you must "
            "set the dtype of argmin/argmax to 'int64'.",
            all_element_num, INT_MAX);
      }
    }

    std::vector<int64_t> vec;
    if (flatten) {
      // if is flatten, will return the only on element
      vec.emplace_back(static_cast<int64_t>(1));
    } else {
      for (int64_t i = 0; i < axis; i++) vec.emplace_back(x_dims[i]);
      if (keepdims) {
        vec.emplace_back(static_cast<int64_t>(1));
      }
      for (int64_t i = axis + 1; i < x_rank; i++) vec.emplace_back(x_dims[i]);
    }
    ctx->SetOutputDim("Out", framework::make_ddim(vec));
  }
};

class BaseArgMinMaxOpMaker : public framework::OpProtoAndCheckerMaker {
 protected:
  virtual const char* OpName() const = 0;
  virtual const char* Name() const = 0;

 public:
  void Make() override {
    AddInput("X", "Input tensor.");
    AddOutput("Out", "Output tensor.");
    AddAttr<int64_t>("axis", "The axis in which to compute the arg indics.");
    AddAttr<bool>("keepdims", "Keep the dim that to reduce.").SetDefault(false);
    AddAttr<bool>("flatten",
                  "Flatten the input value, and search the min or max indices")
        .SetDefault(false);
    AddAttr<int>("dtype",
                 "(int, 3), the dtype of indices, the indices dtype must be "
                 "int32, int64."
                 "default dtype is int64, and proto value is 3.")
        .SetDefault(3)
        .InEnum({2, 3});
    AddComment(string::Sprintf(R"DOC(
      %s Operator.

      Computes the indices of the %s elements of the input tensor's element
      along the provided axis.
)DOC",
                               OpName(), Name()));
  }
};

class ArgMinOpMaker : public BaseArgMinMaxOpMaker {
 protected:
  const char* OpName() const override { return "ArgMin"; }
  const char* Name() const override { return "min"; }
};

class ArgMaxOpMaker : public BaseArgMinMaxOpMaker {
 protected:
  const char* OpName() const override { return "ArgMax"; }
  const char* Name() const override { return "max"; }
};
}  // namespace operators
}  // namespace paddle
