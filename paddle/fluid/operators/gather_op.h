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
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/gather.h"
#include "paddle/fluid/operators/scatter.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

template <typename T>
class GatherOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    PADDLE_ENFORCE_EQ(
        platform::is_cpu_place(ctx.GetPlace()), true,
        platform::errors::PreconditionNotMet("This kernel only runs on CPU."));

    auto *x = ctx.Input<Tensor>("X");
    auto *index = ctx.Input<Tensor>("Index");
    auto *output = ctx.Output<Tensor>("Out");

    int axis = ctx.Attr<int>("axis");
    // get axis from tensor
    if (ctx.HasInput("Axis")) {
      const Tensor *axis_tensor = ctx.Input<Tensor>("Axis");
      const auto &axis_type = axis_tensor->type();
      if (axis_type == framework::proto::VarType::INT32) {
        axis = static_cast<int>(axis_tensor->data<int32_t>()[0]);
      } else if (axis_type == framework::proto::VarType::INT64) {
        axis = static_cast<int>(axis_tensor->data<int64_t>()[0]);
      }
    }
    const auto &place = ctx.GetPlace();
    const auto &index_type = index->type();
    if (axis != 0) {
      if (index_type == framework::proto::VarType::INT32) {
        GatherV2Function<T, int32_t>(x, index, axis, output, place);
      } else if (index_type == framework::proto::VarType::INT64) {
        GatherV2Function<T, int64_t>(x, index, axis, output, place);
      }
      return;
    }

    output->mutable_data<T>(ctx.GetPlace());
    if (x->numel() == 0) return;
    if (index_type == framework::proto::VarType::INT32) {
      CPUGather<T, int>(ctx.device_context(), *x, *index, output);
    } else if (index_type == framework::proto::VarType::INT64) {
      CPUGather<T, int64_t>(ctx.device_context(), *x, *index, output);
    }
  }
};

template <typename T>
class GatherGradientOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    PADDLE_ENFORCE_EQ(
        platform::is_cpu_place(ctx.GetPlace()), true,
        platform::errors::PreconditionNotMet("This kernel only runs on CPU."));

    auto *index = ctx.Input<Tensor>("Index");
    auto *dX = ctx.Output<Tensor>(framework::GradVarName("X"));
    auto *dO = ctx.Input<Tensor>(framework::GradVarName("Out"));

    int axis = ctx.Attr<int>("axis");
    if (ctx.HasInput("Axis")) {
      const Tensor *axis_tensor = ctx.Input<Tensor>("Axis");
      const auto &axis_type = axis_tensor->type();
      if (axis_type == framework::proto::VarType::INT32) {
        axis = static_cast<int>(axis_tensor->data<int32_t>()[0]);
      } else if (axis_type == framework::proto::VarType::INT64) {
        axis = static_cast<int>(axis_tensor->data<int64_t>()[0]);
      }
    }
    const auto &index_type = index->type();

    if (axis != 0) {
      if (index_type == framework::proto::VarType::INT32) {
        GatherV2GradFunction<T, int32_t>(dO, index, axis, dX, ctx.GetPlace());
      } else if (index_type == framework::proto::VarType::INT64) {
        GatherV2GradFunction<T, int64_t>(dO, index, axis, dX, ctx.GetPlace());
      }
      return;
    }

    dX->mutable_data<T>(ctx.GetPlace());
    auto dxt = framework::EigenVector<T>::Flatten(*dX);
    auto &place = *ctx.template device_context<platform::CPUDeviceContext>()
                       .eigen_device();
    dxt.device(place) = dxt.constant(static_cast<T>(0));
    if (dO->numel() == 0) return;
    bool overwrite = ctx.Attr<bool>("overwrite");

    if (index_type == framework::proto::VarType::INT32) {
      if (overwrite) {
        ScatterAssign<T, int32_t>(ctx.device_context(), *dO, *index, dX);
      } else {
        ScatterAssignAdd<T, int32_t>(ctx, *dO, *index, dX);
      }
    } else if (index_type == framework::proto::VarType::INT64) {
      if (overwrite) {
        ScatterAssign<T, int64_t>(ctx.device_context(), *dO, *index, dX);
      } else {
        ScatterAssignAdd<T, int64_t>(ctx, *dO, *index, dX);
      }
    }
  }
};

}  // namespace operators
}  // namespace paddle
