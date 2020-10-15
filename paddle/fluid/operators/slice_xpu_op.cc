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

#ifdef PADDLE_WITH_XPU

#include <algorithm>
#include <memory>
#include <string>
#include <vector>
#include "paddle/fluid/operators/slice_op.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

template <typename DeviceContext, typename T>
class SliceXPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto in = ctx.Input<framework::Tensor>("Input");
    auto out = ctx.Output<framework::Tensor>("Out");
    auto axes = ctx.Attr<std::vector<int>>("axes");
    auto starts = ctx.Attr<std::vector<int>>("starts");
    auto ends = ctx.Attr<std::vector<int>>("ends");
    auto in_dims = in->dims();

    // prepare starts, ends on XPU
    int dim_value = 0, start = 0, end = 0;
    // If a negative value is passed for any of the start or end indices,
    // it represents number of elements before the end of that dimension.
    // If the value passed to start or end is larger than the n
    // (the number of elements in this dimension), it represents n.
    for (size_t i = 0; i < axes.size(); ++i) {
      dim_value = in_dims[axes[i]];
      start = starts[i];
      end = ends[i];
      start = start < 0 ? (start + dim_value) : start;
      end = end < 0 ? (end + dim_value) : end;
      start = std::max(start, 0);
      end = std::max(end, 0);
      end = std::min(end, dim_value);
      PADDLE_ENFORCE_GT(end, start, platform::errors::InvalidArgument(
                                        "end should greater than start"));
      starts[i] = start;
      ends[i] = end;
    }
    size_t shape_size = in_dims.size();
    // the slice XPU kernel require that the length of `start`, `end` must be
    // equal
    // to the dims size of input tensor, therefore, if shape_size > axes.size(),
    // the `starts_extension` and `ends_extension` is necessary.
    std::vector<int> starts_extension(shape_size, 0);
    std::vector<int> ends_extension(shape_size, 0);
    if (shape_size > axes.size()) {
      for (size_t i = 0; i < shape_size; ++i) {
        ends_extension[i] = in_dims[i];
      }
      for (size_t i = 0; i < axes.size(); ++i) {
        starts_extension[axes[i]] = starts[i];
        ends_extension[axes[i]] = ends[i];
      }
    } else {
      starts_extension = std::move(starts);
      ends_extension = std::move(ends);
    }

    // prepare shape on XPU
    std::vector<int> shape(shape_size, 0);
    for (size_t i = 0; i < shape_size; ++i) {
      shape[i] = in_dims[i];
    }

    auto& dev_ctx = ctx.template device_context<DeviceContext>();
    auto* in_data = in->data<T>();
    auto* out_data = out->mutable_data<T>(ctx.GetPlace());

    int r = xpu::slice_forward(dev_ctx.x_context(), shape.data(),
                               starts_extension.data(), ends_extension.data(),
                               shape_size, in_data, out_data);
    PADDLE_ENFORCE_EQ(r, XPU_SUCCESS,
                      platform::errors::External("XPU slice kernel error!"));
  }
};

template <typename DeviceContext, typename T>
class SliceGradXPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* d_out = ctx.Input<framework::Tensor>(framework::GradVarName("Out"));
    auto* d_in = ctx.Output<framework::Tensor>(framework::GradVarName("Input"));
    d_in->mutable_data<T>(ctx.GetPlace());

    auto in_dims = d_in->dims();
    auto axes = ctx.Attr<std::vector<int>>("axes");
    auto starts = ctx.Attr<std::vector<int>>("starts");
    auto ends = ctx.Attr<std::vector<int>>("ends");

    // prepare starts, ends on XPU
    int dim_value = 0, start = 0, end = 0;
    // If a negative value is passed for any of the start or end indices,
    // it represents number of elements before the end of that dimension.
    // If the value passed to start or end is larger than the n
    // (the number of elements in this dimension), it represents n.
    for (size_t i = 0; i < axes.size(); ++i) {
      dim_value = in_dims[axes[i]];
      start = starts[i];
      end = ends[i];
      start = start < 0 ? (start + dim_value) : start;
      end = end < 0 ? (end + dim_value) : end;
      start = std::max(start, 0);
      end = std::max(end, 0);
      end = std::min(end, dim_value);
      PADDLE_ENFORCE_GT(end, start, platform::errors::InvalidArgument(
                                        "end should greater than start"));
      starts[i] = start;
      ends[i] = end;
    }
    size_t shape_size = in_dims.size();
    // the slice XPU kernel require that the length of `start`, `end` must be
    // equal
    // to the dims size of input tensor, therefore, if shape_size > axes.size(),
    // the `starts_extension` and `ends_extension` is necessary.
    std::vector<int> starts_extension(shape_size, 0);
    std::vector<int> ends_extension(shape_size, 0);
    if (shape_size > axes.size()) {
      for (size_t i = 0; i < shape_size; ++i) {
        ends_extension[i] = in_dims[i];
      }
      for (size_t i = 0; i < axes.size(); ++i) {
        starts_extension[axes[i]] = starts[i];
        ends_extension[axes[i]] = ends[i];
      }
    }
    int* starts_device = nullptr;
    int* ends_device = nullptr;
    int* starts_host =
        shape_size > axes.size() ? starts_extension.data() : starts.data();
    int* ends_host =
        shape_size > axes.size() ? ends_extension.data() : ends.data();
    PADDLE_ENFORCE_EQ(
        xpu_malloc((void**)(&starts_device), shape_size * sizeof(int)),
        XPU_SUCCESS, platform::errors::External("XPU has no enough memory"));
    PADDLE_ENFORCE_EQ(
        xpu_malloc((void**)(&ends_device), shape_size * sizeof(int)),
        XPU_SUCCESS, platform::errors::External("XPU has no enough memory"));
    memory::Copy(BOOST_GET_CONST(platform::XPUPlace, ctx.GetPlace()),
                 starts_device, platform::CPUPlace(), starts_host,
                 shape_size * sizeof(int));
    memory::Copy(BOOST_GET_CONST(platform::XPUPlace, ctx.GetPlace()),
                 ends_device, platform::CPUPlace(), ends_host,
                 shape_size * sizeof(int));

    // prepare shape on XPU
    std::vector<int> shape(shape_size, 0);
    for (size_t i = 0; i < shape_size; ++i) {
      shape[i] = in_dims[i];
    }
    int* shape_device = nullptr;
    PADDLE_ENFORCE_EQ(
        xpu_malloc((void**)(&shape_device), shape_size * sizeof(int)),
        XPU_SUCCESS, platform::errors::External("XPU has no enough memory"));
    memory::Copy(BOOST_GET_CONST(platform::XPUPlace, ctx.GetPlace()),
                 shape_device, platform::CPUPlace(), shape.data(),
                 shape_size * sizeof(int));

    auto& dev_ctx = ctx.template device_context<DeviceContext>();
    int r =
        xpu::slice_backward(dev_ctx.x_context(), shape_device, starts_device,
                            ends_device, shape_size, d_out->data<T>(),
                            d_in->data<T>(), d_in->numel(), d_out->numel());
    PADDLE_ENFORCE_EQ(r, XPU_SUCCESS,
                      platform::errors::External("xpu slice kernel error"));
    dev_ctx.Wait();
    // free device data
    xpu_free(shape_device);
    xpu_free(starts_device);
    xpu_free(ends_device);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_XPU_KERNEL(
    slice, ops::SliceXPUKernel<paddle::platform::XPUDeviceContext, float>);
REGISTER_OP_XPU_KERNEL(
    slice_grad,
    ops::SliceGradXPUKernel<paddle::platform::XPUDeviceContext, float>);
#endif
