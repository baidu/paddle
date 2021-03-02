/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#ifdef PADDLE_WITH_ASCEND_CL
#include <memory>
#include <string>

#include "paddle/fluid/operators/fill_constant_op.h"
#include "paddle/fluid/operators/npu_op_runner.h"
#include "paddle/fluid/operators/utils.h"

namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
class FillConstantNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto data_type =
        static_cast<framework::proto::VarType::Type>(ctx.Attr<int>("dtype"));
    auto str_value = ctx.Attr<std::string>("str_value");
    auto float_value = ctx.Attr<float>("value");

    auto* out_var = ctx.Output<framework::Tensor>("Out");
    auto place = ctx.GetPlace();
    auto stream =
         ctx.template device_context<paddle::platform::NPUDeviceContext>()
             .stream();

    T value;
    if (str_value.empty()) {
      value = static_cast<T>(float_value);
    } else {
      // handle NaN/Inf first, which cannot be read from stream.
      if (str_value == "inf") {
        value = static_cast<T>(std::numeric_limits<double>::infinity());
      } else if (str_value == "-inf") {
        value = static_cast<T>(-std::numeric_limits<double>::infinity());
      } else if (str_value == "nan") {
        value = static_cast<T>(std::numeric_limits<double>::quiet_NaN());
      } else {
        std::stringstream convert_stream(str_value);
        if (std::is_same<int64_t, T>::value) {
          int64_t tmp_value;
          convert_stream >> tmp_value;
          value = static_cast<T>(tmp_value);
        } else {
          double tmp_value;
          convert_stream >> tmp_value;
          value = static_cast<T>(tmp_value);
        }
      }
    }
    //if (ctx.HasInput("ValueTensor")) {
    //  auto *value_tensor = ctx.Input<framework::Tensor>("ValueTensor");
    //  PADDLE_ENFORCE_EQ(
    //      value_tensor->numel(), 1,
    //      platform::errors::InvalidArgument(
    //          "When use Tensor as value to set Tensor value in fill_cosntant, "
    //          "value input(ValueTensor) size must be 1, but get %d",
    //          value_tensor->numel()));
    //  const T *tensor_data = value_tensor->data<T>();
    //  framework::Tensor cpu_tensor;
    //  auto tmp_place = value_tensor->place();
    //  if (platform::is_gpu_place(tmp_place) ||
    //      platform::is_xpu_place(tmp_place)) {
    //    TensorCopySync(*value_tensor, platform::CPUPlace(), &cpu_tensor);
    //    tensor_data = cpu_tensor.data<T>();
    //  }
    //  value = tensor_data[0];
    //}
    auto shape = GetShape(ctx);

    // Get the shape of x
    Tensor x_shape(framework::proto::VarType::INT32);
    x_shape.mutable_data<int32_t>({shape.size()}, ctx.GetPlace());
    TensorFromVector(framework::vectorize<int32_t>(shape),
                      ctx.device_context(), &x_shape);

    Tensor tensor_tmp(data_type);
    tensor_tmp.mutable_data<T>({1}, ctx.GetPlace());
    TensorFromVector(std::vector<T>{value},
                      ctx.device_context(), &tensor_tmp);
   

    //Tensor factor_bc_tensor(data_type);
    out_var->mutable_data<T>(shape, place);
    auto runner_bc = NpuOpRunner("BroadcastTo", {tensor_tmp, x_shape},
                                  {*out_var}, {});
    runner_bc.Run(stream);
   
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_NPU_KERNEL(
    fill_constant,
    ops::FillConstantNPUKernel<paddle::platform::NPUDeviceContext, float>);
#endif
