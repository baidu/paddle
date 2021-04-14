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

#include "paddle/fluid/operators/truncated_gaussian_random_op.h"
#include <memory>
#include <string>
#include "paddle/fluid/operators/npu_op_runner.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

template <typename DeviceContext, typename T>
class TruncatedGaussianRandomNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    // TODO(zhiqiu): support dynamic shape and call ParameterizedTruncatedNormal
    std::vector<int> shape = ctx.Attr<std::vector<int>>("shape");
    Tensor shape_tensor(framework::proto::VarType::INT32);
    shape_tensor.mutable_data<int32_t>({static_cast<int>(shape.size())},
                                       ctx.GetPlace());
    TensorFromVector(shape, ctx.device_context(), &shape_tensor);
    float mean = ctx.Attr<float>("mean");
    Tensor mean_tensor(framework::proto::VarType::FP32);
    mean_tensor.mutable_data<float>({1}, ctx.GetPlace());
    TensorFromVector(std::vector<float>{mean}, ctx.device_context(),
                     &mean_tensor);

    float std = ctx.Attr<float>("std");
    Tensor std_tensor(framework::proto::VarType::FP32);
    std_tensor.mutable_data<float>({1}, ctx.GetPlace());
    TensorFromVector(std::vector<float>{std}, ctx.device_context(),
                     &std_tensor);

    int32_t seed_var = ctx.Attr<int32_t>("seed");

    Tensor min_tensor(framework::proto::VarType::FP32);
    min_tensor.mutable_data<float>({1}, ctx.GetPlace());
    float min_value = mean - std * 2.0;
    TensorFromVector(std::vector<float>{min_value}, ctx.device_context(),
                     &min_tensor);

    Tensor max_tensor(framework::proto::VarType::FP32);
    max_tensor.mutable_data<float>({1}, ctx.GetPlace());
    float max_value = mean + std * 2.0;
    TensorFromVector(std::vector<float>{max_value}, ctx.device_context(),
                     &max_tensor);

    auto* out = ctx.Output<framework::Tensor>("Out");
    out->mutable_data<T>(ctx.GetPlace());
    auto stream =
        ctx.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();
    auto runner = NpuOpRunner(
        "ParameterizedTruncatedNormal",
        {shape_tensor, mean_tensor, std_tensor, min_tensor, max_tensor}, {*out},
        {{"seed", seed_var}});
    runner.Run(stream);
  }
};

// NOTE(zhiqiu): actually, this is cpu version kernel, and we need to make the
// above
// npu version work in the future.
template <typename T>
class NPUTruncatedGaussianRandomKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    float mean = context.Attr<float>("mean");
    float std = context.Attr<float>("std");
    auto* tensor = context.Output<framework::Tensor>("Out");
    tensor->mutable_data<T>(context.GetPlace());

    Tensor cpu_tensor(tensor->type());
    cpu_tensor.Resize(tensor->dims());
    T* cpu_data = cpu_tensor.mutable_data<T>(platform::CPUPlace());
    std::uniform_real_distribution<T> dist(std::numeric_limits<float>::min(),
                                           1.0);
    TruncatedNormal<T> truncated_normal(mean, std);
    int64_t size = tensor->numel();

    unsigned int seed = static_cast<unsigned int>(context.Attr<int>("seed"));
    auto engine = framework::GetCPURandomEngine(seed);
    for (int64_t i = 0; i < size; ++i) {
      cpu_data[i] = truncated_normal(dist(*engine));
    }
    framework::TensorCopy(
        cpu_tensor, context.GetPlace(),
        context.template device_context<platform::DeviceContext>(), tensor);
    context.template device_context<paddle::platform::NPUDeviceContext>()
        .Wait();
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_NPU_KERNEL(truncated_gaussian_random,
                       ops::NPUTruncatedGaussianRandomKernel<float>);
