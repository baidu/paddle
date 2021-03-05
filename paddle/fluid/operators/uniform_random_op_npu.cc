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

#include "paddle/fluid/operators/uniform_random_op.h"
#include "paddle/fluid/operators/npu_op_runner.h"

namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
class UniformRandomNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto shape = ctx.Attr<std::vector<int64_t>>("shape");
    int shape_size = static_cast<int>(shape.size());
    auto min_v = ctx.Attr<float>("min");
    auto max_v = ctx.Attr<float>("max");
    auto seed1 = ctx.Attr<int>("seed");

    auto seed = static_cast<int>(min_v+max_v+seed1);
    VLOG(3) << seed;

    auto stream =
        ctx.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();
    // auto dtype = ctx.Attr<>("dtype");
    // how to define dtype for ascend
    // ACL_FLOAT
    framework::AttributeMap attr_input = {{"dtype", ACL_FLOAT}};//, {"seed", seed}, {"seed2", seed}};
    framework::Tensor shape_tensor;
    shape_tensor.Resize(framework::make_ddim({shape_size}));
    shape_tensor.mutable_data<int64_t>(ctx.GetPlace());
    framework::TensorFromVector(shape, ctx.device_context(), &shape_tensor);

    framework::Tensor const_shape_tensor;
    const_shape_tensor.Resize(framework::make_ddim({shape_size}));
    const_shape_tensor.mutable_data<int64_t>(ctx.GetPlace());

    auto const_runner = NpuOpRunner("Const", {shape_tensor}, {const_shape_tensor}, {});
    const_runner.Run(stream);


    // how to define a tmp tensor for ascend
    //framework::LoDTensor ur_out;
    //ur_out.mutable_data<T>(ctx.GetPlace());

    auto* out = ctx.Output<framework::LoDTensor>("Out");
    out->Resize(framework::make_ddim(shape));
    out->mutable_data<T>(ctx.GetPlace());

    auto ur_runner = NpuOpRunner("RandomUniform", {const_shape_tensor}, {*out}, attr_input);
    ur_runner.Run(stream);

    /*
    auto scale = max_v - min_v;
    float power = static_cast<float>(1.0);

    auto* out = ctx.Output<framework::LoDTensor>("Out");
    out->mutable_data<T>(ctx.GetPlace());

    auto pow_runner = NpuOpRunner("Power", {ur_out}, {*out},
                                   {{"power", power},
                                    {"scale", scale},
                                    {"shift", min_v}});

    pow_runner.Run(stream);
    */
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_NPU_KERNEL(
    uniform_random,
    ops::UniformRandomNPUKernel<paddle::platform::NPUDeviceContext, float>,
    ops::UniformRandomNPUKernel<paddle::platform::NPUDeviceContext, double>);
#endif
