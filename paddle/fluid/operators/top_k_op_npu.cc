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

#include <memory>
#include <string>

#include "paddle/fluid/operators/top_k_op.h"
#include "paddle/fluid/operators/npu_op_runner.h"

namespace paddle {
namespace operators {

void gen_assist_seq(framework::Tensor* assit_tensor,
                     int64_t dim, const framework::ExecutionContext& ctx) {
  const int64_t dimx2 = dim;
  std::vector<paddle::platform::float16> assit;
  assit.resize(2 * dimx2);
  for (int64_t i = 0; i < dimx2; i++) {
    // for i in range [0, dim]
    assit[i] = static_cast<paddle::platform::float16>(i);

    // for i in range [dim, dimx2]
    int64_t idx = static_cast<int64_t>(
                        static_cast<paddle::platform::float16>(i));
    int64_t gap = i - idx;
    assit[i + dim] = static_cast<paddle::platform::float16>(gap);
  }
  framework::TensorFromVector(assit, ctx.device_context(), assit_tensor);
}


template <typename DeviceContext, typename T>
class TopkNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    // read input
std::cout << "bbbbbbbbbbbbbbbbbb" << std::endl;
    auto* input = ctx.Input<framework::LoDTensor>("X");
    auto* output = ctx.Output<framework::LoDTensor>("Out");
    auto* indices = ctx.Output<framework::LoDTensor>("Indices");
std::cout << "aaaaaaaaaaaaaaaaaaaaa" << std::endl;

    size_t k = static_cast<int>(ctx.Attr<int>("k"));
std::cout << "000000000000000011" << std::endl;

    output->mutable_data<paddle::platform::float16>(ctx.GetPlace());
    indices->mutable_data<int>(ctx.GetPlace());
std::cout << "11111111" << std::endl;

    Tensor k_tensor(framework::proto::VarType::INT32);
std::cout << "2222222222222" << std::endl;
    k_tensor.Resize({1});
    k_tensor.mutable_data<int32_t>(ctx.GetPlace());
std::cout << "333333333333333" << std::endl;
    framework::NPUAttributeMap const_attr_input = {{"value", static_cast<int>(k)}};
std::cout << "44444444444444444" << std::endl;
    auto runner_const = NpuOpRunner("Const", {}, {k_tensor}, const_attr_input);
    auto stream_const =
        ctx.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();
std::cout << "55555555555" << std::endl;
    runner_const.Run(stream_const);


std::cout << "6666666666666666" << std::endl;
    // run ascend
    auto runner = NpuOpRunner("TopK",
                              {*input, k_tensor},
                              {*output, *indices},
                              {});

    auto stream =
        ctx.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();

    runner.Run(stream);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_NPU_KERNEL(
    top_k,
    ops::TopkNPUKernel<paddle::platform::NPUDeviceContext,
                                  paddle::platform::float16>);

