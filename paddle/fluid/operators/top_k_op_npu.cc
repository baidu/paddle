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
    auto* input = ctx.Input<framework::LoDTensor>("X");
    auto* output = ctx.Output<framework::LoDTensor>("Out");
    auto* indices = ctx.Output<framework::LoDTensor>("Indices");

    size_t k = static_cast<int>(ctx.Attr<int>("k"));
    bool largest = static_cast<bool>(ctx.Attr<bool>("largest"));
    int axis = static_cast<int>(ctx.Attr<int>("axis"));

    PADDLE_ENFORCE_EQ(
      axis, -1,
      platform::errors::InvalidArgument("TopKD only support axis == -1"));

    PADDLE_ENFORCE_EQ(
      largest, true,
      platform::errors::InvalidArgument("TopKD only support largest == true"));

    output->mutable_data<paddle::platform::float16>(ctx.GetPlace());
    indices->mutable_data<paddle::platform::float16>(ctx.GetPlace());


    // prepare assit
    auto dim = input->dims().size();
    framework::Tensor assist_seq_tensor;
    assist_seq_tensor.Resize({2 * dim});
    assist_seq_tensor.mutable_data<paddle::platform::float16>(ctx.GetPlace());
    gen_assist_seq(&assist_seq_tensor, dim, ctx);

    framework::NPUAttributeMap attr_input = {{"sorted", "true"},
                                             {"k", static_cast<int>(k)},
                                             {"dim", 1},
                                             {"largest", largest}};

    // run ascend
    auto runner = NpuOpRunner("TopKD",
                              {*input, assist_seq_tensor},
                              {*output, *indices},
                              attr_input);

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
