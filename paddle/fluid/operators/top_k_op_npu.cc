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

#include "paddle/fluid/operators/top_k_op.h"
#include "paddle/fluid/operators/npu_op_runner.h"

namespace paddle {
namespace operators {


template <typename DeviceContext, typename T>
class TopkNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* input = ctx.Input<framework::LoDTensor>("X");
    //auto* k = ctx.Input<framework::Tensor>("K");
    //size_t k = static_cast<int>(ctx.Attr<int>("k"));
    auto* k_t = ctx.Input<Tensor>("K");
    //if (k_t) {
    //  k = k_t->data<int>()[0];
    //}

    //framework::Tensor k_t;
    //framework::TensorCopySync(*k_t, platform::CPUPlace(), &k);

    //const auto& sorted = static_cast<bool>(ctx.Attr<bool>("sorted"));

    framework::AttributeMap attr_input = {{"sorted", false}};
    auto* output = ctx.Output<framework::LoDTensor>("Out");
    auto* indices = ctx.Output<framework::LoDTensor>("Indices");
    output->mutable_data<T>(ctx.GetPlace());
    indices->mutable_data<int>(ctx.GetPlace());

    auto runner = NpuOpRunner("TopK", {*input, *k_t}, {*output, *indices}, attr_input);

    auto stream =
        ctx.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();

    runner.Run(stream);
    /*
    std::cout << "after run "<<std::endl;
    framework::Tensor cpu_tensor;
    framework::TensorCopySync(*indices, platform::CPUPlace(), &cpu_tensor);
    auto* data = cpu_tensor.data<T>();
    auto vec_data = std::vector<T>(data, data + indices->numel());
    for(int i=0; i<static_cast<int>(vec_data.size()); ++i){
       VLOG(3) << " vec_data["<< i << "] = " << vec_data[i];
    }
    */
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_NPU_KERNEL(
    top_k,
    ops::TopkNPUKernel<paddle::platform::NPUDeviceContext, float>,
    ops::TopkNPUKernel<paddle::platform::NPUDeviceContext, double>,
    ops::TopkNPUKernel<paddle::platform::NPUDeviceContext, paddle::platform::float16>,
    ops::TopkNPUKernel<paddle::platform::NPUDeviceContext, int>,
    ops::TopkNPUKernel<paddle::platform::NPUDeviceContext, int64_t>);
#endif
