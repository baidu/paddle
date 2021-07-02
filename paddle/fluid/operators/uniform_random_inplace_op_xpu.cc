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

#ifdef PADDLE_WITH_XPU

#include "paddle/fluid/framework/generator.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/operators/uniform_random_op.h"

namespace paddle {
namespace operators {

template <typename T>
class XPUUniformRandomInplaceKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    auto out_var = ctx.OutputVar("Out");
    auto *tensor = out_var->GetMutable<framework::LoDTensor>();
    T *data = tensor->mutable_data<T>(ctx.GetPlace());

    int64_t size = tensor->numel();
    std::unique_ptr<T[]> data_cpu(new T[size]);

    UniformRealDistribution<T>(
        data_cpu, size, ctx.Attr<float>("min"), ctx.Attr<float>("max"),
        static_cast<unsigned int>(ctx.Attr<int>("seed")));
    unsigned int diag_num =
        static_cast<unsigned int>(ctx.Attr<int>("diag_num"));
    unsigned int diag_step =
        static_cast<unsigned int>(ctx.Attr<int>("diag_step"));
    auto diag_val = static_cast<T>(ctx.Attr<float>("diag_val"));
    if (diag_num > 0) {
      PADDLE_ENFORCE_GT(
          size, (diag_num - 1) * (diag_step + 1),
          platform::errors::InvalidArgument(
              "ShapeInvalid: the diagonal's elements is equal (num-1) "
              "* (step-1) with num %d, step %d,"
              "It should be smaller than %d, but received %d",
              diag_num, diag_step, (diag_num - 1) * (diag_step + 1), size));
      for (int64_t i = 0; i < diag_num; ++i) {
        int64_t pos = i * diag_step + i;
        data_cpu[pos] = diag_val;
      }
    }
    memory::Copy(BOOST_GET_CONST(platform::XPUPlace, ctx.GetPlace()), data,
                 platform::CPUPlace(), reinterpret_cast<void *>(data_cpu.get()),
                 size * sizeof(T));
  }
};

}  // namespace operators
}  // namespace paddle

REGISTER_OP_XPU_KERNEL(uniform_random_inplace,
                       paddle::operators::XPUUniformRandomInplaceKernel<float>);

#endif  // PADDLE_WITH_XPU
