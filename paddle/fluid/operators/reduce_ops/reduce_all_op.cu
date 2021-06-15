// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/fluid/operators/reduce_ops/reduce_all_op.h"
#include "paddle/fluid/operators/reduce_ops/reduce_functor_op.h"
#include "paddle/fluid/operators/reduce_ops/reduce_op.cu.h"
#include "paddle/fluid/operators/reduce_ops/reduce_op.h"

namespace paddle {
namespace operators {

template <typename T>
class BoolReduceAllKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    bool reduce_all = context.Attr<bool>("reduce_all");
    auto* input = context.Input<Tensor>("X");
    auto* output = context.Output<Tensor>("Out");
    auto dims = context.Attr<std::vector<int>>("dim");

    std::vector<int> reduce_dims =
        detail::GetReduceDim(dims, input->dims().size(), reduce_all);

    auto stream = context.cuda_device_context().stream();
    TensorReduceFunc<T, T, CustomLogicalAnd>(*input, output, reduce_dims,
                                             stream);
  }
};

}  // namespace operators
}  // namespace paddle

REGISTER_OP_CUDA_KERNEL(reduce_all, ops::BoolReduceAllKernel<bool>);
