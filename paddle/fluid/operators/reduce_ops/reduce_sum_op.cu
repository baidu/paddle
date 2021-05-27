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

#include "paddle/fluid/operators/reduce_ops/cub_reduce.h"
#include "paddle/fluid/operators/reduce_ops/reduce_sum_op.h"

namespace paddle {
namespace operators {

struct IdentityFunctor {
  HOSTDEVICE explicit inline IdentityFunctor() {}

  template <typename T2>
  HOSTDEVICE inline T2 operator()(const T2& x) const {
    return x;
  }
};

template <typename T>
class ReduceSumKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    bool reduce_all = context.Attr<bool>("reduce_all");
    auto* input = context.Input<Tensor>("X");
    auto* output = context.Output<Tensor>("Out");
    auto out_dtype = context.Attr<int>("out_dtype");

    auto dims = context.Attr<std::vector<int>>("dim");
    bool keep_dim = context.Attr<bool>("keep_dim");

    std::vector<int> reduce_dims;
    if (reduce_all) {
      reduce_dims.resize(input->dims().size());
      for (int i = 0; i < reduce_dims.size(); ++i) reduce_dims[i] = i;
    } else {
      for (auto e : dims) {
        reduce_dims.push_back(e >= 0 ? e : e + input->dims().size());
      }
    }

    int reduce_num = 1;
    for (int i = 0; i < reduce_dims.size(); ++i) {
      reduce_num *= input->dims()[reduce_dims[i]];
    }

    auto stream = context.cuda_device_context().stream();
    if (out_dtype >= 0) {
      framework::VisitDataTypeSmall(
          static_cast<framework::proto::VarType::Type>(out_dtype),
          TensorReduceFunctor<T, cub::Sum, IdentityFunctor>(
              *input, output, reduce_dims, static_cast<double>(0.0), cub::Sum(),
              IdentityFunctor(), stream));
    } else {
      TensorReduce<T, T, cub::Sum, IdentityFunctor>(
          *input, output, reduce_dims, static_cast<T>(0), cub::Sum(),
          IdentityFunctor(), stream);
    }
  }
};

}  // namespace operators
}  // namespace paddle

REGISTER_OP_CUDA_KERNEL(reduce_sum, ops::ReduceSumKernel<bool>,
                        ops::ReduceSumKernel<float>,
                        ops::ReduceSumKernel<paddle::platform::float16>,
                        ops::ReduceSumKernel<double>, ops::ReduceSumKernel<int>,
                        ops::ReduceSumKernel<int64_t>,
                        ops::ReduceSumKernel<paddle::platform::complex64>,
                        ops::ReduceSumKernel<paddle::platform::complex128>);
