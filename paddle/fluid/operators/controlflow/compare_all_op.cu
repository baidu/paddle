/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <thrust/fill.h>
#include "paddle/fluid/operators/controlflow/compare_all_op.h"
#include "paddle/fluid/operators/reduce_ops/cub_reduce.h"
namespace paddle {
namespace operators {

template <typename T>
struct IdentityFunctor {
  HOSTDEVICE explicit inline IdentityFunctor() {}

  HOSTDEVICE inline T operator()(const T& x) const { return x; }
};

struct BitwiseAdd {
  // Bitwise add operator, returns <tt>a + b</tt>
  template <typename T>
  __host__ __device__ __forceinline__ T operator()(const T& a,
                                                   const T& b) const {
    return a & b;
  }
};
template <typename DeviceContext, typename Functor>
class CompareReduceOpKernel
    : public framework::OpKernel<typename Functor::ELEM_TYPE> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    using T = typename Functor::ELEM_TYPE;
    using Tensor = framework::Tensor;

    auto* x = context.Input<Tensor>("X");
    auto* y = context.Input<Tensor>("Y");
    auto* z = context.Output<Tensor>("Out");
    bool shape_same = true;

    Tensor tmp;
    framework::DDim x_dims = x->dims();
    framework::DDim y_dims = y->dims();

    if (x_dims.size() != y_dims.size()) {
      shape_same = false;
    } else {
      for (auto i = 0; i < x_dims.size(); i++) {
        if (x_dims[i] != y_dims[i]) {
          shape_same = false;
          break;
        }
      }
    }

    bool* z_data = z->mutable_data<bool>(context.GetPlace());
    if (!shape_same) {
      thrust::device_ptr<bool> z_dev_ptr(z_data);
      thrust::fill(z_dev_ptr, z_dev_ptr + 1, false);
      return;
    } else {
      tmp.mutable_data<bool>(x_dims, context.GetPlace());
      ElementwiseComputeEx<Functor, DeviceContext, T, bool>(context, x, y, 0,
                                                            Functor(), &tmp);
      // Reduce by 'bitwise and' operator
      std::vector<int> reduce_dims;
      reduce_dims.resize(tmp.dims().size());
      for (int i = 0; i < reduce_dims.size(); ++i) reduce_dims[i] = i;
      auto stream = context.cuda_device_context().stream();
      TensorReduce<bool, bool, BitwiseAdd, IdentityFunctor<bool>>(
          tmp, z, reduce_dims, true, BitwiseAdd(), IdentityFunctor<bool>(),
          stream);
    }
  }
};

}  // namespace operators
}  // namespace paddle

#define REGISTER_COMPARE_REDUCE_CUDA_KERNEL(op_type, functor)          \
  REGISTER_OP_CUDA_KERNEL(                                             \
      op_type, paddle::operators::CompareReduceOpKernel<               \
                   paddle::platform::CUDADeviceContext, functor<int>>, \
      paddle::operators::CompareReduceOpKernel<                        \
          paddle::platform::CUDADeviceContext, functor<int64_t>>,      \
      paddle::operators::CompareReduceOpKernel<                        \
          paddle::platform::CUDADeviceContext, functor<float>>,        \
      paddle::operators::CompareReduceOpKernel<                        \
          paddle::platform::CUDADeviceContext, functor<double>>);
REGISTER_COMPARE_REDUCE_CUDA_KERNEL(equal_all,
                                    paddle::operators::EqualReduceFunctor);
