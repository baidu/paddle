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

#include "paddle/fluid/operators/fused/fusion_transpose_flatten_concat_op.h"
#include <vector>
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/platform/cudnn_helper.h"

namespace paddle {
namespace operators {

template <typename T>
using CudnnDataType = platform::CudnnDataType<T>;

template <typename T>
class TransposeFlattenConcatFusionKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto ins = ctx.MultiInput<framework::Tensor>("X");
    auto* out = ctx.Output<framework::Tensor>("Out");
    out->mutable_data<T>(ctx.GetPlace());
    auto odims = out->dims();

    std::vector<int> trans_axis = ctx.Attr<std::vector<int>>("trans_axis");
    int concat_axis = ctx.Attr<int>("concat_axis");

    int rank = ins[0]->dims().size();
    // use at least 4D in cudnnTransformTensor
    int max_dim = rank < 4 ? 4 : rank;
    std::vector<int> stride_x(max_dim, 0);
    std::vector<int> stride_y(max_dim, 0);
    std::vector<int> dims_y(max_dim, 0);

    cudnnTensorDescriptor_t in_desc;
    cudnnTensorDescriptor_t out_desc;
    CUDNN_ENFORCE(cudnnCreateTensorDescriptor(&in_desc));
    CUDNN_ENFORCE(cudnnCreateTensorDescriptor(&out_desc));
    cudnnDataType_t cudnn_dtype = CudnnDataType<T>::type;

    auto& dev_ctx = ctx.template device_context<platform::CUDADeviceContext>();
    auto handle = dev_ctx.cudnn_handle();

    T* odata = out->data<T>();
    for (size_t i = 0; i < ins.size(); ++i) {
      auto perm_shape = GetPermuteShape(trans_axis, ins[i]->dims());
      int osize = 1;
      auto idims = ins[i]->dims();
      for (int i = 0; i < rank; i++) {
        stride_x[i] = 1;
        for (int j = trans_axis[i] + 1; j < rank; j++) {
          stride_x[i] *= idims[j];
        }
        dims_y[i] = perm_shape[i];
        osize *= perm_shape[i];
      }
      stride_y[rank - 1] = 1;
      for (int i = rank - 2; i >= 1; i--) {
        stride_y[i] = stride_y[i + 1] * perm_shape[i + 1];
      }
      // Since concat is aftern flatten, the output is 2D tensor.
      // If concat_axis is 0, each input's permutated tensor is continuous.
      // If concat_axis is 1, the stride of 0-th dim of each input's
      // permutated tensor is odims()[1].
      stride_y[0] = concat_axis == 0 ? stride_y[1] * perm_shape[1] : odims[1];

      for (int i = rank; i < max_dim; i++) {
        stride_x[i] = 1;
        stride_y[i] = 1;
        dims_y[i] = 1;
      }
      CUDNN_ENFORCE(cudnnSetTensorNdDescriptor(in_desc, cudnn_dtype, max_dim,
                                               dims_y.data(), stride_x.data()));
      CUDNN_ENFORCE(cudnnSetTensorNdDescriptor(out_desc, cudnn_dtype, max_dim,
                                               dims_y.data(), stride_y.data()));

      CUDNN_ENFORCE(cudnnTransformTensor(
          handle, CudnnDataType<T>::kOne(), in_desc,
          static_cast<const void*>(ins[0]->data<T>()),
          CudnnDataType<T>::kZero(), out_desc, static_cast<void*>(odata)));
      if (concat_axis == 0) {
        odata += osize;
      }
    }
    CUDNN_ENFORCE(cudnnDestroyTensorDescriptor(in_desc));
    CUDNN_ENFORCE(cudnnDestroyTensorDescriptor(out_desc));
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(fusion_transpose_flatten_concat,
                        ops::TransposeFlattenConcatFusionKernel<float>,
                        ops::TransposeFlattenConcatFusionKernel<double>);
