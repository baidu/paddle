/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <iostream>
#include "paddle/fluid/operators/center_loss_op.h"
#include "paddle/fluid/platform/cuda_primitives.h"
#include "paddle/fluid/platform/gpu_info.h"
namespace paddle {
namespace operators {

using platform::PADDLE_CUDA_NUM_THREADS;

template <typename T, int BlockDimX, int BlockDimY, int GridDimX>
__global__ void ComputeDifferent(T *centers_diff, const T *X, const T *centers,
                                 const int64_t *ids, const int64_t N,
                                 const int64_t K, const int64_t D) {
  int idx = threadIdx.x;
  int idy = blockIdx.x + threadIdx.y * GridDimX;

  while (idy < K) {
    int64_t id = ids[idy];
    PADDLE_ENFORCE(id >= 0, "Id should larger than 0 but received id: %d.", id);
    PADDLE_ENFORCE(id < N, "Id should smaller than %d but received id: %d.", N,
                   id);

    T *out = centers_diff + idy * D;
    const T *x = X + idy * D;
    const T *cent = centers + id * D;
    for (int i = idx; i < D; i += BlockDimX) {
      out[i] = x[i] - cent[i];
    }
    idy += BlockDimY * GridDimX;
  }
}

template <typename T, int BlockDimX, int BlockDimY, int GridDimX>
__global__ void UpdateCenters(T *centers, T *centers_diff, const int64_t *ids,
                              const int64_t N, const int64_t K, const int64_t D,
                              const T *alpha) {
  int idx = threadIdx.x;
  int idy = blockIdx.x + threadIdx.y * GridDimX;
  int count;
  while (idy < K) {
    int count = 1;
    int64_t id = ids[idy];
    PADDLE_ENFORCE(id >= 0, "Id should larger than 0 but received id: %d.", id);
    PADDLE_ENFORCE(id < N, "Id should smaller than %d but received id: %d.", N,
                   id);

    for (int i = 0; i < K; i++) {
      if (ids[i] == id) {
        count++;
      }
    }
    const T *diff = centers_diff + idy * D;
    T *cent = centers + id * D;
    for (int i = idx; i < D; i += BlockDimX) {
      paddle::platform::CudaAtomicAdd(&cent[i], alpha[0] * diff[i] / count);
    }
    idy += BlockDimY * GridDimX;
  }
}

template <typename DeviceContext, typename T>
class CenterLossCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    auto &device_context = ctx.template device_context<DeviceContext>();
    auto stream = device_context.stream();
    auto *X = ctx.Input<Tensor>("X");  // deep feature
    auto *labels = ctx.Input<Tensor>("Label");
    auto *centers = ctx.Input<Tensor>("Centers");
    auto *update_rate = ctx.Input<Tensor>("CenterUpdateRate");
    int cluster_num = ctx.Attr<int>("cluster_num");
    auto *lr_center = update_rate->data<T>();
    bool need_update = static_cast<T>(ctx.Attr<bool>("need_update"));

    auto x_data = X->data<T>();
    auto label_data = labels->data<int64_t>();

    auto x_dims = X->dims();
    int batch_size = x_dims[0];
    const int deep_feat_dim = x_dims[1];

    auto *centers_diff = ctx.Output<Tensor>("SampleCenterDiff");
    auto centers_diff_data = centers_diff->mutable_data<T>(ctx.GetPlace());

    auto centers_data = centers->data<T>();
    auto centers_dim = centers->dims();
    auto *out_loss = ctx.Output<Tensor>("Loss");
    auto loss_data = out_loss->mutable_data<T>(ctx.GetPlace());

    auto *centers_out = ctx.Output<Tensor>("CentersOut");
    auto *centers_out_data = centers_out->mutable_data<T>(ctx.GetPlace());

    auto ctx_place = ctx.GetPlace();
    if (centers != centers_out) {
      framework::TensorCopy(
          *static_cast<const framework::Tensor *>(centers), ctx_place,
          *platform::DeviceContextPool::Instance().Get(ctx_place),
          static_cast<framework::Tensor *>(centers_out));
    }

    int64_t numel = X->numel();

    size_t N = centers->dims()[0];
    size_t D = centers->dims()[1];
    size_t K = labels->numel();

    dim3 threads(128, 8);
    dim3 grids(8, 1);

    ComputeDifferent<T, 128, 8, 8><<<grids, threads, 0, stream>>>(
        centers_diff_data, x_data, centers_data, label_data, N, K, D);

    auto &place = *ctx.template device_context<DeviceContext>().eigen_device();
    auto sub_result = EigenMatrix<T>::From(*centers_diff);

    auto sub_res_pow2 = (sub_result * sub_result) / T(2.0);
    auto z = EigenVector<T>::Flatten(*out_loss);
    z.device(place) = sub_res_pow2.sum(Eigen::array<int, 1>({{1}}));
    if (need_update) {
      UpdateCenters<T, 128, 8, 8><<<grids, threads, 0, stream>>>(
          centers_out_data, centers_diff_data, label_data, N, K, D, lr_center);
    }
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
using GPUCtx = paddle::platform::CUDADeviceContext;
REGISTER_OP_CUDA_KERNEL(center_loss, ops::CenterLossCUDAKernel<GPUCtx, float>,
                        ops::CenterLossCUDAKernel<GPUCtx, double>);

REGISTER_OP_CUDA_KERNEL(center_loss_grad,
                        ops::CenterLossGradKernel<GPUCtx, float>,
                        ops::CenterLossGradKernel<GPUCtx, double>);
