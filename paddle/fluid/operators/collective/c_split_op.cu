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

#include <vector>

#include "paddle/fluid/operators/collective/c_split_op.h"
#include "paddle/fluid/operators/math/concat_and_split.h"
#include "paddle/fluid/platform/cuda_primitives.h"

namespace paddle {
namespace operators {

static constexpr int kNumCUDAThreads = 512;
static constexpr int kNumMaxinumNumBlocks = 4096;

static inline int NumBlocks(const int N) {
  return std::min((N + kNumCUDAThreads - 1) / kNumCUDAThreads,
                  kNumMaxinumNumBlocks);
}

template <typename T>
__global__ void SplitFromRank(const T* input, T* output, const int rows,
                              const int columns, const int rank,
                              const int nranks, const int limit) {
  CUDA_KERNEL_LOOP(i, limit) {
    int row = i / columns;
    int col = i % columns;

    int block = columns / nranks;
    int start = block * rank;
    int end = start + block;

    if (col >= start && col < end) {
      int idx = block * row + col % block;
      output[idx] = input[i];
    }
  }
}

template <typename T>
class CSplitOpCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto x = ctx.Input<framework::Tensor>("X");
    auto out = ctx.Output<framework::Tensor>("Out");

    int nranks = ctx.Attr<int>("nranks");
    int rank = ctx.Attr<int>("rank");
    auto place = ctx.GetPlace();

    PADDLE_ENFORCE_GE(rank, 0, platform::errors::PreconditionNotMet(
                                   "The value of rank (%d) for c_split must be "
                                   "greater than or equal to 0.",
                                   rank));
    PADDLE_ENFORCE_GE(nranks, 2,
                      platform::errors::PreconditionNotMet(
                          "The value of nranks (%d) for c_split must be "
                          "greater than or equal to 2.",
                          nranks));
    PADDLE_ENFORCE_LT(rank, nranks,
                      platform::errors::PreconditionNotMet(
                          "The value of rank (%d) for c_split must be "
                          "less than that of nranks (%d).",
                          rank, nranks));

    auto& dev_ctx = ctx.template device_context<platform::CUDADeviceContext>();
    auto dims = x->dims();
    auto dims_size = dims.size();
    // final dim
    int64_t end_size = dims[dims_size - 1];

    // remain dim
    auto remain_ddim = framework::slice_ddim(dims, 0, dims_size - 1);
    int64_t remain_numel = framework::product(remain_ddim);

    int limit = x->numel();
    int blocks = NumBlocks(limit);
    int threads = kNumCUDAThreads;

    dims[dims_size - 1] /= nranks;
    out->mutable_data<T>(dims, place);

    SplitFromRank<T><<<blocks, threads, 0, dev_ctx.stream()>>>(
        x->data<T>(), out->data<T>(), remain_numel, end_size, rank, nranks,
        limit);
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_CUDA_KERNEL(c_split, ops::CSplitOpCUDAKernel<float>,
                        ops::CSplitOpCUDAKernel<double>,
                        ops::CSplitOpCUDAKernel<int>,
                        ops::CSplitOpCUDAKernel<int64_t>,
                        ops::CSplitOpCUDAKernel<plat::float16>);
