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

#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/collective/c_embedding_op.h"
#include "paddle/fluid/platform/cuda_primitives.h"
#include "paddle/fluid/platform/float16.h"

namespace paddle {
namespace operators {

static constexpr int kNumCUDAThreads = 512;
static constexpr int kNumMaxinumNumBlocks = 4096;

static inline int NumBlocks(const int N) {
  return std::min((N + kNumCUDAThreads - 1) / kNumCUDAThreads,
                  kNumMaxinumNumBlocks);
}

template <typename T, typename IndexT>
__global__ void CEmbedding(T *out, const T *table, const IndexT *ids,
                           const int rows, const int columns, const int64_t N,
                           const int64_t start_idx, const int64_t end_idx,
                           const int64_t limit) {
  CUDA_KERNEL_LOOP(i, limit) {
    size_t row = i / columns;
    size_t col = i % columns;
    auto id = ids[row];

    if (id >= start_idx && id < end_idx) {
      auto real_idx = id - start_idx;
      PADDLE_ENFORCE(real_idx < N,
                     "The index is out of bounds, "
                     "please check whether the dimensions of index and "
                     "input meet the requirements. It should "
                     "be less than [%d], but received [%d]",
                     N, real_idx);
      out[i] = table[real_idx * columns + col];
    } else {
      out[i] = static_cast<T>(0);
    }
  }
}

template <typename T, typename IndexT>
__global__ void CEmbeddingGrad(T *table, const T *output, const IndexT *ids,
                               const int rows, const int columns,
                               const int64_t N, const int64_t start_idx,
                               const int64_t end_idx, const int64_t limit) {
  CUDA_KERNEL_LOOP(i, limit) {
    size_t row = i / columns;
    size_t col = i % columns;
    auto id = ids[row];
    if (id >= start_idx && id < end_idx) {
      auto real_idx = id - start_idx;
      paddle::platform::CudaAtomicAdd(&table[real_idx * columns + col],
                                      output[i]);
    }
  }
}

template <typename T>
class CEmbeddingCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &context) const override {
    auto *table_t = context.Input<LoDTensor>("W");
    auto *ids_t = context.Input<LoDTensor>("Ids");
    auto *output_t = context.Output<LoDTensor>("Out");

    const auto &dev_ctx =
        context.template device_context<platform::CUDADeviceContext>();
    const int64_t start_idx = context.Attr<int64_t>("start_index");
    size_t N = table_t->dims()[0];
    size_t D = table_t->dims()[1];
    size_t K = ids_t->numel();

    const int64_t end_idx = start_idx + N;

    auto *table = table_t->data<T>();
    auto *output = output_t->mutable_data<T>(context.GetPlace());

    auto limit = K * D;
    int blocks = NumBlocks(limit);
    int threads = kNumCUDAThreads;

    const auto &index_type = ids_t->type();
    if (index_type == framework::proto::VarType::INT32) {
      CEmbedding<T, int32_t><<<blocks, threads, 0, dev_ctx.stream()>>>(
          output, table, ids_t->data<int32_t>(), K, D, N, start_idx, end_idx,
          limit);

    } else if (index_type == framework::proto::VarType::INT64) {
      CEmbedding<T, int64_t><<<blocks, threads, 0, dev_ctx.stream()>>>(
          output, table, ids_t->data<int64_t>(), K, D, N, start_idx, end_idx,
          limit);
    }
  }
};

template <typename T>
class CEmbeddingGradCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &context) const override {
    const auto &dev_ctx =
        context.template device_context<platform::CUDADeviceContext>();
    const int64_t start_idx = context.Attr<int64_t>("start_index");
    auto ids_t = context.Input<LoDTensor>("Ids");
    auto d_output_t = context.Input<LoDTensor>(framework::GradVarName("Out"));
    auto d_table_t = context.Output<LoDTensor>(framework::GradVarName("W"));

    int N = d_table_t->dims()[0];
    int D = d_table_t->dims()[1];
    int K = ids_t->numel();

    const int64_t end_idx = start_idx + N;
    auto limit = K * D;
    int blocks = NumBlocks(limit);
    int threads = kNumCUDAThreads;

    const T *d_output = d_output_t->data<T>();
    T *d_table = d_table_t->mutable_data<T>(context.GetPlace());

    auto t = framework::EigenVector<T>::Flatten(*d_table_t);
    t.device(*dev_ctx.eigen_device()) = t.constant(static_cast<T>(0));

    const auto &index_type = ids_t->type();
    if (index_type == framework::proto::VarType::INT32) {
      CEmbeddingGrad<T, int32_t><<<blocks, threads, 0, dev_ctx.stream()>>>(
          d_table, d_output, ids_t->data<int32_t>(), K, D, N, start_idx,
          end_idx, limit);
    } else if (index_type == framework::proto::VarType::INT64) {
      CEmbeddingGrad<T, int64_t><<<blocks, threads, 0, dev_ctx.stream()>>>(
          d_table, d_output, ids_t->data<int64_t>(), K, D, N, start_idx,
          end_idx, limit);
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;
REGISTER_OP_CUDA_KERNEL(c_embedding, ops::CEmbeddingCUDAKernel<float>,
                        ops::CEmbeddingCUDAKernel<double>,
                        ops::CEmbeddingCUDAKernel<plat::float16>);
REGISTER_OP_CUDA_KERNEL(c_embedding_grad, ops::CEmbeddingGradCUDAKernel<float>,
                        ops::CEmbeddingGradCUDAKernel<double>,
                        ops::CEmbeddingGradCUDAKernel<plat::float16>);
