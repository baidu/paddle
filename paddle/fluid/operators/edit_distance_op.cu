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

#include <algorithm>
#include "paddle/fluid/framework/mixed_vector.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/edit_distance_op.h"
#include "paddle/fluid/operators/math/math_function.h"
#include "paddle/fluid/platform/cuda_primitives.h"
#include "paddle/fluid/platform/gpu_info.h"

namespace paddle {
namespace operators {

using platform::PADDLE_CUDA_NUM_THREADS;

template <typename T>
__global__ void FillFirstRow(T* dist, const int N) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx < N + 1) {
    dist[idx] = idx;
  }
}

template <typename T>
__global__ void FillFirstColumn(T* dist, const int M, const int N) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx < M + 1) {
    dist[idx * (N + 1)] = idx;
  }
}

template <typename T>
__global__ void Levenshtein(T* dist, const int64_t* x1, const int64_t* x2,
                            const int M, const int N, const int start) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  int offset = N;
  int index = start + idx * offset;
  int row = index / (N + 1);
  int col = index % (N + 1);
  if (row > 0 && col > 0 && row < M + 1 && col < N + 1) {
    int cost = x1[row - 1] == x2[col - 1] ? 0 : 1;
    int dels = dist[(row - 1) * (N + 1) + col] + 1;
    int ins = dist[row * (N + 1) + col - 1] + 1;
    int subs = dist[(row - 1) * (N + 1) + (col - 1)] + cost;
    dist[index] = min(dels, min(ins, subs));
  }
}

template <typename T>
__global__ void SetOutput(T* out, const T* dist, const int M, const int N,
                          bool normalized) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx == 0) {
    out[0] = normalized ? dist[M * (N + 1) + N] / N : dist[M * (N + 1) + N];
  }
}

template <typename Place, typename T>
class EditDistanceGPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const {
    auto* out_t = ctx.Output<framework::Tensor>("Out");

    auto* x1_t = ctx.Input<framework::LoDTensor>("Hyps");
    auto* x2_t = ctx.Input<framework::LoDTensor>("Refs");
    auto* sequence_num = ctx.Output<framework::Tensor>("SequenceNum");
    sequence_num->mutable_data<int64_t>(ctx.GetPlace());
    auto batch_size = x1_t->dims()[0];

    auto normalized = ctx.Attr<bool>("normalized");
    auto stream = reinterpret_cast<const platform::CUDADeviceContext&>(
                      ctx.device_context())
                      .stream();

    framework::Vector<size_t> hyp_lod(batch_size + 1);
    framework::Vector<size_t> ref_lod(batch_size + 1);

    bool use_length = ctx.HasInput("HypsLength");

    if (use_length) {
      // build lod when using padding
      auto* hyp_length = ctx.Input<framework::Tensor>("HypsLength");
      auto* ref_length = ctx.Input<framework::Tensor>("RefsLength");

      framework::Tensor hyp_length_cpu;
      framework::Tensor ref_length_cpu;
      framework::TensorCopy(*hyp_length, platform::CPUPlace(), &hyp_length_cpu);
      framework::TensorCopy(*ref_length, platform::CPUPlace(), &ref_length_cpu);

      for (auto i = 0; i < batch_size; i++) {
        hyp_lod[i + 1] = hyp_lod[i] + hyp_length_cpu.data<int64_t>()[i];
        ref_lod[i + 1] = ref_lod[i] + ref_length_cpu.data<int64_t>()[i];
      }

    } else {
      hyp_lod = x1_t->lod()[0];
      ref_lod = x2_t->lod()[0];
    }

    if (normalized) {
      for (size_t i = 1; i < ref_lod.size(); ++i) {
        PADDLE_ENFORCE_GT(ref_lod[i], ref_lod[i - 1],
                          platform::errors::InvalidArgument(
                              "Reference string %d is empty.", i));
      }
    }

    const size_t num_strs = hyp_lod.size() - 1;
    math::SetConstant<platform::CUDADeviceContext, int64_t> set_constant;
    set_constant(ctx.template device_context<platform::CUDADeviceContext>(),
                 sequence_num, static_cast<int64_t>(num_strs));

    out_t->Resize({static_cast<int64_t>(num_strs), 1});
    out_t->mutable_data<T>(ctx.GetPlace());
    auto out = out_t->data<T>();

    T distance = 0.0;
    for (size_t num = 0; num < num_strs; num++) {
      auto m = static_cast<int64_t>(hyp_lod[num + 1] - hyp_lod[num]);
      auto n = static_cast<int64_t>(ref_lod[num + 1] - ref_lod[num]);
      if (m == 0 || n == 0) {
        distance = std::max(m, n);
        if (normalized) {
          distance = distance / n;
        }
        memory::Copy(BOOST_GET_CONST(Place, ctx.GetPlace()), out + num,
                     platform::CPUPlace(), &distance, sizeof(T), stream);
      } else {
        framework::Tensor dist_t;
        dist_t.Resize({m + 1, n + 1});
        dist_t.mutable_data<T>(ctx.GetPlace());
        auto dist = dist_t.data<T>();
        auto hyp_offset = use_length ? num * x1_t->dims()[1] : hyp_lod[num];
        auto ref_offset = use_length ? num * x2_t->dims()[1] : ref_lod[num];
        auto x1 = x1_t->data<int64_t>() + hyp_offset;
        auto x2 = x2_t->data<int64_t>() + ref_offset;

        FillFirstColumn<T><<<1 + m / PADDLE_CUDA_NUM_THREADS,
                             PADDLE_CUDA_NUM_THREADS, 0, stream>>>(dist, m, n);

        FillFirstRow<T><<<1 + n / PADDLE_CUDA_NUM_THREADS,
                          PADDLE_CUDA_NUM_THREADS, 0, stream>>>(dist, n);

        // Compute the elements of distance matrix in the anti-diagonal diretion
        for (int64_t slice = 2; slice < m + n + 1; ++slice) {
          int z_m = slice < m + 1 ? 0 : slice - m;
          int z_n = slice < n + 1 ? 0 : slice - n;
          int size = slice - (z_m + z_n) + 1;  // number of elments in the same
                                               // anti-diagonal line to update
          // the start index at which computes from
          int start = slice < n + 1 ? slice : (z_n + 1) * (n + 1) - 1;
          Levenshtein<T><<<1 + (size - 1) / PADDLE_CUDA_NUM_THREADS,
                           PADDLE_CUDA_NUM_THREADS, 0, stream>>>(dist, x1, x2,
                                                                 m, n, start);
        }
        SetOutput<T><<<1, 1, 0, stream>>>(out + num, dist, m, n, normalized);
      }
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_CUDA_KERNEL(
    edit_distance,
    ops::EditDistanceGPUKernel<paddle::platform::CUDAPlace, float>);
