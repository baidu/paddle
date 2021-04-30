// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.1 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.1
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include "paddle/fluid/operators/elementwise/elementwise_op_broadcast_impl.cu.h"

namespace paddle {
namespace operators {

template <typename T, typename fetch_t, int N>
__device__ inline void BScalarizedKernelImpl(fetch_t data_processor, int tid) {
  T args[N];
  data_processor.load_scalar(args, tid);

#pragma unroll(N)
  for (int j = 1; j < N; ++j) {
    args[0] += args[j];
  }
  data_processor.store_scalar(args, tid);
}

template <typename T, typename fetch_t, int N, int vec_size>
__device__ inline void BVectorizedKernelImpl(fetch_t data_processor, int tid) {
  using VecT = CudaAlignedVector<T, vec_size>;
  VecT args[N];
  data_processor.load_vector(args, tid);

#pragma unroll(N)
  for (int j = 1; j < N; ++j) {
#pragma unroll(vec_size)
    for (int i = 0; i < vec_size; ++i) {
      args[0].val[i] += args[j].val[i];
    }
  }
  data_processor.store_vector(args, tid);
}

template <typename T, typename fetch_t, int N, int vec_size>
__global__ void CommonElementwiseKernel(fetch_t data_processor, int main_tid,
                                        int tail_tid) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < main_tid) {
    BVectorizedKernelImpl<T, fetch_t, N, vec_size>(data_processor, tid);
  }
  if (tid < tail_tid) {
    BScalarizedKernelImpl<T, fetch_t, N>(data_processor, tid);
  }
}

template <typename T, typename FuncT, typename OffsetT, int vec_size, int N>
void CommonElementwiseCore(const platform::CUDADeviceContext &ctx,
                           const std::vector<const framework::Tensor *> &ins,
                           framework::Tensor *out, const OffsetT &p_offset_pre,
                           FuncT func) {
  int numel = out->numel();
  const int threads = 256;
  int blocks = ((numel + vec_size - 1) / vec_size + threads - 1) / threads;
  int main_tid = numel / vec_size;
  int tail_tid = numel % vec_size;
  int vec_len = main_tid * vec_size;

  int dim_size = p_offset_pre.strides[0].size();
  auto stream = ctx.stream();

  switch (dim_size) {
    case 1: {
      auto data_processor = DataProcessor<T, OffsetT, N, vec_size, 1>(
          ins, p_offset_pre, out, vec_len);
      CommonElementwiseKernel<T, decltype(data_processor), N,
                              vec_size><<<blocks, threads, 0, stream>>>(
          data_processor, main_tid, tail_tid);
      break;
    }
    case 2: {
      auto data_processor = DataProcessor<T, OffsetT, N, vec_size, 2>(
          ins, p_offset_pre, out, vec_len);
      CommonElementwiseKernel<T, decltype(data_processor), N,
                              vec_size><<<blocks, threads, 0, stream>>>(
          data_processor, main_tid, tail_tid);
      break;
    }
    case 3: {
      auto data_processor = DataProcessor<T, OffsetT, N, vec_size, 3>(
          ins, p_offset_pre, out, vec_len);
      CommonElementwiseKernel<T, decltype(data_processor), N,
                              vec_size><<<blocks, threads, 0, stream>>>(
          data_processor, main_tid, tail_tid);
      break;
    }
    case 4: {
      auto data_processor = DataProcessor<T, OffsetT, N, vec_size, 4>(
          ins, p_offset_pre, out, vec_len);
      CommonElementwiseKernel<T, decltype(data_processor), N,
                              vec_size><<<blocks, threads, 0, stream>>>(
          data_processor, main_tid, tail_tid);
      break;
    }
    case 5: {
      auto data_processor = DataProcessor<T, OffsetT, N, vec_size, 5>(
          ins, p_offset_pre, out, vec_len);
      CommonElementwiseKernel<T, decltype(data_processor), N,
                              vec_size><<<blocks, threads, 0, stream>>>(
          data_processor, main_tid, tail_tid);
      break;
    }
    case 6: {
      auto data_processor = DataProcessor<T, OffsetT, N, vec_size, 6>(
          ins, p_offset_pre, out, vec_len);
      CommonElementwiseKernel<T, decltype(data_processor), N,
                              vec_size><<<blocks, threads, 0, stream>>>(
          data_processor, main_tid, tail_tid);
      break;
    }
    case 7: {
      auto data_processor = DataProcessor<T, OffsetT, N, vec_size, 7>(
          ins, p_offset_pre, out, vec_len);
      CommonElementwiseKernel<T, decltype(data_processor), N,
                              vec_size><<<blocks, threads, 0, stream>>>(
          data_processor, main_tid, tail_tid);
      break;
    }
    case 8: {
      auto data_processor = DataProcessor<T, OffsetT, N, vec_size, 8>(
          ins, p_offset_pre, out, vec_len);
      CommonElementwiseKernel<T, decltype(data_processor), N,
                              vec_size><<<blocks, threads, 0, stream>>>(
          data_processor, main_tid, tail_tid);
      break;
    }
    default: {
      PADDLE_ENFORCE_GT(dim_size, MAX_DIMS, platform::errors::InvalidArgument(
                                                "Quantity of Input tensor"
                                                "is %d, the limitation is %d\n",
                                                dim_size, MAX_DIMS));
    }
  }
}

template <typename T, typename FuncT, int vec_size = 1>
void BroadcastDimsTransform(const platform::CUDADeviceContext &ctx,
                            const std::vector<const framework::Tensor *> &ins,
                            framework::Tensor *out, FuncT func) {
  int input_num = ins.size();
  const auto merge_dims = DimensionTransform(ins, out->dims(), input_num);
  const auto offset_pre = OffsetPreCalculator<decltype(merge_dims)>(merge_dims);

  switch (input_num) {
    case 2: {
      CommonElementwiseCore<T, FuncT, decltype(offset_pre), vec_size, 2>(
          ctx, ins, out, offset_pre, func);
      break;
    }
    case 3: {
      CommonElementwiseCore<T, FuncT, decltype(offset_pre), vec_size, 3>(
          ctx, ins, out, offset_pre, func);
      break;
    }
    case 4: {
      CommonElementwiseCore<T, FuncT, decltype(offset_pre), vec_size, 4>(
          ctx, ins, out, offset_pre, func);
      break;
    }
    default: {
      PADDLE_ENFORCE_GT(
          input_num, MAX_TENSORS,
          platform::errors::InvalidArgument("Quantity of Input tensor"
                                            "is %d, the limitation is %d\n",
                                            input_num, MAX_TENSORS));
    }
  }
}

template <typename T, typename FuncT>
void LaunchBroadcastElementwiseCudaKernel(
    const platform::CUDADeviceContext &ctx,
    const std::vector<const framework::Tensor *> &ins, framework::Tensor *out,
    FuncT func) {
  int in_vec_size = 8;
  for (auto *in : ins) {
    auto temp_size = GetVectorizedSizeImpl<T>(in->data<T>());
    in_vec_size = in->dims() == out->dims() ? std::min(temp_size, in_vec_size)
                                            : in_vec_size;
  }
  T *out_data = out->data<T>();
  int out_vec_size = GetVectorizedSizeImpl<T>(out_data);
  int vec_size = std::min(out_vec_size, in_vec_size);

  switch (vec_size) {
    case 4: {
      BroadcastDimsTransform<T, FuncT, 4>(ctx, ins, out, func);
      break;
    }
    case 2: {
      BroadcastDimsTransform<T, FuncT, 2>(ctx, ins, out, func);
      break;
    }
    default: {
      BroadcastDimsTransform<T, FuncT>(ctx, ins, out, func);
      break;
    }
  }
}

}  // namespace operators
}  // namespace paddle
