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

struct DimensionsTransform {
  using DimVector = std::vector<int64_t>;
  typedef void (*MergeFunctor)(bool &, std::vector<DimVector> &, DimVector &,
                               int, int);
  int64_t dim_size;
  DimVector out_dims;
  std::vector<DimVector> in_dims;

 private:
  // 1. To compensate the lackage of input_tensors` dimension;
  void InputDimensionsExtend(int N, int axis) {
    for (auto &in_dim : in_dims) {
      int64_t in_idx = 0;
      if (in_dim.size() < dim_size) {
        DimVector tmp_dim(dim_size, 1);
        do {
          if (in_dim[in_idx] == out_dims[axis] || in_dim[in_idx] == 1) {
            tmp_dim[axis] = in_dim[in_idx];
            in_idx++;
            axis++;
          } else {
            PADDLE_THROW(platform::errors::InvalidArgument(
                "The %dth dimension of input tensor is expected to be equal "
                "with"
                "the %dth dimension of output tensor %d or 1, but recieved "
                "%d.\n",
                in_idx + 1, axis + 1, out_dims[axis], in_dim[in_idx]));
          }
        } while (in_idx < in_dim.size());
        in_dim.resize(dim_size);
        std::copy(tmp_dim.begin(), tmp_dim.end(), in_dim.begin());
      } else {
        do {
          if (in_dim[in_idx] == out_dims[in_idx] || in_dim[in_idx] == 1) {
            in_idx++;
          } else {
            PADDLE_THROW(platform::errors::InvalidArgument(
                "The %dth dimension of input tensor is expected to be equal "
                "with"
                "the %dth dimension of output tensor %d or 1, but recieved "
                "%d.\n",
                in_idx + 1, in_idx + 1, out_dims[in_idx], in_dim[in_idx]));
          }
        } while (in_idx < dim_size);
      }
      std::reverse(in_dim.begin(), in_dim.end());
    }
    std::reverse(out_dims.begin(), out_dims.end());
  }

  template <typename MergeFunctor>
  __inline__ void DimensionsReorganise(MergeFunctor merge_func, int N) {
    auto VectorReorganise = [](DimVector *vec, int l_idx, int m_idx) {
      (*vec)[m_idx - 1] =
          std::accumulate(vec->begin() + l_idx, vec->begin() + m_idx, 1,
                          std::multiplies<int64_t>());
      vec->erase(vec->begin() + l_idx, vec->begin() + m_idx - 1);
    };

    int64_t i = 0;
    while (i < dim_size) {
      int cnt = 0;
      int low_idx = i;
      bool equal = true;
      do {
        merge_func(equal, in_dims, out_dims, i, N);
        if (equal) {
          i++;
          cnt++;
        } else {
          break;
        }
      } while (i < dim_size);

      if (cnt > 1) {
        for (auto &in_dim : in_dims) {
          VectorReorganise(&in_dim, low_idx, i);
        }
        VectorReorganise(&out_dims, low_idx, i);
        dim_size -= --cnt;
        i -= cnt;
      } else if (cnt < 1) {
        i++;
      }
    }
  }

 public:
  explicit DimensionsTransform(
      const std::vector<const framework::Tensor *> &ins,
      const framework::DDim &dims, int axis) {
    const int N = ins.size();
    dim_size = dims.size();
    out_dims = framework::vectorize<int64_t>(dims);
    in_dims.resize(N);
    for (int j = 0; j < N; ++j) {
      in_dims[j] = framework::vectorize<int64_t>(ins[j]->dims());
    }
    InputDimensionsExtend(N, axis);

    auto merge_sequential_dims = [](bool &equal,
                                    std::vector<DimVector> &in_dims,
                                    DimVector &out, int i, int num) {
      for (int j = 1; j < num; ++j) {
        equal = (in_dims[0][i] == in_dims[j][i]) ? true : false;
      }
    };
    auto merge_sequential_one_dims = [](bool &equal,
                                        std::vector<DimVector> &in_dims,
                                        DimVector &out, int i, int num) {
      equal = in_dims[0][i] == 1;
      if (equal) {
        for (int j = 1; j < num; ++j) {
          equal = in_dims[j][i] == out[i];
        }
      }
    };
    // To Merge the dimensions of input_tensors while the consequtive
    // equal-dimensions appears.
    MergeFunctor merge_ptr = merge_sequential_dims;
    DimensionsReorganise<MergeFunctor>(merge_ptr, N);

    int min_idx = 0;
    int min_val = std::accumulate(in_dims[0].begin(), in_dims[0].end(), 1,
                                  std::multiplies<int64_t>());
    for (int j = 1; j < N; ++j) {
      int temp = std::accumulate(in_dims[j].begin(), in_dims[j].end(), 1,
                                 std::multiplies<int64_t>());
      min_val = min_val > temp ? temp : min_val;
      min_idx = min_val == temp ? j : min_idx;
    }
    std::swap(in_dims[0], in_dims[min_idx]);

    // To Merge the dimension of input_tensors while the consequtive
    // 1-value-dimensions appears.
    merge_ptr = merge_sequential_one_dims;
    DimensionsReorganise<MergeFunctor>(merge_ptr, N);
    std::swap(in_dims[min_idx], in_dims[0]);
  }
};

struct CalculateInputStrides {
  std::vector<std::vector<uint32_t>> strides;
  std::vector<FastDivMod> divmoders;

 private:
  // To calculate the strides of each input_tensor.
  __inline__ void CalculateStrides(
      int N, int dim_size, const std::vector<std::vector<int64_t>> &in_dims) {
    for (int j = 0; j < N; ++j) {
      for (int i = 0; i < dim_size; ++i) {
        strides[j][i] = in_dims[j][i] == 1 ? 0 : strides[j][i];
        strides[j][i] =
            (i != 0 && strides[j][i] != 0)
                ? std::accumulate(in_dims[j].begin(), in_dims[j].begin() + i, 1,
                                  std::multiplies<int64_t>())
                : strides[j][i];
      }
    }
  }

 public:
  explicit CalculateInputStrides(
      const int64_t &dim_size, const std::vector<std::vector<int64_t>> &in_dims,
      const std::vector<int64_t> &out_dims) {
    const auto N = in_dims.size();
    divmoders.resize(dim_size);
    strides.resize(N, std::vector<uint32_t>(dim_size, 1));

    for (int i = 0; i < dim_size; ++i) {
      divmoders[i] = FastDivMod(out_dims[i]);
    }
    CalculateStrides(N, dim_size, in_dims);
  }
};

template <typename T, typename Functor, ElementwiseType ET, int VecSize,
          int kDims>
struct BroadcastArgsWarpper {
  using DimsVec = CudaAlignedVector<T, VecSize>;

  T *out_data;
  DimsVec *vec_out_data;
  const T *__restrict__ in_data[ET];
  const DimsVec *__restrict__ vec_in_data[ET];
  bool no_broadcast[ET];
  FastDivMod divmoders[kDims];
  uint32_t strides[ET][framework::DDim::kMaxRank];
  uint32_t scalar_cal_offset;
  Functor func;

  HOSTDEVICE BroadcastArgsWarpper(
      const std::vector<const framework::Tensor *> &ins, framework::Tensor *out,
      int scalar_cal_offset, Functor func,
      const CalculateInputStrides &offset_calculator)
      : scalar_cal_offset(scalar_cal_offset), func(func) {
    for (int j = 0; j < ET; ++j) {
      in_data[j] = ins[j]->data<T>();
      vec_in_data[j] = reinterpret_cast<const DimsVec *>(in_data[j]);
      no_broadcast[j] = ins[j]->dims() == out->dims() ? true : false;
      memcpy(strides[j], offset_calculator.strides[j].data(),
             kDims * sizeof(uint32_t));
    }
    out_data = out->data<T>();
    vec_out_data = reinterpret_cast<DimsVec *>(out_data);
    memcpy(divmoders, offset_calculator.divmoders.data(),
           kDims * sizeof(FastDivMod));
  }

  __device__ __forceinline__ uint32_t GetDivmodOffset(int idx, int in_idx) {
    uint32_t offset = 0;

#pragma unroll(kDims)
    for (int i = 0; i < kDims; ++i) {
      auto fast_divmoder = divmoders[i].Divmod(idx);
      idx = fast_divmoder.val[0];
      offset += fast_divmoder.val[1] * strides[in_idx][i];
    }
    return offset;
  }

  __device__ __forceinline__ void VectorizedCommonLoadData(DimsVec *args,
                                                           int tid, int idx) {
    args[idx] = vec_in_data[idx][tid];
  }

  __device__ __forceinline__ void VectorizedDivmodLoadData(T *args, int tid,
                                                           int idx) {
    int index = tid * VecSize;

    for (int i = 0; i < VecSize; ++i) {
      uint32_t offset = GetDivmodOffset(index + i, idx);
      args[i] = in_data[idx][offset];
    }
  }

  __device__ __forceinline__ void ScalarizedCommonLoadData(T args[], int tid,
                                                           int idx) {
    args[idx] = in_data[idx][tid + scalar_cal_offset];
  }

  __device__ __forceinline__ void ScalarizedDivmodLoadData(T args[], int tid,
                                                           int idx) {
    auto offset = GetDivmodOffset(tid + scalar_cal_offset, idx);
    args[idx] = in_data[idx][offset];
  }

  __device__ __forceinline__ void VectorizedLoadData(T (*args)[VecSize],
                                                     int tid) {
#pragma unroll(ET)
    for (int j = 0; j < ET; ++j) {
      if (no_broadcast[j]) {
        DimsVec *vec_args = reinterpret_cast<DimsVec *>(args[j]);
        VectorizedCommonLoadData(vec_args, tid, j);
      } else {
        VectorizedDivmodLoadData(args[j], tid, j);
      }
    }
  }

  __device__ __forceinline__ void ScalarizedLoadData(T args[], int tid) {
#pragma unroll(ET)
    for (int j = 0; j < ET; ++j) {
      if (no_broadcast[j]) {
        ScalarizedCommonLoadData(args, tid, j);
      } else {
        ScalarizedDivmodLoadData(args, tid, j);
      }
    }
  }

  __device__ __forceinline__ void VectorizedStoreData(T (*args)[VecSize],
                                                      int tid) {
    DimsVec *args_out = reinterpret_cast<DimsVec *>(args[0]);
    vec_out_data[tid] = *args_out;
  }

  __device__ __forceinline__ void ScalarizedStoreData(T args[], int tid) {
    out_data[scalar_cal_offset + tid] = args[0];
  }
};

template <typename T, typename BroadcastArgsWarpper, ElementwiseType ET>
__device__ inline void ScalarizedBroadcastKernelImpl(
    BroadcastArgsWarpper broadcast_warpper, int tid) {
  T args[ET];
  broadcast_warpper.ScalarizedLoadData(args, tid);

#pragma unroll(ET)
  for (int j = 1; j < ET; ++j) {
    args[0] = broadcast_warpper.func(args[0], args[j]);
  }
  broadcast_warpper.ScalarizedStoreData(args, tid);
}

template <typename T, typename BroadcastArgsWarpper, ElementwiseType ET,
          int VecSize>
__device__ inline void VectorizedBroadcastKernelImpl(
    BroadcastArgsWarpper broadcast_warpper, int tid) {
  T args[ET][VecSize];
  broadcast_warpper.VectorizedLoadData(args, tid);

#pragma unroll(ET)
  for (int j = 1; j < ET; ++j) {
#pragma unroll(VecSize)
    for (int i = 0; i < VecSize; ++i) {
      broadcast_warpper.func(args[0][i], args[j][i]);
    }
  }
  broadcast_warpper.VectorizedStoreData(args, tid);
}

template <typename T, typename BroadcastArgsWarpper, ElementwiseType ET,
          int VecSize>
__global__ void ElementwiseBroadcastKernel(
    BroadcastArgsWarpper broadcast_warpper, int main_tid, int tail_tid) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;

  // Vectorized calculation of major data whose length is the max multipler of
  // VecSize,
  // eg: Calcualtion the front 1024-length data in total 1027 data once VecSize
  // is 4.
  if (tid < main_tid) {
    VectorizedBroadcastKernelImpl<T, BroadcastArgsWarpper, ET, VecSize>(
        broadcast_warpper, tid);
  }
  // Scalarzed calculation of rest data whose lenght cannot fulfill VecSize.
  // eg: Calcualtion the rest 3-length data in total 1027 data once VecSize is
  // 4.
  if (tid < tail_tid) {
    ScalarizedBroadcastKernelImpl<T, BroadcastArgsWarpper, ET>(
        broadcast_warpper, tid);
  }
}

template <typename T, ElementwiseType ET, int VecSize, typename Functor>
void LaunchBroadcastKernelForDifferentDimSize(
    const platform::CUDADeviceContext &ctx,
    const std::vector<const framework::Tensor *> &ins, framework::Tensor *out,
    int axis, Functor func) {
  int numel = out->numel();
  const int threads = 256;
  int blocks = ((numel + VecSize - 1) / VecSize + threads - 1) / threads;
  int main_tid = numel / VecSize;
  int tail_tid = numel % VecSize;
  int vec_len = main_tid * VecSize;
  auto stream = ctx.stream();

  const auto merge_dims = DimensionsTransform(ins, out->dims(), axis);
  const auto offset_calculator = CalculateInputStrides(
      merge_dims.dim_size, merge_dims.in_dims, merge_dims.out_dims);

  switch (merge_dims.dim_size) {
    case 1: {
      auto broadcast_warpper = BroadcastArgsWarpper<T, Functor, ET, VecSize, 1>(
          ins, out, vec_len, func, offset_calculator);
      ElementwiseBroadcastKernel<T, decltype(broadcast_warpper), ET,
                                 VecSize><<<blocks, threads, 0, stream>>>(
          broadcast_warpper, main_tid, tail_tid);
      break;
    }
    case 2: {
      auto broadcast_warpper = BroadcastArgsWarpper<T, Functor, ET, VecSize, 2>(
          ins, out, vec_len, func, offset_calculator);
      ElementwiseBroadcastKernel<T, decltype(broadcast_warpper), ET,
                                 VecSize><<<blocks, threads, 0, stream>>>(
          broadcast_warpper, main_tid, tail_tid);
      break;
    }
    case 3: {
      auto broadcast_warpper = BroadcastArgsWarpper<T, Functor, ET, VecSize, 3>(
          ins, out, vec_len, func, offset_calculator);
      ElementwiseBroadcastKernel<T, decltype(broadcast_warpper), ET,
                                 VecSize><<<blocks, threads, 0, stream>>>(
          broadcast_warpper, main_tid, tail_tid);
      break;
    }
    case 4: {
      auto broadcast_warpper = BroadcastArgsWarpper<T, Functor, ET, VecSize, 4>(
          ins, out, vec_len, func, offset_calculator);
      ElementwiseBroadcastKernel<T, decltype(broadcast_warpper), ET,
                                 VecSize><<<blocks, threads, 0, stream>>>(
          broadcast_warpper, main_tid, tail_tid);
      break;
    }
    case 5: {
      auto broadcast_warpper = BroadcastArgsWarpper<T, Functor, ET, VecSize, 5>(
          ins, out, vec_len, func, offset_calculator);
      ElementwiseBroadcastKernel<T, decltype(broadcast_warpper), ET,
                                 VecSize><<<blocks, threads, 0, stream>>>(
          broadcast_warpper, main_tid, tail_tid);
      break;
    }
    case 6: {
      auto broadcast_warpper = BroadcastArgsWarpper<T, Functor, ET, VecSize, 6>(
          ins, out, vec_len, func, offset_calculator);
      ElementwiseBroadcastKernel<T, decltype(broadcast_warpper), ET,
                                 VecSize><<<blocks, threads, 0, stream>>>(
          broadcast_warpper, main_tid, tail_tid);
      break;
    }
    case 7: {
      auto broadcast_warpper = BroadcastArgsWarpper<T, Functor, ET, VecSize, 7>(
          ins, out, vec_len, func, offset_calculator);
      ElementwiseBroadcastKernel<T, decltype(broadcast_warpper), ET,
                                 VecSize><<<blocks, threads, 0, stream>>>(
          broadcast_warpper, main_tid, tail_tid);
      break;
    }
    case 8: {
      auto broadcast_warpper = BroadcastArgsWarpper<T, Functor, ET, VecSize, 8>(
          ins, out, vec_len, func, offset_calculator);
      ElementwiseBroadcastKernel<T, decltype(broadcast_warpper), ET,
                                 VecSize><<<blocks, threads, 0, stream>>>(
          broadcast_warpper, main_tid, tail_tid);
      break;
    }
    default: {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "The maximum dimension of input tensor is expected to be less than "
          "%d, but recieved %d.\n",
          merge_dims.dim_size, framework::DDim::kMaxRank));
    }
  }
}

template <ElementwiseType ET, typename T, typename Functor>
void LaunchBroadcastElementwiseCudaKernel(
    const platform::CUDADeviceContext &ctx,
    const std::vector<const framework::Tensor *> &ins, framework::Tensor *out,
    Functor func, int axis) {
  int in_vec_size = 4;
  for (auto *in : ins) {
    auto temp_size = GetVectorizedSizeImpl<T>(in->data<T>());
    in_vec_size = in->dims() == out->dims() ? std::min(temp_size, in_vec_size)
                                            : in_vec_size;
  }
  int out_vec_size = GetVectorizedSizeImpl<T>(out->data<T>());
  int vec_size = std::min(out_vec_size, in_vec_size);

  switch (vec_size) {
    case 4: {
      LaunchBroadcastKernelForDifferentDimSize<T, ET, 4>(ctx, ins, out, axis,
                                                         func);
      break;
    }
    case 2: {
      LaunchBroadcastKernelForDifferentDimSize<T, ET, 2>(ctx, ins, out, axis,
                                                         func);
      break;
    }
    case 1: {
      LaunchBroadcastKernelForDifferentDimSize<T, ET, 1>(ctx, ins, out, axis,
                                                         func);
      break;
    }
    default: {
      PADDLE_THROW(platform::errors::Unimplemented(
          "Unsupported vectorized size: %d !", vec_size));
      break;
    }
  }
}

}  // namespace operators
}  // namespace paddle