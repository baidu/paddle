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

#include "paddle/fluid/operators/elementwise/elementwise_op_impl.cu.h"

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
  // To compensate the lackage of input_tensors` dimension with input variable
  // 'axis'
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
  __inline__ void MergeDimensions(MergeFunctor merge_func, int N) {
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
    MergeDimensions<MergeFunctor>(merge_ptr, N);

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
    MergeDimensions<MergeFunctor>(merge_ptr, N);
    std::swap(in_dims[min_idx], in_dims[0]);
  }
};

struct StridesCalculation {
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
  explicit StridesCalculation(const int64_t &dim_size,
                              const std::vector<std::vector<int64_t>> &in_dims,
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

template <typename InT, typename OutT, typename Functor, ElementwiseType ET,
          int VecSize, int kDims>
struct BroadcastArgsWarpper {
  using InVecType = CudaAlignedVector<InT, VecSize>;
  using OutVecType = CudaAlignedVector<OutT, VecSize>;

  OutT *out_data;
  OutVecType *vec_out_data;
  const InT *__restrict__ in_data[ET];
  const InVecType *__restrict__ vec_in_data[ET];
  bool no_broadcast[ET];
  FastDivMod divmoders[kDims];
  uint32_t strides[ET][framework::DDim::kMaxRank];
  uint32_t scalar_cal_offset;
  Functor func;

  HOSTDEVICE BroadcastArgsWarpper(
      const std::vector<const framework::Tensor *> &ins, framework::Tensor *out,
      int scalar_cal_offset, Functor func,
      const StridesCalculation &offset_calculator)
      : scalar_cal_offset(scalar_cal_offset), func(func) {
    for (int j = 0; j < ET; ++j) {
      in_data[j] = ins[j]->data<InT>();
      vec_in_data[j] = reinterpret_cast<const InVecType *>(in_data[j]);
      no_broadcast[j] = ins[j]->dims() == out->dims() ? true : false;
      memcpy(strides[j], offset_calculator.strides[j].data(),
             kDims * sizeof(uint32_t));
    }
    out_data = out->data<OutT>();
    vec_out_data = reinterpret_cast<OutVecType *>(out_data);
    memcpy(divmoders, offset_calculator.divmoders.data(),
           kDims * sizeof(FastDivMod));
  }

  __device__ __forceinline__ uint32_t GetOffsetByDivmod(int idx, int in_idx) {
    uint32_t offset = 0;

#pragma unroll(kDims)
    for (int i = 0; i < kDims; ++i) {
      auto fast_divmoder = divmoders[i].Divmod(idx);
      idx = fast_divmoder.val[0];
      offset += fast_divmoder.val[1] * strides[in_idx][i];
    }
    return offset;
  }

  __device__ __forceinline__ void LoadVectorizedDataCommon(
      InVecType *vector_args, int tid, int idx) {
    *vector_args = vec_in_data[idx][tid];
  }

  __device__ __forceinline__ void LoadVectorizedDataByDivmod(InT *scalar_args,
                                                             int tid, int idx) {
    int index = tid * VecSize;
#pragma unroll(VecSize)
    for (int i = 0; i < VecSize; ++i) {
      uint32_t offset = GetOffsetByDivmod(index + i, idx);
      scalar_args[i] = in_data[idx][offset];
    }
  }

  __device__ __forceinline__ void LoadScalarizedDataCommon(InT args[], int tid,
                                                           int idx) {
    args[idx] = in_data[idx][tid + scalar_cal_offset];
  }

  __device__ __forceinline__ void LoadScalarizedDataByDivmod(InT args[],
                                                             int tid, int idx) {
    auto offset = GetOffsetByDivmod(tid + scalar_cal_offset, idx);
    args[idx] = in_data[idx][offset];
  }

  __device__ __forceinline__ void LoadVectorizedData(InT (*args)[VecSize],
                                                     int tid) {
#pragma unroll(ET)
    for (int j = 0; j < ET; ++j) {
      if (no_broadcast[j]) {
        InVecType *vector_args = reinterpret_cast<InVecType *>(args[j]);
        LoadVectorizedDataCommon(vector_args, tid, j);
      } else {
        LoadVectorizedDataByDivmod(args[j], tid, j);
      }
    }
  }

  __device__ __forceinline__ void LoadScalarizedData(InT args[], int tid) {
#pragma unroll(ET)
    for (int j = 0; j < ET; ++j) {
      if (no_broadcast[j]) {
        LoadScalarizedDataCommon(args, tid, j);
      } else {
        LoadScalarizedDataByDivmod(args, tid, j);
      }
    }
  }

  __device__ __forceinline__ void StoreVectorizedData(OutVecType vec_args_out,
                                                      int tid) {
    vec_out_data[tid] = vec_args_out;
  }

  __device__ __forceinline__ void StoreScalarizedData(OutT args_out, int tid) {
    out_data[scalar_cal_offset + tid] = args_out;
  }
};

template <typename InT, typename OutT, typename BroadcastArgsWarpper,
          ElementwiseType ET>
__device__ inline void ScalarizedBroadcastKernelImpl(
    BroadcastArgsWarpper broadcast_warpper, int tid) {
  InT args[ET];
  OutT args_out;
  broadcast_warpper.LoadScalarizedData(args, tid);

#pragma unroll(ET)
  for (int j = 1; j < ET; ++j) {
    args_out = broadcast_warpper.func(args);
  }
  broadcast_warpper.StoreScalarizedData(args_out, tid);
}

template <typename InT, typename OutT, typename BroadcastArgsWarpper,
          ElementwiseType ET, int VecSize>
__device__ inline void VectorizedBroadcastKernelImpl(
    BroadcastArgsWarpper broadcast_warpper, int tid) {
  using OutVecType = CudaAlignedVector<OutT, VecSize>;
  OutVecType args_out;
  InT ins[ET];
  InT args[ET][VecSize];
  broadcast_warpper.LoadVectorizedData(args, tid);

#pragma unroll(VecSize)
  for (int i = 0; i < VecSize; ++i) {
#pragma unroll(ET)
    for (int j = 0; j < ET; ++j) {
      ins[j] = args[j][i];
    }
    args_out.val[i] = broadcast_warpper.func(ins);
  }
  broadcast_warpper.StoreVectorizedData(args_out, tid);
}

template <typename InT, typename OutT, typename BroadcastArgsWarpper,
          ElementwiseType ET, int VecSize>
__global__ void ElementwiseBroadcastKernel(
    BroadcastArgsWarpper broadcast_warpper, int main_tid, int tail_tid) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  // Vectorized calculation of major data whose length is the max multipler of
  // VecSize,
  // eg: Calcualting the front 1024-length data in total 1027 data once VecSize
  // is 4.
  if (tid < main_tid) {
    VectorizedBroadcastKernelImpl<InT, OutT, BroadcastArgsWarpper, ET, VecSize>(
        broadcast_warpper, tid);
  }
  // Scalarzed calculation of rest data whose lenght cannot fulfill VecSize.
  // eg: Calcualting the rest 3-length data in total 1027 data once VecSize is
  // 4.
  if (tid < tail_tid) {
    ScalarizedBroadcastKernelImpl<InT, OutT, BroadcastArgsWarpper, ET>(
        broadcast_warpper, tid);
  }
}

template <typename InT, typename OutT, ElementwiseType ET, int VecSize,
          typename Functor>
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
  const auto offset_calculator = StridesCalculation(
      merge_dims.dim_size, merge_dims.in_dims, merge_dims.out_dims);

  switch (merge_dims.dim_size) {
    case 1: {
      auto broadcast_warpper =
          BroadcastArgsWarpper<InT, OutT, Functor, ET, VecSize, 1>(
              ins, out, vec_len, func, offset_calculator);
      ElementwiseBroadcastKernel<InT, OutT, decltype(broadcast_warpper), ET,
                                 VecSize><<<blocks, threads, 0, stream>>>(
          broadcast_warpper, main_tid, tail_tid);
      break;
    }
    case 2: {
      auto broadcast_warpper =
          BroadcastArgsWarpper<InT, OutT, Functor, ET, VecSize, 2>(
              ins, out, vec_len, func, offset_calculator);
      ElementwiseBroadcastKernel<InT, OutT, decltype(broadcast_warpper), ET,
                                 VecSize><<<blocks, threads, 0, stream>>>(
          broadcast_warpper, main_tid, tail_tid);
      break;
    }
    case 3: {
      auto broadcast_warpper =
          BroadcastArgsWarpper<InT, OutT, Functor, ET, VecSize, 3>(
              ins, out, vec_len, func, offset_calculator);
      ElementwiseBroadcastKernel<InT, OutT, decltype(broadcast_warpper), ET,
                                 VecSize><<<blocks, threads, 0, stream>>>(
          broadcast_warpper, main_tid, tail_tid);
      break;
    }
    case 4: {
      auto broadcast_warpper =
          BroadcastArgsWarpper<InT, OutT, Functor, ET, VecSize, 4>(
              ins, out, vec_len, func, offset_calculator);
      ElementwiseBroadcastKernel<InT, OutT, decltype(broadcast_warpper), ET,
                                 VecSize><<<blocks, threads, 0, stream>>>(
          broadcast_warpper, main_tid, tail_tid);
      break;
    }
    case 5: {
      auto broadcast_warpper =
          BroadcastArgsWarpper<InT, OutT, Functor, ET, VecSize, 5>(
              ins, out, vec_len, func, offset_calculator);
      ElementwiseBroadcastKernel<InT, OutT, decltype(broadcast_warpper), ET,
                                 VecSize><<<blocks, threads, 0, stream>>>(
          broadcast_warpper, main_tid, tail_tid);
      break;
    }
    case 6: {
      auto broadcast_warpper =
          BroadcastArgsWarpper<InT, OutT, Functor, ET, VecSize, 6>(
              ins, out, vec_len, func, offset_calculator);
      ElementwiseBroadcastKernel<InT, OutT, decltype(broadcast_warpper), ET,
                                 VecSize><<<blocks, threads, 0, stream>>>(
          broadcast_warpper, main_tid, tail_tid);
      break;
    }
    case 7: {
      auto broadcast_warpper =
          BroadcastArgsWarpper<InT, OutT, Functor, ET, VecSize, 7>(
              ins, out, vec_len, func, offset_calculator);
      ElementwiseBroadcastKernel<InT, OutT, decltype(broadcast_warpper), ET,
                                 VecSize><<<blocks, threads, 0, stream>>>(
          broadcast_warpper, main_tid, tail_tid);
      break;
    }
    case 8: {
      auto broadcast_warpper =
          BroadcastArgsWarpper<InT, OutT, Functor, ET, VecSize, 8>(
              ins, out, vec_len, func, offset_calculator);
      ElementwiseBroadcastKernel<InT, OutT, decltype(broadcast_warpper), ET,
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

template <ElementwiseType ET, typename InT, typename OutT, typename Functor>
void LaunchBroadcastElementwiseCudaKernel(
    const platform::CUDADeviceContext &ctx,
    const std::vector<const framework::Tensor *> &ins,
    std::vector<framework::Tensor *> *outs, int axis, Functor func) {
  PADDLE_ENFORCE_EQ(ET, ElementwiseType::kBinary,
                    platform::errors::InvalidArgument(
                        "Currently, only Support binary calculation, "
                        "but received %d input tensors.\n",
                        static_cast<int>(ET)));
  int in_vec_size = 4;
  framework::Tensor *out = (*outs)[0];
  for (auto *in : ins) {
    auto temp_size = GetVectorizedSizeImpl<InT>(in->data<InT>());
    in_vec_size = in->dims() == out->dims() ? std::min(temp_size, in_vec_size)
                                            : in_vec_size;
  }
  int out_vec_size = GetVectorizedSizeImpl<OutT>(out->data<OutT>());
  int vec_size = std::min(out_vec_size, in_vec_size);

  switch (vec_size) {
    case 4: {
      LaunchBroadcastKernelForDifferentDimSize<InT, OutT, ET, 4>(ctx, ins, out,
                                                                 axis, func);
      break;
    }
    case 2: {
      LaunchBroadcastKernelForDifferentDimSize<InT, OutT, ET, 2>(ctx, ins, out,
                                                                 axis, func);
      break;
    }
    case 1: {
      LaunchBroadcastKernelForDifferentDimSize<InT, OutT, ET, 1>(ctx, ins, out,
                                                                 axis, func);
      break;
    }
    default: {
      PADDLE_THROW(platform::errors::Unimplemented(
          "Unsupported vectorized size: %d !", vec_size));
      break;
    }
  }
}

template <ElementwiseType ET, typename InT, typename OutT, typename Functor>
void LaunchElementwiseCudaKernel(
    const platform::CUDADeviceContext &cuda_ctx,
    const std::vector<const framework::Tensor *> &ins,
    std::vector<framework::Tensor *> *outs, int axis, Functor func) {
  std::vector<int> dims_size;
  bool no_broadcast_flag = true;
  for (auto *in : ins) {
    no_broadcast_flag = ins[0]->dims() == in->dims();
    dims_size.emplace_back(in->dims().size());
  }

  if (no_broadcast_flag) {
    LaunchSameDimsElementwiseCudaKernel<ET, InT, OutT>(cuda_ctx, ins, outs,
                                                       func);
  } else {
    axis = axis == -1
               ? *std::max_element(dims_size.begin(), dims_size.end()) -
                     *std::min_element(dims_size.begin(), dims_size.end())
               : axis;
    LaunchBroadcastElementwiseCudaKernel<ET, InT, OutT>(cuda_ctx, ins, outs,
                                                        axis, func);
  }
}

}  // namespace operators
}  // namespace paddle
