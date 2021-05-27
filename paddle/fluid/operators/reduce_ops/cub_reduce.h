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

#pragma once

#include <algorithm>
#include <cmath>
#include <numeric>
#include <set>
#include <vector>

#ifdef __NVCC__
#include "cub/cub.cuh"  // NOLINT
#endif

#ifdef __HIPCC__
#include <hipcub/hipcub.hpp>
namespace cub = hipcub;
#endif

#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/fluid/operators/amp/fp16_type_traits.h"

namespace paddle {
namespace operators {

namespace detail {
template <typename T, size_t ElementCount>
struct Array {
 public:
  HOSTDEVICE inline Array() {}

  HOSTDEVICE inline T& operator[](size_t index) { return data_[index]; }

  HOSTDEVICE inline const T& operator[](size_t index) const {
    return data_[index];
  }

  HOSTDEVICE constexpr inline size_t size() const { return ElementCount; }

  template <typename VectorLikeType>
  static inline Array<T, ElementCount> From(const VectorLikeType& vec) {
    PADDLE_ENFORCE_EQ(vec.size(), ElementCount,
                      platform::errors::InvalidArgument(
                          "Cub reduce Array: size not match. Received "
                          "vec.size() %d !=  ElementCount %d.",
                          vec.size(), ElementCount));
    size_t n = static_cast<size_t>(vec.size());
    Array<T, ElementCount> ret;
    for (size_t i = 0; i < n; ++i) ret[i] = vec[i];
    return ret;
  }

 private:
  T data_[ElementCount];
};

// reduce the 1d array to one element
template <typename Tx, typename MPType, typename Ty, typename ReduceOp,
          typename TransformOp, int BlockDim>
__global__ void ReduceKernel1D(const Tx* x, Ty* y, ReduceOp reducer,
                               TransformOp transformer, MPType init,
                               int reduce_num) {
  int thread_id = blockIdx.x * blockDim.x + threadIdx.x;

  typedef cub::BlockReduce<MPType, BlockDim> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  MPType local_data = init;
  for (int i = thread_id; i < reduce_num; i += gridDim.x * blockDim.x) {
    local_data = static_cast<MPType>(
        reducer(local_data, static_cast<MPType>(transformer(x[i]))));
  }
  __syncthreads();

  local_data = BlockReduce(temp_storage).Reduce(local_data, reducer);

  if (threadIdx.x == 0) {
    y[blockIdx.x] = static_cast<Ty>(local_data);
  }
}

// reduce the last axis of 2d array
template <typename Tx, typename MPType, typename Ty, typename ReduceOp,
          typename TransformOp, int BlockDim>
__global__ void ReduceKernel2D(const Tx* x, Ty* y, ReduceOp reducer,
                               TransformOp transformer, MPType init,
                               int reduce_num) {
  __shared__
      typename cub::BlockReduce<MPType, BlockDim>::TempStorage temp_storage;
  int idx_x = blockIdx.x * reduce_num;
  int idx_y = threadIdx.x;
  MPType reduce_var = init;
  for (int idx_y = threadIdx.x; idx_y < reduce_num; idx_y += BlockDim)
    reduce_var =
        reducer(reduce_var, static_cast<MPType>(transformer(x[idx_x + idx_y])));
  __syncthreads();

  reduce_var = cub::BlockReduce<MPType, BlockDim>(temp_storage)
                   .Reduce(reduce_var, reducer);

  if (threadIdx.x == 0) {
    y[blockIdx.x] = static_cast<Ty>(reduce_var);
  }
}

template <typename Tx, typename MPType, typename Ty, typename ReduceOp,
          typename TransformOp, int BlockDim, int Rank, int ReduceRank>
__global__ void ReduceKernel(const Tx* x, Ty* y, ReduceOp reducer,
                             TransformOp transformer, MPType init,
                             int reduce_num, Array<int, Rank> x_strides,
                             Array<int, ReduceRank> reduce_dim,
                             Array<int, ReduceRank> reduce_strides,
                             Array<int, Rank - ReduceRank> left_dim,
                             Array<int, Rank - ReduceRank> left_strides) {
  __shared__
      typename cub::BlockReduce<MPType, BlockDim>::TempStorage temp_storage;
  Array<int, Rank> sub_index;
  int left_idx = blockIdx.x;
  for (int i = 0; i < Rank - ReduceRank; ++i) {
    sub_index[left_dim[i]] = left_idx / left_strides[i];
    left_idx %= left_strides[i];
  }

  int reduce_idx = threadIdx.x;
  for (int j = 0; j < ReduceRank; ++j) {
    sub_index[reduce_dim[j]] = reduce_idx / reduce_strides[j];
    reduce_idx %= reduce_strides[j];
  }

  int idx_x = 0;
  for (int k = 0; k < Rank; ++k) idx_x += (sub_index[k] * x_strides[k]);
  MPType reduce_var = static_cast<MPType>(transformer(x[idx_x]));

  for (int i = threadIdx.x + BlockDim; i < reduce_num; i += BlockDim) {
    int reduce_idx = i;
    for (int j = 0; j < ReduceRank; ++j) {
      sub_index[reduce_dim[j]] = reduce_idx / reduce_strides[j];
      reduce_idx %= reduce_strides[j];
    }

    int idx_x = 0;
    for (int k = 0; k < Rank; ++k) idx_x += (sub_index[k] * x_strides[k]);
    reduce_var = static_cast<MPType>(
        reducer(reduce_var, static_cast<MPType>(transformer(x[idx_x]))));
  }
  __syncthreads();

  reduce_var = cub::BlockReduce<MPType, BlockDim>(temp_storage)
                   .Reduce(reduce_var, reducer);

  if (threadIdx.x == 0) {
    y[blockIdx.x] = static_cast<Ty>(reduce_var);
  }
}

static inline std::vector<int> GetStrides(const std::vector<int>& dims) {
  int n = static_cast<int>(dims.size());
  if (n == 0) return std::vector<int>();
  std::vector<int> strides(n);
  strides.back() = 1;
  for (int i = n - 2; i >= 0; --i) {
    strides[i] = strides[i + 1] * dims[i + 1];
  }
  return strides;
}

static inline std::vector<int> GetStrides(const std::vector<int>& dims,
                                          const std::vector<int>& idx) {
  int n = static_cast<int>(idx.size());
  if (n == 0) return std::vector<int>();
  std::vector<int> strides(n);
  strides.back() = 1;
  for (int i = n - 2; i >= 0; --i) {
    strides[i] = strides[i + 1] * dims[idx[i + 1]];
  }
  return strides;
}

#ifdef __HIPCC__
constexpr int kMaxBlockDim = 256;
#else
constexpr int kMaxBlockDim = 512;
#endif

static inline int GetDesiredBlockDim(int block_dim) {
  return block_dim >= kMaxBlockDim
             ? kMaxBlockDim
             : (1 << static_cast<int>(std::log2(block_dim)));
}

static inline void CheckReduceRankIsValid(int reduce_rank, int rank) {
  if (rank % 2 == 0) {
    PADDLE_ENFORCE_EQ(reduce_rank, rank / 2,
                      platform::errors::InvalidArgument(
                          "ReduceOp: invalid reduce rank. When rank = %d, "
                          "reduce_rank must be %d, but got %d.",
                          rank, rank / 2, reduce_rank));
  } else {
    auto lower_rank = (rank - 1) / 2;
    auto upper_rank = (rank + 1) / 2;
    PADDLE_ENFORCE_EQ(
        reduce_rank == lower_rank || reduce_rank == upper_rank, true,
        platform::errors::InvalidArgument(
            "ReduceOp: invalid reduce rank. When rank = %d, reduce_rank "
            "must be %d or %d, but got %d.",
            rank, lower_rank, upper_rank, reduce_rank));
  }
}

template <typename Tx, typename MPType, typename Ty, typename ReduceOp,
          typename TransformOp, int BlockDim>
typename std::enable_if<!std::is_same<Tx, paddle::platform::float16>::value,
                        void>::type
LaunchCubReduceKernel(const Tx* x_data, Ty* y_data,
                      const platform::Place& place, const ReduceOp& reducer,
                      const TransformOp& transformer, const Ty& init,
                      int reduce_num, gpuStream_t stream) {
  cub::TransformInputIterator<Ty, TransformOp, const Tx*> trans_x(x_data,
                                                                  transformer);
  size_t temp_storage_bytes = 0;
  cub::DeviceReduce::Reduce(nullptr, temp_storage_bytes, trans_x, y_data,
                            reduce_num, reducer, init, stream);
  framework::Tensor tmp;
  auto* temp_storage = tmp.mutable_data<uint8_t>(
      framework::make_ddim({static_cast<int64_t>(temp_storage_bytes)}), place);
  cub::DeviceReduce::Reduce(temp_storage, temp_storage_bytes, trans_x, y_data,
                            reduce_num, reducer, init, stream);
}

template <typename Tx, typename MPType, typename Ty, typename ReduceOp,
          typename TransformOp, int BlockDim>
typename std::enable_if<std::is_same<Tx, paddle::platform::float16>::value,
                        void>::type
LaunchCubReduceKernel(const Tx* x_data, Ty* y_data,
                      const platform::Place& place, const ReduceOp& reducer,
                      const TransformOp& transformer, const MPType& init,
                      int reduce_num, gpuStream_t stream) {
  int element_per_block = BlockDim * 10;
  int block_per_grid = (reduce_num + element_per_block - 1) / element_per_block;

  framework::Tensor tmp;
  auto* temp_storage = tmp.mutable_data<MPType>(
      framework::make_ddim(
          {static_cast<int64_t>(block_per_grid * sizeof(MPType))}),
      place);

  // each block reduce number to interim result
  ReduceKernel1D<Tx, MPType, MPType, ReduceOp, TransformOp,
                 BlockDim><<<block_per_grid, BlockDim, 0, stream>>>(
      x_data, temp_storage, reducer, transformer, init, reduce_num);
  // reduce all number to final result
  ReduceKernel1D<MPType, MPType, Ty, ReduceOp, TransformOp,
                 BlockDim><<<1, BlockDim, 0, stream>>>(
      temp_storage, y_data, reducer, transformer, init, block_per_grid);
}

template <typename Tx, typename Ty, int BlockDim, typename ReduceOp,
          typename TransformOp>
static void TensorReduceImpl(
    const Tx* x_data, Ty* y_data, const platform::Place& place,
    const ReduceOp& reducer, const TransformOp& transformer, const Ty& init,
    int left_num, int reduce_num, const std::vector<int>& x_strides,
    const std::vector<int>& reduce_dim, const std::vector<int>& reduce_strides,
    const std::vector<int>& left_dim, const std::vector<int>& left_strides,
    gpuStream_t stream) {
  using MPType = typename details::MPTypeTrait<Tx>::Type;
  MPType init_mp = static_cast<MPType>(init);

#define CUB_RANK_CASE(i, ...)             \
  case i: {                               \
    constexpr auto kRank = i;             \
    switch (reduce_rank) { __VA_ARGS__; } \
  } break

#define CUB_REDUCE_RANK_CASE(i, ...)                                     \
  case i: {                                                              \
    constexpr auto kReduceRank = i;                                      \
    ReduceKernel<Tx, MPType, Ty, ReduceOp, TransformOp, BlockDim, kRank, \
                 kReduceRank><<<left_num, BlockDim, 0, stream>>>(        \
        x_data, y_data, reducer, transformer, init_mp, reduce_num,       \
        Array<int, kRank>::From(x_strides),                              \
        Array<int, kReduceRank>::From(reduce_dim),                       \
        Array<int, kReduceRank>::From(reduce_strides),                   \
        Array<int, kRank - kReduceRank>::From(left_dim),                 \
        Array<int, kRank - kReduceRank>::From(left_strides));            \
  } break

  int rank = x_strides.size();
  int reduce_rank = reduce_strides.size();
  if (rank == reduce_rank) {
    LaunchCubReduceKernel<Tx, MPType, Ty, ReduceOp, TransformOp, BlockDim>(
        x_data, y_data, place, reducer, transformer, init_mp, reduce_num,
        stream);
    return;
  }
  if (rank == 2 && reduce_rank == 1 && reduce_dim[0] == 1) {
    ReduceKernel2D<Tx, MPType, Ty, ReduceOp, TransformOp,
                   BlockDim><<<left_num, BlockDim, 0, stream>>>(
        x_data, y_data, reducer, transformer, init_mp, reduce_num);
    return;
  }
  /*
  if (rank == 3 && reduce_rank == 1 && reduce_dim[0] == 1) {
    // TODO(liangdun): we can optimize 3d case which the 2nd axis is reduced.
    // Currently, it is handled by code below, but inefficient
    return;
  }
  */

  /**
   * Since we have combined the adjacent reduce dimensions inside TensorReduce,
   * The reduce ranks and non-reduce ranks must be interleaving. That is to say,
   * the rank of Tensor must be `1010...` or `0101...` where 1 represents that
   * the dimension is about to be reduced.
   *
   * Therefore,
   * If rank is odd, only need to switch-case (rank - 1)/2 and (rank + 1)/2.
   * If rank is even, only need to switch-case rank/2.
   *
   * The total switch-case numbers reduce from 1+2+3+...+8=36 to (1+2)*4=12,
   * it would speed up compiling and make the binary size lower.
   */
  CheckReduceRankIsValid(reduce_rank, rank);
  switch (rank) {
    CUB_RANK_CASE(2, CUB_REDUCE_RANK_CASE(1););

    CUB_RANK_CASE(3, CUB_REDUCE_RANK_CASE(1); CUB_REDUCE_RANK_CASE(2););

    CUB_RANK_CASE(4, CUB_REDUCE_RANK_CASE(2););

    CUB_RANK_CASE(5, CUB_REDUCE_RANK_CASE(2); CUB_REDUCE_RANK_CASE(3););

    CUB_RANK_CASE(6, CUB_REDUCE_RANK_CASE(3););

    CUB_RANK_CASE(7, CUB_REDUCE_RANK_CASE(3); CUB_REDUCE_RANK_CASE(4););

    CUB_RANK_CASE(8, CUB_REDUCE_RANK_CASE(4););

    CUB_RANK_CASE(9, CUB_REDUCE_RANK_CASE(4); CUB_REDUCE_RANK_CASE(5););
  }

#undef CUB_REDUCE_RANK_CASE
#undef CUB_RANK_CASE
}

}  // namespace detail

template <typename Tx, typename Ty, typename ReduceOp, typename TransformOp>
void TensorReduce(const framework::Tensor& x, framework::Tensor* y,
                  std::vector<int> origin_reduce_dims, const Ty& init,
                  const ReduceOp& reducer, const TransformOp& transformer,
                  gpuStream_t stream) {
  auto x_dim = framework::vectorize<int>(x.dims());
  std::vector<int> new_x_dim, new_reduce_dims;
  int is_reduced = 0;
  for (auto e : origin_reduce_dims) {
    auto pos = e >= 0 ? e : e + x_dim.size();
    is_reduced |= 1 << e;
  }
  for (int i = 0; i < x_dim.size(); i++) {
    if ((i == 0) || (((is_reduced >> i) ^ (is_reduced >> (i - 1))) & 1)) {
      new_x_dim.push_back(x_dim[i]);
      if ((is_reduced >> i) & 1)
        new_reduce_dims.push_back(new_x_dim.size() - 1);
    } else {
      new_x_dim[new_x_dim.size() - 1] *= x_dim[i];
    }
  }
  x_dim = new_x_dim;
  origin_reduce_dims = new_reduce_dims;
  int x_rank = static_cast<int>(x_dim.size());
  std::set<int> left_set, reduce_set;
  for (int i = 0; i < x_rank; ++i) left_set.insert(i);

  for (auto e : origin_reduce_dims) {
    left_set.erase(e);
    reduce_set.insert(e);
  }

  std::vector<int> reduce_dim(reduce_set.begin(), reduce_set.end());
  std::vector<int> left_dim(left_set.begin(), left_set.end());

  std::vector<int> x_strides = detail::GetStrides(x_dim);
  std::vector<int> reduce_strides = detail::GetStrides(x_dim, reduce_dim);
  std::vector<int> left_strides = detail::GetStrides(x_dim, left_dim);
  int reduce_num = reduce_strides[0] * x_dim[reduce_dim[0]];
  int left_num = 1;
  if (left_dim.size()) left_num = left_strides[0] * x_dim[left_dim[0]];

  std::vector<int> y_dim(left_dim.size());
  for (int i = 0; i < left_dim.size(); ++i) {
    y_dim[i] = x_dim[left_dim[i]];
  }
  auto x_data = x.data<Tx>();
  auto y_data = y->mutable_data<Ty>(x.place());
  if (reduce_num == 1) {
    auto out_dims = y->dims();
    framework::TensorCopy(x, y->place(), y);
    y->Resize(out_dims);
    return;
  }

#define CUB_BLOCK_DIM_CASE(block_dim)                                    \
  case block_dim: {                                                      \
    constexpr auto kBlockDim = block_dim;                                \
    detail::TensorReduceImpl<Tx, Ty, block_dim, ReduceOp, TransformOp>(  \
        x_data, y_data, x.place(), reducer, transformer, init, left_num, \
        reduce_num, x_strides, reduce_dim, reduce_strides, left_dim,     \
        left_strides, stream);                                           \
  } break

  switch (detail::GetDesiredBlockDim(reduce_num)) {
    CUB_BLOCK_DIM_CASE(512);
    CUB_BLOCK_DIM_CASE(256);
    CUB_BLOCK_DIM_CASE(128);
    CUB_BLOCK_DIM_CASE(64);
    CUB_BLOCK_DIM_CASE(32);
    CUB_BLOCK_DIM_CASE(16);
    CUB_BLOCK_DIM_CASE(8);
    CUB_BLOCK_DIM_CASE(4);
    CUB_BLOCK_DIM_CASE(2);
  }
#undef CUB_BLOCK_DIM_CASE
}

template <typename Tx, typename ReduceOp, typename TransformOp>
struct TensorReduceFunctor {
  const framework::Tensor& x;
  framework::Tensor* y;
  std::vector<int> origin_reduce_dims;
  const double& init;
  const ReduceOp& reducer;
  const TransformOp& transformer;
  gpuStream_t stream;
  TensorReduceFunctor(const framework::Tensor& x, framework::Tensor* y,
                      std::vector<int> origin_reduce_dims, const double& init,
                      const ReduceOp& reducer, const TransformOp& transformer,
                      gpuStream_t stream)
      : x(x),
        y(y),
        origin_reduce_dims(origin_reduce_dims),
        init(init),
        reducer(reducer),
        transformer(transformer),
        stream(stream) {}

  template <typename Ty>

  void apply() const {
    const Ty& init_cast = static_cast<Ty>(init);
    TensorReduce<Tx, Ty, ReduceOp, TransformOp>(
        x, y, origin_reduce_dims, init_cast, reducer, transformer, stream);
  }
};

}  // namespace operators
}  // namespace paddle
