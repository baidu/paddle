// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
#include "cub/cub.cuh"
#endif

#ifdef __HIPCC__
#include <hipcub/hipcub.hpp>
namespace cub = hipcub;
#endif

#include "paddle/fluid/framework/array.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/fluid/platform/cuda_device_function.h"

// Reduce split or not, Whether to use ReduceHigherDim
#define REDUCE_SPLIT_BOUNDARY 512

namespace paddle {
namespace operators {
namespace detail {

// Post processing function for sum, max, min, prod, any
template <typename Tx, typename Ty = Tx>
struct IdentityFunctor {
  HOSTDEVICE explicit inline IdentityFunctor(int n) {}

  HOSTDEVICE inline Ty operator()(const Tx& x) const {
    return static_cast<Ty>(x);
  }
};

// Post processing function for mean
template <typename T>
struct DivideFunctor {
  HOSTDEVICE explicit inline DivideFunctor(int n) : n_inv((T)(1.0 / n)) {}

  HOSTDEVICE inline T operator()(const T& x) const { return x * n_inv; }

 private:
  T n_inv;
};

static inline int GetLastPow2(int n) {
  n |= (n >> 1);
  n |= (n >> 2);
  n |= (n >> 4);
  n |= (n >> 8);
  n |= (n >> 16);
  return std::max(1, n - (n >> 1));
}

// get strides of x_dim, reduce_dim and left_dim for reduceLastDim and reduceAny
static inline std::vector<int> GetDimStrides(const std::vector<int>& dims,
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
constexpr int kMaxThread = 256;
constexpr int kWarpSize = 64;
#else
constexpr int kMaxThread = 128;
constexpr int kWarpSize = 32;
#endif

// get blockDim for reduceLastDim and reduceAny
static inline int GetBlockDim(int block_dim) {
  return block_dim >= kMaxThread ? kMaxThread : GetLastPow2(block_dim);
}

// check reduce rand is valid
static inline void CheckReduceRank(int reduce_rank, int rank) {
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

// convert dims from vector to array
template <typename T, size_t ElementCount, typename VectorLikeType>
static inline paddle::framework::Array<T, ElementCount> VectorToArray(
    const VectorLikeType& vec) {
  PADDLE_ENFORCE_EQ(vec.size(), ElementCount,
                    platform::errors::InvalidArgument(
                        "Cub reduce Array: size not match. Received "
                        "vec.size() %d !=  ElementCount %d.",
                        vec.size(), ElementCount));
  size_t n = static_cast<size_t>(vec.size());
  paddle::framework::Array<T, ElementCount> ret;
  for (size_t i = 0; i < n; ++i) {
    ret[i] = vec[i];
  }
  return ret;
}

}  // namespace detail

using Tensor = framework::Tensor;

enum ReduceType {
  kReduceAll = 0x00,        // when reduce_rank == x_rank
  kReduceLastDim = 0x01,    // when reduce_dim[0] == x_dim.size() - 1;
  kReduceHigherDim = 0x02,  // ReduceFirstDim or reduceSecondDim
  kReduceAny = 0x03,        // when reduce_dim.size() > 1
};

// reduce config
template <typename Ty>
struct ReduceConfig {
  ReduceConfig(const std::vector<int>& origin_reduce_dims,
               const std::vector<int>& origin_x_dim)
      : reduce_dims_origin(origin_reduce_dims), x_dim(origin_x_dim) {}

  // get the parameters of reduceKernel
  void Run() {
    // step1: update the reduce_dim left_dim and x_dim
    SetReduceDim();

    // step2: get the strides of dim for reduceAny and reduceLastDim
    SetStrides();

    // step3: get the type of reduce
    SetReduceType();

    // step4: set the block and grid for launch kernel
    SetBlockDim();
  }

  // when should_reduce_again is true, we need malloc temp space for temp data
  void SetOutputData(Ty* y_data, const platform::Place& place,
                     framework::Tensor* tmp) {
    if (should_reduce_again) {
      output_data = tmp->mutable_data<Ty>(
          framework::make_ddim(
              {static_cast<int64_t>(left_num * grid.z * grid.y * sizeof(Ty))}),
          place);
    } else {
      output_data = y_data;
    }
  }

 private:
  // set reduce_dim, left_dim and update x_dim
  // eg: x_dim = [2, 4, 6] origin_reduce_dims = [0, 1]
  //     --SetReduceDim--> x_dim = [8,6], reduce_dim = [0], left_dim = [1]
  void SetReduceDim() {
    std::set<int> reduce_set;
    for (auto e : reduce_dims_origin) {
      auto pos = e >= 0 ? e : e + x_dim.size();
      reduce_set.insert(pos);
    }

    std::vector<int> reduce_dim_temp(reduce_set.begin(), reduce_set.end());
    std::sort(reduce_dim_temp.begin(), reduce_dim_temp.end());

    // update reduce_dim and x_dim
    std::vector<int> x_new_dim;

    reduce_dim.push_back(reduce_dim_temp[0]);
    x_new_dim.push_back(x_dim[0]);

    int idx_reduce = 1;
    int num = 0;

    if (reduce_dim_temp.size() > 1) {
      for (int i = 1; i < x_dim.size(); i++) {
        if ((idx_reduce < reduce_dim_temp.size()) &&
            (i == reduce_dim_temp[idx_reduce])) {
          int result =
              reduce_dim_temp[idx_reduce] - reduce_dim[reduce_dim.size() - 1];
          bool is_equal = ((result - num) == 1);
          if (is_equal) {
            x_new_dim[x_new_dim.size() - 1] *= x_dim[i];
            num++;
          } else {
            reduce_dim.push_back(reduce_dim_temp[idx_reduce] - num);
            x_new_dim.push_back(x_dim[i]);
          }
          idx_reduce++;
        } else {
          x_new_dim.push_back(x_dim[i]);
        }
      }
    } else {
      x_new_dim = x_dim;
    }

    // update x_dim
    x_dim = x_new_dim;
    std::vector<int>().swap(x_new_dim);

    std::vector<int> reduce_dim_new;
    int is_reduced = 0;
    for (auto e : reduce_dim) {
      is_reduced |= 1 << e;
    }

    std::vector<int>().swap(reduce_dim);

    for (int i = 0; i < x_dim.size(); i++) {
      if ((i == 0) || (((is_reduced >> i) ^ (is_reduced >> (i - 1))) & 1)) {
        x_new_dim.push_back(x_dim[i]);
        if ((is_reduced >> i) & 1)
          reduce_dim_new.push_back(x_new_dim.size() - 1);
      } else {
        x_new_dim[x_new_dim.size() - 1] *= x_dim[i];
      }
    }

    x_dim = x_new_dim;
    reduce_dim = reduce_dim_new;

    int x_rank = static_cast<int>(x_dim.size());
    std::set<int> left_set;

    for (int i = 0; i < x_rank; ++i) {
      left_set.insert(i);
    }

    for (auto e : reduce_dim) {
      left_set.erase(e);
    }

    left_dim.assign(left_set.begin(), left_set.end());
  }

  // set x_strides, reduce_strides, left_strides for reduceLastDim and reduceAny
  // eg: x_dim = [8, 6], reduce_dim = [0], left_dim = [1]
  //     --SetStrides--> x_strides= [6,1], reduce_strides = [1],
  //     left_strides = [1]
  void SetStrides() {
    std::vector<int> idx_dim;
    for (int i = 0; i < x_dim.size(); i++) {
      idx_dim.push_back(i);
    }

    x_strides = detail::GetDimStrides(x_dim, idx_dim);
    reduce_strides = detail::GetDimStrides(x_dim, reduce_dim);
    left_strides = detail::GetDimStrides(x_dim, left_dim);
    reduce_num = reduce_strides[0] * x_dim[reduce_dim[0]];

    left_num = 1;
    if (left_dim.size()) {
      left_num = left_strides[0] * x_dim[left_dim[0]];
    }
  }

  // get the reduceType
  // eg: x_dim = [8, 6] reduce_dim = [0] --> ReduceHigherDim -->reduceFirstDim
  //     x_dim = [8, 6] reduce_dim = [1] --> reduceLastDim
  //     x_dim = [8] reduce_dim = [0] --> reduceAll
  //     x_dim = [8, 6, 4, 2] reduce_dim = [0, 2] --> reduceAny
  void SetReduceType() {
    int rank = x_dim.size();
    int reduce_rank = reduce_dim.size();
    bool is_large_enough = (reduce_num > REDUCE_SPLIT_BOUNDARY / 2) ||
                           (left_num > REDUCE_SPLIT_BOUNDARY);

    if (rank == reduce_rank) {
      reduce_type = static_cast<int>(ReduceType::kReduceAll);

    } else if (rank == 2 && reduce_rank == 1 && reduce_dim[0] == 1) {
      reduce_type = static_cast<int>(ReduceType::kReduceLastDim);

    } else if (reduce_rank == 1 &&
               ((rank == 2 && is_large_enough) || rank != 2)) {
      // ReduceFirstDim and reduceSecondDim
      reduce_type = static_cast<int>(ReduceType::kReduceHigherDim);

    } else {
      reduce_type = static_cast<int>(ReduceType::kReduceAny);
    }
  }

  // set block and grid for launch kernel
  // for ReduceHigherDim: if block is enough -> splite reduce_num
  //                     else init block(32, 1) grid(block_num, 1)
  // for others: block(block_num, 1) , grid(left_num, 1)
  void SetBlockDim() {
    // init
    int block_num = detail::GetBlockDim(reduce_num);
    should_reduce_again = false;

    dim3 block_dim(block_num, 1);
    dim3 grid_dim(left_num, 1);
    blocking_size = reduce_num;

    if (reduce_type == ReduceType::kReduceHigherDim) {
      int last_dim_num = x_dim.back();
      // update left_num
      int grid_z = left_num / last_dim_num;
      left_num = last_dim_num;

      block_dim.z = 1;
      grid_dim.z = grid_z;

      int device_id = platform::GetCurrentDeviceId();
      int max_mp = platform::GetCUDAMultiProcessors(device_id);
      int max_threads_per_mp =
          platform::GetCUDAMaxThreadsPerMultiProcessor(device_id);
      int max_threads = max_threads_per_mp * max_mp;

      // init
      int num_block = (max_threads / left_num);

      if (num_block > 1 && reduce_num >= REDUCE_SPLIT_BOUNDARY) {
        blocking_size = detail::GetLastPow2(reduce_num / num_block);

        if (blocking_size <= 1) {
          blocking_size = detail::GetLastPow2(sqrt(reduce_num));
        } else if (blocking_size * 2 < reduce_num) {
          blocking_size *= 2;
        }

        should_reduce_again = true;

        block_dim.x = 32;
        block_dim.y = 1;
        grid_dim.x = (left_num + block_dim.x - 1) / block_dim.x;
        grid_dim.y = (reduce_num + blocking_size - 1) / blocking_size;

      } else {
        block_dim.x = 32;
        block_dim.y = 1;
        blocking_size = reduce_num;
        grid_dim.x = (left_num + block_dim.x - 1) / block_dim.x;
        grid_dim.y = 1;
      }
    }

    block = block_dim;
    grid = grid_dim;
  }

 public:
  std::vector<int> reduce_dims_origin;
  std::vector<int> reduce_dim;
  std::vector<int> x_dim;
  std::vector<int> left_dim;
  std::vector<int> x_strides;
  std::vector<int> left_strides;
  std::vector<int> reduce_strides;

  int reduce_type;
  int reduce_num;
  int left_num;
  int blocking_size;
  bool should_reduce_again;

  Ty* output_data;

  dim3 block;
  dim3 grid;
};

template <typename T, typename ReduceOp>
__device__ __forceinline__ T WarpReduce(T val, ReduceOp reducer) {
  unsigned mask = 0u;
  CREATE_SHFL_MASK(mask, true);
  for (int stride = detail::kWarpSize / 2; stride > 0; stride >>= 1) {
    T temp = paddle::platform::CudaShuffleDownSync(mask, val, stride);
    val = reducer(val, temp);
  }
  return val;
}

/* e.g.
 * |---------block---------|
 * |warp0|warp1|warp2|warp3|
 * |0~31|32~63|64~95|96~127|  ---->blockDim.x = 128
 *  \|/  \|/   \|/    \|/     ---->1. First WarpReduce in each warp
 * res0  res1  res2  res3     ---->2. Store result of each warp to shared memory
 *   \    \    /     /        ---->3. Load the result above from shared memory
 *        res                         to warp0 and process the second WarpReduce
 */
template <typename T, typename ReduceOp>
__device__ __forceinline__ T BlockReduce(T val, ReduceOp reducer) {
  using detail::kWarpSize;
  __shared__ T shared[kWarpSize];
  int block_dim_x = blockDim.x;
  if (blockDim.x > kWarpSize) {
    block_dim_x = blockDim.x / kWarpSize;
    int lane = threadIdx.x % kWarpSize;
    int wid = threadIdx.x / kWarpSize;
    val = WarpReduce(val, reducer);
    if (lane == 0) {
      shared[wid] = val;
    }
    __syncthreads();
    val = shared[lane];
  }

  unsigned mask = 0u;
  CREATE_SHFL_MASK(mask, true);
  for (int stride = 1; stride < block_dim_x; stride <<= 1) {
    T temp = paddle::platform::CudaShuffleDownSync(mask, val, stride);
    val = reducer(val, temp);
  }
  return val;
}

// when reduce_dim.size() == 1 and reduce_dim[0] == x_dim.size() - 1, this
// function will be used
// blockId.x -> left_num, threadId.x -> reduce_num
template <typename Tx, typename Ty, typename ReduceOp, typename TransformOp>
__device__ __forceinline__ void ReduceLastDim(const Tx* x, Ty* y,
                                              ReduceOp reducer,
                                              TransformOp transformer, Ty init,
                                              int reduce_num) {
  int idx_x = blockIdx.x * reduce_num;
  int idx_y = threadIdx.x;
  Ty reduce_var = init;
  for (int idx_y = threadIdx.x; idx_y < reduce_num; idx_y += blockDim.x) {
    reduce_var =
        reducer(reduce_var, static_cast<Ty>(transformer(x[idx_x + idx_y])));
  }
  __syncthreads();

  reduce_var = BlockReduce(reduce_var, reducer);

  if (threadIdx.x == 0) {
    y[blockIdx.x] = reduce_var;
  }
}

// when reduce_dim.size() == 1 and reduce_dim[0] != x_dim.size() - 1, this
// function will be used
// eg: x_dim = {nz, ny, nx}, nx != 1, axis can be 0 or 1
//     if axis = 1 then grid.z = nz, grid.y = ny / block_size, grid.x = nx / 32
//     else grid.z = 1, grid.y = ny / block_size, grid.x = nx /32
template <typename Tx, typename Ty, typename ReduceOp, typename TransformOp>
__device__ __forceinline__ void ReduceHigherDim(const Tx* x, Ty* y,
                                                ReduceOp reducer,
                                                TransformOp transformer,
                                                Ty init, int reduce_num,
                                                int left_num, int block_size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int idy = blockIdx.y * block_size;

  Ty reduce_var = init;

  if (idx < left_num) {
    int loop = reduce_num - idy;
    loop = loop > block_size ? block_size : loop;

    for (int iy = 0; iy < loop; iy++) {
      int id = (idy + iy) * left_num + idx + blockIdx.z * reduce_num * left_num;
      reduce_var = reducer(reduce_var, static_cast<Ty>(transformer(x[id])));
    }

    y[idx + blockIdx.y * left_num + blockIdx.z * gridDim.y * left_num] =
        reduce_var;
  }
}

// when reduce_dim.size() != 1 and reduce_dim.size() != x_dim.size(), this
// function will be used
// blockId.x -> left_num, threadId.x -> reduce_num
template <typename Tx, typename Ty, typename ReduceOp, typename TransformOp,
          int Rank, int ReduceRank>
__device__ __forceinline__ void ReduceAny(
    const Tx* x, Ty* y, ReduceOp reducer, TransformOp transformer,
    int reduce_num, paddle::framework::Array<int, Rank> x_strides,
    paddle::framework::Array<int, ReduceRank> reduce_dim,
    paddle::framework::Array<int, ReduceRank> reduce_strides,
    paddle::framework::Array<int, Rank - ReduceRank> left_dim,
    paddle::framework::Array<int, Rank - ReduceRank> left_strides) {
  int sub_index[Rank];
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
  for (int k = 0; k < Rank; ++k) {
    idx_x += (sub_index[k] * x_strides[k]);
  }
  Ty reduce_var = static_cast<Ty>(transformer(x[idx_x]));

  for (int i = threadIdx.x + blockDim.x; i < reduce_num; i += blockDim.x) {
    int reduce_idx = i;

    for (int j = 0; j < ReduceRank; ++j) {
      sub_index[reduce_dim[j]] = reduce_idx / reduce_strides[j];
      reduce_idx %= reduce_strides[j];
    }

    int idx_x = 0;
    for (int k = 0; k < Rank; ++k) {
      idx_x += (sub_index[k] * x_strides[k]);
    }

    reduce_var = static_cast<Ty>(
        reducer(reduce_var, static_cast<Ty>(transformer(x[idx_x]))));
  }
  __syncthreads();

  reduce_var = BlockReduce(reduce_var, reducer);
  if (threadIdx.x == 0) {
    y[blockIdx.x] = reduce_var;
  }
}

// module function designed for global function
template <typename Tx, typename Ty, typename ReduceOp, typename TransformOp,
          int Rank, int ReduceRank>
__device__ __forceinline__ void ReduceModule(
    const Tx* x, Ty* y, ReduceOp reducer, TransformOp transformer, Ty init,
    int reduce_num, int left_num, int blocking_size, int reduce_type,
    paddle::framework::Array<int, Rank> x_strides,
    paddle::framework::Array<int, ReduceRank> reduce_dim,
    paddle::framework::Array<int, ReduceRank> reduce_strides,
    paddle::framework::Array<int, Rank - ReduceRank> left_dim,
    paddle::framework::Array<int, Rank - ReduceRank> left_strides) {
  // reduce_rank == 1 && reduce_dim[0] == x_dim.size() - 1
}

template <typename Tx, typename Ty, typename ReduceOp, typename TransformOp,
          int Rank, int ReduceRank>
__global__ void ReduceKernelFunction(
    const Tx* x, Ty* y, ReduceOp reducer, TransformOp transformer, Ty init,
    int reduce_num, int left_num, int block_size, int reduce_type
    ,
    paddle::framework::Array<int, Rank> x_strides,
    paddle::framework::Array<int, ReduceRank> reduce_dim,
    paddle::framework::Array<int, ReduceRank> reduce_strides,
    paddle::framework::Array<int, Rank - ReduceRank> left_dim,
    paddle::framework::Array<int, Rank - ReduceRank> left_strides
    ) {
  if (reduce_type == ReduceType::kReduceLastDim) {
    ReduceLastDim<Tx, Ty, ReduceOp, TransformOp>(x, y, reducer, transformer,
                                                 init, reduce_num);

    // reduce_rank == 1 && reduce_dim[0] != x_dim.size() - 1
  } else if (reduce_type == ReduceType::kReduceHigherDim) {
    ReduceHigherDim<Tx, Ty, ReduceOp, TransformOp>(
        x, y, reducer, transformer, init, reduce_num, left_num, block_size);

    // reduce_rank >= 2
  } else {
     ReduceAny<Tx, Ty, ReduceOp, TransformOp, Rank, ReduceRank>(
         x, y, reducer, transformer, reduce_num, x_strides, reduce_dim,
         reduce_strides, left_dim, left_strides);
  }
}

template <typename Tx, typename Ty, typename ReduceOp, int Rank, int ReduceRank>
static inline void LaunchReduceKernel(const Tx* x_data, Ty* y_data,
                               const ReduceOp& reducer, Ty init,
                               gpuStream_t stream, ReduceConfig<Ty> config) {
  using TransformOp = typename ReduceOp::Transformer;

  ReduceKernelFunction<Tx, Ty, ReduceOp, TransformOp, Rank,
                       ReduceRank><<<config.grid, config.block, 0, stream>>>(
      x_data, config.output_data, reducer, TransformOp(config.reduce_num), init,
      config.reduce_num, config.left_num, config.blocking_size,
      config.reduce_type
      , detail::VectorToArray<int, Rank>(config.x_strides),
      detail::VectorToArray<int, ReduceRank>(config.reduce_dim),
      detail::VectorToArray<int, ReduceRank>(config.reduce_strides),
      detail::VectorToArray<int, Rank - ReduceRank>(config.left_dim),
      detail::VectorToArray<int, Rank - ReduceRank>(config.left_strides)
      );

  if (config.should_reduce_again) {
    dim3 block(config.block.x, 1, 1);
    dim3 grid(config.grid.x, 1, config.grid.z);

    ReduceKernelFunction<Ty, Ty, ReduceOp, detail::IdentityFunctor<Ty>, Rank,
                         ReduceRank><<<grid, block, 0, stream>>>(
        config.output_data, y_data, reducer,
        detail::IdentityFunctor<Ty>(config.grid.y), init, config.grid.y,
        config.left_num, config.grid.y, ReduceType::kReduceHigherDim
        ,
        detail::VectorToArray<int, Rank>(config.x_strides),
        detail::VectorToArray<int, ReduceRank>(config.reduce_dim),
        detail::VectorToArray<int, ReduceRank>(config.reduce_strides),
        detail::VectorToArray<int, Rank - ReduceRank>(config.left_dim),
        detail::VectorToArray<int, Rank - ReduceRank>(config.left_strides)
        );
  }
}

template <typename Tx, typename Ty, typename ReduceOp>
static void ReduceKernelImpl(const Tx* x_data, Ty* y_data,
                             const ReduceOp& reducer, Ty init,
                             gpuStream_t stream, ReduceConfig<Ty> config) {
  int reduce_rank = config.reduce_strides.size();
  int rank = config.x_strides.size();

#define CUB_RANK_CASE(i, ...)             \
  case i: {                               \
    constexpr auto Rank = i;              \
    switch (reduce_rank) { __VA_ARGS__; } \
  } break

#define CUB_REDUCE_RANK_CASE(i, ...)                        \
  case i: {                                                 \
    constexpr auto ReduceRank = i;                          \
    LaunchReduceKernel<Tx, Ty, ReduceOp, Rank, ReduceRank>( \
        x_data, y_data, reducer, init, stream, config);     \
  } break

  detail::CheckReduceRank(reduce_rank, rank);
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

template <typename Tx, typename Ty,
          template <typename, typename> class ReduceOp>
void TensorReduceFunctorImpl(const framework::Tensor& x, framework::Tensor* y,
                             std::vector<int> origin_reduce_dims,
                             gpuStream_t stream) {
  auto x_dim = framework::vectorize<int>(x.dims());
  auto config = ReduceConfig<Ty>(origin_reduce_dims, x_dim);
  config.Run();  // get the parameters of LaunchReduceKernel

  // after config.run()
  // SetOutputData for ReduceHigherDim when should_reduce_again is true,
  //   temp_output should be stored temp_data in output_data space or stored in
  //   y_data;
  framework::Tensor tmp;
  auto x_data = x.data<Tx>();
  auto y_data = y->mutable_data<Ty>(x.place());

  if (config.reduce_num == 1) {
    auto out_dims = y->dims();
    framework::TensorCopy(x, y->place(), y);
    y->Resize(out_dims);
    return;
  }

  config.SetOutputData(y_data, x.place(), &tmp);

  using TransformOp = typename ReduceOp<Tx, Ty>::Transformer;
  auto reducer = ReduceOp<Tx, Ty>();
  // launch CUB::Reduce
  if (config.reduce_type == static_cast<int>(ReduceType::kReduceAll)) {
    cub::TransformInputIterator<Ty, TransformOp, const Tx*> trans_x(
        x_data, TransformOp(config.reduce_num));
    size_t temp_storage_bytes = 0;
    cub::DeviceReduce::Reduce(nullptr, temp_storage_bytes, trans_x, y_data,
                              config.reduce_num, reducer, reducer.initial(),
                              stream);
    framework::Tensor tmp;
    auto* temp_storage = tmp.mutable_data<uint8_t>(
        framework::make_ddim({static_cast<int64_t>(temp_storage_bytes)}),
        x.place());
    cub::DeviceReduce::Reduce(temp_storage, temp_storage_bytes, trans_x, y_data,
                              config.reduce_num, reducer, reducer.initial(),
                              stream);

    return;
  }
  ReduceKernelImpl<Tx, Ty, ReduceOp<Tx, Ty>>(x_data, y_data, reducer,
                                             reducer.initial(), stream, config);
}

template <typename Tx, template <typename, typename> class ReduceOp>
struct TensorReduceFunc {
  const framework::Tensor& x;
  framework::Tensor* y;
  std::vector<int> origin_reduce_dims;
  gpuStream_t stream;
  TensorReduceFunc(const framework::Tensor& x, framework::Tensor* y,
                   std::vector<int> origin_reduce_dims, gpuStream_t stream)
      : x(x), y(y), origin_reduce_dims(origin_reduce_dims), stream(stream) {}

  template <typename Ty>
  void apply() const {
    TensorReduceFunctorImpl<Tx, Ty, ReduceOp>(x, y, origin_reduce_dims, stream);
  }
};

}  // namespace operators
}  // namespace paddle
