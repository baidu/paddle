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

#pragma once
#include <stdio.h>
#include <cstdio>
#include <vector>
#include "cub/cub.cuh"
#include "paddle/fluid/operators/top_k_op.h"
#include "paddle/fluid/platform/cuda_device_function.h"
#include "paddle/fluid/platform/float16.h"

// set cub base traits in order to handle float16
namespace cub {
template <>
struct NumericTraits<paddle::platform::float16>
    : BaseTraits<FLOATING_POINT, true, false, uint16_t,
                 paddle::platform::float16> {};
}  // namespace cub

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

struct SegmentOffsetIter {
  EIGEN_DEVICE_FUNC
  explicit SegmentOffsetIter(int num_cols) : num_cols_(num_cols) {}

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE int operator()(int idx) const {
    return idx * num_cols_;
  }

  int num_cols_;
};

// Iter using into a column
struct ColumnIndexIter {
  explicit ColumnIndexIter(int num_cols) : num_cols_(num_cols) {}

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE int operator()(
      const Eigen::array<int, 1>& ix) const {
    return ix[0] % num_cols_;
  }

  int num_cols_;
};

inline static int GetDesiredBlockDim(int dim) {
  if (dim > 128) {
    return 256;
  } else if (dim > 64) {
    return 128;
  } else if (dim > 32) {
    return 64;
  } else {
    return 32;
  }
}

template <typename T, typename IndType = int64_t>
__global__ void InitIndex(T* indices, int64_t num_rows, int64_t num_cols) {
  int col_id = threadIdx.x;
  int row_id = blockIdx.x;

  for (int64_t j = row_id; j < num_rows; j += gridDim.x) {
    for (int64_t i = col_id; i < num_cols; i += blockDim.x) {
      indices[j * num_cols + i] = static_cast<IndType>(i);
    }
  }
}

template <typename T, typename IndType = int64_t>
struct Pair {
  __device__ __forceinline__ Pair() {}
  __device__ __forceinline__ Pair(T value, IndType id) : v(value), id(id) {}

  __device__ __forceinline__ void set(T value, IndType id) {
    v = value;
    id = id;
  }

  __device__ __forceinline__ void operator=(const Pair<T, IndType>& in) {
    v = in.v;
    id = in.id;
  }

  __device__ __forceinline__ bool operator<(const T value) const {
    return (v < value);
  }

  __device__ __forceinline__ bool operator>(const T value) const {
    return (v > value);
  }
  __device__ __forceinline__ bool operator<(const Pair<T, IndType>& in) const {
    return (v < in.v) || ((v == in.v) && (id > in.id));
  }

  __device__ __forceinline__ bool operator>(const Pair<T, IndType>& in) const {
    return (v > in.v) || ((v == in.v) && (id < in.id));
  }

  T v;
  int64_t id;
};

template <typename T, typename IndType = int64_t>
__device__ __forceinline__ void AddTo(Pair<T, IndType> topk[],
                                      const Pair<T, IndType>& p, int beam_size,
                                      const bool& largest) {
  for (int k = beam_size - 2; k >= 0; k--) {
    if (largest) {
      if (topk[k] < p) {
        topk[k + 1] = topk[k];
      } else {
        topk[k + 1] = p;
        return;
      }
    } else {
      if (topk[k] > p) {
        topk[k + 1] = topk[k];
      } else {
        topk[k + 1] = p;
        return;
      }
    }
  }
  topk[0] = p;
}

template <typename T, int BlockSize, typename IndType = int64_t>
__device__ __forceinline__ void GetTopK(Pair<T, IndType> topk[], const T* src,
                                        int idx, int dim, int beam_size,
                                        const bool& largest) {
  while (idx < dim) {
    if (largest) {
      if (topk[beam_size - 1] < src[idx]) {
        Pair<T, IndType> tmp(src[idx], static_cast<IndType>(idx));
        AddTo<T>(topk, tmp, beam_size, largest);
      }
    } else {
      if (topk[beam_size - 1] > src[idx]) {
        Pair<T, IndType> tmp(src[idx], static_cast<IndType>(idx));
        AddTo<T>(topk, tmp, beam_size, largest);
      }
    }
    idx += BlockSize;
  }
}

template <typename T, int BlockSize, typename IndType = int64_t>
__device__ __forceinline__ void GetTopK(Pair<T, IndType> topk[], const T* src,
                                        int idx, int dim,
                                        const Pair<T, IndType>& max,
                                        int beam_size, const bool& largest) {
  while (idx < dim) {
    if (largest) {
      if (topk[beam_size - 1] < src[idx]) {
        Pair<T, IndType> tmp(src[idx], static_cast<IndType>(idx));
        if (tmp < max) {
          AddTo<T, IndType>(topk, tmp, beam_size, largest);
        }
      }
    } else {
      if (topk[beam_size - 1] > src[idx]) {
        Pair<T, IndType> tmp(src[idx], static_cast<IndType>(idx));
        if (tmp > max) {
          AddTo<T, IndType>(topk, tmp, beam_size, largest);
        }
      }
    }
    idx += BlockSize;
  }
}

template <typename T, int MaxLength, int BlockSize, typename IndType = int64_t>
__device__ __forceinline__ void ThreadGetTopK(Pair<T, IndType> topk[],
                                              int* beam, int beam_size,
                                              const T* src, bool* firstStep,
                                              bool* is_empty,
                                              Pair<T, IndType>* max, int dim,
                                              const int tid, bool largest) {
  if (*beam > 0) {
    int length = (*beam) < beam_size ? *beam : beam_size;
    if (*firstStep) {
      *firstStep = false;
      GetTopK<T, BlockSize, IndType>(topk, src, tid, dim, length, largest);
    } else {
      for (int k = 0; k < MaxLength; k++) {
        if (k < MaxLength - (*beam)) {
          topk[k] = topk[k + *beam];
        } else {
          topk[k].set(-static_cast<T>(INFINITY), static_cast<IndType>(-1));
        }
      }
      if (!(*is_empty)) {
        GetTopK<T, BlockSize, IndType>(topk + MaxLength - *beam, src, tid, dim,
                                       *max, length, largest);
      }
    }

    *max = topk[MaxLength - 1];
    if ((*max).v == -static_cast<T>(1)) *is_empty = true;
    *beam = 0;
  }
}

template <typename T, int MaxLength, int BlockSize, typename IndType = int64_t>
__device__ __forceinline__ void BlockReduce(Pair<T, IndType>* sh_topk,
                                            int* maxid, Pair<T, IndType> topk[],
                                            T** topVal, IndType** topIds,
                                            int* beam, int* k, const int tid,
                                            const int warp,
                                            const bool& largest) {
  while (true) {
    __syncthreads();
    if (tid < BlockSize / 2) {
      if (largest) {
        if (sh_topk[tid] < sh_topk[tid + BlockSize / 2]) {
          maxid[tid] = tid + BlockSize / 2;
        } else {
          maxid[tid] = tid;
        }
      } else {
        if (sh_topk[tid] > sh_topk[tid + BlockSize / 2]) {
          maxid[tid] = tid + BlockSize / 2;
        } else {
          maxid[tid] = tid;
        }
      }
    }
    __syncthreads();
    for (int stride = BlockSize / 4; stride > 0; stride = stride / 2) {
      if (tid < stride) {
        if (largest) {
          if (sh_topk[maxid[tid]] < sh_topk[maxid[tid + stride]]) {
            maxid[tid] = maxid[tid + stride];
          }
        } else {
          if (sh_topk[maxid[tid]] > sh_topk[maxid[tid + stride]]) {
            maxid[tid] = maxid[tid + stride];
          }
        }
      }
      __syncthreads();
    }
    __syncthreads();

    if (tid == 0) {
      **topVal = sh_topk[maxid[0]].v;
      **topIds = sh_topk[maxid[0]].id;
      (*topVal)++;
      (*topIds)++;
    }
    if (tid == maxid[0]) (*beam)++;
    if (--(*k) == 0) break;
    __syncthreads();

    if (tid == maxid[0]) {
      if (*beam < MaxLength) {
        sh_topk[tid] = topk[*beam];
      }
    }
    // NOTE(zcd): temporary solution
    unsigned mask = 0u;
    CREATE_SHFL_MASK(mask, true);

    if (maxid[0] / 32 == warp) {
      if (platform::CudaShuffleSync(mask, *beam, (maxid[0]) % 32, 32) ==
          MaxLength)
        break;
    }
  }
}

/**
 * Each block compute one sample.
 * In a block:
 * 1. every thread get top MaxLength value;
 * 2. merge to sh_topk, block reduce and get max value;
 * 3. go to the second setp, until one thread's topk value is null;
 * 4. go to the first setp, until get the topk value.
 */

template <typename T, int MaxLength, int BlockSize, typename IndType = int64_t>
__global__ void KeMatrixTopK(T* output, int output_stride, IndType* indices,
                             const T* src, int lds, int dim, int k,
                             int grid_dim, int num, bool largest = true) {
  __shared__ Pair<T, IndType> sh_topk[BlockSize];
  const int tid = threadIdx.x;
  const int warp = threadIdx.x / 32;

  const int bid = blockIdx.x;
  for (int i = bid; i < num; i += grid_dim) {
    int top_num = k;
    __shared__ int maxid[BlockSize / 2];
    T* out = output + i * output_stride;
    IndType* inds = indices + i * k;
    Pair<T, IndType> topk[MaxLength];
    int beam = MaxLength;
    Pair<T, IndType> max;
    bool is_empty = false;
    bool firststep = true;

    for (int j = 0; j < MaxLength; j++) {
      if (largest) {
        topk[j].set(-static_cast<T>(INFINITY), static_cast<IndType>(-1));
      } else {
        topk[j].set(static_cast<T>(INFINITY), static_cast<IndType>(-1));
      }
    }
    while (top_num) {
      ThreadGetTopK<T, MaxLength, BlockSize, IndType>(
          topk, &beam, k, src + i * lds, &firststep, &is_empty, &max, dim, tid,
          largest);

      sh_topk[tid] = topk[0];
      BlockReduce<T, MaxLength, BlockSize, IndType>(sh_topk, maxid, topk, &out,
                                                    &inds, &beam, &top_num, tid,
                                                    warp, largest);
    }
  }
}

template <typename T, int MaxLength, int BlockSize>
__global__ void AssignGrad(T* x_grad, const int64_t* indices, const T* out_grad,
                           size_t rows, size_t cols, size_t k) {
  for (size_t i = 0; i < rows; ++i) {
    for (size_t j = 0; j < cols; ++j) {
      x_grad[i * cols + j] = 0;
    }
    for (size_t j = 0; j < k; ++j) {
      size_t idx = indices[i * k + j];
      x_grad[i * cols + idx] = out_grad[i * k + j];
    }
  }
}

// the grad assign with the axis
template <typename T, typename IndType>
__global__ void AssignGradWithAxis(const T* grad_out, const IndType* indices,
                                   T* grad_in, int64_t pre, int64_t post,
                                   int64_t raw_height, const int64_t k) {
  // raw_height is the length of topk axis
  for (int64_t i = blockIdx.x; i < pre; i += gridDim.x) {
    const auto& base_index = i * post * k;
    const auto& base_grad = i * post * raw_height;
    for (int64_t j = threadIdx.x; j < raw_height * post; j += blockDim.x) {
      grad_in[base_grad + j] = static_cast<T>(0);
    }
    for (int64_t j = threadIdx.x; j < k * post; j += blockDim.x) {
      const int64_t idx_ij = static_cast<int64_t>(indices[base_index + j]);
      const int64_t in_ij = base_grad + (idx_ij * post) + (j % post);
      grad_in[in_ij] = grad_out[idx_ij];
    }
  }
}
// use the radix sort for the topk
template <typename T, typename IndType = int64_t>
bool SortTopk(const platform::CUDADeviceContext& ctx,
              const framework::Tensor* input_tensor, const int64_t num_cols,
              const int64_t num_rows, const int k,
              framework::Tensor* out_tensor, framework::Tensor* indices_tensor,
              bool largest = true) {
  auto cu_stream = ctx.stream();

  Tensor input_indices;
  const std::vector<int64_t> dims = {num_rows, num_cols};
  auto dim = framework::make_ddim(dims);
  input_indices.Resize(dim);
  // input_indices.Resize(num_rows*num_cols);
  input_indices.mutable_data<IndType>(ctx.GetPlace());
  size_t temp_storage_bytes = -1;

  auto ComputeBlockSize = [](int col) {
    if (col > 512)
      return 1024;
    else if (col > 256 && col <= 512)
      return 512;
    else if (col > 128 && col <= 256)
      return 256;
    else if (col > 64 && col <= 128)
      return 128;
    else
      return 64;
  };
  int64_t block_size = ComputeBlockSize(num_cols);

  unsigned int maxGridDimX = ctx.GetCUDAMaxGridDimSize().x;
  // actually, int num_rows < max_grid_size
  unsigned int grid_size = num_rows < maxGridDimX
                               ? static_cast<unsigned int>(num_rows)
                               : maxGridDimX;
  // Init a index array
  InitIndex<T, IndType><<<grid_size, block_size, 0, cu_stream>>>(
      input_indices.data<T>(), num_rows, num_cols);

  // create iter for counting input
  cub::CountingInputIterator<int64_t> counting_iter(0);
  // segment_offset is used for move to next row
  cub::TransformInputIterator<int64_t, SegmentOffsetIter,
                              cub::CountingInputIterator<int64_t>>
      segment_offsets_t(counting_iter, SegmentOffsetIter(num_cols));

  T* sorted_values_ptr;
  IndType* sorted_indices_ptr;

  Tensor temp_values;
  Tensor temp_indices;

  const T* input = input_tensor->data<T>();
  T* values = out_tensor->data<T>();
  auto* indices = indices_tensor->mutable_data<IndType>(ctx.GetPlace());

  if (k == num_cols) {
    // Doing a full sort.
    sorted_values_ptr = values;
    sorted_indices_ptr = indices;
  } else {
    temp_values.Resize(dim);
    temp_indices.Resize(dim);
    sorted_values_ptr = temp_values.mutable_data<T>(ctx.GetPlace());
    sorted_indices_ptr = temp_indices.mutable_data<IndType>(ctx.GetPlace());
  }

  // Get temp storage buffer size, maybe can allocate a fixed buffer to save
  // time.
  if (largest) {
    auto err = cub::DeviceSegmentedRadixSort::SortPairsDescending(
        nullptr, temp_storage_bytes, input, sorted_values_ptr,
        input_indices.data<IndType>(), sorted_indices_ptr, num_cols * num_rows,
        num_rows, segment_offsets_t, segment_offsets_t + 1, 0, sizeof(T) * 8,
        cu_stream);
    if (err != cudaSuccess) {
      LOG(ERROR)
          << "TopKOP failed as could not launch "
             "cub::DeviceSegmentedRadixSort::SortPairsDescending to calculate "
             "temp_storage_bytes, status: "
          << cudaGetErrorString(err);
      return false;
    }
  } else {
    auto err = cub::DeviceSegmentedRadixSort::SortPairs(
        nullptr, temp_storage_bytes, input, sorted_values_ptr,
        input_indices.data<IndType>(), sorted_indices_ptr, num_cols * num_rows,
        num_rows, segment_offsets_t, segment_offsets_t + 1, 0, sizeof(T) * 8,
        cu_stream);
    if (err != cudaSuccess) {
      LOG(ERROR) << "TopKOP failed as could not launch "
                    "cub::DeviceSegmentedRadixSort::SortPairs to calculate "
                    "temp_storage_bytes, status: "
                 << cudaGetErrorString(err);
      return false;
    }
  }
  Tensor temp_storage;
  temp_storage.mutable_data<uint8_t>(ctx.GetPlace(), temp_storage_bytes);

  if (largest) {
    auto err = cub::DeviceSegmentedRadixSort::SortPairsDescending(
        temp_storage.data<uint8_t>(), temp_storage_bytes, input,
        sorted_values_ptr, input_indices.data<IndType>(), sorted_indices_ptr,
        num_cols * num_rows, num_rows, segment_offsets_t, segment_offsets_t + 1,
        0, sizeof(T) * 8, cu_stream);
    if (err != cudaSuccess) {
      LOG(ERROR) << "TopKOP failed as could not launch "
                    "cub::DeviceSegmentedRadixSort::SortPairsDescending to "
                    "sort input, "
                    "temp_storage_bytes: "
                 << temp_storage_bytes
                 << ", status: " << cudaGetErrorString(err);
      return false;
    }
  } else {
    auto err = cub::DeviceSegmentedRadixSort::SortPairs(
        temp_storage.data<uint8_t>(), temp_storage_bytes, input,
        sorted_values_ptr, input_indices.data<IndType>(), sorted_indices_ptr,
        num_cols * num_rows, num_rows, segment_offsets_t, segment_offsets_t + 1,
        0, sizeof(T) * 8, cu_stream);
    if (err != cudaSuccess) {
      LOG(ERROR) << "TopKOP failed as could not launch "
                    "cub::DeviceSegmentedRadixSort::SortPairs to "
                    "sort input, "
                    "temp_storage_bytes: "
                 << temp_storage_bytes
                 << ", status: " << cudaGetErrorString(err);
      return false;
    }
  }
  auto& dev = *ctx.eigen_device();
  if (k < num_cols) {
    // copy sliced data to output.
    const Eigen::DSizes<Eigen::DenseIndex, 2> slice_indices{0, 0};
    const Eigen::DSizes<Eigen::DenseIndex, 2> slice_sizes{num_rows, k};
    auto e_indices = EigenMatrix<IndType>::From(*indices_tensor, dim);
    auto e_tmp_indices = EigenMatrix<IndType>::From(temp_indices);

    std::vector<int64_t> odims = {num_rows, k};
    auto dim = framework::make_ddim(odims);
    auto e_values = EigenMatrix<T>::From(*out_tensor, dim);
    auto e_tmp_values = EigenMatrix<T>::From(temp_values);

    e_indices.device(dev) = e_tmp_indices.slice(slice_indices, slice_sizes);
    e_values.device(dev) = e_tmp_values.slice(slice_indices, slice_sizes);
  }
  return true;
}
}  // namespace operators
}  // namespace paddle
