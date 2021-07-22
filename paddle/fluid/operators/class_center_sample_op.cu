//   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#ifdef PADDLE_WITH_HIP
#include <hiprand.h>
#include <hiprand_kernel.h>
#include <hipcub/hipcub.hpp>
typedef hiprandState curandState;
namespace cub = hipcub;
#else
#include <curand.h>
#include <curand_kernel.h>
#include <cub/cub.cuh>
#endif

#include <iterator>
#include <random>
#include "paddle/fluid/operators/class_center_sample_op.h"

#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
#include "paddle/fluid/platform/collective_helper.h"
#include "paddle/fluid/platform/nccl_helper.h"
#endif

namespace paddle {
namespace operators {
#define CUDA_KERNEL_LOOP(i, n)                            \
  for (int32_t i = blockIdx.x * blockDim.x + threadIdx.x, \
               step = blockDim.x * gridDim.x;             \
       i < (n); i += step)

using Tensor = framework::Tensor;

static constexpr int kNumCUDAThreads = 512;
static constexpr int kNumMaxinumNumBlocks = 4096;

inline int32_t NumBlocks(const int32_t n) {
  return std::min((n + kNumCUDAThreads - 1) / kNumCUDAThreads,
                  kNumMaxinumNumBlocks);
}

template <typename T>
__global__ void RandomSampleClassCenter(const int64_t n, int64_t seed,
                                        int64_t increment,
                                        const int64_t max_val, T* buffer) {
  const int id = blockIdx.x * blockDim.x + threadIdx.x;
  curandState localState;
  size_t local_seed =
      (static_cast<size_t>(seed) + 0x9E3779B9U +
       (static_cast<size_t>(id) << 6U) + (static_cast<size_t>(id) >> 2U));
#ifdef PADDLE_WITH_HIP
  hiprand_init(local_seed, id, increment, &localState);
  CUDA_KERNEL_LOOP(i, n) {
    buffer[i] = static_cast<T>(hiprand(&localState) % max_val);
  }
#else
  curand_init(local_seed, id, increment, &localState);
  CUDA_KERNEL_LOOP(i, n) {
    buffer[i] = static_cast<T>(curand(&localState) % max_val);
  }
#endif
}

template <typename T>
__global__ void MarkPositiveClassCenter(const int64_t n, const int64_t rank,
                                        const T* class_interval_ptr,
                                        const int64_t num_classes,
                                        const T* labels, T* out, T* index) {
  CUDA_KERNEL_LOOP(i, n) {
    T label = labels[i] - class_interval_ptr[rank];
    if (label >= 0 && label < num_classes) {
      out[label] = label - num_classes;
    }
    index[i] = static_cast<T>(i);
  }
}

template <typename T>
__device__ void FindIntervalIndex(const T* class_interval_ptr,
                                  const int64_t nranks, const int value,
                                  int64_t* find_index) {
  int64_t start = 0;
  int64_t end = nranks;
  int64_t mid = ((end - start) >> 2) + start + 1;
  while (start < end) {
    if (class_interval_ptr[mid] == value) break;
    if (class_interval_ptr[mid] > value)
      end = mid - 1;
    else
      start = mid;
    mid = ((end - start) >> 2) + start + 1;
  }
  *find_index = min(mid, end);
}

template <typename T>
__global__ void GetClassCenterBound(const int64_t n, const int64_t nranks,
                                    const T* class_interval_ptr,
                                    const T* key_ptr, const T* value_ptr,
                                    T* bound_index, T* bound_value) {
  CUDA_KERNEL_LOOP(i, n) {
    if (i != 0) {
      int64_t cur_index, pre_index;
      FindIntervalIndex(class_interval_ptr, nranks, key_ptr[i], &cur_index);
      FindIntervalIndex(class_interval_ptr, nranks, key_ptr[i - 1], &pre_index);
      if (cur_index > pre_index) {
        assert(cur_index < nranks);
#pragma unroll
        for (int32_t j = pre_index + 1; j <= cur_index; ++j) {
          bound_index[j] = static_cast<T>(i);
          bound_value[j] = value_ptr[i];
        }
      }
    }
  }
  CUDA_KERNEL_LOOP(i, nranks + 1) {
    int64_t first_index, last_index;
    FindIntervalIndex(class_interval_ptr, nranks, key_ptr[0], &first_index);
    FindIntervalIndex(class_interval_ptr, nranks, key_ptr[n - 1], &last_index);
    if (i <= first_index) {
      bound_index[i] = 0;
      bound_value[i] = value_ptr[0];
    } else if (i > last_index) {
      bound_index[i] = n;
      bound_value[i] = value_ptr[n - 1] + 1;
    }
  }
}

template <typename T>
__global__ void GetRemappedLabel(const int64_t n, const int64_t nranks,
                                 const T* sampled_class_interval_ptr,
                                 const T* bound_index, const T* bound_value,
                                 const T* label_map_key, T* label_map_value,
                                 T* mapped_label) {
  CUDA_KERNEL_LOOP(i, n) {
#pragma unroll
    for (int64_t j = 0; j < nranks; j++) {
      if (i >= bound_index[j] && i < bound_index[j + 1]) {
        label_map_value[i] =
            label_map_value[i] - bound_value[j] + sampled_class_interval_ptr[j];
      }
    }
    mapped_label[label_map_key[i]] = label_map_value[i];
  }
}

template <typename T>
__global__ void GetSampledClassCenter(const int64_t n, const T* src, T* dst) {
  CUDA_KERNEL_LOOP(i, n) { dst[i] = src[i]; }
}

// aligned vector generates vectorized load/store on CUDA
template <typename T, int Size>
struct alignas(sizeof(T) * Size) AlignedVector {
  T val[Size];
};

template <typename T>
inline int VectorizedSize(const T* pointer) {
  uint64_t address = reinterpret_cast<uint64_t>(pointer);
  constexpr int vec4 = std::alignment_of<AlignedVector<T, 4>>::value;  // NOLINT
  if (address % vec4 == 0) {
    return 4;
  }
  return 1;
}

#undef CUDA_KERNEL_LOOP

template <typename T>
class NotEqualToPreviousAdjacentIterator {
 public:
  using self_type = NotEqualToPreviousAdjacentIterator;
  using value_type = T;
  using difference_type = std::ptrdiff_t;
  using pointer = T*;
  using reference = T;
  using iterator_category = std::input_iterator_tag;

 public:
  __host__ __device__ __forceinline__
  NotEqualToPreviousAdjacentIterator(const T* arr, int64_t offset)
      : arr_(arr), offset_(offset) {}

  __host__ __device__ __forceinline__ reference operator*() const {
    return offset_ == 0 ? 0 : (arr_[offset_] == arr_[offset_ - 1] ? 0 : 1);
  }

  template <typename Distance>
  __host__ __device__ __forceinline__ self_type operator+(Distance n) const {
    self_type ret(arr_, offset_ + n);
    return ret;
  }

  template <typename Distance>
  __host__ __device__ __forceinline__ reference operator[](Distance n) const {
    return *(*this + n);
  }

 private:
  const T* arr_;
  int64_t offset_;
};

template <typename T>
struct ActualNumSampledFunctor {
  __host__ __device__ __forceinline__ T operator()(const T& a,
                                                   const T& b) const {
    return max(num_samples, (b - a));
  }
  T num_samples;
  explicit ActualNumSampledFunctor(const T num) : num_samples(num) {}
};

template <typename T>
class MemoryBuffer {
 public:
  MemoryBuffer(const int num_buffer_ele, const int num_temp_ele,
               const int nranks, const int num_classes,
               const platform::Place& place, const gpuStream_t& stream) {
    offset1 = 0;
    offset2 = offset1 + num_buffer_ele * sizeof(T);
    offset3 = offset2 + num_buffer_ele * sizeof(T);
    offset4 = offset3 + num_buffer_ele * sizeof(T);
    offset5 = offset4 + num_buffer_ele * sizeof(T);
    offset6 = offset5 + (nranks + 1) * sizeof(T);
    offset7 = offset6 + (nranks + 1) * sizeof(T);
    offset8 = offset7 + (nranks + 1) * sizeof(T);
    offset9 = offset8 + num_temp_ele * sizeof(T);

    buffer_ptr = buffer.mutable_data<T>(
        {4 * num_buffer_ele + 3 * (nranks + 1) + num_temp_ele}, place);
  }

  T* cub_sort_keys_ptr() { return buffer_ptr + offset1; }
  T* cub_sort_keys_out_ptr() { return buffer_ptr + offset2; }
  T* cub_sort_values_ptr() { return buffer_ptr + offset3; }
  T* cub_sort_values_out_ptr() { return buffer_ptr + offset4; }
  T* bound_index_ptr() { return buffer_ptr + offset5; }
  T* bound_value_ptr() { return buffer_ptr + offset6; }
  T* class_interval_ptr() { return buffer_ptr + offset7; }
  void* cub_temp_storage_ptr() {
    return reinterpret_cast<void*>(buffer_ptr + offset8);
  }

 private:
  Tensor buffer;
  T* buffer_ptr;
  int offset1;
  int offset2;
  int offset3;
  int offset4;
  int offset5;
  int offset6;
  int offset7;
  int offset8;
  int offset9;
};

template <typename DeviceContext, typename T>
class ClassCenterSampleCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* label = ctx.Input<Tensor>("Label");
    auto* remapped_label = ctx.Output<Tensor>("RemappedLabel");
    auto* sampled_local_class_center =
        ctx.Output<Tensor>("SampledLocalClassCenter");
    int num_classes = ctx.Attr<int>("num_classes");
    int num_samples = ctx.Attr<int>("num_samples");

    int rid = ctx.Attr<int>("ring_id");
    int nranks = ctx.Attr<int>("nranks");
    int rank = ctx.Attr<int>("rank");

    int seed = ctx.Attr<int>("seed");
    bool fix_seed = ctx.Attr<bool>("fix_seed");
    PADDLE_ENFORCE_GT(num_classes, 0,
                      platform::errors::InvalidArgument(
                          "The value 'num_classes' for Op(class_center_sample) "
                          "must be greater than 0, "
                          "but the value given is %d.",
                          num_classes));

    PADDLE_ENFORCE_GT(num_samples, 0,
                      platform::errors::InvalidArgument(
                          "The value 'num_samples' for Op(class_center_sample) "
                          "must be greater than 0, "
                          "but the value given is %d.",
                          num_samples));

    PADDLE_ENFORCE_LE(num_samples, num_classes,
                      platform::errors::InvalidArgument(
                          "The value 'num_samples' for Op(class_center_sample) "
                          "must be less than or equal to %d, "
                          "but the value given is %d.",
                          num_classes, num_samples));

    auto& dev_ctx = ctx.template device_context<DeviceContext>();
    auto place = BOOST_GET_CONST(platform::CUDAPlace, dev_ctx.GetPlace());

    int batch_size = label->numel();
    // Algorithm:
    // We first randomly generate a value in [0, num_classes) on each position
    // in a array(shape[num_classes]). Then, we mark the element as negative
    // value in the array according input label. Now, we can sort the array
    // by ascending to ensure that the positive class center always in the
    // front of the sorted array. So, we can get the sampled class center
    // index by sorted keys. Finally, we can get the rempped label by remap
    // the input label according sampled class center.

    // step 1: Calculate num classes per device using nccl all reduce
    std::vector<T> shard_dim_vec(nranks + 1, 0);
    shard_dim_vec[rank + 1] = num_classes;

    Tensor num_classes_per_device;
    framework::TensorFromVector(shard_dim_vec, ctx.cuda_device_context(),
                                &num_classes_per_device);
    T* num_classes_per_device_ptr = num_classes_per_device.data<T>();

#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
    if (nranks > 1) {
      const auto& comm =
          platform::NCCLCommContext::Instance().Get(rid, ctx.GetPlace());
      // use global calculate stream
      const auto calcu_stream =
          static_cast<platform::CUDADeviceContext*>(
              platform::DeviceContextPool::Instance().Get(ctx.GetPlace()))
              ->stream();
      PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::ncclAllReduce(
          num_classes_per_device_ptr, num_classes_per_device_ptr,
          num_classes_per_device.numel(),
          platform::ToNCCLDataType(num_classes_per_device.type()), ncclSum,
          comm->comm(), calcu_stream));
    }
#endif

    // step 2: Determine temporary device storage requirements
    int num_buffer_ele = std::max(batch_size, num_classes);
    size_t cub_sort_temp_store_size = 0;
    cub::DeviceRadixSort::SortPairs<T, T>(
        nullptr, cub_sort_temp_store_size, nullptr, nullptr, nullptr, nullptr,
        num_buffer_ele, 0, sizeof(T) * 8, ctx.cuda_device_context().stream());

    size_t cub_sum_temp_store_size = 0;
    NotEqualToPreviousAdjacentIterator<T> unique_counting_iter_temp(nullptr, 0);
    cub::DeviceScan::InclusiveSum<NotEqualToPreviousAdjacentIterator<T>, T*>(
        nullptr, cub_sum_temp_store_size, unique_counting_iter_temp, nullptr,
        batch_size, ctx.cuda_device_context().stream());

    size_t cub_scan_temp_store_size = 0;
    ActualNumSampledFunctor<T> actual_num_sampled_op_temp(num_samples);
    cub::DeviceScan::InclusiveScan(
        nullptr, cub_scan_temp_store_size, num_classes_per_device_ptr,
        num_classes_per_device_ptr, actual_num_sampled_op_temp, nranks + 1,
        ctx.cuda_device_context().stream());

    size_t cub_temp_storage_bytes =
        std::max(std::max(cub_sort_temp_store_size, cub_scan_temp_store_size),
                 cub_sum_temp_store_size);
    int num_temp_ele = cub_temp_storage_bytes / sizeof(T) + 1;

    // step 3: Alloc buffer memory so that we can reuse allocated memory
    MemoryBuffer<T> memory_buffer =
        MemoryBuffer<T>(num_buffer_ele, num_temp_ele, nranks, num_classes,
                        ctx.GetPlace(), ctx.cuda_device_context().stream());

    T* cub_sort_keys_ptr = memory_buffer.cub_sort_keys_ptr();
    T* cub_sort_keys_out_ptr = memory_buffer.cub_sort_keys_out_ptr();
    T* cub_sort_values_ptr = memory_buffer.cub_sort_values_ptr();
    T* cub_sort_values_out_ptr = memory_buffer.cub_sort_values_out_ptr();
    T* bound_index_ptr = memory_buffer.bound_index_ptr();
    T* bound_value_ptr = memory_buffer.bound_value_ptr();
    T* class_interval_ptr = memory_buffer.class_interval_ptr();
    void* cub_temp_storage_ptr = memory_buffer.cub_temp_storage_ptr();

    // step 4: Calculate class interval among nranks
    cub::DeviceScan::InclusiveSum<T*, T*>(
        cub_temp_storage_ptr, cub_temp_storage_bytes,
        num_classes_per_device_ptr, class_interval_ptr, nranks + 1,
        ctx.cuda_device_context().stream());

    // step 5: random sample negative class center
    int vec_size = VectorizedSize<T>(cub_sort_keys_ptr);
    int increment = ((num_classes - 1) /
                         (NumBlocks(num_classes) * kNumCUDAThreads * vec_size) +
                     1) *
                    vec_size;
    if (!fix_seed) {
      std::random_device rnd;
      seed = rnd();
    }
    RandomSampleClassCenter<T><<<NumBlocks(num_classes), kNumCUDAThreads, 0,
                                 ctx.cuda_device_context().stream()>>>(
        num_classes, seed + rank, increment, num_classes, cub_sort_keys_ptr);

    // step 6: mark positive class center as negative value
    // fill the sort values to index 0, 1, ..., batch_size-1
    MarkPositiveClassCenter<<<NumBlocks(batch_size), kNumCUDAThreads, 0,
                              ctx.cuda_device_context().stream()>>>(
        batch_size, rank, class_interval_ptr, num_classes, label->data<T>(),
        cub_sort_keys_ptr, cub_sort_values_ptr);

// step 7: sort class center by ascending, so that positive class center
// always be sampled.
#ifdef PADDLE_WITH_HIP
    hipError_t ret;
#else
    cudaError_t ret;
#endif
    ret = cub::DeviceRadixSort::SortPairs<T, T>(
        cub_temp_storage_ptr, cub_temp_storage_bytes, cub_sort_keys_ptr,
        cub_sort_keys_out_ptr, cub_sort_values_ptr, cub_sort_values_out_ptr,
        num_classes, 0, sizeof(T) * 8, ctx.cuda_device_context().stream());
    PADDLE_ENFORCE_CUDA_SUCCESS(ret);

    // step 8: sort input label ascending
    ret = cub::DeviceRadixSort::SortPairs<T, T>(
        cub_temp_storage_ptr, cub_temp_storage_bytes, label->data<T>(),
        cub_sort_keys_out_ptr, cub_sort_values_ptr, cub_sort_keys_ptr,
        batch_size, 0, sizeof(T) * 8, ctx.cuda_device_context().stream());
    PADDLE_ENFORCE_CUDA_SUCCESS(ret);

    // step 9: Calculate new index using InclusiveSum on ascending sorted input
    // label
    NotEqualToPreviousAdjacentIterator<T> unique_counting_iter(
        cub_sort_keys_out_ptr, 0);
    ret = cub::DeviceScan::InclusiveSum<NotEqualToPreviousAdjacentIterator<T>,
                                        T*>(
        cub_temp_storage_ptr, cub_temp_storage_bytes, unique_counting_iter,
        cub_sort_values_ptr, batch_size, ctx.cuda_device_context().stream());
    PADDLE_ENFORCE_CUDA_SUCCESS(ret);

    // step 10: Calculate new class center bound among ranks
    GetClassCenterBound<T><<<NumBlocks(batch_size), kNumCUDAThreads, 0,
                             ctx.cuda_device_context().stream()>>>(
        batch_size, nranks, class_interval_ptr, cub_sort_keys_out_ptr,
        cub_sort_values_ptr, bound_index_ptr, bound_value_ptr);

    // step 11: Calculate actual number of sampled class per device.
    // Since maybe num_positive_class_center > num_samples,
    // we need to ensure all positive class center per device are sampled.
    ActualNumSampledFunctor<T> actual_num_sampled_op(num_samples);
    ret = cub::DeviceScan::InclusiveScan(
        cub_temp_storage_ptr, cub_temp_storage_bytes, bound_value_ptr,
        num_classes_per_device_ptr, actual_num_sampled_op, nranks + 1,
        ctx.cuda_device_context().stream());
    PADDLE_ENFORCE_CUDA_SUCCESS(ret);

    // step 12: Calculate actual sampled class interval among nranks
    ret = cub::DeviceScan::InclusiveSum(
        cub_temp_storage_ptr, cub_temp_storage_bytes,
        num_classes_per_device_ptr, class_interval_ptr, nranks + 1,
        ctx.cuda_device_context().stream());
    PADDLE_ENFORCE_CUDA_SUCCESS(ret);

    // step 13: Get remapped label for output
    GetRemappedLabel<T><<<NumBlocks(batch_size), kNumCUDAThreads, 0,
                          ctx.cuda_device_context().stream()>>>(
        batch_size, nranks, class_interval_ptr, bound_index_ptr,
        bound_value_ptr, cub_sort_keys_ptr, cub_sort_values_ptr,
        remapped_label->mutable_data<T>(ctx.GetPlace()));

    // step 14: Get sampled class center for output
    framework::TensorCopySync(num_classes_per_device, platform::CPUPlace(),
                              &num_classes_per_device);
    T actual_num_samples = num_classes_per_device.data<T>()[rank + 1];
    T* sampled_local_class_center_ptr =
        sampled_local_class_center->mutable_data<T>({actual_num_samples},
                                                    ctx.GetPlace());
    GetSampledClassCenter<T><<<NumBlocks(batch_size), kNumCUDAThreads, 0,
                               ctx.cuda_device_context().stream()>>>(
        actual_num_samples, cub_sort_values_out_ptr,
        sampled_local_class_center_ptr);
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(
    class_center_sample,
    ops::ClassCenterSampleCUDAKernel<paddle::platform::CUDADeviceContext,
                                     int64_t>,
    ops::ClassCenterSampleCUDAKernel<paddle::platform::CUDADeviceContext, int>);
