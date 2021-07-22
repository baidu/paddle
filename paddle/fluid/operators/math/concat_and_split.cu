/* Copyright (c) 2018 paddlepaddle Authors. All Rights Reserved.

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
#include <vector>
#include "gflags/gflags.h"
#include "paddle/fluid/framework/mixed_vector.h"
#include "paddle/fluid/memory/malloc.h"
#include "paddle/fluid/operators/math/concat_and_split.h"
#include "paddle/fluid/platform/cuda_primitives.h"
#include "paddle/fluid/platform/float16.h"

namespace paddle {
namespace operators {
namespace math {

template <typename T>
__global__ void ConcatKernel(const T** inputs, const int64_t* input_cols,
                             int col_size, const int64_t output_rows,
                             const int64_t output_cols, T* output) {
  int tid_x = blockIdx.x * blockDim.x + threadIdx.x;
  int curr_segment = 0;
  int curr_offset = input_cols[0];
  for (; tid_x < output_cols; tid_x += blockDim.x * gridDim.x) {
    int curr_col_offset = input_cols[curr_segment + 1];
    while (curr_col_offset <= tid_x) {
      curr_offset = curr_col_offset;
      ++curr_segment;
      curr_col_offset = input_cols[curr_segment + 1];
    }

    int local_col = tid_x - curr_offset;
    int segment_width = curr_col_offset - curr_offset;

    const T* input_ptr = inputs[curr_segment];
    int tid_y = blockIdx.y * blockDim.y + threadIdx.y;
    for (; tid_y < output_rows; tid_y += blockDim.y * gridDim.y)
      output[tid_y * output_cols + tid_x] =
          input_ptr[tid_y * segment_width + local_col];
  }
}

template <typename T>
__device__ void ConcatKernelDetail(const T** inputs_data,
                                   const int fixed_in_col, const int out_rows,
                                   const int out_cols, T* output_data) {
  int tid_x = blockIdx.x * blockDim.x + threadIdx.x;
  for (; tid_x < out_cols; tid_x += blockDim.x * gridDim.x) {
    int split = tid_x * 1.0 / fixed_in_col;
    int in_offset = tid_x - split * fixed_in_col;
    const T* input_ptr = inputs_data[split];
    int tid_y = blockIdx.y * blockDim.y + threadIdx.y;
    for (; tid_y < out_rows; tid_y += blockDim.y * gridDim.y) {
      output_data[tid_y * out_cols + tid_x] =
          input_ptr[tid_y * fixed_in_col + in_offset];
    }
  }
}

template <typename T>
__global__ void ConcatKernel(const T* input_addr0, const T* input_addr1,
                             const int64_t fixed_in_col, const int64_t out_rows,
                             const int64_t out_cols, T* output_data) {
  const T* inputs_data[2];
  inputs_data[0] = input_addr0;
  inputs_data[1] = input_addr1;
  ConcatKernelDetail<T>(inputs_data, fixed_in_col, out_rows, out_cols,
                        output_data);
}

template <typename T>
__global__ void ConcatKernel(const T* input_addr0, const T* input_addr1,
                             const T* input_addr2, const int64_t fixed_in_col,
                             const int64_t out_rows, const int64_t out_cols,
                             T* output_data) {
  const T* inputs_data[3];
  inputs_data[0] = input_addr0;
  inputs_data[1] = input_addr1;
  inputs_data[2] = input_addr2;
  ConcatKernelDetail<T>(inputs_data, fixed_in_col, out_rows, out_cols,
                        output_data);
}

template <typename T>
__global__ void ConcatKernel(const T* input_addr0, const T* input_addr1,
                             const T* input_addr2, const T* input_addr3,
                             const int64_t fixed_in_col, const int64_t out_rows,
                             const int64_t out_cols, T* output_data) {
  const T* inputs_data[4];
  inputs_data[0] = input_addr0;
  inputs_data[1] = input_addr1;
  inputs_data[2] = input_addr2;
  inputs_data[3] = input_addr3;
  ConcatKernelDetail<T>(inputs_data, fixed_in_col, out_rows, out_cols,
                        output_data);
}

template <typename T>
__global__ void ConcatKernel(const T** inputs_data, const int in_num,
                             const int64_t fixed_in_col, const int64_t out_rows,
                             const int64_t out_cols, T* output_data) {
  ConcatKernelDetail<T>(inputs_data, fixed_in_col, out_rows, out_cols,
                        output_data);
}

template <typename T>
__global__ void SplitKernel(const T* input_data, const int64_t in_row,
                            const int64_t in_col, const int64_t* out_cols,
                            int out_cols_size, T** outputs_data) {
  int tid_x = blockIdx.x * blockDim.x + threadIdx.x;
  int curr_segment = 0;
  int curr_offset = out_cols[0];
  for (; tid_x < in_col; tid_x += blockDim.x * gridDim.x) {
    int curr_col_offset = out_cols[curr_segment + 1];
    while (curr_col_offset <= tid_x) {
      curr_offset = curr_col_offset;
      ++curr_segment;
      curr_col_offset = out_cols[curr_segment + 1];
    }

    int local_col = tid_x - curr_offset;
    int segment_width = curr_col_offset - curr_offset;
    T* output_ptr = outputs_data[curr_segment];
    if (output_ptr != nullptr) {
      int tid_y = blockIdx.y * blockDim.y + threadIdx.y;
      for (; tid_y < in_row; tid_y += blockDim.y * gridDim.y)
        output_ptr[tid_y * segment_width + local_col] =
            input_data[tid_y * in_col + tid_x];
    }
  }
}

template <typename T>
__device__ void SplitKernelDetail(const T* input_data, const int in_row,
                                  const int in_col, const int fixed_out_col,
                                  T** outputs_data) {
  int tid_x = blockIdx.x * blockDim.x + threadIdx.x;
  for (; tid_x < in_col; tid_x += blockDim.x * gridDim.x) {
    int split = tid_x / fixed_out_col;
    int in_offset = tid_x - split * fixed_out_col;
    T* output_ptr = outputs_data[split];
    if (output_ptr != nullptr) {
      int tid_y = blockIdx.y * blockDim.y + threadIdx.y;
      for (; tid_y < in_row; tid_y += blockDim.y * gridDim.y)
        output_ptr[tid_y * fixed_out_col + in_offset] =
            input_data[tid_y * in_col + tid_x];
    }
  }
}

template <typename T>
__global__ void SplitKernel(const T* input_data, const int64_t in_row,
                            const int64_t in_col, const int64_t fixed_out_col,
                            T** outputs_data) {
  SplitKernelDetail<T>(input_data, in_row, in_col, fixed_out_col, outputs_data);
}

template <typename T>
__global__ void SplitKernel(const T* input_data, const int64_t in_row,
                            const int64_t in_col, const int64_t fixed_out_col,
                            T* outputs_addr0, T* outputs_addr1) {
  T* outputs_data[2];
  outputs_data[0] = outputs_addr0;
  outputs_data[1] = outputs_addr1;
  SplitKernelDetail<T>(input_data, in_row, in_col, fixed_out_col, outputs_data);
}

template <typename T>
__global__ void SplitKernel(const T* input_data, const int64_t in_row,
                            const int64_t in_col, const int64_t fixed_out_col,
                            T* outputs_addr0, T* outputs_addr1,
                            T* outputs_addr2) {
  T* outputs_data[3];
  outputs_data[0] = outputs_addr0;
  outputs_data[1] = outputs_addr1;
  outputs_data[2] = outputs_addr2;
  SplitKernelDetail<T>(input_data, in_row, in_col, fixed_out_col, outputs_data);
}

template <typename T>
__global__ void SplitKernel(const T* input_data, const int64_t in_row,
                            const int64_t in_col, const int64_t fixed_out_col,
                            T* outputs_addr0, T* outputs_addr1,
                            T* outputs_addr2, T* outputs_addr3) {
  T* outputs_data[4];
  outputs_data[0] = outputs_addr0;
  outputs_data[1] = outputs_addr1;
  outputs_data[2] = outputs_addr2;
  outputs_data[3] = outputs_addr3;
  SplitKernelDetail<T>(input_data, in_row, in_col, fixed_out_col, outputs_data);
}

static inline void GetBlockDims(const platform::CUDADeviceContext& context,
                                int64_t num_rows, int64_t num_cols,
                                dim3* block_dims, dim3* grid_dims) {
  // Set the thread block and grid according to CurrentDeviceId
  const int kThreadsPerBlock = 1024;
  int block_cols = kThreadsPerBlock;
  if (num_cols < kThreadsPerBlock) {  // block_cols is aligned by 32.
    block_cols = ((num_cols + 31) >> 5) << 5;
  }
  int block_rows = kThreadsPerBlock / block_cols;
  *block_dims = dim3(block_cols, block_rows, 1);

  int max_threads = context.GetMaxPhysicalThreadCount();
  int64_t max_blocks = std::max(max_threads / kThreadsPerBlock, 1);

  int grid_cols =
      std::min((num_cols + block_cols - 1) / block_cols, max_blocks);
  int grid_rows = std::min(max_blocks / grid_cols,
                           std::max(num_rows / block_rows, (int64_t)1));
  *grid_dims = dim3(grid_cols, grid_rows, 1);
}

/*
 * All tensors' dimension should be the same and the values of
 * each dimension must be the same, except the axis dimension.
 */
template <typename T>
class ConcatFunctor<platform::CUDADeviceContext, T> {
 public:
  void operator()(const platform::CUDADeviceContext& context,
                  const std::vector<framework::Tensor>& input, int axis,
                  framework::Tensor* output) {
    // TODO(zcd): Add input data validity checking
    int in_num = input.size();
    int64_t in_row = 1;
    auto dim_0 = input[0].dims();
    for (int i = 0; i < axis; ++i) {
      in_row *= dim_0[i];
    }
    int64_t in_col = input[0].numel() / in_row;
    int64_t out_row = in_row, out_col = 0;

    int inputs_col_num = in_num + 1;
    std::vector<const T*> inputs_data_vec(in_num);
    std::vector<int64_t> inputs_col_vec(inputs_col_num);
    const T** inputs_data = inputs_data_vec.data();
    int64_t* inputs_col = inputs_col_vec.data();

// There are some differences between hip runtime and NV runtime.
// In NV, when the pageable memory data less than 64K is transferred from
// hosttodevice, it will be automatically asynchronous.
// However, only pinned memory in hip can copy asynchronously
// https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#concurrent-execution-host-device
// 3.2.6.1. Concurrent Execution between Host and Device
// Memory copies from host to device of a memory block of 64 KB or less
#ifdef PADDLE_WITH_HIP
    memory::AllocationPtr data_alloc, col_alloc;
    data_alloc =
        memory::Alloc(platform::CUDAPinnedPlace(), in_num * sizeof(T*));
    inputs_data = reinterpret_cast<const T**>(data_alloc->ptr());
    col_alloc = memory::Alloc(platform::CUDAPinnedPlace(),
                              inputs_col_num * sizeof(int));
    inputs_col = reinterpret_cast<int64_t*>(col_alloc->ptr());
#endif

    inputs_col[0] = 0;
    bool has_same_shape = true;
    for (int i = 0; i < in_num; ++i) {
      int64_t t_cols = input[i].numel() / in_row;
      if (has_same_shape) {
        if (t_cols != in_col) has_same_shape = false;
      }
      out_col += t_cols;
      inputs_col[i + 1] = out_col;
      inputs_data[i] = input[i].data<T>();
    }

    dim3 block_dims;
    dim3 grid_dims;
    GetBlockDims(context, out_row, out_col, &block_dims, &grid_dims);

    memory::allocation::AllocationPtr tmp_dev_ins_data;
    const T** dev_ins_data = nullptr;
    if (!has_same_shape || in_num < 2 || in_num > 4) {
      tmp_dev_ins_data = memory::Alloc(context, in_num * sizeof(T*));
      memory::Copy(BOOST_GET_CONST(platform::CUDAPlace, context.GetPlace()),
                   tmp_dev_ins_data->ptr(), platform::CPUPlace(),
                   static_cast<void*>(inputs_data), in_num * sizeof(T*),
                   context.stream());
      dev_ins_data = reinterpret_cast<const T**>(tmp_dev_ins_data->ptr());
    }

    if (has_same_shape) {
      if (in_num == 2) {
        ConcatKernel<<<grid_dims, block_dims, 0, context.stream()>>>(
            inputs_data[0], inputs_data[1], in_col, out_row, out_col,
            output->data<T>());
      } else if (in_num == 3) {
        ConcatKernel<<<grid_dims, block_dims, 0, context.stream()>>>(
            inputs_data[0], inputs_data[1], inputs_data[2], in_col, out_row,
            out_col, output->data<T>());
      } else if (in_num == 4) {
        ConcatKernel<<<grid_dims, block_dims, 0, context.stream()>>>(
            inputs_data[0], inputs_data[1], inputs_data[2], inputs_data[3],
            in_col, out_row, out_col, output->data<T>());
      } else {
        ConcatKernel<<<grid_dims, block_dims, 0, context.stream()>>>(
            dev_ins_data, in_num, in_col, out_row, out_col, output->data<T>());
      }
    } else {
      auto tmp_dev_ins_col_data =
          memory::Alloc(context, inputs_col_num * sizeof(int64_t));
      memory::Copy(BOOST_GET_CONST(platform::CUDAPlace, context.GetPlace()),
                   tmp_dev_ins_col_data->ptr(), platform::CPUPlace(),
                   static_cast<void*>(inputs_col),
                   inputs_col_num * sizeof(int64_t), context.stream());
      int64_t* dev_ins_col_data =
          static_cast<int64_t*>(tmp_dev_ins_col_data->ptr());

      ConcatKernel<<<grid_dims, block_dims, 0, context.stream()>>>(
          dev_ins_data, dev_ins_col_data, static_cast<int>(inputs_col_num),
          out_row, out_col, output->data<T>());
    }

#ifdef PADDLE_WITH_HIP
    // Prevent the pinned memory value from being covered and release the memory
    // after the launch kernel of the stream is executed (reapply pinned memory
    // next time)
    auto* data_alloc_released = data_alloc.release();
    auto* col_alloc_released = col_alloc.release();
    context.AddStreamCallback([data_alloc_released, col_alloc_released] {
      memory::allocation::AllocationDeleter deleter;
      deleter(data_alloc_released);
      deleter(col_alloc_released);
    });
#endif
  }
};

/*
 * All tensors' dimension should be the same and the values of
 * each dimension must be the same, except the axis dimension.
 */
template <typename T>
class SplitFunctor<platform::CUDADeviceContext, T> {
 public:
  SplitFunctor();
  void operator()(const platform::CUDADeviceContext& context,
                  const framework::Tensor& input,
                  const std::vector<const framework::Tensor*>& ref_inputs,
                  int axis, std::vector<framework::Tensor*>* outputs) {
    // TODO(zcd): Add input data validity checking
    int o_num = outputs->size();
    int64_t out_row = 1;
    auto dim_0 = ref_inputs[0]->dims();
    for (int i = 0; i < axis; ++i) {
      out_row *= dim_0[i];
    }

    int64_t out0_col = ref_inputs[0]->numel() / out_row;
    int64_t in_col = 0, in_row = out_row;
    bool has_same_shape = true;

    int outputs_cols_num = o_num + 1;
    std::vector<T*> outputs_data_vec(o_num);
    std::vector<int64_t> outputs_cols_vec(outputs_cols_num);
    T** outputs_data = outputs_data_vec.data();
    int64_t* outputs_cols = outputs_cols_vec.data();

// There are some differences between hip runtime and NV runtime.
// In NV, when the pageable memory data less than 64K is transferred from
// hosttodevice, it will be automatically asynchronous.
// However, only pinned memory in hip can copy asynchronously
// https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#concurrent-execution-host-device
// 3.2.6.1. Concurrent Execution between Host and Device
// Memory copies from host to device of a memory block of 64 KB or less
#ifdef PADDLE_WITH_HIP
    memory::AllocationPtr data_alloc, cols_alloc;
    data_alloc = memory::Alloc(platform::CUDAPinnedPlace(), o_num * sizeof(T*));
    outputs_data = reinterpret_cast<T**>(data_alloc->ptr());
    cols_alloc = memory::Alloc(platform::CUDAPinnedPlace(),
                               (outputs_cols_num) * sizeof(int64_t));
    outputs_cols = reinterpret_cast<int64_t*>(cols_alloc->ptr());
#endif

    outputs_cols[0] = 0;
    for (int i = 0; i < o_num; ++i) {
      int64_t t_col = ref_inputs.at(i)->numel() / out_row;
      if (has_same_shape) {
        if (t_col != out0_col) has_same_shape = false;
      }
      in_col += t_col;
      outputs_cols[i + 1] = in_col;
      if (outputs->at(i) != nullptr) {
        outputs_data[i] = outputs->at(i)->data<T>();
      } else {
        outputs_data[i] = nullptr;
      }
    }

    dim3 block_dims;
    dim3 grid_dims;
    GetBlockDims(context, out_row, in_col, &block_dims, &grid_dims);

    memory::allocation::AllocationPtr tmp_dev_outs_data;
    T** dev_out_gpu_data = nullptr;
    if (!has_same_shape || o_num < 2 || o_num > 4) {
      tmp_dev_outs_data = memory::Alloc(context, o_num * sizeof(T*));
      memory::Copy(BOOST_GET_CONST(platform::CUDAPlace, context.GetPlace()),
                   tmp_dev_outs_data->ptr(), platform::CPUPlace(),
                   reinterpret_cast<void*>(outputs_data), o_num * sizeof(T*),
                   context.stream());
      dev_out_gpu_data = reinterpret_cast<T**>(tmp_dev_outs_data->ptr());
    }

    if (has_same_shape) {
      if (o_num == 2) {
        SplitKernel<<<grid_dims, block_dims, 0, context.stream()>>>(
            input.data<T>(), in_row, in_col, out0_col, outputs_data[0],
            outputs_data[1]);
      } else if (o_num == 3) {
        SplitKernel<<<grid_dims, block_dims, 0, context.stream()>>>(
            input.data<T>(), in_row, in_col, out0_col, outputs_data[0],
            outputs_data[1], outputs_data[2]);
      } else if (o_num == 4) {
        SplitKernel<<<grid_dims, block_dims, 0, context.stream()>>>(
            input.data<T>(), in_row, in_col, out0_col, outputs_data[0],
            outputs_data[1], outputs_data[2], outputs_data[3]);
      } else {
        SplitKernel<<<grid_dims, block_dims, 0, context.stream()>>>(
            input.data<T>(), in_row, in_col, out0_col, dev_out_gpu_data);
      }
    } else {
      auto tmp_dev_ins_col_data =
          memory::Alloc(context, outputs_cols_num * sizeof(int64_t));
      memory::Copy(BOOST_GET_CONST(platform::CUDAPlace, context.GetPlace()),
                   tmp_dev_ins_col_data->ptr(), platform::CPUPlace(),
                   reinterpret_cast<void*>(outputs_cols),
                   outputs_cols_num * sizeof(int64_t), context.stream());
      int64_t* dev_outs_col_data =
          reinterpret_cast<int64_t*>(tmp_dev_ins_col_data->ptr());

      SplitKernel<<<grid_dims, block_dims, 0, context.stream()>>>(
          input.data<T>(), in_row, in_col, dev_outs_col_data,
          static_cast<int>(outputs_cols_num), dev_out_gpu_data);
    }
#ifdef PADDLE_WITH_HIP
    // Prevent the pinned memory value from being covered and release the memory
    // after the launch kernel of the stream is executed (reapply pinned memory
    // next time)
    auto* data_alloc_released = data_alloc.release();
    auto* cols_alloc_released = cols_alloc.release();
    context.AddStreamCallback([data_alloc_released, cols_alloc_released] {
      memory::allocation::AllocationDeleter deleter;
      deleter(data_alloc_released);
      deleter(cols_alloc_released);
    });
#endif
  }
};

#define DEFINE_FUNCTOR(type)                                       \
  template class ConcatFunctor<platform::CUDADeviceContext, type>; \
  template class SplitFunctor<platform::CUDADeviceContext, type>

FOR_ALL_TYPES(DEFINE_FUNCTOR);

}  // namespace math
}  // namespace operators
}  // namespace paddle
