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
#include <vector>
#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/memory/malloc.h"
#include "paddle/fluid/memory/memcpy.h"
#include "paddle/fluid/operators/math/blas.h"
#include "paddle/fluid/operators/math/math_function.h"
#include "paddle/fluid/operators/math/math_function_impl.h"
#include "paddle/fluid/platform/bfloat16.h"
#include "paddle/fluid/platform/complex128.h"
#include "paddle/fluid/platform/complex64.h"
#include "paddle/fluid/platform/float16.h"

namespace paddle {
namespace operators {
namespace math {

using float16 = paddle::platform::float16;
using bfloat16 = paddle::platform::bfloat16;
using complex64 = paddle::platform::complex64;
using complex128 = paddle::platform::complex128;

template struct SetConstant<platform::CUDADeviceContext, platform::float16>;
template struct SetConstant<platform::CUDADeviceContext, platform::bfloat16>;
template struct SetConstant<platform::CUDADeviceContext, float>;
template struct SetConstant<platform::CUDADeviceContext, double>;
template struct SetConstant<platform::CUDADeviceContext, uint8_t>;
template struct SetConstant<platform::CUDADeviceContext, int>;
template struct SetConstant<platform::CUDADeviceContext, int64_t>;
template struct SetConstant<platform::CUDADeviceContext, bool>;
template struct SetConstant<platform::CUDADeviceContext, platform::complex64>;
template struct SetConstant<platform::CUDADeviceContext, platform::complex128>;
template struct SetConstant<platform::CUDADeviceContext,
                            platform::complex<float>>;
template struct SetConstant<platform::CUDADeviceContext,
                            platform::complex<double>>;

#define DEFINE_GPU_TRANS(RANK)                                             \
  template struct Transpose<platform::CUDADeviceContext, float, RANK>;     \
  template struct Transpose<platform::CUDADeviceContext, double, RANK>;    \
  template struct Transpose<platform::CUDADeviceContext, float16, RANK>;   \
  template struct Transpose<platform::CUDADeviceContext, bfloat16, RANK>;  \
  template struct Transpose<platform::CUDADeviceContext, int8_t, RANK>;    \
  template struct Transpose<platform::CUDADeviceContext, int32_t, RANK>;   \
  template struct Transpose<platform::CUDADeviceContext, int64_t, RANK>;   \
  template struct Transpose<platform::CUDADeviceContext,                   \
                            paddle::platform::complex<float>, RANK>;       \
  template struct Transpose<platform::CUDADeviceContext,                   \
                            paddle::platform::complex<double>, RANK>;      \
  template struct Transpose<platform::CUDADeviceContext, complex64, RANK>; \
  template struct Transpose<platform::CUDADeviceContext, complex128, RANK>;

DEFINE_GPU_TRANS(1);
DEFINE_GPU_TRANS(2);
DEFINE_GPU_TRANS(3);
DEFINE_GPU_TRANS(4);
DEFINE_GPU_TRANS(5);
DEFINE_GPU_TRANS(6);

#define REINTERPRET(T, DST_PTR, SRC_PTR) \
  T* DST_PTR = reinterpret_cast<T*>(SRC_PTR)

template <typename T>
__global__ void TransposeNormalKernel(const T* in_ptr, T* out_ptr,
                                      int64_t element,
                                      const int64_t* in_stride_ptr,
                                      const int64_t* out_stride_ptr,
                                      const int64_t* axis_ptr, int rank) {
  CUDA_KERNEL_LOOP(out_idx, element) {
    int64_t in_idx = 0;
    int64_t tmp_idx = out_idx;
    for (int i = 0; i < rank; ++i) {
      const int64_t coordinate = tmp_idx / out_stride_ptr[i];
      tmp_idx -= coordinate * out_stride_ptr[i];
      in_idx += coordinate * in_stride_ptr[axis_ptr[i]];
    }
    out_ptr[out_idx] = in_ptr[in_idx];
  }
}

template <typename T>
struct TransposeNormal<platform::CUDADeviceContext, T> {
  void operator()(const platform::CUDADeviceContext& context,
                  const framework::Tensor& in, framework::Tensor* out,
                  const std::vector<int>& axis) {
    const int rank = axis.size();
    auto in_stride = framework::stride(in.dims());
    auto out_stride = framework::stride(out->dims());
    auto* in_ptr = in.data<T>();
    auto* out_ptr = out->data<T>();

    // copy in_stride, out_stride, axis to gpu device
    const platform::CUDAPlace& cuda_place =
        BOOST_GET_CONST(platform::CUDAPlace, context.GetPlace());
    platform::CPUPlace cpu_place = platform::CPUPlace();
    size_t size = 3 * rank * sizeof(int64_t);
    auto cpu_buf_holder = memory::AllocShared(cpu_place, size);
    auto cuda_buf_holder = memory::AllocShared(cuda_place, size);
    REINTERPRET(int64_t, cpu_buf, cpu_buf_holder->ptr());
    REINTERPRET(int64_t, cuda_buf, cuda_buf_holder->ptr());
    for (int i = 0; i < rank; ++i) {
      cpu_buf[i] = in_stride[i];
      cpu_buf[rank + i] = out_stride[i];
      cpu_buf[2 * rank + i] = axis[i];
    }
    memory::Copy(cuda_place, cuda_buf, cpu_place, cpu_buf, size,
                 context.stream());
    REINTERPRET(const int64_t, in_stride_ptr, cuda_buf);
    REINTERPRET(const int64_t, out_stride_ptr, cuda_buf + rank);
    REINTERPRET(const int64_t, axis_ptr, cuda_buf + 2 * rank);

    const int MAX_BLOCK_DIM = context.GetMaxThreadsPerBlock();
    const int MAX_GRID_DIM =
        context.GetMaxPhysicalThreadCount() / MAX_BLOCK_DIM;
    int64_t elements = in.numel();
    int block_size = (elements >= MAX_BLOCK_DIM)
                         ? MAX_BLOCK_DIM
                         : (1 << static_cast<int>(std::log2(elements)));
    int grid_size = elements / block_size;
    grid_size = (grid_size >= MAX_GRID_DIM) ? MAX_GRID_DIM : grid_size;
    TransposeNormalKernel<T><<<grid_size, block_size, 0, context.stream()>>>(
        in_ptr, out_ptr, elements, in_stride_ptr, out_stride_ptr, axis_ptr,
        rank);
  }
};

// define transpose normal
#define DEFINE_GPU_TRANS_NORMAL(TYPE) \
  template struct TransposeNormal<platform::CUDADeviceContext, TYPE>

DEFINE_GPU_TRANS_NORMAL(float16);
DEFINE_GPU_TRANS_NORMAL(bfloat16);
DEFINE_GPU_TRANS_NORMAL(float);
DEFINE_GPU_TRANS_NORMAL(double);
DEFINE_GPU_TRANS_NORMAL(int);
DEFINE_GPU_TRANS_NORMAL(int64_t);
DEFINE_GPU_TRANS_NORMAL(bool);
DEFINE_GPU_TRANS_NORMAL(int16_t);
DEFINE_GPU_TRANS_NORMAL(uint8_t);
DEFINE_GPU_TRANS_NORMAL(int8_t);
DEFINE_GPU_TRANS_NORMAL(complex64);
DEFINE_GPU_TRANS_NORMAL(complex128);
DEFINE_GPU_TRANS_NORMAL(paddle::platform::complex<float>);
DEFINE_GPU_TRANS_NORMAL(paddle::platform::complex<double>);

struct TensorSetConstantGPU {
  TensorSetConstantGPU(const platform::DeviceContext& context,
                       framework::Tensor* tensor, float value)
      : context_(context), tensor_(tensor), value_(value) {}

  template <typename T>
  void apply() const {
    SetConstant<platform::CUDADeviceContext, T> functor;
    functor(reinterpret_cast<const platform::CUDADeviceContext&>(context_),
            tensor_, static_cast<T>(value_));
  }

  const platform::DeviceContext& context_;
  framework::Tensor* tensor_;
  float value_;
};

template <>
void set_constant_with_place<platform::CUDAPlace>(
    const platform::DeviceContext& context, framework::Tensor* tensor,
    float value) {
  framework::VisitDataType(tensor->type(),
                           TensorSetConstantGPU(context, tensor, value));
}

template <typename T>
__global__ void RowwiseAddKernel(const T* a, const T* b, T* c, int width,
                                 int num) {
  T tmp = 1.0 / width;
  CUDA_KERNEL_LOOP(i, num) {
    int h = i * tmp;
    int w = i - h * width;
    c[i] = a[i] + b[w];
  }
}

template <typename T>
struct RowwiseAdd<platform::CUDADeviceContext, T> {
  void operator()(const platform::CUDADeviceContext& context,
                  const framework::Tensor& input,
                  const framework::Tensor& vector, framework::Tensor* output) {
    auto in_dims = input.dims();
    auto out_dims = output->dims();
    auto size = input.numel() / in_dims[0];
    PADDLE_ENFORCE_EQ(
        vector.numel(), size,
        platform::errors::InvalidArgument(
            "The input vector size"
            " should be equal to the size of each row of input tensor."
            " Expected vector size=%d, but received %d",
            size, vector.numel()));
    const char* in_dims_cstr = in_dims.to_str().c_str();
    const char* out_dims_cstr = out_dims.to_str().c_str();
    PADDLE_ENFORCE_EQ(
        out_dims, in_dims,
        platform::errors::InvalidArgument(
            "The output tensor shape should be same as the input tensor"
            " shape. Expected output tensor shape: %s,"
            " but received %s",
            in_dims_cstr, out_dims_cstr));
    int blocks = 512;
    int grids = (input.numel() + blocks - 1) / blocks;
    RowwiseAddKernel<T><<<grids, blocks, 0, context.stream()>>>(
        input.data<T>(), vector.data<T>(), output->data<T>(),
        static_cast<int>(in_dims[1]), static_cast<int>(input.numel()));
  }
};

template struct RowwiseAdd<platform::CUDADeviceContext, float>;
template struct RowwiseAdd<platform::CUDADeviceContext, double>;
template struct ColwiseSum<platform::CUDADeviceContext, float>;
template struct ColwiseSum<platform::CUDADeviceContext, int>;
template struct ColwiseSum<platform::CUDADeviceContext, int64_t>;
// template struct ColwiseSum<platform::CUDADeviceContext, double>;
// The ColwiseSum<platform::CUDADeviceContext, double> failed in debug mode,
// and only failed for this case. So reimplemented it.
template <>
void ColwiseSum<platform::CUDADeviceContext, double>::operator()(
    const platform::CUDADeviceContext& context, const framework::Tensor& input,
    framework::Tensor* vector) {
  auto in_dims = input.dims();
  auto size = input.numel() / in_dims[0];
  PADDLE_ENFORCE_EQ(vector->numel(), size,
                    platform::errors::InvalidArgument(
                        "The size of input vector"
                        " should be equal to the size of input tensor column"
                        " dimension. Expected vector size=%d, but received %d",
                        size, vector->numel()));
  framework::Tensor one;
  one.mutable_data<double>({in_dims[0]}, context.GetPlace());
  SetConstant<platform::CUDADeviceContext, double> set;
  set(context, &one, static_cast<double>(1.0));
  GetBlas<platform::CUDADeviceContext, double>(context).GEMV(
      true, static_cast<int>(in_dims[0]), static_cast<int>(in_dims[1]), 1.0,
      input.data<double>(), one.data<double>(), 0.0, vector->data<double>());
}

template struct RowwiseSum<platform::CUDADeviceContext, float>;
// template struct RowwiseSum<platform::CUDADeviceContext, double>;
// TODO(zcd): Following ColwiseSum format, need to confirm.
// The RowwiseSum<platform::CUDADeviceContext, double> failed in debug mode,
// and only failed for this case. So reimplemented it.
template <>
void RowwiseSum<platform::CUDADeviceContext, double>::operator()(
    const platform::CUDADeviceContext& context, const framework::Tensor& input,
    framework::Tensor* vector) {
  auto in_dims = input.dims();
  auto size = input.numel() / in_dims[0];
  PADDLE_ENFORCE_EQ(vector->numel(), in_dims[0],
                    platform::errors::InvalidArgument(
                        "The size of input vector"
                        " should be equal to the size of input tensor row"
                        " dimension. Expected vector size=%d, but received %d",
                        in_dims[0], vector->numel()));
  framework::Tensor one;
  one.mutable_data<double>({size}, context.GetPlace());
  SetConstant<platform::CUDADeviceContext, double> set;
  set(context, &one, static_cast<double>(1.0));
  GetBlas<platform::CUDADeviceContext, double>(context).GEMV(
      true, static_cast<int>(in_dims[1]), static_cast<int>(in_dims[0]), 1.0,
      one.data<double>(), input.data<double>(), 0.0, vector->data<double>());
}

template struct RowwiseMean<platform::CUDADeviceContext, float>;
template struct RowwiseMean<platform::CUDADeviceContext, double>;

template <typename T>
struct ElementwiseAddTo<platform::CUDADeviceContext, T> {
  void operator()(platform::CUDADeviceContext* ctx,
                  const framework::Tensor& src, framework::Tensor* dst) {
    auto in = framework::EigenVector<T>::Flatten(src);
    auto out = framework::EigenVector<T>::Flatten(*dst);
    auto& place = *(ctx->eigen_device());
    out.device(place) = out + in;
  }
};

template struct ElementwiseAddTo<platform::CUDADeviceContext,
                                 platform::float16>;
}  // namespace math
}  // namespace operators
}  // namespace paddle
