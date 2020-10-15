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
#include <vector>

#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/framework/dlpack_tensor.h"
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/framework.pb.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/platform/device_context.h"

namespace paddle {
namespace framework {

class PrintOptions {
 public:
  static PrintOptions& Instance() {
    static PrintOptions instance;
    return instance;
  }
  ~PrintOptions() {}
  PrintOptions(const PrintOptions& o) = delete;
  const PrintOptions& operator=(const PrintOptions& o) = delete;

  int precision = 8;
  int threshold = 1000;
  int edgeitems = 3;
  int linewidth = 75;
  bool sci_mode = false;

 private:
  PrintOptions() {}
};

// NOTE(zcd): Because TensorCopy is an async operation, when the src_place
// and dst_place are two different GPU, to ensure that the operation can
// be carried out correctly, there is a src_ctx wait operation in TensorCopy.
// If ctx_place and src_place are the same, src_ctx.Wait() is added
// after memory::Copy; if ctx_place and dst_place are the same,
// src_ctx.Wait() is added before memory::Copy.
class Tensor;

void TensorCopy(const Tensor& src, const platform::Place& dst_place,
                const platform::DeviceContext& ctx, Tensor* dst);

// NOTE(zcd): If the src.place() and dst_place are two different GPU,
// the copy operation is carried out on the dst_place's stream. This is
// very important, because TensorCopy is an async operator, and in most
// case, once this copy operator returns, dst is to be used in dst_place's
// stream, if this copy operation is carried out on the src_place's stream,
// when dst is used in dst_place's stream the copy operation may be
// not completed.
void TensorCopy(const Tensor& src, const platform::Place& dst_place,
                Tensor* dst);

void TensorCopySync(const Tensor& src, const platform::Place& dst_place,
                    Tensor* dst);

template <typename T>
void TensorFromVector(const std::vector<T>& src,
                      const platform::DeviceContext& ctx, Tensor* dst);
template <typename T>
void TensorFromVector(const std::vector<T>& src, Tensor* dst);

template <typename T>
void TensorToVector(const Tensor& src, const platform::DeviceContext& ctx,
                    std::vector<T>* dst);
template <typename T>
void TesnorToVector(const Tensor& src, std::vector<T>* dst);

// copy the result bool to cpu
bool TensorContainsNAN(const framework::Tensor& tensor);
bool TensorContainsInf(const framework::Tensor& tensor);
bool TensorIsfinite(const framework::Tensor& tensor);

// store the result bool in gpu tensor, async operation. Faster than above ones.
void TensorContainsNAN(const framework::Tensor& tensor, framework::Tensor* out);
void TensorContainsInf(const framework::Tensor& tensor, framework::Tensor* out);
void TensorIsfinite(const framework::Tensor& tensor, framework::Tensor* out);

void TensorToStream(std::ostream& os, const Tensor& tensor,
                    const platform::DeviceContext& dev_ctx);
void TensorFromStream(std::istream& is, Tensor* tensor,
                      const platform::DeviceContext& dev_ctx);
void TensorFromStream(std::istream& is, Tensor* tensor,
                      const platform::DeviceContext& dev_ctx,
                      const size_t& seek, const std::vector<int64_t>& shape);

// store the bool result tensor in out tensor
void TensorContainsNANV2(const framework::Tensor& tensor,
                         framework::Tensor* out);
void TensorContainsInfV2(const framework::Tensor& tensor,
                         framework::Tensor* out);
void TensorIsfiniteV2(const framework::Tensor& tensor, framework::Tensor* out);

// convert dlpack's DLTensor to tensor
void TensorFromDLPack(const ::DLTensor& dl_tensor, framework::Tensor* dst);

//
// The implementation of template functions.
//

template <typename T>
void TensorFromArray(const T* src, const size_t& array_size,
                     const platform::DeviceContext& ctx, Tensor* dst) {
  auto dst_place = ctx.GetPlace();
  auto src_ptr = static_cast<const void*>(src);
  platform::CPUPlace src_place;
  dst->Resize({static_cast<int64_t>(array_size)});
  auto dst_ptr = static_cast<void*>(dst->mutable_data<T>(dst_place));
  auto size = array_size * sizeof(T);

  if (platform::is_cpu_place(dst_place)) {
    memory::Copy(BOOST_GET_CONST(platform::CPUPlace, dst_place), dst_ptr,
                 src_place, src_ptr, size);
  }
#ifdef PADDLE_WITH_CUDA
  else if (platform::is_gpu_place(dst_place)) {  // NOLINT
    memory::Copy(
        BOOST_GET_CONST(platform::CUDAPlace, dst_place), dst_ptr, src_place,
        src_ptr, size,
        reinterpret_cast<const platform::CUDADeviceContext&>(ctx).stream());
  }
#endif
}
template <typename T>
void TensorFromVector(const std::vector<T>& src,
                      const platform::DeviceContext& ctx, Tensor* dst) {
  auto dst_place = ctx.GetPlace();
  auto src_ptr = static_cast<const void*>(src.data());
  platform::CPUPlace src_place;
  dst->Resize({static_cast<int64_t>(src.size())});
  auto dst_ptr = static_cast<void*>(dst->mutable_data<T>(dst_place));
  auto size = src.size() * sizeof(T);

  if (platform::is_cpu_place(dst_place)) {
    memory::Copy(BOOST_GET_CONST(platform::CPUPlace, dst_place), dst_ptr,
                 src_place, src_ptr, size);
  }
#ifdef PADDLE_WITH_CUDA
  else if (platform::is_gpu_place(dst_place)) {  // NOLINT
    memory::Copy(
        BOOST_GET_CONST(platform::CUDAPlace, dst_place), dst_ptr, src_place,
        src_ptr, size,
        reinterpret_cast<const platform::CUDADeviceContext&>(ctx).stream());
  }
#endif
}

template <typename T>
void TensorFromVector(const std::vector<T>& src, Tensor* dst) {
  platform::CPUPlace dst_place = platform::CPUPlace();
  auto src_ptr = static_cast<const void*>(src.data());
  platform::CPUPlace src_place;
  dst->Resize({static_cast<int64_t>(src.size())});
  auto dst_ptr = static_cast<void*>(dst->mutable_data<T>(dst_place));
  auto size = src.size() * sizeof(T);

  memory::Copy(dst_place, dst_ptr, src_place, src_ptr, size);
}

template <typename T>
void TensorToVector(const Tensor& src, const platform::DeviceContext& ctx,
                    std::vector<T>* dst) {
  auto src_ptr = static_cast<const void*>(src.data<T>());
  auto size = src.numel() * sizeof(T);

  platform::CPUPlace dst_place;
  dst->resize(src.numel());
  auto dst_ptr = static_cast<void*>(dst->data());

  if (platform::is_cpu_place(src.place())) {
    memory::Copy(dst_place, dst_ptr,
                 BOOST_GET_CONST(platform::CPUPlace, src.place()), src_ptr,
                 size);
  }
#ifdef PADDLE_WITH_CUDA
  else if (platform::is_gpu_place(src.place())) {  // NOLINT
    memory::Copy(
        dst_place, dst_ptr, BOOST_GET_CONST(platform::CUDAPlace, src.place()),
        src_ptr, size,
        reinterpret_cast<const platform::CUDADeviceContext&>(ctx).stream());
  }
#endif
}

template <typename T>
void TensorToVector(const Tensor& src, std::vector<T>* dst) {
  auto src_ptr = static_cast<const void*>(src.data<T>());
  auto size = src.numel() * sizeof(T);

  platform::CPUPlace dst_place;
  dst->resize(src.numel());
  auto dst_ptr = static_cast<void*>(dst->data());

  PADDLE_ENFORCE_EQ(
      platform::is_cpu_place(src.place()), true,
      platform::errors::InvalidArgument(
          "The input tensor should be CPU device, but actually it is in %s.",
          src.place()));

  memory::Copy(dst_place, dst_ptr,
               BOOST_GET_CONST(platform::CPUPlace, src.place()), src_ptr, size);
}

std::ostream& operator<<(std::ostream& os, const Tensor& t);
}  // namespace framework
}  // namespace paddle
