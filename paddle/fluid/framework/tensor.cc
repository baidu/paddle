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

#include "paddle/fluid/framework/tensor.h"

namespace paddle {
namespace memory {
namespace allocation {
class Allocation;
}  // namespace allocation
}  // namespace memory
}  // namespace paddle

namespace paddle {
namespace framework {
extern size_t SizeOfType(proto::VarType::Type type);
void Tensor::check_memory_size() const {
  PADDLE_ENFORCE_NOT_NULL(holder_, platform::errors::PreconditionNotMet(
                                       "Tensor holds no memory. "
                                       "Call Tensor::mutable_data firstly."));
  PADDLE_ENFORCE_LE(
      numel() * SizeOfType(type()), memory_size(),
      platform::errors::PreconditionNotMet(
          "Tensor's dimension is out of bound."
          "Tensor's dimension must be equal or less than the size of its "
          "memory."
          "But received  Tensor's dimension is d%, memory's size is %d.",
          numel() * SizeOfType(type()), memory_size()));
}

Tensor::Tensor(const proto::VarType::Type& dtype)
    : type_(dtype),
      offset_(0),
      inplace_version_counter_(std::make_shared<TensorInplaceVersion>(0)) {}

size_t Tensor::memory_size() const {
  return holder_ == nullptr ? 0UL : holder_->size() - offset_;
}

void* Tensor::mutable_data(const platform::Place& place,
                           proto::VarType::Type type, size_t requested_size) {
  type_ = type;
  PADDLE_ENFORCE_GE(
      numel(), 0,
      platform::errors::PreconditionNotMet(
          "The Tensor's element number must be equal or greater than zero. "
          "The Tensor's shape is [",
          dims(), "] now"));
  size_t size = numel() * SizeOfType(type);
  if (requested_size) {
    PADDLE_ENFORCE_GE(
        requested_size, size,
        platform::errors::InvalidArgument(
            "The requested memory size is less than the memory size of Tensor. "
            "But received requested memory size is %d, "
            "memory size of Tensor is %d.",
            requested_size, size));
    size = requested_size;
  }
  /* some versions of boost::variant don't have operator!= */
  if (holder_ == nullptr || !(holder_->place() == place) ||
      holder_->size() < size + offset_) {
    // Reset holder first before re-allocate to save memory
    holder_.reset();
    holder_ = memory::AllocShared(place, size);
    offset_ = 0;
  }
  return reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(holder_->ptr()) +
                                 offset_);
}

void* Tensor::mutable_data(const platform::Place& place,
                           size_t requested_size) {
  PADDLE_ENFORCE_NOT_NULL(this->holder_, platform::errors::PreconditionNotMet(
                                             "The tensor is not initialized."));
  return mutable_data(place, type_, requested_size);
}

Tensor& Tensor::ShareDataWith(const Tensor& src) {
  src.check_memory_size();
  *this = src;
  return *this;
}
Tensor& Tensor::ShareInplaceVersionCounterWith(const Tensor& src) {
  PADDLE_ENFORCE_NOT_NULL(
      inplace_version_counter_,
      platform::errors::PreconditionNotMet(
          "Tensor does not hold inplace_version_counter_."));

  inplace_version_counter_ = src.inplace_version_counter_;
  return *this;
}

Tensor Tensor::Slice(int64_t begin_idx, int64_t end_idx) const {
  check_memory_size();
  PADDLE_ENFORCE_GE(
      begin_idx, 0,
      platform::errors::OutOfRange("The start row index must be greater than 0."
                                   "But received the start index is d%.",
                                   begin_idx));
  PADDLE_ENFORCE_LE(
      end_idx, dims_[0],
      platform::errors::OutOfRange("The end row index is out of bound."));
  PADDLE_ENFORCE_LT(
      begin_idx, end_idx,
      platform::errors::InvalidArgument(
          "The start row index must be less than the end row index."
          "But received the start index = %d, the end index = %d.",
          begin_idx, end_idx));

  if (dims_[0] == 1) {
    return *this;
  } else {
    size_t base = numel() / dims_[0];
    Tensor dst;
    dst.holder_ = holder_;
    dst.set_layout(layout_);
    dst.type_ = type_;
    DDim dst_dims = dims_;
    dst_dims[0] = end_idx - begin_idx;
    dst.Resize(dst_dims);
    dst.offset_ = offset_ + begin_idx * base * SizeOfType(type());
    return dst;
  }
}

Tensor& Tensor::Resize(const DDim& dims) {
  dims_ = dims;
  return *this;
}

const DDim& Tensor::dims() const { return dims_; }

int64_t Tensor::numel() const {
#ifdef PADDLE_WITH_ASCEND_CL
  int64_t npu_storage_numel = product(npu_storage_dims_);
  if (npu_storage_layout_ == DataLayout::kFractalNZ) {
    PADDLE_ENFORCE_GT(npu_storage_numel, 0,
                      paddle::platform::errors::InvalidArgument(
                          "The size of NPU Tensor's storage dims should be "
                          "larger than 0, but get %d",
                          npu_storage_numel));
    VLOG(3) << "use npu storage numel: " << npu_storage_numel;
    return npu_storage_numel;
  } else if (npu_storage_numel != 0) {
    int64_t basic_numel = product(dims_);
    PADDLE_ENFORCE_EQ(npu_storage_numel, basic_numel,
                      paddle::platform::errors::InvalidArgument(
                          "NPU storage size %d should be equal with basic "
                          "size %d in NCHW data format",
                          npu_storage_numel, basic_numel));
  }
#endif
  return product(dims_);
}

Tensor& Tensor::ResizeNPUDims(const DDim& npu_storage_dims) {
  npu_storage_dims_ = npu_storage_dims;
  return *this;
}

const DDim& Tensor::npu_storage_dims() const { return npu_storage_dims_; }

int64_t Tensor::npu_storage_numel() const { return product(npu_storage_dims_); }

void Tensor::ResetHolder(std::shared_ptr<memory::Allocation> holder) {
  PADDLE_ENFORCE_EQ(
      offset_, 0,
      platform::errors::Fatal(
          "Only the offset is supported to zero when the holder is reset."));
  if (holder_) {
    PADDLE_ENFORCE_LE(
        numel() * SizeOfType(type()) + offset_, holder->size(),
        paddle::platform::errors::InvalidArgument(
            "The size of Holder is not enough to store the Tensor."));
  }
  holder_ = holder;
}

void Tensor::ResetHolderWithType(std::shared_ptr<memory::Allocation> holder,
                                 const proto::VarType::Type type) {
  ResetHolder(holder);
  type_ = type;
}

}  // namespace framework
}  // namespace paddle
