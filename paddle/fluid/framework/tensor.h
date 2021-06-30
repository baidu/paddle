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

#include <cstdint>
#include <cstring>
#include <memory>
#include <typeindex>
#include <utility>
#include <vector>

#include "paddle/fluid/framework/data_layout.h"
#include "paddle/fluid/framework/ddim.h"
#include "paddle/fluid/framework/framework.pb.h"
#include "paddle/fluid/memory/memory.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/place.h"

namespace paddle {
namespace memory {
namespace allocation {
class Allocation;
}  // namespace allocation
}  // namespace memory
}  // namespace paddle

namespace paddle {

namespace framework {

class LoDTensor;

/*
 NOTE(liym27): [ What is TensorInplaceVersion used for? ]

 TensorInplaceVersion is a version counter and every Tensor has a version
 counter. It's used to check whether an inplace operation will result in an
 incorrect gradient calculation. Version is incremented when the data of the
 Variable is modified in place.

 - Question: In what scenarios will version counters be shared?
 - Answer: When two Variables/VarBases share the same C++ Tensor(its Allocation
 may change), both of them share the same version counter. For examples:
  1. `z = paddle.assign(input=x, output=y)`, `z` shares the same version counter
    of `y` because z and y is the same VarBase;
  2. `y = x.detach()`, `y` shares the same version counter of `x`.

 - Question: In what scenarios will version counters NOT be shared?
 - Answer: Replacing a `Variable`'s data by calling `Tensor::ShareDataWith(...)`
 or `Tensor::ShareBufferWith(...)`. Because they share the same Allocation but
 not framework::Tensor.

 - Question: Why put the inplace_version_counter_ in framework::Tensor instead
 of Allocation or Variable?
 - Answer:
  1. Tensor can call ResetHolder() to reset the corresponding Allocation so that
  the inplace_version_counter_ changes if it's in Allocation, which will lead to
  confusing information about inplace version.
  2. If inplace_version_counter_ is in Variable, different VariableWrappers
  should be able to share the same Variable. However, a VariableWrapper hold a
  Variable object but not a pointer.
*/

class TensorInplaceVersion {
 public:
  explicit TensorInplaceVersion(uint32_t inplace_version = 0)
      : inplace_version_(inplace_version) {}
  bool IsUnique() const { return inplace_version_ == 0; }
  void Bump() { ++inplace_version_; }
  uint32_t CurrentVersion() const { return inplace_version_; }

 private:
  uint32_t inplace_version_;
};

class Tensor {
#ifdef PADDLE_WITH_MKLDNN

 public:
  inline mkldnn::memory::format_tag format() const { return format_; }

  inline void set_format(const mkldnn::memory::format_tag format) {
    format_ = format;
  }

 protected:
  /**
   * @brief the detail format of memory block which have layout as kMKLDNN
   *
   * @note MKLDNN lib support various memory format like nchw, nhwc, nChw8C,
   *       nChw16c, etc. For a MKLDNN memory block, layout will be set as
   *       DataLayout::kMKLDNN meanwhile detail memory format will be kept in
   *       this field.
   */

  mkldnn::memory::format_tag format_ = mkldnn::memory::format_tag::undef;
#endif

 public:
  template <typename T, size_t D, int MajorType, typename IndexType>
  friend struct EigenTensor;

  template <typename T, int MajorType, typename IndexType>
  friend struct EigenMatrix;

  template <typename T, int MajorType, typename IndexType>
  friend struct EigenVector;

 public:
  Tensor()
      : type_(proto::VarType::FP32),
        offset_(0),
        inplace_version_counter_(std::make_shared<TensorInplaceVersion>(0)) {}

  explicit Tensor(const proto::VarType::Type&);

  /*! Return a pointer to mutable memory block. */
  template <typename T>
  T* data();

  /*! Return a pointer to constant memory block. */
  template <typename T>
  const T* data() const;

  inline bool IsInitialized() const;

  /**
   * @brief   Return a pointer to mutable memory block.
   * @note    If not exist, then allocation.
   */
  template <typename T>
  T* mutable_data(const platform::Place& place, size_t requested_size = 0);

  void* mutable_data(const platform::Place& place, proto::VarType::Type type,
                     size_t requested_size = 0);

  void* mutable_data(const platform::Place& place, size_t requested_size = 0);

  /**
   * @brief     Return a pointer to mutable memory block.
   *
   * @param[in] dims           The dimensions of the memory block.
   * @param[in] place          The place of the memory block.
   * @param[in] requested_size The size of the block in bytes.
   *
   * @note      If not exist, then allocation.
   */
  template <typename T>
  T* mutable_data(const DDim& dims, const platform::Place& place,
                  size_t requested_size = 0);

  /*! Return the dimensions of the memory block. */
  const DDim& dims() const;

  /*! Return the numel of the memory block. */
  int64_t numel() const;

  /*! Resize the dimensions of the memory block. */
  Tensor& Resize(const DDim& dims);

  /*! The internal of two tensors share the same memory block. */
  Tensor& ShareDataWith(const Tensor& src);

  /*! The internal of two tensors share the same inplace version counter. */
  Tensor& ShareInplaceVersionCounterWith(const Tensor& src);

  /**
   * @brief  Return a sub-tensor of the given tensor.
   *
   * @param[in] begin_idx   The index of the start row(inclusive) to slice.
   *                        The index number begins from 0.
   * @param[in] end_idx     The index of the end row(exclusive) to slice.
   *                        The index number begins from 0.
   */
  Tensor Slice(int64_t begin_idx, int64_t end_idx) const;

  /**
   * @brief  Return a tensor list of the given tensor.
   *
   * @param[in] split_size  The size of tensor to be split along axis.
   * @param[in] axis        The axis along which to split.
   */
  std::vector<Tensor> Split(int64_t split_size, int64_t axis) const;

  /**
   * @brief  Return a tensor list of the given tensor.
   *
   * @param[in] chunks   The number of tensor to be split along axis.
   * @param[in] axis     The axis along which to split.
   */
  std::vector<Tensor> Chunk(int64_t chunks, int64_t axis) const;

  const platform::Place& place() const {
    PADDLE_ENFORCE_NOT_NULL(
        holder_,
        platform::errors::PreconditionNotMet(
            "Tensor not initialized yet when Tensor::place() is called."));
    return holder_->place();
  }

  proto::VarType::Type type() const {
    PADDLE_ENFORCE_NOT_NULL(
        holder_,
        platform::errors::PreconditionNotMet(
            "Tensor not initialized yet when Tensor::type() is called."));
    return type_;
  }

  /**
   * [Add method get the saved type of tensor]
   *
   * After the introduction of complex number calculations, Ops that support
   * complex number calculations generally support type promotion, such as
   * x(float32) + y(complex64) = out(complex64), then the type of the grad
   * tensor should be dout(complex64), dx(float32), dy (complex64), but the
   * type of dx to be recognized to be float32 by the grad Op relay on the type
   * of forward tensor x. But many of our ops have registered InplaceInferer,
   * covering the tensor memory of x with out, so as to save storage.
   *
   * In this case, the dim and type information recorded by x still exist,
   * but because x becomes an uninitialized tensor, The type of x record cannot
   * be obtained with x.type(), but the type is still valid here, so we
   * add saved_type(), This method SHOULD NOT be called by general scenarios.
   */
  proto::VarType::Type saved_type() const { return type_; }

  // memory size returns the holding memory size in byte.
  size_t memory_size() const;

  void check_memory_size() const;

  DataLayout layout() const { return layout_; }

  void set_layout(const DataLayout layout) { layout_ = layout; }

#ifdef PADDLE_WITH_ASCEND_CL
  DataLayout npu_storage_layout() const { return npu_storage_layout_; }

  void set_npu_storage_layout(const DataLayout npu_storage_layout) {
    npu_storage_layout_ = npu_storage_layout;
  }

  /*! Return the dimensions of the memory block used in NPU FRACTAL_NZ
   * data_format. */
  const DDim& npu_storage_dims() const;

  /*! Return the numel of the memory block used in NPU FRACTAL_NZ data_format.
   */
  int64_t npu_storage_numel() const;

  /*! Resize the dimensions of the memory block used in NPU FRACTAL_NZ
   * data_format. */
  Tensor& ResizeNPUDims(const DDim& npu_storage_dims);

/* Use this method carefully, where the data_format of the NPU Tensor is
 * FRACTAL_NZ. */
// Tensor& ShareDataWithNPUTensor(const Tensor& src);

// void check_memory_size_of_nz_tensor() const;
#endif

  void clear() {
    holder_ = nullptr;
    offset_ = 0;
  }

  void ShareBufferWith(const Tensor& tensor) {
    holder_ = tensor.holder_;
    offset_ = tensor.offset_;
    type_ = tensor.type_;
  }

  bool IsSharedBufferWith(const Tensor& src) const {
    return holder_ && holder_ == src.Holder();
  }

  const std::shared_ptr<memory::Allocation>& Holder() const { return holder_; }
  size_t offset() const { return offset_; }

  std::shared_ptr<memory::Allocation> MoveMemoryHolder() {
    return std::move(holder_);
  }

  void ResetHolder(std::shared_ptr<memory::Allocation> holder);

  void ResetHolderWithType(std::shared_ptr<memory::Allocation> holder,
                           const proto::VarType::Type type);

  TensorInplaceVersion& InplaceVersionCounter() {
    return *inplace_version_counter_;
  }

 private:
  /*! holds the memory block if allocated. */
  std::shared_ptr<memory::Allocation> holder_;
  proto::VarType::Type type_;
  /**
   * @brief points to elements dimensions.
   *
   * @note dims_ do not indicate the memory block size.
   */

  DDim dims_;

  /**
   * @brief the layout of memory block, default is NHWC.
   *
   * @note the memory allocation order, describe how weight/data is stored
   *       For example, in 4-D Tensor(rank=4), there are three commonly
   *       used layout. They are
   *            NCHW, NHWC, CHWN.
   *       N,C,H,W for respectively the batch size, the number of
   *       feature maps, the height.
   */
  // Fix me: here just change the default layout to kNCHW
  // it doesn't fix the real issue, i.e. feeder should set up tensor layout
  // according to actual input data
  DataLayout layout_ = DataLayout::kNCHW;

#ifdef PADDLE_WITH_ASCEND_CL
  DDim npu_storage_dims_;
  DataLayout npu_storage_layout_ = DataLayout::kNCHW;
#endif

  /**
   * @brief   A PlaceHolder may be shared by more than one tensor.
   *
   * @note    Some of them may be slices of the others. So the offset_
   *          is introduced here to indicate the byte offset between
   *          PlaceHolder::ptr_ and where the tensor data really begins.
   */
  size_t offset_;
  std::shared_ptr<TensorInplaceVersion> inplace_version_counter_;
};

}  // namespace framework
}  // namespace paddle

#include "paddle/fluid/framework/tensor_impl.h"
