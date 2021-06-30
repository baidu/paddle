/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#ifdef PADDLE_WITH_ASCEND_CL
#pragma once
#include <paddle/fluid/framework/operator.h>
#include <paddle/fluid/framework/type_defs.h>

#include <string>
#include <vector>

#include "acl/acl.h"
#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/fluid/operators/npu_op_runner.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using DataLayout = framework::DataLayout;
using NPUAttribute = framework::NPUAttribute;
using NPUAttributeMap = framework::NPUAttributeMap;
using DeviceContextPool = platform::DeviceContextPool;

class NpuOpRunner {
 public:
  explicit NpuOpRunner(std::string op_type);
  explicit NpuOpRunner(std::string op_type,
                       const std::vector<Tensor> &inputs = {},
                       const std::vector<Tensor> &outputs = {},
                       const NPUAttributeMap &attrs = {});

  // NOTE(zhiqiu): why forbid copy and operator= ?
  // Since we will free the tensor_descs and data_buffers in the ~NpuOpRunner,
  // if shallow copy is performed on tensor_descs and data_buffers, it may
  // result
  // in use-after-free bugs.
  NpuOpRunner(const NpuOpRunner &runner) = delete;
  NpuOpRunner &operator=(const NpuOpRunner &runner) = delete;

  ~NpuOpRunner();

  const std::string &Type();

  NpuOpRunner &AddAttr(const std::string &name, const NPUAttribute &attr);

  NpuOpRunner &AddAttrs(const NPUAttributeMap &attrs);

  NpuOpRunner &AddInput(const Tensor &tensor);

  NpuOpRunner &AddOutput(const Tensor &tensor);

  NpuOpRunner &AddInputs(const std::vector<Tensor> &tensors);

  NpuOpRunner &AddInputNames(const std::vector<std::string> &names);

  NpuOpRunner &AddOutputs(const std::vector<Tensor> &tensors);

  aclTensorDesc *GetInputDesc(size_t index);

  aclTensorDesc *GetOutputDesc(size_t index);

  std::vector<aclTensorDesc *> &GetInputDescs();

  std::vector<aclTensorDesc *> &GetOutputDescs();

  std::vector<aclDataBuffer *> &GetInputBuffers();

  std::vector<aclDataBuffer *> &GetOutputBuffers();

  void Run(aclrtStream stream = nullptr) const;

 private:
  aclTensorDesc *CreateTensorDesc(Tensor tensor);
  aclDataBuffer *CreateDataBuffer(Tensor tensor);

 private:
  std::string op_type_;
  std::vector<aclDataBuffer *> input_buffers_;
  std::vector<aclDataBuffer *> output_buffers_;
  std::vector<aclTensorDesc *> input_descs_;
  std::vector<aclTensorDesc *> output_descs_;
  aclopAttr *attr_{nullptr};
};

aclDataType ConvertToNpuDtype(framework::proto::VarType::Type dtype);

aclFormat ConvertToNpuFormat(DataLayout layout);

DataLayout ConvertNpuFormatToDataLayout(aclFormat acl_format);

aclFormat FindBaseFormat(aclFormat acl_format);

aclrtStream GetCurrentNPUStream(int device_id = -1);

platform::Place GetCurrentNPUPlace(int device_id = -1);

std::vector<int64_t> InferShapeLessTo4(std::vector<int64_t> dims);

std::vector<int64_t> InferShapeNCHWToND(std::vector<int64_t> base_dims);

Tensor FormatCastBetweenGroup(const Tensor &src_tensor, Tensor dst_tensor,
                              Tensor trans_src_tensor);

std::vector<int64_t> InferShapeNDToNZ(std::vector<int64_t> dims);

void RunTransDataNPUOP(const Tensor &src_tensor, Tensor *dst_tensor,
                       aclrtStream stream);

Tensor RunTransDataToCastFormat(const Tensor &src_tensor, Tensor dst_tensor);

Tensor CastNPUFormat(const Tensor &src_tensor, int acl_format);

Tensor GenerateNZTensor(const Tensor &src_tensor);

void InferNPUStorageFormatAndDims(Tensor *dst, DataLayout layout);

template <typename T>
void FillNpuTensorWithConstant(Tensor *tensor, T val) {
  PADDLE_ENFORCE_EQ(
      tensor->IsInitialized(), true,
      platform::errors::InvalidArgument("The tensor should be initialized."));
  PADDLE_ENFORCE_EQ(
      platform::is_npu_place(tensor->place()), true,
      platform::errors::InvalidArgument("The tensor should be on NPUPlace."));

  int numel = tensor->numel();
  if (numel == 1) {
    Tensor npu_pinned_tensor(tensor->type());
    platform::NPUPinnedPlace npu_pinned_place;
    auto npu_pinned_ptr =
        npu_pinned_tensor.mutable_data<T>({1}, npu_pinned_place);
    *npu_pinned_ptr = val;

    memory::Copy(BOOST_GET_CONST(platform::NPUPlace, tensor->place()),
                 tensor->data<void>(), npu_pinned_place, npu_pinned_ptr,
                 sizeof(T), GetCurrentNPUStream());

    auto npu_pinned_allocator =
        static_cast<paddle::memory::allocation::NPUPinnedAllocator *>(
            paddle::memory::allocation::AllocatorFacade::Instance()
                .GetAllocator(npu_pinned_place)
                .get());
    paddle::memory::allocation::Allocation *allocation =
        npu_pinned_tensor.Holder().get();

    npu_pinned_allocator->RecordEvent(allocation, GetCurrentNPUStream());
  } else {
    std::vector<T> vec(numel, static_cast<T>(val));
    auto device_id = platform::GetCurrentNPUDeviceId();
    platform::DeviceContextPool &pool = platform::DeviceContextPool::Instance();
    auto *dev_ctx = static_cast<platform::NPUDeviceContext *>(
        pool.Get(platform::NPUPlace(device_id)));

    paddle::framework::TensorFromVector<T>(vec, *dev_ctx, tensor);
  }
}

}  // namespace operators
}  // namespace paddle
#endif
