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
#include "paddle/fluid/operators/npu_op_runner.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using DataLayout = framework::DataLayout;
using NPUAttribute = framework::NPUAttribute;
using NPUAttributeMap = framework::NPUAttributeMap;

class NpuOpRunner {
 public:
  explicit NpuOpRunner(std::string op_type);
  explicit NpuOpRunner(std::string op_type,
                       const std::vector<Tensor> &inputs = {},
                       const std::vector<Tensor> &outputs = {},
                       const NPUAttributeMap &attrs = {});

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

  void Run(aclrtStream stream = nullptr);

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

aclrtStream GetCurrentNPUStream(int device_id = -1);

template <typename T>
void FillNpuTensorWithConstant(Tensor *tensor, T val) {
  PADDLE_ENFORCE_EQ(
      tensor->IsInitialized(), true,
      platform::errors::InvalidArgument("The tensor should be initialized."));
  PADDLE_ENFORCE_EQ(
      platform::is_npu_place(tensor->place()), true,
      platform::errors::InvalidArgument("The tensor should be on NPUPlace."));

  auto stream = GetCurrentNPUStream(
      BOOST_GET_CONST(platform::NPUPlace, tensor->place()).device);
  auto src_place = platform::CPUPlace();
  Tensor src_tensor(tensor->type());
  src_tensor.Resize(tensor->dims());
  auto src_tensor_ptr = static_cast<T *>(src_tensor.mutable_data<T>(src_place));
  for (unsigned int i = 0; i < tensor->numel(); ++i) {
    src_tensor_ptr[i] = static_cast<T>(val);
  }
  memory::Copy(BOOST_GET_CONST(platform::NPUPlace, tensor->place()),
               tensor->data<void>(), src_place, src_tensor_ptr,
               tensor->numel() * sizeof(T), stream);
  auto callback = [src_tensor, src_place]() {
    VLOG(4) << "Run callback of var at place " << src_place
            << " in FillNpuTensorWithConstant";
  };
  auto dev_ctx = platform::DeviceContextPool::Instance().Get(tensor->place());
  auto npu_stream =
      static_cast<platform::NPUDeviceContext *>(dev_ctx)->NPUstream();
  npu_stream->AddCallback(callback);
}

}  // namespace operators
}  // namespace paddle
#endif
