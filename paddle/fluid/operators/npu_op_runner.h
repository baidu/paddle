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

}  // namespace operators
}  // namespace paddle
#endif
