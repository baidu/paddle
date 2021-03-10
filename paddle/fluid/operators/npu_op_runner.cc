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

#include "paddle/fluid/operators/npu_op_runner.h"

#include <paddle/fluid/framework/data_type.h>
#include <paddle/fluid/framework/operator.h>

#include <map>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "acl/acl_op_compiler.h"

#include "paddle/fluid/framework/framework.pb.h"

namespace paddle {
namespace operators {

static std::map<framework::proto::VarType::Type, aclDataType>
    DTYPE_2_ACL_DTYPE = {
        {framework::proto::VarType::BOOL, ACL_BOOL},
        {framework::proto::VarType::INT16, ACL_INT16},
        {framework::proto::VarType::INT32, ACL_INT32},
        {framework::proto::VarType::INT64, ACL_INT64},
        {framework::proto::VarType::FP16, ACL_FLOAT16},
        {framework::proto::VarType::FP32, ACL_FLOAT},
        {framework::proto::VarType::FP64, ACL_DOUBLE},
};

static std::map<DataLayout, aclFormat> DATA_LAYOUT_2_ACL_FORMAT = {
    {DataLayout::kNCHW, ACL_FORMAT_NCHW},
    {DataLayout::kNHWC, ACL_FORMAT_NHWC},
    {DataLayout::kAnyLayout, ACL_FORMAT_ND},
};

aclDataType ConvertToNpuDtype(framework::proto::VarType::Type dtype) {
  auto iter = DTYPE_2_ACL_DTYPE.find(dtype);
  PADDLE_ENFORCE_NE(iter, DTYPE_2_ACL_DTYPE.end(),
                    platform::errors::NotFound(
                        "The data type (%s) can not convert to ACL data type.",
                        framework::DataTypeToString(dtype)));
  return iter->second;
}

aclFormat ConvertToNpuFormat(DataLayout layout) {
  auto iter = DATA_LAYOUT_2_ACL_FORMAT.find(layout);
  PADDLE_ENFORCE_NE(
      iter, DATA_LAYOUT_2_ACL_FORMAT.end(),
      platform::errors::NotFound(
          "The data type (%s) can not convert to ACL data type.", layout));
  return iter->second;
}

NpuOpRunner::NpuOpRunner(std::string op_type) : op_type_(op_type) {
  attr_ = aclopCreateAttr();
}

NpuOpRunner::NpuOpRunner(std::string op_type, const std::vector<Tensor> &inputs,
                         const std::vector<Tensor> &outputs,
                         const NPUAttributeMap &attrs)
    : op_type_(op_type) {
  attr_ = aclopCreateAttr();
  AddInputs(inputs);
  AddOutputs(outputs);
  AddAttrs(attrs);
}

NpuOpRunner::~NpuOpRunner() {
  // TODO(zhiqiu): handle free
}

const std::string &NpuOpRunner::Type() { return op_type_; }

NpuOpRunner &NpuOpRunner::AddAttr(const std::string &name,
                                  const NPUAttribute &attr) {
  if (attr.type() == typeid(bool)) {
    PADDLE_ENFORCE_NPU_SUCCESS(
        aclopSetAttrBool(attr_, name.c_str(), BOOST_GET_CONST(bool, attr)));
  } else if (attr.type() == typeid(int)) {
    PADDLE_ENFORCE_NPU_SUCCESS(
        aclopSetAttrInt(attr_, name.c_str(), BOOST_GET_CONST(int, attr)));

  } else if (attr.type() == typeid(int64_t)) {
    PADDLE_ENFORCE_NPU_SUCCESS(
        aclopSetAttrInt(attr_, name.c_str(), BOOST_GET_CONST(int64_t, attr)));
  } else if (attr.type() == typeid(float)) {
    PADDLE_ENFORCE_NPU_SUCCESS(
        aclopSetAttrFloat(attr_, name.c_str(), BOOST_GET_CONST(float, attr)));
  } else if (attr.type() == typeid(std::vector<bool>)) {
    auto a = BOOST_GET_CONST(std::vector<bool>, attr);
    std::vector<uint8_t> cast_a;
    for (auto it : a) {
      cast_a.push_back(static_cast<uint8_t>(it));
    }
    PADDLE_ENFORCE_NPU_SUCCESS(aclopSetAttrListBool(
        attr_, name.c_str(), cast_a.size(), cast_a.data()));
  } else if (attr.type() == typeid(std::vector<int>)) {
    auto a = BOOST_GET_CONST(std::vector<int>, attr);
    std::vector<int64_t> cast_a;
    for (auto it : a) {
      cast_a.push_back(static_cast<int64_t>(it));
    }
    PADDLE_ENFORCE_NPU_SUCCESS(
        aclopSetAttrListInt(attr_, name.c_str(), cast_a.size(), cast_a.data()));
  } else if (attr.type() == typeid(std::vector<int64_t>)) {
    auto a = BOOST_GET_CONST(std::vector<int64_t>, attr);
    PADDLE_ENFORCE_NPU_SUCCESS(
        aclopSetAttrListInt(attr_, name.c_str(), a.size(), a.data()));
  } else if (attr.type() == typeid(std::vector<float>)) {
    auto a = BOOST_GET_CONST(std::vector<float>, attr);
    PADDLE_ENFORCE_NPU_SUCCESS(
        aclopSetAttrListFloat(attr_, name.c_str(), a.size(), a.data()));
  } else if (attr.type() == typeid(std::string)) {
    auto a = BOOST_GET_CONST(std::string, attr);
    PADDLE_ENFORCE_NPU_SUCCESS(
        aclopSetAttrString(attr_, name.c_str(), a.c_str()));
  } else if (attr.type() == typeid(std::vector<std::string>)) {
    auto a = BOOST_GET_CONST(std::vector<std::string>, attr);
    std::vector<const char *> s;
    for (auto &it : a) {
      s.push_back(it.data());
    }
    PADDLE_ENFORCE_NPU_SUCCESS(
        aclopSetAttrListString(attr_, name.c_str(), s.size(), s.data()));
  } else if (attr.type() == typeid(std::vector<std::vector<int64_t>>)) {
    auto a = BOOST_GET_CONST(std::vector<std::vector<int64_t>>, attr);
    std::vector<int64_t *> data;
    std::vector<int> num;
    for (auto &&v : a) {
      data.push_back(v.data());
      num.push_back(v.size());
    }
    PADDLE_ENFORCE_NPU_SUCCESS(
        aclopSetAttrListListInt(attr_, name.c_str(),
                                data.size(),
                                num.data(),
                                data.data()));
  } else {
    PADDLE_THROW(platform::errors::Unimplemented(
        "Can not convert attribubte '%s' to convert to aclopAttr", name));
  }
  return *this;
}

NpuOpRunner &NpuOpRunner::AddAttrs(const NPUAttributeMap &attrs) {
  for (const auto &pair : attrs) {
    AddAttr(pair.first, pair.second);
  }
  return *this;
}

NpuOpRunner &NpuOpRunner::AddInput(const Tensor &tensor) {
  // create aclTensorDesc
  input_descs_.emplace_back(CreateTensorDesc(tensor));
  // create aclDataBuffer
  input_buffers_.emplace_back(CreateDataBuffer(tensor));
  return *this;
}

NpuOpRunner &NpuOpRunner::AddOutput(const Tensor &tensor) {
  // create aclTensorDesc
  output_descs_.emplace_back(CreateTensorDesc(tensor));
  // create aclDataBuffer
  output_buffers_.emplace_back(CreateDataBuffer(tensor));
  return *this;
}

NpuOpRunner &NpuOpRunner::AddInputs(const std::vector<Tensor> &tensors) {
  for (auto tensor : tensors) {
    // create aclTensorDesc
    input_descs_.emplace_back(CreateTensorDesc(tensor));
    // create aclDataBuffer
    input_buffers_.emplace_back(CreateDataBuffer(tensor));
  }
  return *this;
}

NpuOpRunner &NpuOpRunner::AddOutputs(const std::vector<Tensor> &tensors) {
  for (auto tensor : tensors) {
    // create aclTensorDesc
    output_descs_.emplace_back(CreateTensorDesc(tensor));
    // create aclDataBuffer
    output_buffers_.emplace_back(CreateDataBuffer(tensor));
  }
  return *this;
}

aclTensorDesc *NpuOpRunner::GetInputDesc(size_t index) {
  PADDLE_ENFORCE_LT(index, input_descs_.size(),
                    platform::errors::OutOfRange(
                        "The index should be less than the size of inputs of "
                        "operator %s, but got index is %d and size is %d",
                        Type(), index, input_descs_.size()));
  return input_descs_[index];
}

aclTensorDesc *NpuOpRunner::GetOutputDesc(size_t index) {
  PADDLE_ENFORCE_LT(index, output_descs_.size(),
                    platform::errors::OutOfRange(
                        "The index should be less than the size of output of "
                        "operator %s, but got index is %d and size is %d",
                        Type(), index, output_descs_.size()));
  return output_descs_[index];
}

std::vector<aclTensorDesc *> &NpuOpRunner::GetInputDescs() {
  return input_descs_;
}

std::vector<aclTensorDesc *> &NpuOpRunner::GetOutputDescs() {
  return output_descs_;
}

std::vector<aclDataBuffer *> &NpuOpRunner::GetInputBuffers() {
  return input_buffers_;
}

std::vector<aclDataBuffer *> &NpuOpRunner::GetOutputBuffers() {
  return output_buffers_;
}

aclTensorDesc *NpuOpRunner::CreateTensorDesc(Tensor tensor) {
  auto dtype = ConvertToNpuDtype(tensor.type());
  auto format = ConvertToNpuFormat(tensor.layout());
  auto dims = framework::vectorize(tensor.dims());

  VLOG(4) << dtype << " " << dims.size() << " " << dims[0] << "," << dims[1]
          << " " << format;

  auto *desc = aclCreateTensorDesc(dtype, dims.size(), dims.data(), format);
  PADDLE_ENFORCE_NOT_NULL(
      desc, platform::errors::External("Call aclCreateTensorDesc failed."));
  return desc;
}

aclDataBuffer *NpuOpRunner::CreateDataBuffer(Tensor tensor) {
  void *ptr = tensor.data<void>();
  VLOG(4) << "ptr: " << ptr << ", size: " << tensor.memory_size();
  auto *buffer = aclCreateDataBuffer(ptr, tensor.memory_size());
  PADDLE_ENFORCE_NOT_NULL(
      buffer, platform::errors::External("Call aclCreateDataBuffer failed."));
  return buffer;
}

void NpuOpRunner::Run(aclrtStream stream) {
  VLOG(4) << "op_type: " << op_type_;
  VLOG(4) << "input_desc.size: " << input_descs_.size();
  VLOG(4) << "output_desc.size: " << output_descs_.size();
  VLOG(4) << "stream: " << stream;
  VLOG(4) << "attr: " << attr_;
  aclError ret = aclopCompileAndExecute(
      op_type_.c_str(), input_descs_.size(), input_descs_.data(),
      input_buffers_.data(), output_descs_.size(), output_descs_.data(),
      output_buffers_.data(), attr_, ACL_ENGINE_SYS, ACL_COMPILE_SYS, NULL,
      stream);
  VLOG(4) << "after aclopCompileAndExecute: " << ret;
  PADDLE_ENFORCE_NPU_SUCCESS(ret);
}
}  // namespace operators
}  // namespace paddle
