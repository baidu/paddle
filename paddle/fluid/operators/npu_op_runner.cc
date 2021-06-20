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
    {DataLayout::kFractalNZ, ACL_FORMAT_FRACTAL_NZ},
};

static std::map<aclFormat, DataLayout> ACL_FORMAT_2_DATA_LAYOUT = {
    {ACL_FORMAT_NCHW, DataLayout::kNCHW},
    {ACL_FORMAT_NHWC, DataLayout::kNHWC},
    {ACL_FORMAT_ND, DataLayout::kAnyLayout},
    {ACL_FORMAT_FRACTAL_NZ, DataLayout::kFractalNZ},
};

static std::map<aclFormat, aclFormat> ACL_FORMAT_2_BASE_FORMAT = {
    {ACL_FORMAT_NCHW, ACL_FORMAT_NCHW},
    {ACL_FORMAT_ND, ACL_FORMAT_ND},
    {ACL_FORMAT_FRACTAL_NZ, ACL_FORMAT_ND},
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

DataLayout ConvertNpuFormatToDataLayout(aclFormat acl_format) {
  auto iter = ACL_FORMAT_2_DATA_LAYOUT.find(acl_format);
  PADDLE_ENFORCE_NE(
      iter, ACL_FORMAT_2_DATA_LAYOUT.end(),
      platform::errors::NotFound(
          "The ACL data type (%s) can not convert to data type.", acl_format));
  return iter->second;
}

aclFormat FindBaseFormat(aclFormat acl_format) {
  auto iter = ACL_FORMAT_2_BASE_FORMAT.find(acl_format);
  PADDLE_ENFORCE_NE(
      iter, ACL_FORMAT_2_BASE_FORMAT.end(),
      platform::errors::NotFound(
          "Can't find base format of the data type (%s).", acl_format));
  return iter->second;
}

aclrtStream GetCurrentNPUStream(int device_id) {
  if (device_id == -1) {
    device_id = platform::GetCurrentNPUDeviceId();
  }
  platform::DeviceContextPool &pool = platform::DeviceContextPool::Instance();
  auto *dev_ctx = static_cast<platform::NPUDeviceContext *>(
      pool.Get(platform::NPUPlace(device_id)));
  return dev_ctx->stream();
}

platform::Place GetCurrentNPUPlace(int device_id) {
  if (device_id == -1) {
    device_id = platform::GetCurrentNPUDeviceId();
  }
  platform::DeviceContextPool &pool = platform::DeviceContextPool::Instance();
  auto *dev_ctx = static_cast<platform::NPUDeviceContext *>(
      pool.Get(platform::NPUPlace(device_id)));
  return dev_ctx->GetPlace();
}

std::vector<int64_t> InferShapeLessTo4(std::vector<int64_t> dims) {
  std::vector<int64_t> res;
  res.resize(4);
  PADDLE_ENFORCE_LE(
      dims.size(), 4,
      platform::errors::InvalidArgument(
          "The input shape should <= 4, but got %d", dims.size()));
  switch (dims.size()) {
    case 0:
      res[0] = 1;
      res[1] = 1;
      res[2] = 1;
      res[3] = 1;
      break;
    case 1:
      // RESHAPE_TYPE_C;
      res[0] = 1;
      res[1] = dims[0];
      res[2] = 1;
      res[3] = 1;
      break;
    case 2:
      // RESHAPE_TYPE_CH;
      res[0] = 1;
      res[1] = dims[0];
      res[2] = dims[1];
      res[3] = 1;
      break;
    case 3:
      // RESHAPE_TYPE_CHW;
      res[0] = 1;
      res[1] = dims[0];
      res[2] = dims[1];
      res[3] = dims[2];
      break;
    case 4:
      res[0] = dims[0];
      res[1] = dims[1];
      res[2] = dims[2];
      res[3] = dims[3];
      break;
    default:
      PADDLE_ENFORCE_LE(
          dims.size(), 4,
          platform::errors::InvalidArgument(
              "The input shape should <= 4 in InferShapeLessTo4, but got %d",
              dims.size()));
  }
  return res;
}

std::vector<int64_t> InferShapeNCHWToND(std::vector<int64_t> base_dims) {
  std::vector<int64_t> res;
  res.resize(4);
  auto cur_storage_dims = base_dims;
  if (base_dims.size() != 4) {
    cur_storage_dims = InferShapeLessTo4(base_dims);
  }
  PADDLE_ENFORCE_EQ(
      cur_storage_dims.size(), 4,
      platform::errors::InvalidArgument(
          "The storage_dims should = 4 in InferShapeNCHWToND, but got %d",
          cur_storage_dims.size()));

  if (base_dims.size() == 0) {
    std::vector<int64_t> temp_dims;
    temp_dims.emplace_back(1);
    return InferShapeLessTo4(temp_dims);
  }
  switch (base_dims.size()) {
    case 1:
      // reshape_type = RESHAPE_TYPE_C;
      res.resize(1);
      res[0] = cur_storage_dims[1];
      PADDLE_ENFORCE_EQ(cur_storage_dims[0], 1,
                        platform::errors::InvalidArgument(
                            "N must be 1, but got %d", cur_storage_dims[0]));
      PADDLE_ENFORCE_EQ(cur_storage_dims[2], 1,
                        platform::errors::InvalidArgument(
                            "H must be 1, but got %d", cur_storage_dims[2]));
      PADDLE_ENFORCE_EQ(cur_storage_dims[3], 1,
                        platform::errors::InvalidArgument(
                            "W must be 1, but got %d", cur_storage_dims[3]));
      break;
    case 2:
      // reshape_type = RESHAPE_TYPE_CH;
      res.resize(2);
      res[0] = cur_storage_dims[1];
      res[1] = cur_storage_dims[2];
      PADDLE_ENFORCE_EQ(cur_storage_dims[0], 1,
                        platform::errors::InvalidArgument(
                            "N must be 1, but got %d", cur_storage_dims[0]));
      PADDLE_ENFORCE_EQ(cur_storage_dims[3], 1,
                        platform::errors::InvalidArgument(
                            "W must be 1, but got %d", cur_storage_dims[3]));
      break;
    case 3:
      // reshape_type = RESHAPE_TYPE_CHW;
      res.resize(3);
      res[0] = cur_storage_dims[1];
      res[1] = cur_storage_dims[2];
      res[2] = cur_storage_dims[3];
      PADDLE_ENFORCE_EQ(cur_storage_dims[0], 1,
                        platform::errors::InvalidArgument(
                            "N must be 1, but got %d", cur_storage_dims[0]));
      break;
    case 4:
      res = cur_storage_dims;
      return res;
    default:
      PADDLE_ENFORCE_LE(
          base_dims.size(), 4,
          platform::errors::InvalidArgument("base_dims should <= 4, but got %d",
                                            base_dims.size()));
  }
  return res;
}

Tensor FormatCastBetweenGroup(const Tensor &src_tensor, Tensor dst_tensor,
                              Tensor trans_src_tensor) {
  std::string src_format_name =
      framework::DataLayoutToString(src_tensor.layout());
  std::string dst_format_name =
      framework::DataLayoutToString(dst_tensor.npu_storage_layout());

  if (src_format_name == "NCHW" && dst_format_name == "FRACTAL_NZ") {
    auto dims = framework::vectorize(src_tensor.dims());
    std::vector<int64_t> storage_dims = InferShapeNCHWToND(dims);
    trans_src_tensor.Resize(framework::make_ddim(storage_dims));
    trans_src_tensor.set_layout(DataLayout::kAnyLayout);
    VLOG(4) << "Cast NPU format from NCHW to ND.";
  }

  return trans_src_tensor;
}

std::vector<int64_t> InferShapeNDToNZ(std::vector<int64_t> dims) {
  std::vector<int64_t> res;
  // sum(keepdim = false) may make tensor dim = 0
  std::vector<int64_t> dim;
  for (uint64_t i = 0; i < dims.size(); i++) {
    dim.emplace_back(dims[i]);
  }

  // TODO(ascend): this expand code can be remove now
  // this action will move to GuessStorageSizeWhenConvertFormat
  if (dim.size() == 0) {
    dim.emplace_back(1);
  }
  if (dim.size() == 1) {
    dim.emplace_back(1);
  }

  uint64_t i = 0;
  for (; i < dim.size() - 2; i++) {
    res.emplace_back(dim[i]);
  }

  res.emplace_back((dim[i + 1] + 15) / 16);
  res.emplace_back((dim[i] + 15) / 16);
  res.emplace_back(16);
  res.emplace_back(16);

  return res;
}

Tensor RunTransDataToCastFormat(const Tensor &src_tensor, Tensor dst_tensor) {
  // auto src_format = ConvertToNpuFormat(src_tensor.layout());
  // auto dst_npu_format = ConvertToNpuFormat(dst_tensor.npu_storage_layout());
  // std::string src_format_name =
  // framework::DataLayoutToString(src_tensor.layout());
  std::string dst_format_name =
      framework::DataLayoutToString(dst_tensor.npu_storage_layout());
  // auto src_base_format = FindBaseFormat(src_format);
  // auto dst_npu_base_format = FindBaseFormat(dst_npu_format);

  // Tensor trans_src_tensor(src_tensor.type());
  // trans_src_tensor.ShareDataWith(src_tensor);
  // if (src_base_format != dst_npu_base_format) {
  //   trans_src_tensor =
  //       FormatCastBetweenGroup(src_tensor, dst_tensor, trans_src_tensor);
  // }

  if (dst_format_name == "FRACTAL_NZ") {
    auto dims = framework::vectorize(dst_tensor.dims());
    std::vector<int64_t> dst_cast_npu_dims = InferShapeNDToNZ(dims);
    dst_tensor.ResizeNPUDims(framework::make_ddim(dst_cast_npu_dims));
    VLOG(4) << "Staring to cast NPU format from NCHW to FRACTAL_NZ.";

    // std::string src_format_name = "ND";
    auto stream = GetCurrentNPUStream();
    auto place = GetCurrentNPUPlace();

    size_t npu_storage_size = dst_tensor.npu_storage_numel() *
                              framework::SizeOfType(src_tensor.type());
    dst_tensor.mutable_data(place, src_tensor.type(), npu_storage_size);
    Tensor trans_src_tensor(src_tensor.type());
    trans_src_tensor.ShareDataWith(src_tensor);
    trans_src_tensor.ResizeNPUDims(src_tensor.dims());

    RunTransDataNPUOP(trans_src_tensor, &dst_tensor, stream);

    VLOG(4) << "Complete to cast NPU format from NCHW to FRACTAL_NZ.";

    return dst_tensor;
  } else {
    return src_tensor;
  }
}

void RunTransDataNPUOP(const Tensor &src_tensor, Tensor *dst_tensor,
                       aclrtStream stream) {
  std::string src_format_name =
      framework::DataLayoutToString(src_tensor.npu_storage_layout());
  std::string dst_format_name =
      framework::DataLayoutToString(dst_tensor->npu_storage_layout());
  const auto &runner_trans_data =
      NpuOpRunner("TransData", {src_tensor}, {*dst_tensor},
                  {{"src_format", src_format_name},
                   {"dst_format", dst_format_name},
                   {"groups", 1}});
  runner_trans_data.Run(stream);
  VLOG(4) << "Run TransData OP to cast NPU format from " << src_format_name
          << " to " << dst_format_name;
}

Tensor CastNPUFormat(const Tensor &src_tensor, int acl_format_id) {
  auto dtype = src_tensor.type();

  auto src_format = ConvertToNpuFormat(src_tensor.npu_storage_layout());

  aclFormat acl_format = static_cast<aclFormat>(acl_format_id);

  if (src_format == acl_format) {
    VLOG(4) << "The format of input tensor has met the requirements. There's "
               "no need to cast format.";
    return src_tensor;
  }

  PADDLE_ENFORCE_EQ(dtype == framework::proto::VarType::FP32 ||
                        dtype == framework::proto::VarType::FP16,
                    true, platform::errors::InvalidArgument(
                              "The data type of the Tensor that needs to cast "
                              "format must be float or float16, but got %s",
                              framework::DataTypeToString(dtype)));

  Tensor dst_tensor(dtype);
  dst_tensor.Resize(src_tensor.dims());
  dst_tensor.set_npu_storage_layout(ConvertNpuFormatToDataLayout(acl_format));
  // if (src_tensor->layout() != tmp_x.layout()) {
  //   auto runner_cast_x = NpuOpRunner(
  //       "TransData", {src_tensor}, {tmp_x},
  //       {{"src_format", framework::DataLayoutToString(x->layout())},
  //       {"dst_format", framework::DataLayoutToString(tmp_x.layout())},
  //       {"groups", 1}});
  //   runner_cast_x.Run(stream);
  // }

  dst_tensor = RunTransDataToCastFormat(src_tensor, dst_tensor);
  return dst_tensor;
}

Tensor GenerateNZTensor(const Tensor &src_tensor) {
  Tensor out_tensor(src_tensor.type());
  out_tensor.Resize(src_tensor.dims());
  out_tensor.ResizeNPUDims(framework::make_ddim(
      InferShapeNDToNZ(framework::vectorize(src_tensor.dims()))));
  out_tensor.set_npu_storage_layout(DataLayout::kFractalNZ);

  auto place = GetCurrentNPUPlace();
  size_t npu_storage_size =
      out_tensor.npu_storage_numel() * framework::SizeOfType(src_tensor.type());
  out_tensor.mutable_data(place, src_tensor.type(), npu_storage_size);

  return out_tensor;
}

void InferNPUStorageFormatAndDims(Tensor *dst, DataLayout layout) {
  dst->set_npu_storage_layout(layout);
  if (layout == DataLayout::kFractalNZ) {
    dst->ResizeNPUDims(framework::make_ddim(
        InferShapeNDToNZ(framework::vectorize(dst->dims()))));
  } else {
    dst->ResizeNPUDims(dst->dims());
  }
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
  VLOG(5) << "Free NpuOpRunner(" << this << ") of " << op_type_;
  // Is it safe to free the descs/buffers after run called in host ?
  aclopDestroyAttr(attr_);  // return void
  for (auto desc : input_descs_) {
    aclDestroyTensorDesc(desc);
  }
  for (auto desc : output_descs_) {
    aclDestroyTensorDesc(desc);
  }
  for (auto buffer : input_buffers_) {
    PADDLE_ENFORCE_NPU_SUCCESS(aclDestroyDataBuffer(buffer));
  }
  for (auto buffer : output_buffers_) {
    PADDLE_ENFORCE_NPU_SUCCESS(aclDestroyDataBuffer(buffer));
  }
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
    PADDLE_ENFORCE_NPU_SUCCESS(aclopSetAttrListListInt(
        attr_, name.c_str(), data.size(), num.data(), data.data()));
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
  input_descs_.reserve(tensors.size());
  input_buffers_.reserve(tensors.size());
  for (auto tensor : tensors) {
    // create aclTensorDesc
    input_descs_.emplace_back(CreateTensorDesc(tensor));
    // create aclDataBuffer
    input_buffers_.emplace_back(CreateDataBuffer(tensor));
  }
  return *this;
}

// NOTE(zhiqiu): For operators whose input is a list (such as concat, stack),
// It is needed to set the name of each input tensor.
NpuOpRunner &NpuOpRunner::AddInputNames(const std::vector<std::string> &names) {
  PADDLE_ENFORCE_EQ(names.size(), input_descs_.size(),
                    platform::errors::InvalidArgument(
                        "The size of input names should be "
                        "equal to the size of input descs, but got the size "
                        "of input names is %d, the size of input descs is %d.",
                        names.size(), input_descs_.size()));
  for (size_t i = 0; i < names.size(); ++i) {
    aclSetTensorDescName(input_descs_[i], names[i].c_str());
  }
  return *this;
}

NpuOpRunner &NpuOpRunner::AddOutputs(const std::vector<Tensor> &tensors) {
  output_descs_.reserve(tensors.size());
  output_buffers_.reserve(tensors.size());
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
  auto storage_format = ConvertToNpuFormat(tensor.npu_storage_layout());
  auto storage_dims = framework::vectorize(tensor.npu_storage_dims());

  VLOG(4) << "NPU dtype:" << dtype << " "
          << "rank:" << dims.size() << " dims:" << tensor.dims()
          << " format:" << format
          << " storage_dims:" << tensor.npu_storage_dims()
          << " storage_format:" << storage_format;

  aclTensorDesc *desc;
  if (op_type_ == "TransData") {
    VLOG(4) << "Create Tensor Desc for TransData NPU OP";
    desc = aclCreateTensorDesc(dtype, storage_dims.size(), storage_dims.data(),
                               storage_format);
  } else {
    desc = aclCreateTensorDesc(dtype, dims.size(), dims.data(), format);
  }
  PADDLE_ENFORCE_NOT_NULL(
      desc, platform::errors::External("Call aclCreateTensorDesc failed."));

  if (storage_format == ACL_FORMAT_FRACTAL_NZ) {
    VLOG(4) << "Set Tensor's storage format with NZ format";
    PADDLE_ENFORCE_NPU_SUCCESS(aclSetTensorStorageFormat(desc, storage_format));
    PADDLE_ENFORCE_NPU_SUCCESS(aclSetTensorStorageShape(
        desc, storage_dims.size(), storage_dims.data()));
  } else {
    VLOG(4) << "Set Tensor's storage format with NCHW format";
    PADDLE_ENFORCE_NPU_SUCCESS(aclSetTensorStorageFormat(desc, format));
    PADDLE_ENFORCE_NPU_SUCCESS(
        aclSetTensorStorageShape(desc, dims.size(), dims.data()));
  }
  /*
  if (op_type_ == "TransData" || op_type_ == "MatMul" ||
      op_type_ == "BatchMatMul") {
    VLOG(4) << "Set Storage format and shape for TransData, MatMul, "
               "BatchMatMul NPU OP";
    PADDLE_ENFORCE_NPU_SUCCESS(aclSetTensorStorageFormat(desc, storage_format));
    PADDLE_ENFORCE_NPU_SUCCESS(aclSetTensorStorageShape(
        desc, storage_dims.size(), storage_dims.data()));
  } else {
    PADDLE_ENFORCE_NPU_SUCCESS(aclSetTensorStorageFormat(desc, format));
    PADDLE_ENFORCE_NPU_SUCCESS(
        aclSetTensorStorageShape(desc, dims.size(), dims.data()));
  }*/
  return desc;
}

aclDataBuffer *NpuOpRunner::CreateDataBuffer(Tensor tensor) {
  void *ptr = tensor.data<void>();
  VLOG(4) << "NPU ptr: " << ptr << ", size: " << tensor.memory_size();
  auto *buffer = aclCreateDataBuffer(ptr, tensor.memory_size());
  PADDLE_ENFORCE_NOT_NULL(
      buffer, platform::errors::External("Call aclCreateDataBuffer failed."));
  return buffer;
}

void NpuOpRunner::Run(aclrtStream stream) const {
  if (!stream) {
    VLOG(4) << "Run with default current npu stream: " << stream;
    stream = GetCurrentNPUStream();
  }
  VLOG(5) << "NpuOpRunner(" << this << ") Run:";
  VLOG(4) << "op_type: " << op_type_;
  VLOG(4) << "input_desc.size: " << input_descs_.size();
  VLOG(4) << "output_desc.size: " << output_descs_.size();
  VLOG(4) << "attr: " << attr_;
  VLOG(4) << "stream: " << stream;

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
