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

#include "paddle/fluid/framework/var_desc.h"

#include "glog/logging.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace framework {

proto::VarType::Type VarDesc::GetType() const { return desc_.type().type(); }

void VarDesc::SetType(proto::VarType::Type type) {
  desc_.mutable_type()->set_type(type);
}

void VarDesc::SetShape(const std::vector<int64_t> &dims) {
  VectorToRepeated(dims, mutable_tensor_desc()->mutable_dims());
}

void VarDesc::SetTensorDescNum(size_t num) {
  switch (desc_.type().type()) {
    case proto::VarType::READER: {
      auto *lod_tensors_ptr =
          desc_.mutable_type()->mutable_reader()->mutable_lod_tensor();
      lod_tensors_ptr->Clear();
      for (size_t i = 0; i < num; ++i) {
        lod_tensors_ptr->Add();
      }
      return;
    } break;
    default:
      PADDLE_THROW(
          platform::errors::Unavailable("Setting 'sub_tensor_number' is not "
                                        "supported by the %s type variable.",
                                        this->Name()));
  }
}

size_t VarDesc::GetTensorDescNum() const {
  switch (desc_.type().type()) {
    case proto::VarType::READER:
      return desc_.type().reader().lod_tensor_size();
      break;
    default:
      PADDLE_THROW(
          platform::errors::Unavailable("Getting 'sub_tensor_number' is not "
                                        "supported by the %s type variable.",
                                        this->Name()));
  }
}

void VarDesc::SetShapes(
    const std::vector<std::vector<int64_t>> &multiple_dims) {
  if (multiple_dims.size() != GetTensorDescNum()) {
    VLOG(3) << "WARNING: The number of given shapes(" << multiple_dims.size()
            << ") doesn't match the existing tensor number("
            << GetTensorDescNum()
            << "). The Reader is going to be reinitialized.";
    SetTensorDescNum(multiple_dims.size());
  }
  std::vector<proto::VarType::TensorDesc *> tensors = mutable_tensor_descs();
  for (size_t i = 0; i < multiple_dims.size(); ++i) {
    VectorToRepeated(multiple_dims[i], tensors[i]->mutable_dims());
  }
}

std::vector<int64_t> VarDesc::GetShape() const {
  return RepeatedToVector(tensor_desc().dims());
}

std::vector<std::vector<int64_t>> VarDesc::GetShapes() const {
  std::vector<proto::VarType::TensorDesc> descs = tensor_descs();
  std::vector<std::vector<int64_t>> res;
  res.reserve(descs.size());
  for (const auto &tensor_desc : descs) {
    res.push_back(RepeatedToVector(tensor_desc.dims()));
  }
  return res;
}

void VarDesc::SetDataType(proto::VarType::Type data_type) {
  mutable_tensor_desc()->set_data_type(data_type);
}

void VarDesc::SetDataTypes(
    const std::vector<proto::VarType::Type> &multiple_data_type) {
  if (multiple_data_type.size() != GetTensorDescNum()) {
    VLOG(3) << "WARNING: The number of given data types("
            << multiple_data_type.size()
            << ") doesn't match the existing tensor number("
            << GetTensorDescNum()
            << "). The Reader is going to be reinitialized.";
    SetTensorDescNum(multiple_data_type.size());
  }
  std::vector<proto::VarType::TensorDesc *> tensor_descs =
      mutable_tensor_descs();
  for (size_t i = 0; i < multiple_data_type.size(); ++i) {
    tensor_descs[i]->set_data_type(multiple_data_type[i]);
  }
}

proto::VarType::Type VarDesc::GetDataType() const {
  return tensor_desc().data_type();
}

std::vector<proto::VarType::Type> VarDesc::GetDataTypes() const {
  std::vector<proto::VarType::TensorDesc> descs = tensor_descs();
  std::vector<proto::VarType::Type> res;
  res.reserve(descs.size());
  for (const auto &tensor_desc : descs) {
    res.push_back(tensor_desc.data_type());
  }
  return res;
}

void VarDesc::SetLoDLevel(int32_t lod_level) {
  switch (desc_.type().type()) {
    case proto::VarType::LOD_TENSOR:
      desc_.mutable_type()->mutable_lod_tensor()->set_lod_level(lod_level);
      break;
    case proto::VarType::LOD_TENSOR_ARRAY:
      desc_.mutable_type()->mutable_tensor_array()->set_lod_level(lod_level);
      break;
    default:
      PADDLE_THROW(platform::errors::Unavailable(
          "Setting 'lod_level' is not supported by the %s type variable.",
          this->Name()));
  }
}

void VarDesc::SetLoDLevels(const std::vector<int32_t> &multiple_lod_level) {
  if (multiple_lod_level.size() != GetTensorDescNum()) {
    VLOG(3) << "WARNING: The number of given lod_levels("
            << multiple_lod_level.size()
            << ") doesn't match the existing tensor number("
            << GetTensorDescNum()
            << "). The Reader is going to be reinitialized.";
    SetTensorDescNum(multiple_lod_level.size());
  }
  switch (desc_.type().type()) {
    case proto::VarType::READER: {
      size_t i = 0;
      for (auto &lod_tensor :
           *desc_.mutable_type()->mutable_reader()->mutable_lod_tensor()) {
        lod_tensor.set_lod_level(multiple_lod_level[i++]);
      }
    } break;
    default:
      PADDLE_THROW(platform::errors::Unavailable(
          "Setting 'lod_levels' is not supported by the %s type variable",
          this->Name()));
  }
}

int32_t VarDesc::GetLoDLevel() const {
  switch (desc_.type().type()) {
    case proto::VarType::LOD_TENSOR:
      return desc_.type().lod_tensor().lod_level();
    case proto::VarType::LOD_TENSOR_ARRAY:
      return desc_.type().tensor_array().lod_level();
    default:
      PADDLE_THROW(platform::errors::Unavailable(
          "Getting 'lod_level' is not supported by the %s type variable.",
          this->Name()));
  }
}

std::vector<int32_t> VarDesc::GetLoDLevels() const {
  std::vector<int32_t> res;
  switch (desc_.type().type()) {
    case proto::VarType::READER:
      res.reserve(desc_.type().reader().lod_tensor_size());
      for (auto &lod_tensor : desc_.type().reader().lod_tensor()) {
        res.push_back(lod_tensor.lod_level());
      }
      return res;
      break;
    default:
      PADDLE_THROW(platform::errors::Unavailable(
          "Getting 'lod_levels' is not supported by the %s type variable.",
          this->Name()));
  }
}

const proto::VarType::TensorDesc &VarDesc::tensor_desc() const {
  PADDLE_ENFORCE_EQ(
      desc_.has_type(), true,
      platform::errors::NotFound("The variable's type was not be set."));
  PADDLE_ENFORCE_EQ(
      desc_.type().has_type(), true,
      platform::errors::NotFound("The variable's type was not be set."));
  switch (desc_.type().type()) {
    case proto::VarType::SELECTED_ROWS:
      return desc_.type().selected_rows();
    case proto::VarType::LOD_TENSOR:
      return desc_.type().lod_tensor().tensor();
    case proto::VarType::LOD_TENSOR_ARRAY:
      return desc_.type().tensor_array().tensor();
    default:
      PADDLE_THROW(platform::errors::Unavailable(
          "Getting 'tensor_desc' is not supported by the %s type variable.",
          this->Name()));
  }
}

std::vector<proto::VarType::TensorDesc> VarDesc::tensor_descs() const {
  PADDLE_ENFORCE_EQ(
      desc_.has_type(), true,
      platform::errors::NotFound("The variable's type was not be set."));
  std::vector<proto::VarType::TensorDesc> res;
  res.reserve(GetTensorDescNum());
  switch (desc_.type().type()) {
    case proto::VarType::READER:
      for (const auto &lod_tensor : desc_.type().reader().lod_tensor()) {
        res.push_back(lod_tensor.tensor());
      }
      return res;
    default:
      PADDLE_THROW(platform::errors::Unavailable(
          "Getting 'tensor_descs' is not supported by the %s type variable.",
          this->Name()));
  }
}

proto::VarType::TensorDesc *VarDesc::mutable_tensor_desc() {
  PADDLE_ENFORCE_EQ(
      desc_.has_type(), true,
      platform::errors::NotFound("The variable's type was not be set."));
  PADDLE_ENFORCE_EQ(
      desc_.type().has_type(), true,
      platform::errors::NotFound("The variable's type was not be set."));
  switch (desc_.type().type()) {
    case proto::VarType::SELECTED_ROWS:
      return desc_.mutable_type()->mutable_selected_rows();
    case proto::VarType::LOD_TENSOR:
      return desc_.mutable_type()->mutable_lod_tensor()->mutable_tensor();
    case proto::VarType::LOD_TENSOR_ARRAY:
      return desc_.mutable_type()->mutable_tensor_array()->mutable_tensor();
    default:
      PADDLE_THROW(
          platform::errors::Unavailable("Getting 'mutable_tensor_desc' is not "
                                        "supported by the %s type variable.",
                                        this->Name()));
  }
}

std::vector<proto::VarType::TensorDesc *> VarDesc::mutable_tensor_descs() {
  PADDLE_ENFORCE_EQ(
      desc_.has_type(), true,
      platform::errors::NotFound("The variable's type was not be set."));
  PADDLE_ENFORCE_EQ(
      desc_.type().has_type(), true,
      platform::errors::NotFound("The variable's type was not be set."));
  std::vector<proto::VarType::TensorDesc *> res;
  res.reserve(GetTensorDescNum());
  switch (desc_.type().type()) {
    case proto::VarType::READER:
      for (auto &lod_tensor :
           *desc_.mutable_type()->mutable_reader()->mutable_lod_tensor()) {
        res.push_back(lod_tensor.mutable_tensor());
      }
      return res;
    default:
      PADDLE_THROW(platform::errors::Unavailable(
          "Getting 'tensor_descs' is not supported by the %s type variable.",
          this->Name()));
  }
}

std::vector<std::string> VarDesc::DistributedAttrNames() const {
  std::vector<std::string> retv;
  retv.reserve(distributed_attrs_.size());
  for (auto &attr : distributed_attrs_) {
    retv.push_back(attr.first);
  }
  return retv;
}

void VarDesc::RemoveDistributedAttr(const std::string &name) {
  distributed_attrs_.erase(name);
  need_update_ = true;
}

void VarDesc::SetDistributedAttr(const std::string &name, const Attribute &v) {
  // NOTICE(sandyhouse): pybind11 will take the empty list in python as
  // the std::vector<int> type in C++; so we have to change the attr's type
  // here if we meet this issue
  proto::AttrType attr_type = static_cast<proto::AttrType>(v.which() - 1);
  if (attr_type == proto::AttrType::INTS &&
      BOOST_GET_CONST(std::vector<int>, v).size() == 0u) {
    // Find current attr via attr name and set the correct attribute value
    this->distributed_attrs_[name] = std::vector<int>();
    need_update_ = true;
    return;
  }

  this->distributed_attrs_[name] = v;
  need_update_ = true;
}

Attribute VarDesc::GetDistributedAttr(const std::string &name) const {
  auto it = distributed_attrs_.find(name);
  PADDLE_ENFORCE_NE(
      it, distributed_attrs_.end(),
      platform::errors::NotFound("Attribute %s is not found.", name));
  return it->second;
}

bool operator==(const VarDesc &left, const VarDesc &right) {
  return left.Proto()->SerializeAsString() ==
         right.Proto()->SerializeAsString();
}

}  // namespace framework
}  // namespace paddle
