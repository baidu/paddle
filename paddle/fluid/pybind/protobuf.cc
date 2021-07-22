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
#include "paddle/fluid/pybind/protobuf.h"

#include <deque>
#include <iostream>
#include <string>
#include <tuple>

#include "paddle/fluid/framework/block_desc.h"
#include "paddle/fluid/framework/op_desc.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/framework/var_desc.h"
#include "paddle/fluid/framework/version.h"

#include "paddle/fluid/pybind/pybind_boost_headers.h"

namespace paddle {
namespace pybind {

PyTypeObject *g_vartype_pytype = nullptr;
PyTypeObject *g_blockdesc_pytype = nullptr;

namespace pd = paddle::framework;

template <typename T>
static pybind11::bytes SerializeMessage(
    T &self) {  // NOLINT due to pybind11 convention.
  // Check IsInitialized in Python
  std::string retv;
  PADDLE_ENFORCE_EQ(self.Proto()->SerializePartialToString(&retv), true,
                    platform::errors::InvalidArgument(
                        "Failed to serialize input Desc to string."));
  return retv;
}

// Bind Methods
void BindProgramDesc(pybind11::module *m) {
  pybind11::class_<pd::ProgramDesc>(*m, "ProgramDesc", "")
      .def(pybind11::init<>())
      .def("__init__",
           [](pd::ProgramDesc &self, const pd::ProgramDesc &other) {
             new (&self) pd::ProgramDesc(other);
           })
      .def("__init__",
           [](pd::ProgramDesc &self, const pybind11::bytes &binary_str) {
             std::string str(binary_str);
             new (&self) pd::ProgramDesc(str);
           })
      .def("append_block", &pd::ProgramDesc::AppendBlock,
           pybind11::return_value_policy::reference)
      .def("block", &pd::ProgramDesc::MutableBlock,
           pybind11::return_value_policy::reference)
      .def("num_blocks", &pd::ProgramDesc::Size)
      .def("flush", &pd::ProgramDesc::Flush)
      .def("get_feed_target_names", &pd::ProgramDesc::GetFeedTargetNames)
      .def("get_fetch_target_names", &pd::ProgramDesc::GetFetchTargetNames)
      .def("serialize_to_string", SerializeMessage<pd::ProgramDesc>)
      .def("parse_from_string",
           [](pd::ProgramDesc &program_desc, const std::string &data) {
             pd::proto::ProgramDesc *desc = program_desc.Proto();
             PADDLE_ENFORCE_EQ(
                 desc->ParseFromString(data), true,
                 platform::errors::InvalidArgument(
                     "Failed to parse ProgramDesc from binary string."));
           })
      .def("_set_version",
           [](pd::ProgramDesc &self, int64_t version) {
             return self.SetVersion(version);
           },
           pybind11::arg("version") = pd::kCurProgramVersion)
      .def("_version",
           [](pd::ProgramDesc &self) -> int64_t { return self.Version(); });
}

void BindBlockDesc(pybind11::module *m) {
  pybind11::class_<pd::BlockDesc> blockdesc(*m, "BlockDesc", "");
  g_blockdesc_pytype = (PyTypeObject *)blockdesc.ptr();  // NOLINT
  blockdesc.def_property_readonly("id", &pd::BlockDesc::ID)
      .def_property_readonly("parent", &pd::BlockDesc::Parent)
      .def("get_forward_block_idx", &pd::BlockDesc::ForwardBlockID)
      .def("_set_forward_block_idx", &pd::BlockDesc::SetForwardBlockID)
      .def("append_op", &pd::BlockDesc::AppendOp,
           pybind11::return_value_policy::reference)
      .def("_prepend_op", &pd::BlockDesc::PrependOp,
           pybind11::return_value_policy::reference)
      .def("_insert_op", &pd::BlockDesc::InsertOp,
           pybind11::return_value_policy::reference)
      .def("_remove_op", &pd::BlockDesc::RemoveOp)
      .def("var",
           [](pd::BlockDesc &self, pybind11::bytes byte_name) {
             std::string name = byte_name;
             return self.Var(name);
           },
           pybind11::return_value_policy::reference)
      .def("has_var",
           [](pd::BlockDesc &self, pybind11::bytes byte_name) {
             std::string name = byte_name;
             return self.HasVar(name);
           },
           pybind11::return_value_policy::reference)
      .def("_rename_var",
           [](pd::BlockDesc &self, const pybind11::bytes &byte_name,
              const pybind11::bytes &byte_name_new) {
             std::string name = byte_name;
             std::string new_name = byte_name_new;
             self.RenameVar(name, new_name);
           })
      .def("has_var_recursive",
           [](pd::BlockDesc &self, pybind11::bytes byte_name) {
             std::string name = byte_name;
             return self.HasVarRecursive(name);
           })
      .def("find_var",
           [](pd::BlockDesc &self, pybind11::bytes byte_name) {
             std::string name = byte_name;
             return self.FindVar(name);
           },
           pybind11::return_value_policy::reference)
      .def("find_var_recursive",
           [](pd::BlockDesc &self, pybind11::bytes byte_name) {
             std::string name = byte_name;
             return self.FindVarRecursive(name);
           },
           pybind11::return_value_policy::reference)
      .def("_remove_var",
           [](pd::BlockDesc &self, pybind11::bytes byte_name) {
             std::string name = byte_name;
             return self.RemoveVar(name);
           },
           pybind11::return_value_policy::reference)
      .def("all_vars", &pd::BlockDesc::AllVars,
           pybind11::return_value_policy::reference)
      .def("op_size", &pd::BlockDesc::OpSize)
      .def("op", &pd::BlockDesc::Op, pybind11::return_value_policy::reference)
      .def("serialize_to_string", SerializeMessage<pd::BlockDesc>);
}

void BindVarDsec(pybind11::module *m) {
  pybind11::class_<pd::VarDesc> var_desc(*m, "VarDesc", "");
  var_desc.def(pybind11::init<const std::string &>())
      .def("name", &pd::VarDesc::Name, pybind11::return_value_policy::reference)
      .def("set_name", &pd::VarDesc::SetName)
      .def("set_shape", &pd::VarDesc::SetShape)
      .def("set_shapes", &pd::VarDesc::SetShapes)
      .def("get_shape", &pd::VarDesc::GetShape)
      .def("set_dtype", &pd::VarDesc::SetDataType)
      .def("set_dtypes", &pd::VarDesc::SetDataTypes)
      .def("shape", &pd::VarDesc::GetShape,
           pybind11::return_value_policy::reference)
      .def("shapes", &pd::VarDesc::GetShapes,
           pybind11::return_value_policy::reference)
      .def("dtype", &pd::VarDesc::GetDataType,
           pybind11::return_value_policy::reference)
      .def("dtypes", &pd::VarDesc::GetDataTypes,
           pybind11::return_value_policy::reference)
      .def("lod_level", &pd::VarDesc::GetLoDLevel)
      .def("lod_levels", &pd::VarDesc::GetLoDLevels,
           pybind11::return_value_policy::reference)
      .def("set_lod_level", &pd::VarDesc::SetLoDLevel)
      .def("set_lod_levels", &pd::VarDesc::SetLoDLevels)
      .def("type", &pd::VarDesc::GetType)
      .def("set_type", &pd::VarDesc::SetType)
      .def("serialize_to_string", SerializeMessage<pd::VarDesc>)
      .def("persistable", &pd::VarDesc::Persistable)
      .def("set_persistable", &pd::VarDesc::SetPersistable)
      .def("is_parameter", &pd::VarDesc::IsParameter)
      .def("set_is_parameter", &pd::VarDesc::SetIsParameter)
      .def("clear_is_parameter", &pd::VarDesc::ClearIsParameter)
      .def("has_is_parameter", &pd::VarDesc::HasIsParameter)
      .def("stop_gradient", &pd::VarDesc::StopGradient)
      .def("set_stop_gradient", &pd::VarDesc::SetStopGradient)
      .def("clear_stop_gradient", &pd::VarDesc::ClearStopGradient)
      .def("has_stop_gradient", &pd::VarDesc::HasStopGradient)
      .def("need_check_feed", &pd::VarDesc::NeedCheckFeed)
      .def("set_need_check_feed", &pd::VarDesc::SetNeedCheckFeed);

  pybind11::enum_<pd::proto::VarType::Type> vartype(var_desc, "VarType", "");
  g_vartype_pytype = (PyTypeObject *)vartype.ptr();  // NOLINT
  vartype.value("BOOL", pd::proto::VarType::BOOL)
      .value("UINT8", pd::proto::VarType::UINT8)
      .value("INT8", pd::proto::VarType::INT8)
      .value("INT16", pd::proto::VarType::INT16)
      .value("INT32", pd::proto::VarType::INT32)
      .value("INT64", pd::proto::VarType::INT64)
      .value("FP16", pd::proto::VarType::FP16)
      .value("FP32", pd::proto::VarType::FP32)
      .value("FP64", pd::proto::VarType::FP64)
      .value("BF16", pd::proto::VarType::BF16)
      .value("COMPLEX64", pd::proto::VarType::COMPLEX64)
      .value("COMPLEX128", pd::proto::VarType::COMPLEX128)
      .value("LOD_TENSOR", pd::proto::VarType::LOD_TENSOR)
      .value("SELECTED_ROWS", pd::proto::VarType::SELECTED_ROWS)
      .value("FEED_MINIBATCH", pd::proto::VarType::FEED_MINIBATCH)
      .value("FETCH_LIST", pd::proto::VarType::FETCH_LIST)
      .value("STEP_SCOPES", pd::proto::VarType::STEP_SCOPES)
      .value("LOD_RANK_TABLE", pd::proto::VarType::LOD_RANK_TABLE)
      .value("LOD_TENSOR_ARRAY", pd::proto::VarType::LOD_TENSOR_ARRAY)
      .value("PLACE_LIST", pd::proto::VarType::PLACE_LIST)
      .value("READER", pd::proto::VarType::READER)
      .value("RAW", pd::proto::VarType::RAW);
}

void BindOpDesc(pybind11::module *m) {
  pybind11::enum_<pd::proto::AttrType>(*m, "AttrType", "")
      .value("INT", pd::proto::AttrType::INT)
      .value("INTS", pd::proto::AttrType::INTS)
      .value("LONG", pd::proto::AttrType::LONG)
      .value("LONGS", pd::proto::AttrType::LONGS)
      .value("FLOAT", pd::proto::AttrType::FLOAT)
      .value("FLOATS", pd::proto::AttrType::FLOATS)
      .value("STRING", pd::proto::AttrType::STRING)
      .value("STRINGS", pd::proto::AttrType::STRINGS)
      .value("BOOL", pd::proto::AttrType::BOOLEAN)
      .value("BOOLS", pd::proto::AttrType::BOOLEANS)
      .value("BLOCK", pd::proto::AttrType::BLOCK)
      .value("BLOCKS", pd::proto::AttrType::BLOCKS);

  pybind11::class_<pd::OpDesc> op_desc(*m, "OpDesc", "");
  op_desc
      .def("__init__", [](pd::OpDesc &self) { new (&self) pd::OpDesc(); },
           pybind11::return_value_policy::reference)
      .def("copy_from", &pd::OpDesc::CopyFrom)
      .def("type", &pd::OpDesc::Type)
      .def("set_type", &pd::OpDesc::SetType)
      .def("input", &pd::OpDesc::Input)
      .def("input_names", &pd::OpDesc::InputNames)
      .def("output", &pd::OpDesc::Output)
      .def("output_names", &pd::OpDesc::OutputNames)
      .def("set_input",
           [](pd::OpDesc &self, const std::string &name,
              const std::vector<std::string> &vec_var_name) {
             self.SetInput(name, vec_var_name);
           })
      .def("set_output",
           [](pd::OpDesc &self, const std::string &name,
              const std::vector<std::string> &vec_var_name) {
             self.SetOutput(name, vec_var_name);
           })
      .def("remove_output", &pd::OpDesc::RemoveOutput)
      .def("input_arg_names", &pd::OpDesc::InputArgumentNames)
      .def("output_arg_names", &pd::OpDesc::OutputArgumentNames)
      .def("_rename_input", &pd::OpDesc::RenameInput)
      .def("_rename_output", &pd::OpDesc::RenameOutput)
      .def("has_attr", &pd::OpDesc::HasAttr)
      .def("attr_type", &pd::OpDesc::GetAttrType)
      .def("attr_names", &pd::OpDesc::AttrNames)
      .def("_set_attr", &pd::OpDesc::SetAttr)
      .def("remove_attr", &pd::OpDesc::RemoveAttr)
      .def("attr", &pd::OpDesc::GetAttr)
      .def("set_block_attr", &pd::OpDesc::SetBlockAttr)
      .def("set_blocks_attr", &pd::OpDesc::SetBlocksAttr)
      .def("set_serialized_attr",
           [](pd::OpDesc &self, const std::string &name,
              const pybind11::bytes &seriralized) {
             std::string ser(seriralized);
             self.SetAttr(name, ser);
           })
      .def("_block_attr_id", &pd::OpDesc::GetBlockAttrId)
      .def("_blocks_attr_ids", &pd::OpDesc::GetBlocksAttrIds)
      .def("check_attrs", &pd::OpDesc::CheckAttrs)
      .def("infer_shape", &pd::OpDesc::InferShape)
      .def("infer_var_type", &pd::OpDesc::InferVarType)
      .def("set_is_target", &pd::OpDesc::SetIsTarget)
      .def("serialize_to_string", SerializeMessage<pd::OpDesc>)
      .def("block", [](pd::OpDesc &self) { return self.Block(); },
           pybind11::return_value_policy::reference)
      .def("inputs", &pd::OpDesc::Inputs)
      .def("outputs", &pd::OpDesc::Outputs);
}

}  // namespace pybind
}  // namespace paddle
