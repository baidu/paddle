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

#ifdef PADDLE_WITH_ASCEND
#include <fcntl.h>

#ifdef _POSIX_C_SOURCE
#undef _POSIX_C_SOURCE
#endif

#ifdef _XOPEN_SOURCE
#undef _XOPEN_SOURCE
#endif

#include <ge/ge_api.h>
#include <graph/attr_value.h>
#include <graph/operator_factory.h>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "paddle/fluid/framework/fleet/ascend_wrapper.h"
#include "paddle/fluid/pybind/ascend_wrapper_py.h"

using namespace ge;  // NOLINT
namespace py = pybind11;

namespace paddle {
namespace pybind {

void BindAscendWrapper(py::module *m) {
  py::class_<framework::AscendInstance,
             std::shared_ptr<framework::AscendInstance>>(*m, "AscendInstance")
      .def(py::init([]() { return framework::AscendInstance::GetInstance(); }))
      .def("init_global_resources",
           &framework::AscendInstance::InitGlobalResouces,
           py::call_guard<py::gil_scoped_release>())
      .def("add_ascend_subgraph", &framework::AscendInstance::AddAscendSubgraph,
           py::call_guard<py::gil_scoped_release>());
}  // end AscendWrapper

Status ge_initialize(std::map<std::string, std::string> &options) {  // NOLINT
  py::gil_scoped_release release;
  Status res = GEInitialize(options);
  py::gil_scoped_acquire acquire;
  return res;
}

enum AttrType {
  AT_INT64 = 0,
  AT_INT32,
  AT_UINT32,
  AT_LIST_INT64,
  AT_LIST_INT32,
  AT_LIST_UINT32,
  AT_FLOAT,
  AT_LIST_FLOAT,
  AT_ATTR_VALUE,
  AT_STRING,
  AT_LIST_STRING,
  AT_BOOL,
  AT_LIST_BOOL,
  AT_TENSOR,
  AT_LIST_TENSOR,
  AT_LIST_UINT8,
  AT_LIST_LIST_INT64,
  AT_LIST_DT,
  AT_DT,
  AT_LIST_NAMEATTR,
  AT_NAMEATTR
};

void BindAscendGraph(py::module *m) {
  m->def("ge_initialize", &ge_initialize, "GEInitialize");
  m->def("ge_finalize", &GEFinalize, "GEFinalize");

  //枚举封装
  py::enum_<GraphRunMode>(*m, "GEGraphRunMode")
      .value("PREDICTION", GraphRunMode::PREDICTION)
      .value("TRAIN", GraphRunMode::TRAIN)
      .export_values();

  py::enum_<DataType>(*m, "GEDataType")
      .value("DT_FLOAT", DataType::DT_FLOAT)
      .value("DT_FLOAT16", DataType::DT_FLOAT16)
      .value("DT_INT8", DataType::DT_INT8)
      .value("DT_INT16", DataType::DT_INT16)
      .value("DT_UINT16", DataType::DT_UINT16)
      .value("DT_UINT8", DataType::DT_UINT8)
      .value("DT_INT32", DataType::DT_INT32)
      .value("DT_INT64", DataType::DT_INT64)
      .value("DT_UINT32", DataType::DT_UINT32)
      .value("DT_UINT64", DataType::DT_UINT64)
      .value("DT_BOOL", DataType::DT_BOOL)
      .value("DT_DOUBLE", DataType::DT_DOUBLE)
      .value("DT_STRING", DataType::DT_STRING)
      .value("DT_DUAL_SUB_INT8", DataType::DT_DUAL_SUB_INT8)
      .value("DT_DUAL_SUB_UINT8", DataType::DT_DUAL_SUB_UINT8)
      .value("DT_COMPLEX64", DataType::DT_COMPLEX64)
      .value("DT_COMPLEX128", DataType::DT_COMPLEX128)
      .value("DT_QINT8", DataType::DT_QINT8)
      .value("DT_QINT16", DataType::DT_QINT16)
      .value("DT_QINT32", DataType::DT_QINT32)
      .value("DT_QUINT8", DataType::DT_QUINT8)
      .value("DT_QUINT16", DataType::DT_QUINT16)
      .value("DT_RESOURCE", DataType::DT_RESOURCE)
      .value("DT_STRING_REF", DataType::DT_STRING_REF)
      .value("DT_DUAL", DataType::DT_DUAL)
      .value("DT_UNDEFINED", DataType::DT_UNDEFINED)
      .export_values();

  py::enum_<Format>(*m, "GEFormat")
      .value("FORMAT_NCHW", Format::FORMAT_NCHW)
      .value("FORMAT_NHWC", Format::FORMAT_NHWC)
      .value("FORMAT_ND", Format::FORMAT_ND)
      .value("FORMAT_NC1HWC0", Format::FORMAT_NC1HWC0)
      .value("FORMAT_FRACTAL_Z", Format::FORMAT_FRACTAL_Z)
      .value("FORMAT_NC1C0HWPAD", Format::FORMAT_NC1C0HWPAD)
      .value("FORMAT_NHWC1C0", Format::FORMAT_NHWC1C0)
      .value("FORMAT_FSR_NCHW", Format::FORMAT_FSR_NCHW)
      .value("FORMAT_FRACTAL_DECONV", Format::FORMAT_FRACTAL_DECONV)
      .value("FORMAT_C1HWNC0", Format::FORMAT_C1HWNC0)
      .value("FORMAT_FRACTAL_DECONV_TRANSPOSE",
             Format::FORMAT_FRACTAL_DECONV_TRANSPOSE)
      .value("FORMAT_FRACTAL_DECONV_SP_STRIDE_TRANS",
             Format::FORMAT_FRACTAL_DECONV_SP_STRIDE_TRANS)
      .value("FORMAT_NC1HWC0_C04", Format::FORMAT_NC1HWC0_C04)
      .value("FORMAT_FRACTAL_Z_C04", Format::FORMAT_FRACTAL_Z_C04)
      .value("FORMAT_CHWN", Format::FORMAT_CHWN)
      .value("FORMAT_FRACTAL_DECONV_SP_STRIDE8_TRANS",
             Format::FORMAT_FRACTAL_DECONV_SP_STRIDE8_TRANS)
      .value("FORMAT_HWCN", Format::FORMAT_HWCN)
      .value("FORMAT_NC1KHKWHWC0", Format::FORMAT_NC1KHKWHWC0)
      .value("FORMAT_BN_WEIGHT", Format::FORMAT_BN_WEIGHT)
      .value("FORMAT_FILTER_HWCK", Format::FORMAT_FILTER_HWCK)
      .value("FORMAT_HASHTABLE_LOOKUP_LOOKUPS",
             Format::FORMAT_HASHTABLE_LOOKUP_LOOKUPS)
      .value("FORMAT_HASHTABLE_LOOKUP_KEYS",
             Format::FORMAT_HASHTABLE_LOOKUP_KEYS)
      .value("FORMAT_HASHTABLE_LOOKUP_VALUE",
             Format::FORMAT_HASHTABLE_LOOKUP_VALUE)
      .value("FORMAT_HASHTABLE_LOOKUP_OUTPUT",
             Format::FORMAT_HASHTABLE_LOOKUP_OUTPUT)
      .value("FORMAT_HASHTABLE_LOOKUP_HITS",
             Format::FORMAT_HASHTABLE_LOOKUP_HITS)
      .value("FORMAT_C1HWNCoC0", Format::FORMAT_C1HWNCoC0)
      .value("FORMAT_MD", Format::FORMAT_MD)
      .value("FORMAT_NDHWC", Format::FORMAT_NDHWC)
      .value("FORMAT_FRACTAL_ZZ", Format::FORMAT_FRACTAL_ZZ)
      .value("FORMAT_FRACTAL_NZ", Format::FORMAT_FRACTAL_NZ)
      .value("FORMAT_NCDHW", Format::FORMAT_NCDHW)
      .value("FORMAT_DHWCN", Format::FORMAT_DHWCN)
      .value("FORMAT_NDC1HWC0", Format::FORMAT_NDC1HWC0)
      .value("FORMAT_FRACTAL_Z_3D", Format::FORMAT_FRACTAL_Z_3D)
      .value("FORMAT_CN", Format::FORMAT_CN)
      .value("FORMAT_NC", Format::FORMAT_NC)
      .value("FORMAT_DHWNC", Format::FORMAT_DHWNC)
      .value("FORMAT_FRACTAL_Z_3D_TRANSPOSE",
             Format::FORMAT_FRACTAL_Z_3D_TRANSPOSE)
      .value("FORMAT_FRACTAL_ZN_LSTM", Format::FORMAT_FRACTAL_ZN_LSTM)
      .value("FORMAT_FRACTAL_Z_G", Format::FORMAT_FRACTAL_Z_G)
      .value("FORMAT_RESERVED", Format::FORMAT_RESERVED)
      .value("FORMAT_ALL", Format::FORMAT_ALL)
      .value("FORMAT_NULL", Format::FORMAT_NULL)
      .export_values();

  py::enum_<UnknowShapeOpType>(*m, "GEUnknowShapeOpType")
      .value("DEPEND_IN_SHAPE", UnknowShapeOpType::DEPEND_IN_SHAPE)
      .value("DEPEND_CONST_VALUE", UnknowShapeOpType::DEPEND_CONST_VALUE)
      .value("DEPEND_SHAPE_RANGE", UnknowShapeOpType::DEPEND_SHAPE_RANGE)
      .value("DEPEND_COMPUTE", UnknowShapeOpType::DEPEND_COMPUTE)
      .export_values();

  py::enum_<DeviceType>(*m, "GEDeviceType")
      .value("NPU", DeviceType::NPU)
      .value("CPU", DeviceType::CPU)
      .export_values();

  py::enum_<AttrType>(*m, "GEAttrType")
      .value("AT_INT64", AttrType::AT_INT64)
      .value("AT_INT32", AttrType::AT_INT32)
      .value("AT_UINT32", AttrType::AT_UINT32)
      .value("AT_LIST_INT64", AttrType::AT_LIST_INT64)
      .value("AT_LIST_INT32", AttrType::AT_LIST_INT32)
      .value("AT_LIST_UINT32", AttrType::AT_LIST_UINT32)
      .value("AT_FLOAT", AttrType::AT_FLOAT)
      .value("AT_LIST_FLOAT", AttrType::AT_LIST_FLOAT)
      .value("AT_ATTR_VALUE", AttrType::AT_ATTR_VALUE)
      .value("AT_STRING", AttrType::AT_STRING)
      .value("AT_LIST_STRING", AttrType::AT_LIST_STRING)
      .value("AT_BOOL", AttrType::AT_BOOL)
      .value("AT_LIST_BOOL", AttrType::AT_LIST_BOOL)
      .value("AT_TENSOR", AttrType::AT_TENSOR)
      .value("AT_LIST_TENSOR", AttrType::AT_LIST_TENSOR)
      .value("AT_LIST_UINT8", AttrType::AT_LIST_UINT8)
      .value("AT_LIST_LIST_INT64", AttrType::AT_LIST_LIST_INT64)
      .value("AT_LIST_DT", AttrType::AT_LIST_DT)
      .value("AT_DT", AttrType::AT_DT)
      .value("AT_LIST_NAMEATTR", AttrType::AT_LIST_NAMEATTR)
      .value("AT_NAMEATTR", AttrType::AT_NAMEATTR)
      .export_values();

  // 类封装
  py::class_<Session>(*m, "GESession")
      .def(py::init<const std::map<std::string, std::string> &>())
      .def("add_graph",
           (Status (Session::*)(uint32_t, const Graph &)) & Session::AddGraph)
      .def("add_graph",
           (Status (Session::*)(uint32_t, const Graph &,
                                const std::map<std::string, std::string> &)) &
               Session::AddGraph)
      .def("remove_graph", &Session::RemoveGraph)
      .def("run_graph",
           [](Session &ss, uint32_t graphId,
              const std::vector<Tensor> &inputs) -> py::tuple {
             std::vector<Tensor> outputs;
             Status res = ss.RunGraph(graphId, inputs, outputs);
             return py::make_tuple(outputs, res);
           },
           py::call_guard<py::gil_scoped_release>())
      .def("build_graph", &Session::BuildGraph)
      .def("run_graph_async", &Session::RunGraphAsync)
      .def("register_call_back_func",
           (Status (Session::*)(  // NOLINT
               const std::string &,
               std::function<uint32_t(
                   uint32_t graph_id,
                   const std::map<std::string, ge::Tensor> &params_list)>)) &
               Session::RegisterCallBackFunc)
      .def("is_graph_need_rebuild", &Session::IsGraphNeedRebuild);

  py::class_<Graph>(*m, "GEGraph")
      .def(py::init<>())
      .def(py::init<const std::string &>())
      .def("set_inputs", &Graph::SetInputs)
      .def("set_outputs", (Graph & (Graph::*)(const std::vector<Operator> &)) &
                              Graph::SetOutputs)
      .def("set_outputs",
           (Graph & (Graph::*)(const std::vector<
                               std::pair<Operator, std::vector<size_t>>> &)) &
               Graph::SetOutputs)
      .def("set_outputs",
           (Graph &
            (Graph::*)(const std::vector<std::pair<ge::Operator, std::string>>
                           &)) &
               Graph::SetOutputs)
      .def("set_targets", &Graph::SetTargets)
      .def("is_valid", &Graph::IsValid)
      .def("add_op", &Graph::AddOp)
      .def("find_op_by_name",
           [](Graph &graph, const std::string &name) -> py::tuple {
             ge::Operator op;
             graphStatus status = graph.FindOpByName(name, op);
             return py::make_tuple(op, status);
           })
      .def("find_op_by_type",
           [](Graph &graph, const std::string &type) -> py::tuple {
             std::vector<ge::Operator> ops;
             graphStatus status = graph.FindOpByType(type, ops);
             return py::make_tuple(ops, status);
           })
      .def("get_all_op_name",
           [](Graph &graph) -> py::tuple {
             std::vector<std::string> op_name;
             graphStatus status = graph.GetAllOpName(op_name);
             return py::make_tuple(op_name, status);
           })
      .def("save_to_file", &Graph::SaveToFile)
      .def("load_from_file", &Graph::LoadFromFile)
      .def("get_name", &Graph::GetName)
      .def("set_need_iteration", &Graph::SetNeedIteration);

  py::class_<Operator>(*m, "GEOperator")
      .def(py::init<>())
      .def(py::init<const std::string &>())
      .def(py::init<const std::string &, const std::string &>())
      .def("is_empty", &Operator::IsEmpty)
      .def("get_name", &Operator::GetName)
      .def("get_op_type", &Operator::GetOpType)
      .def("set_input",
           (Operator & (Operator::*)(const std::string &, const Operator &)) &
               Operator::SetInput)
      .def("set_input",
           (Operator & (Operator::*)(const std::string &, const Operator &,
                                     const std::string &)) &
               Operator::SetInput)
      .def("set_input", (Operator & (Operator::*)(const std::string &,
                                                  const Operator &, uint32_t)) &
                            Operator::SetInput)
      .def("add_control_input", &Operator::AddControlInput)
      .def("get_input_const_data",
           [](Operator &op, const std::string &dst_name) -> py::tuple {
             Tensor data;
             graphStatus res = op.GetInputConstData(dst_name, data);
             return py::make_tuple(data, res);
           })
      .def("get_input_desc",
           (TensorDesc (Operator::*)(const std::string &) const) &
               Operator::GetInputDesc)
      .def("get_input_desc",
           (TensorDesc (Operator::*)(uint32_t) const) & Operator::GetInputDesc)
      .def("get_dynamic_output_num", &Operator::GetDynamicOutputNum)
      .def("get_dynamic_input_num", &Operator::GetDynamicInputNum)
      .def("try_get_input_desc",
           [](Operator &op, const std::string &name) -> py::tuple {
             TensorDesc tensor_desc;
             graphStatus status = op.TryGetInputDesc(name, tensor_desc);
             return py::make_tuple(tensor_desc, status);
           })
      .def("update_input_desc", &Operator::UpdateInputDesc)
      .def("get_output_desc",
           (TensorDesc (Operator::*)(const std::string &) const) &
               Operator::GetOutputDesc)
      .def("get_output_desc",
           (TensorDesc (Operator::*)(uint32_t) const) & Operator::GetOutputDesc)
      .def("update_output_desc", &Operator::UpdateOutputDesc)
      .def("get_dynamic_input_desc", &Operator::GetDynamicInputDesc)
      .def("update_dynamic_input_desc", &Operator::UpdateDynamicInputDesc)
      .def("get_dynamic_output_desc", &Operator::GetDynamicOutputDesc)
      .def("update_dynamic_output_desc", &Operator::UpdateDynamicOutputDesc)
      .def("infer_shape_and_type", &Operator::InferShapeAndType)
      .def("set_inference_context", &Operator::SetInferenceContext)
      .def("get_inference_context", &Operator::GetInferenceContext)
      .def("verify_all_attr", &Operator::VerifyAllAttr)
      .def("get_inputs_size", &Operator::GetInputsSize)
      .def("get_outputs_size", &Operator::GetOutputsSize)
      .def("get_all_attr_names_and_types", &Operator::GetAllAttrNamesAndTypes)
      .def("set_attr_int64",
           [](Operator &op, const std::string &name,
              int64_t value) -> Operator & {
             int64_t tar = (int64_t)value;
             return op.SetAttr(name, tar);
           })
      .def("set_attr_int32",
           [](Operator &op, const std::string &name,
              int32_t value) -> Operator & {
             int32_t tar = (int32_t)value;
             return op.SetAttr(name, tar);
           })
      .def("set_attr_uint32",
           [](Operator &op, const std::string &name,
              uint32_t value) -> Operator & {
             uint32_t tar = (uint32_t)value;
             return op.SetAttr(name, tar);
           })
      .def("set_attr_vec_int64",
           [](Operator &op, const std::string &name,
              const std::vector<int64_t> &value) -> Operator & {
             int len = value.size();
             std::vector<int64_t> tar;
             int64_t tmp;
             for (int i = 0; i < len; i++) {
               tmp = (int64_t)value[i];
               tar.push_back(tmp);
             }
             return op.SetAttr(name, tar);
           })
      .def("set_attr_vec_int32",
           [](Operator &op, const std::string &name,
              const std::vector<int32_t> &value) -> Operator & {
             int len = value.size();
             std::vector<int32_t> tar;
             int32_t tmp;
             for (int i = 0; i < len; i++) {
               tmp = (int32_t)value[i];
               tar.push_back(tmp);
             }
             return op.SetAttr(name, tar);
           })
      .def("set_attr_vec_uint32",
           [](Operator &op, const std::string &name,
              const std::vector<uint32_t> &value) -> Operator & {
             int len = value.size();
             std::vector<uint32_t> tar;
             uint32_t tmp;
             for (int i = 0; i < len; i++) {
               tmp = (uint32_t)value[i];
               tar.push_back(tmp);
             }
             return op.SetAttr(name, tar);
           })
      .def("set_attr_list_int64",
           [](Operator &op, const std::string &name,
              std::initializer_list<int64_t> &attrValue) -> Operator & {
             return op.SetAttr(name, std::move(attrValue));
           })
      .def("set_attr_attrvalue",
           [](Operator &op, const std::string &name, AttrValue &attrValue)
               -> Operator & { return op.SetAttr(name, std::move(attrValue)); })
      .def(
          "set_attr_float",
          [](Operator &op, const std::string &name, float value) -> Operator & {
            float tar = static_cast<float>(value);
            return op.SetAttr(name, tar);
          })
      .def("set_attr_vec_float",
           [](Operator &op, const std::string &name,
              const std::vector<float> &value) -> Operator & {
             int len = value.size();
             std::vector<float> tar;
             float tmp;
             for (int i = 0; i < len; i++) {
               tmp = static_cast<float>(value[i]);
               tar.push_back(tmp);
             }
             return op.SetAttr(name, tar);
           })
      .def("set_attr_string", (Operator & (Operator::*)(const std::string &,
                                                        const std::string &)) &
                                  Operator::SetAttr)
      .def("set_attr_vec_string",
           (Operator & (Operator::*)(const std::string &,
                                     const std::vector<std::string> &)) &
               Operator::SetAttr)
      .def("set_attr_bool",
           [](Operator &op, const std::string &name, bool value) -> Operator & {
             if (value)
               return op.SetAttr(name, true);
             else
               return op.SetAttr(name, false);
           })
      .def("set_attr_vec_bool",
           [](Operator &op, const std::string &name,
              const std::vector<bool> &value) -> Operator & {
             int len = value.size();
             std::vector<bool> tar;
             for (int i = 0; i < len; i++) {
               if (value[i])
                 tar.push_back(true);
               else
                 tar.push_back(false);
             }
             return op.SetAttr(name, tar);
           })
      .def("set_attr_tensor",
           (Operator & (Operator::*)(const std::string &, const Tensor &)) &
               Operator::SetAttr)
      .def("set_attr_vec_tensor",
           (Operator &
            (Operator::*)(const std::string &, const std::vector<Tensor> &)) &
               Operator::SetAttr)
      .def("set_attr_vec_uint8",
           [](Operator &op, const std::string &name,
              const std::vector<uint8_t> &value) -> Operator & {
             int len = value.size();
             std::vector<uint8_t> tar;
             uint8_t tmp;
             for (int i = 0; i < len; i++) {
               tmp = (uint8_t)value[i];
               tar.push_back(tmp);
             }
             return op.SetAttr(name, tar);
           })
      .def("set_attr_vec_vec_int64",
           (Operator &
            (Operator::*)(const std::string &,
                          const std::vector<std::vector<int64_t>> &)) &
               Operator::SetAttr)
      .def("set_attr_vec_dtype",
           [](Operator &op, const std::string &name,
              const std::vector<DataType> &value) -> Operator & {
             int len = value.size();
             std::vector<ge::DataType> tar;
             ge::DataType tmp;
             for (int i = 0; i < len; i++) {
               tmp = (ge::DataType)value[i];
               tar.push_back(tmp);
             }
             return op.SetAttr(name, tar);
           })
      .def("set_attr_dtype",
           [](Operator &op, const std::string &name,
              const DataType &value) -> Operator & {
             ge::DataType tar = (ge::DataType)value;
             return op.SetAttr(name, tar);
           })

      .def("get_attr",
           [](Operator &op, const std::string &name,
              AttrType type) -> py::tuple {
             graphStatus res = -1;
             switch (type) {
               case AT_INT64: {
                 int64_t i_64_av;
                 res = op.GetAttr(name, i_64_av);
                 return py::make_tuple(i_64_av, res);
               } break;
               case AT_INT32: {
                 int32_t i_32_av;
                 res = op.GetAttr(name, i_32_av);
                 return py::make_tuple(i_32_av, res);
               } break;
               case AT_UINT32: {
                 uint32_t ui_32_av;
                 res = op.GetAttr(name, ui_32_av);
                 return py::make_tuple(ui_32_av, res);
               } break;
               case AT_LIST_INT64: {
                 std::vector<int64_t> v_i_64_av;
                 res = op.GetAttr(name, v_i_64_av);
                 return py::make_tuple(v_i_64_av, res);
               } break;
               case AT_LIST_INT32: {
                 std::vector<int32_t> v_i_32_av;
                 res = op.GetAttr(name, v_i_32_av);
                 return py::make_tuple(v_i_32_av, res);
               } break;
               case AT_LIST_UINT32: {
                 std::vector<uint32_t> v_ui_32_av;
                 res = op.GetAttr(name, v_ui_32_av);
                 return py::make_tuple(v_ui_32_av, res);
               } break;
               case AT_FLOAT: {
                 float f_av;
                 res = op.GetAttr(name, f_av);
                 return py::make_tuple(f_av, res);
               } break;
               case AT_LIST_FLOAT: {
                 std::vector<float> v_f_av;
                 res = op.GetAttr(name, v_f_av);
                 return py::make_tuple(v_f_av, res);
               } break;
               case AT_ATTR_VALUE: {
                 AttrValue o_av;
                 res = op.GetAttr(name, o_av);
                 return py::make_tuple(o_av, res);
               } break;
               case AT_STRING: {
                 std::string s_av;
                 res = op.GetAttr(name, s_av);
                 return py::make_tuple(s_av, res);
               } break;
               case AT_LIST_STRING: {
                 std::vector<std::string> v_s_av;
                 res = op.GetAttr(name, v_s_av);
                 return py::make_tuple(v_s_av, res);
               } break;
               case AT_BOOL: {
                 bool b_av;
                 res = op.GetAttr(name, b_av);
                 return py::make_tuple(b_av, res);
               } break;
               case AT_LIST_BOOL: {
                 std::vector<bool> v_b_av;
                 res = op.GetAttr(name, v_b_av);
                 return py::make_tuple(v_b_av, res);
               } break;
               case AT_TENSOR: {
                 Tensor t_av;
                 res = op.GetAttr(name, t_av);
                 return py::make_tuple(t_av, res);
               } break;
               case AT_LIST_TENSOR: {
                 std::vector<Tensor> v_t_av;
                 res = op.GetAttr(name, v_t_av);
                 return py::make_tuple(v_t_av, res);
               } break;
               case AT_LIST_UINT8: {
                 std::vector<uint8_t> v_ui_8_av;
                 res = op.GetAttr(name, v_ui_8_av);
                 return py::make_tuple(v_ui_8_av, res);
               } break;
               case AT_LIST_LIST_INT64: {
                 std::vector<std::vector<int64_t>> v_v_i_64_av;
                 res = op.GetAttr(name, v_v_i_64_av);
                 return py::make_tuple(v_v_i_64_av, res);
               } break;
               case AT_DT: {
                 ge::DataType dt_av;
                 res = op.GetAttr(name, dt_av);
                 return py::make_tuple(dt_av, res);
               } break;
               case AT_LIST_DT: {
                 std::vector<ge::DataType> v_dt_av;
                 res = op.GetAttr(name, v_dt_av);
                 return py::make_tuple(v_dt_av, res);
               } break;
               default:
                 return py::make_tuple(0, res);
                 break;
             }
           })
      .def("break_connect", &Operator::BreakConnect)
      .def("get_subgraph_names_count", &Operator::GetSubgraphNamesCount)
      .def("get_subgraph_names", &Operator::GetSubgraphNames)
      .def("get_subgraph_builder", &Operator::GetSubgraphBuilder)
      .def("get_subgraph", &Operator::GetSubgraph)
      .def("get_dynamic_subgraph_builder", &Operator::GetDynamicSubgraphBuilder)
      .def("get_dynamic_subgraph", &Operator::GetDynamicSubgraph);

  py::class_<Tensor>(*m, "GETensor")
      .def(py::init<>())
      .def(py::init<const TensorDesc &>())
      .def(py::init<const TensorDesc &, const std::vector<uint8_t> &>())
      .def(py::init<const TensorDesc &, const uint8_t *, size_t>())
      .def("set_tensor_desc", &Tensor::SetTensorDesc)
      .def("get_tensor_desc", &Tensor::GetTensorDesc)
      // .def("set_data", (graphStatus(Tensor::*)(std::vector<uint8_t> &&)) &
      // Tensor::SetData)
      .def("set_data", (graphStatus (Tensor::*)(const std::vector<uint8_t> &)) &
                           Tensor::SetData)
      .def("set_data",
           (graphStatus (Tensor::*)(const uint8_t *, size_t)) & Tensor::SetData)
      .def("set_data",
           (graphStatus (Tensor::*)(const std::string &)) & Tensor::SetData)
      .def("set_data",
           (graphStatus (Tensor::*)(const std::vector<std::string> &)) &
               Tensor::SetData)

      .def("get_data",
           [](Tensor &ts) -> py::list {
             py::list v_data;
             uint8_t *data = ts.GetData();
             size_t size = ts.GetSize();
             for (size_t i = 0; i < size; ++i) {
               v_data.append(data[i]);
             }
             return v_data;
           })
      .def("get_size", &Tensor::GetSize)
      .def("is_valid", &Tensor::IsValid)
      .def("clone", &Tensor::Clone);

  py::class_<TensorDesc>(*m, "GETensorDesc")
      .def(py::init<>())
      .def(py::init<Shape, Format, DataType>(), py::arg("shape"),
           py::arg("format") = FORMAT_ND, py::arg("dt") = DT_FLOAT)
      .def(py::init<const TensorDesc &>())
      .def("update",
           (void (TensorDesc::*)(Shape, Format, DataType)) & TensorDesc::Update,
           py::arg("shape"), py::arg("format") = FORMAT_ND,
           py::arg("dt") = DT_FLOAT)
      .def("set_shape", &TensorDesc::SetShape)
      .def("get_shape", &TensorDesc::GetShape)
      .def("set_unknown_dim_num_shape", &TensorDesc::SetUnknownDimNumShape)
      .def("set_shape_range", &TensorDesc::SetShapeRange)
      .def("get_shape_range",
           [](TensorDesc &tensorDesc) -> py::tuple {
             std::vector<std::pair<int64_t, int64_t>> range;
             graphStatus status = tensorDesc.GetShapeRange(range);
             return py::make_tuple(range, status);
           })
      .def("set_format", &TensorDesc::SetFormat)
      .def("get_format", &TensorDesc::GetFormat)
      .def("get_origin_shape", &TensorDesc::GetOriginShape)
      .def("set_origin_shape", &TensorDesc::SetOriginShape)
      .def("set_origin_format", &TensorDesc::SetOriginFormat)
      .def("get_origin_format", &TensorDesc::GetOriginFormat)
      .def("set_data_type", &TensorDesc::SetDataType)
      .def("get_data_type", &TensorDesc::GetDataType)
      .def("set_name", &TensorDesc::SetName)
      .def("get_name", &TensorDesc::GetName)
      .def("set_size", &TensorDesc::SetSize)
      .def("get_size", &TensorDesc::GetSize)
      .def("set_real_dim_cnt", &TensorDesc::SetRealDimCnt)
      .def("get_real_dim_cnt", &TensorDesc::GetRealDimCnt);

  py::class_<Shape>(*m, "GEShape")
      .def(py::init<>())
      .def(py::init<const std::vector<int64_t> &>())
      .def("get_dim_num", &Shape::GetDimNum)
      .def("set_dim", &Shape::SetDim)
      .def("get_dim", &Shape::GetDim)
      .def("get_dims", &Shape::GetDims)
      .def("get_shape_size", &Shape::GetShapeSize);

  py::class_<AttrValue>(*m, "GEAttrValue").def(py::init<>());

  py::class_<OperatorFactory>(*m, "GEOperatorFactory")
      .def("create_operator", &OperatorFactory::CreateOperator)
      .def("get_ops_type_list",
           []() -> py::tuple {
             std::vector<std::string> all_ops;
             graphStatus status = OperatorFactory::GetOpsTypeList(all_ops);
             return py::make_tuple(all_ops, status);
           })
      .def("is_exist_op", &OperatorFactory::IsExistOp);
}

}  // end namespace pybind
}  // end namespace paddle
#endif
