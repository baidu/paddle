// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <pybind11/chrono.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/stl.h>

#include <memory>
#include <string>
#include <vector>

#include "paddle/fluid/framework/attribute.h"
#include "paddle/fluid/framework/op_info.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/variable.h"
#include "paddle/fluid/imperative/tracer.h"
#include "paddle/fluid/imperative/type_defs.h"
#include "paddle/fluid/pybind/imperative.h"

namespace py = pybind11;
namespace paddle {
namespace pybind {

class OpAttrTypeMap {
 public:
  static OpAttrTypeMap& Instance() {
    static OpAttrTypeMap g_op_attr_type_map;
    return g_op_attr_type_map;
  }

  std::unordered_map<
      std::string,
      std::unordered_map<std::string, paddle::framework::proto::AttrType>>&
  Map() {
    return ops_attrtype_map_;
  }

 private:
  OpAttrTypeMap() = default;
  std::unordered_map<
      std::string,
      std::unordered_map<std::string, paddle::framework::proto::AttrType>>
      ops_attrtype_map_;
};

static inline std::shared_ptr<imperative::VarBase> CastPyHandleToVarBase(
    const std::string& op_type, const std::string& arg_name, int arg_idx,
    const py::handle& handle, bool dispensable = false) {
  PyObject* py_obj = handle.ptr();  // get underlying PyObject
  if (!py_obj || py_obj == Py_None) {
    if (!dispensable) {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "%s(): argument '%s' (position %d) must be Tensor, but got "
          "%s",
          op_type, arg_name, arg_idx, Py_TYPE(py_obj)->tp_name));
    }
    return nullptr;
  }
  try {
    return py::cast<std::shared_ptr<imperative::VarBase>>(py::handle(py_obj));
  } catch (py::cast_error&) {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "%s(): argument '%s' (position %d) must be Tensor, but got "
        "%s",
        op_type, arg_name, arg_idx, Py_TYPE(py_obj)->tp_name));
  }
}

static inline std::vector<std::shared_ptr<imperative::VarBase>>
CastPyHandleToVarBaseList(const std::string& op_type,
                          const std::string& arg_name, int arg_idx,
                          const py::handle& handle, bool dispensable = false) {
  PyObject* py_obj = handle.ptr();  // get underlying PyObject
  if (!py_obj || py_obj == Py_None) {
    if (!dispensable) {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "%s(): argument '%s' (position %d) must be Tensor, but got "
          "%s",
          op_type, arg_name, arg_idx, Py_TYPE(py_obj)->tp_name));
    }
    return {};
  }
  std::vector<std::shared_ptr<imperative::VarBase>> result;
  if (PyList_Check(py_obj) || PyTuple_Check(py_obj)) {
    auto size = PyTuple_Check(py_obj) ? PyTuple_GET_SIZE(py_obj)
                                      : PyList_GET_SIZE(py_obj);
    for (auto i = 0; i < size; ++i) {
      PyObject* item = PyTuple_Check(py_obj) ? PyTuple_GET_ITEM(py_obj, i)
                                             : PyList_GET_ITEM(py_obj, i);
      if (!item || item == Py_None) {
        result.emplace_back(nullptr);
        continue;
      }
      try {
        result.emplace_back(
            py::cast<std::shared_ptr<imperative::VarBase>>(py::handle(item)));
      } catch (py::cast_error&) {
        PADDLE_THROW(platform::errors::InvalidArgument(
            "%s(): argument '%s' (position %d) must be list of "
            "Tensors, but "
            "got %s in list (item %d)",
            op_type, arg_name, arg_idx, Py_TYPE(item)->tp_name, i));
      }
    }
  } else {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "%s(): argument '%s' (position %d) must be list of Tensors, but got "
        "%s",
        op_type, arg_name, arg_idx, Py_TYPE(py_obj)->tp_name));
  }
  return result;
}  // namespace pybind

static inline void ConstructAttrMapFromPyArgs(const std::string& op_type,
                                              int start_idx,
                                              framework::AttributeMap* attrs,
                                              const py::args& args) {
  PADDLE_ENFORCE_EQ(
      args.size() % 2, 0,
      platform::errors::InvalidArgument(
          "The number of arguments for arributes should be even."));
  for (size_t i = 0; i < args.size(); i += 2) {
    std::string name;
    framework::Attribute value;
    try {
      name = args[i].cast<std::string>();
    } catch (std::exception& e) {
      PyObject* py_obj = args[i].ptr();  // get underlying PyObject
      PADDLE_THROW(platform::errors::InvalidArgument(
          "%s(): argument (position %d) must be str, but got "
          "%s",
          op_type, start_idx + i, Py_TYPE(py_obj)->tp_name));
    }
    try {
      value = args[i + 1].cast<framework::Attribute>();
    } catch (std::exception& e) {
      PyObject* py_obj = args[i + 1].ptr();  // get underlying PyObject
      PADDLE_THROW(platform::errors::InvalidArgument(
          "%s(): argument (position %d) must be "
          "Attribute type (one of str, bool, int, int64, float, or list of "
          "them), but got %s",
          op_type, start_idx + i + 1, Py_TYPE(py_obj)->tp_name));
    }
    (*attrs)[name] = value;
  }
}

static inline std::vector<std::shared_ptr<imperative::VarBase>>
ConstructDuplicableOutput(const size_t num) {
  auto tracer = imperative::GetCurrentTracer();
  std::vector<std::shared_ptr<imperative::VarBase>> res;
  res.reserve(num);
  for (size_t i = 0; i < num; i++) {
    auto var_base_name = tracer->GenerateUniqueName();
    res.emplace_back(new imperative::VarBase(var_base_name));
  }
  return res;
}

static inline void HandleViewBetweenInputAndOutput(
    const std::shared_ptr<imperative::VarBase>& input_var,
    const std::shared_ptr<imperative::VarBase>& view_output_var) {
  PADDLE_ENFORCE_EQ(
      input_var->Var().IsInitialized(), true,
      platform::errors::InvalidArgument("Tensor %s has not been initialized!",
                                        input_var->Name()));

  if (input_var->Var().IsType<framework::LoDTensor>()) {
    const auto& input_tensor = input_var->Var().Get<framework::LoDTensor>();
    PADDLE_ENFORCE_EQ(
        input_tensor.IsInitialized(), true,
        platform::errors::InvalidArgument(
            "LoDTensor %s has not been initialized!", input_var->Name()));

    auto* view_output_tensor =
        view_output_var->MutableVar()->GetMutable<framework::LoDTensor>();
    view_output_tensor->ShareDataWith(input_tensor);
    view_output_tensor->ShareInplaceVersionCounterWith(input_tensor);

    VLOG(3) << "Perform View between Output Var(" << view_output_var->Name()
            << ") and Input Var(" << input_var->Name()
            << "), share allocation and inplace version.";
  }
}

extern PyTypeObject* g_varbase_pytype;
extern PyTypeObject* g_vartype_pytype;
extern PyTypeObject* g_blockdesc_pytype;

inline bool PyObject_CheckBool(PyObject** obj) { return PyBool_Check(*obj); }

inline bool PyObject_CheckLongOrToLong(PyObject** obj) {
  if ((PyLong_Check(*obj) && !PyBool_Check(*obj)) ||
      PyObject_IsInstance(*obj, (PyObject*)g_vartype_pytype) ||  // NOLINT
      PyObject_IsInstance(*obj, (PyObject*)g_varbase_pytype)) {  // NOLINT
    return true;
  }
  auto to = PyNumber_Long(*obj);
  if (to) {
    *obj = to;
    return true;
  }
  return false;
}

inline bool PyObject_CheckFloatOrToFloat(PyObject** obj) {
  // sometimes users provide PyLong or numpy.int64 but attr is float
  if (PyFloat_Check(*obj) || PyLong_Check(*obj) ||
      PyObject_IsInstance(*obj, (PyObject*)g_varbase_pytype)) {  // NOLINT
    return true;
  }
  auto to = PyNumber_Float(*obj);
  if (to) {
    *obj = to;
    return true;
  }
  return false;
}

inline bool PyObject_CheckString(PyObject* obj) { return PyUnicode_Check(obj); }

static inline void CastPyArg2AttrBoolean(
    PyObject* obj,
    paddle::framework::AttributeMap& attrs,  // NOLINT
    const std::string& key, const std::string& op_type, ssize_t arg_pos) {
  if (obj == Py_None) {
    attrs[key] = false;  // To be compatible with QA integration testing. Some
                         // test case pass in None.
  } else if (obj == Py_True) {
    attrs[key] = true;
  } else if (obj == Py_False) {
    attrs[key] = false;
  } else {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "%s(): argument (position %d) must be "
        "bool, but got %s",
        op_type, arg_pos + 1,
        ((PyTypeObject*)obj->ob_type)->tp_name));  // NOLINT
  }
}

static inline void CastPyArg2AttrInt(
    PyObject* obj,
    paddle::framework::AttributeMap& attrs,  // NOLINT
    const std::string& key, const std::string& op_type, ssize_t arg_pos) {
  if (PyObject_CheckLongOrToLong(&obj)) {
    attrs[key] = (int)PyLong_AsLong(obj);  // NOLINT
  } else {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "%s(): argument (position %d) must be "
        "int, but got %s",
        op_type, arg_pos + 1,
        ((PyTypeObject*)obj->ob_type)->tp_name));  // NOLINT
  }
}

static inline void CastPyArg2AttrLong(
    PyObject* obj,
    paddle::framework::AttributeMap& attrs,  // NOLINT
    const std::string& key, const std::string& op_type, ssize_t arg_pos) {
  if (PyObject_CheckLongOrToLong(&obj)) {
    attrs[key] = (int64_t)PyLong_AsLong(obj);  // NOLINT
  } else {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "%s(): argument (position %d) must be "
        "long, but got %s",
        op_type, arg_pos + 1,
        ((PyTypeObject*)obj->ob_type)->tp_name));  // NOLINT
  }
}

static inline void CastPyArg2AttrFloat(
    PyObject* obj,
    paddle::framework::AttributeMap& attrs,  // NOLINT
    const std::string& key, const std::string& op_type, ssize_t arg_pos) {
  if (PyObject_CheckFloatOrToFloat(&obj)) {
    attrs[key] = (float)PyFloat_AsDouble(obj);  // NOLINT
  } else {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "%s(): argument (position %d) must be "
        "float, but got %s",
        op_type, arg_pos + 1,
        ((PyTypeObject*)obj->ob_type)->tp_name));  // NOLINT
  }
}

static inline void CastPyArg2AttrString(
    PyObject* obj,
    paddle::framework::AttributeMap& attrs,  // NOLINT
    const std::string& key, const std::string& op_type, ssize_t arg_pos) {
  if (PyObject_CheckString(obj)) {
    Py_ssize_t size;
    const char* data;
    data = PyUnicode_AsUTF8AndSize(obj, &size);
    attrs[key] = std::string(data, (size_t)size);
  } else {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "%s(): argument (position %d) must be "
        "str, but got %s",
        op_type, arg_pos + 1,
        ((PyTypeObject*)obj->ob_type)->tp_name));  // NOLINT
  }
}

static inline void CastPyArg2AttrBooleans(
    PyObject* obj, paddle::framework::AttributeMap& attrs,  // NOLINT
    const std::string& key, const std::string& op_type, ssize_t arg_pos) {
  if (PyList_Check(obj)) {
    Py_ssize_t len = PyList_Size(obj);
    PyObject* item = nullptr;
    std::vector<bool> value;
    for (Py_ssize_t i = 0; i < len; i++) {
      item = PyList_GetItem(obj, i);
      if (PyObject_CheckBool(&item)) {
        value.emplace_back(PyLong_AsLong(item));
      } else {
        PADDLE_THROW(platform::errors::InvalidArgument(
            "%s(): argument (position %d) must be "
            "list of bool, but got %s at pos %d",
            op_type, arg_pos + 1,
            ((PyTypeObject*)item->ob_type)->tp_name,  // NOLINT
            i));
      }
    }
    attrs[key] = value;
  } else if (PyTuple_Check(obj)) {
    Py_ssize_t len = PyTuple_Size(obj);
    PyObject* item = nullptr;
    std::vector<bool> value;
    for (Py_ssize_t i = 0; i < len; i++) {
      item = PyTuple_GetItem(obj, i);
      if (PyObject_CheckBool(&item)) {
        value.emplace_back(PyLong_AsLong(item));
      } else {
        PADDLE_THROW(platform::errors::InvalidArgument(
            "%s(): argument (position %d) must be "
            "list of bool, but got %s at pos %d",
            op_type, arg_pos + 1,
            ((PyTypeObject*)item->ob_type)->tp_name,  // NOLINT
            i));
      }
    }
    attrs[key] = value;
  } else {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "%s(): argument (position %d) must be "
        "list or tuple, but got %s",
        op_type, arg_pos + 1,
        ((PyTypeObject*)obj->ob_type)->tp_name));  // NOLINT
  }
}

static inline void CastPyArg2AttrInts(
    PyObject* obj,
    paddle::framework::AttributeMap& attrs,  // NOLINT
    const std::string& key, const std::string& op_type, ssize_t arg_pos) {
  if (PyList_Check(obj)) {
    Py_ssize_t len = PyList_Size(obj);
    PyObject* item = nullptr;
    std::vector<int> value;
    for (Py_ssize_t i = 0; i < len; i++) {
      item = PyList_GetItem(obj, i);
      if (PyObject_CheckLongOrToLong(&item)) {
        value.emplace_back(PyLong_AsLong(item));
      } else {
        PADDLE_THROW(platform::errors::InvalidArgument(
            "%s(): argument (position %d) must be "
            "list of int, but got %s at pos %d",
            op_type, arg_pos + 1,
            ((PyTypeObject*)item->ob_type)->tp_name,  // NOLINT
            i));
      }
    }
    attrs[key] = value;
  } else if (PyTuple_Check(obj)) {
    Py_ssize_t len = PyTuple_Size(obj);
    PyObject* item = nullptr;
    std::vector<int> value;
    for (Py_ssize_t i = 0; i < len; i++) {
      item = PyTuple_GetItem(obj, i);
      if (PyObject_CheckLongOrToLong(&item)) {
        value.emplace_back(PyLong_AsLong(item));
      } else {
        PADDLE_THROW(platform::errors::InvalidArgument(
            "%s(): argument (position %d) must be "
            "list of int, but got %s at pos %d",
            op_type, arg_pos + 1,
            ((PyTypeObject*)item->ob_type)->tp_name,  // NOLINT
            i));
      }
    }
    attrs[key] = value;
  } else if (PySequence_Check(obj)) {
    Py_ssize_t len = PySequence_Size(obj);
    PyObject* item = nullptr;
    std::vector<int> value;
    for (Py_ssize_t i = 0; i < len; i++) {
      item = PySequence_GetItem(obj, i);
      if (PyObject_CheckLongOrToLong(&item)) {
        value.emplace_back(PyLong_AsLong(item));
      } else {
        PADDLE_THROW(platform::errors::InvalidArgument(
            "%s(): argument (position %d) must be "
            "list of int, but got %s at pos %d",
            op_type, arg_pos + 1,
            ((PyTypeObject*)item->ob_type)->tp_name,  // NOLINT
            i));
      }
    }
    attrs[key] = value;
  } else {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "%s(): argument (position %d) must be "
        "list or tuple, but got %s",
        op_type, arg_pos + 1,
        ((PyTypeObject*)obj->ob_type)->tp_name));  // NOLINT
  }
}

static inline void CastPyArg2AttrLongs(
    PyObject* obj,
    paddle::framework::AttributeMap& attrs,  // NOLINT
    const std::string& key, const std::string& op_type, ssize_t arg_pos) {
  if (PyList_Check(obj)) {
    Py_ssize_t len = PyList_Size(obj);
    PyObject* item = nullptr;
    std::vector<int64_t> value;
    for (Py_ssize_t i = 0; i < len; i++) {
      item = PyList_GetItem(obj, i);
      if (PyObject_CheckLongOrToLong(&item)) {
        value.emplace_back(PyLong_AsLong(item));
      } else {
        PADDLE_THROW(platform::errors::InvalidArgument(
            "%s(): argument (position %d) must be "
            "list of int, but got %s at pos %d",
            op_type, arg_pos + 1,
            ((PyTypeObject*)item->ob_type)->tp_name,  // NOLINT
            i));
      }
    }
    attrs[key] = value;
  } else if (PyTuple_Check(obj)) {
    Py_ssize_t len = PyTuple_Size(obj);
    PyObject* item = nullptr;
    std::vector<int64_t> value;
    for (Py_ssize_t i = 0; i < len; i++) {
      item = PyTuple_GetItem(obj, i);
      if (PyObject_CheckLongOrToLong(&item)) {
        value.emplace_back(PyLong_AsLong(item));
      } else {
        PADDLE_THROW(platform::errors::InvalidArgument(
            "%s(): argument (position %d) must be "
            "list of int, but got %s at pos %d",
            op_type, arg_pos + 1,
            ((PyTypeObject*)item->ob_type)->tp_name,  // NOLINT
            i));
      }
    }
    attrs[key] = value;
  } else if (PySequence_Check(obj)) {
    Py_ssize_t len = PySequence_Size(obj);
    PyObject* item = nullptr;
    std::vector<int64_t> value;
    for (Py_ssize_t i = 0; i < len; i++) {
      item = PySequence_GetItem(obj, i);
      if (PyObject_CheckLongOrToLong(&item)) {
        value.emplace_back(PyLong_AsLong(item));
      } else {
        PADDLE_THROW(platform::errors::InvalidArgument(
            "%s(): argument (position %d) must be "
            "list of int, but got %s at pos %d",
            op_type, arg_pos + 1,
            ((PyTypeObject*)item->ob_type)->tp_name,  // NOLINT
            i));
      }
    }
    attrs[key] = value;
  } else {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "%s(): argument (position %d) must be "
        "list or tuple, but got %s",
        op_type, arg_pos + 1,
        ((PyTypeObject*)obj->ob_type)->tp_name));  // NOLINT
  }
}

static inline void CastPyArg2AttrFloats(
    PyObject* obj,
    paddle::framework::AttributeMap& attrs,  // NOLINT
    const std::string& key, const std::string& op_type, ssize_t arg_pos) {
  if (PyList_Check(obj)) {
    Py_ssize_t len = PyList_Size(obj);
    PyObject* item = nullptr;
    std::vector<float> value;
    for (Py_ssize_t i = 0; i < len; i++) {
      item = PyList_GetItem(obj, i);
      if (PyObject_CheckFloatOrToFloat(&item)) {
        value.emplace_back(PyFloat_AsDouble(item));
      } else {
        PADDLE_THROW(platform::errors::InvalidArgument(
            "%s(): argument (position %d) must be "
            "list of float, but got %s at pos %d",
            op_type, arg_pos + 1,
            ((PyTypeObject*)item->ob_type)->tp_name,  // NOLINT
            i));
      }
    }
    attrs[key] = value;
  } else if (PyTuple_Check(obj)) {
    Py_ssize_t len = PyTuple_Size(obj);
    PyObject* item = nullptr;
    std::vector<float> value;
    for (Py_ssize_t i = 0; i < len; i++) {
      item = PyTuple_GetItem(obj, i);
      if (PyObject_CheckFloatOrToFloat(&item)) {
        value.emplace_back(PyFloat_AsDouble(item));
      } else {
        PADDLE_THROW(platform::errors::InvalidArgument(
            "%s(): argument (position %d) must be "
            "list of float, but got %s at pos %d",
            op_type, arg_pos + 1,
            ((PyTypeObject*)item->ob_type)->tp_name,  // NOLINT
            i));
      }
    }
    attrs[key] = value;
  } else if (PySequence_Check(obj)) {
    Py_ssize_t len = PySequence_Size(obj);
    PyObject* item = nullptr;
    std::vector<float> value;
    for (Py_ssize_t i = 0; i < len; i++) {
      item = PySequence_GetItem(obj, i);
      if (PyObject_CheckFloatOrToFloat(&item)) {
        value.emplace_back(PyFloat_AsDouble(item));
      } else {
        PADDLE_THROW(platform::errors::InvalidArgument(
            "%s(): argument (position %d) must be "
            "list of float, but got %s at pos %d",
            op_type, arg_pos + 1,
            ((PyTypeObject*)item->ob_type)->tp_name,  // NOLINT
            i));
      }
    }
    attrs[key] = value;
  } else {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "%s(): argument (position %d) must be "
        "list or tuple, but got %s",
        op_type, arg_pos + 1,
        ((PyTypeObject*)obj->ob_type)->tp_name));  // NOLINT
  }
}

static inline void CastPyArg2AttrFloat64s(
    PyObject* obj, paddle::framework::AttributeMap& attrs,  // NOLINT
    const std::string& key, const std::string& op_type, ssize_t arg_pos) {
  if (PyList_Check(obj)) {
    Py_ssize_t len = PyList_Size(obj);
    PyObject* item = nullptr;
    std::vector<double> value;
    for (Py_ssize_t i = 0; i < len; i++) {
      item = PyList_GetItem(obj, i);
      if (PyObject_CheckFloatOrToFloat(&item)) {
        value.emplace_back(PyFloat_AsDouble(item));
      } else {
        PADDLE_THROW(platform::errors::InvalidArgument(
            "%s(): argument (position %d) must be "
            "list of float, but got %s at pos %d",
            op_type, arg_pos + 1,
            ((PyTypeObject*)item->ob_type)->tp_name,  // NOLINT
            i));
      }
    }
    attrs[key] = value;
  } else if (PyTuple_Check(obj)) {
    Py_ssize_t len = PyTuple_Size(obj);
    PyObject* item = nullptr;
    std::vector<double> value;
    for (Py_ssize_t i = 0; i < len; i++) {
      item = PyTuple_GetItem(obj, i);
      if (PyObject_CheckFloatOrToFloat(&item)) {
        value.emplace_back(PyFloat_AsDouble(item));
      } else {
        PADDLE_THROW(platform::errors::InvalidArgument(
            "%s(): argument (position %d) must be "
            "list of float, but got %s at pos %d",
            op_type, arg_pos + 1,
            ((PyTypeObject*)item->ob_type)->tp_name,  // NOLINT
            i));
      }
    }
    attrs[key] = value;
  } else if (PySequence_Check(obj)) {
    Py_ssize_t len = PySequence_Size(obj);
    PyObject* item = nullptr;
    std::vector<double> value;
    for (Py_ssize_t i = 0; i < len; i++) {
      item = PySequence_GetItem(obj, i);
      if (PyObject_CheckFloatOrToFloat(&item)) {
        value.emplace_back(PyFloat_AsDouble(item));
      } else {
        PADDLE_THROW(platform::errors::InvalidArgument(
            "%s(): argument (position %d) must be "
            "list of float, but got %s at pos %d",
            op_type, arg_pos + 1,
            ((PyTypeObject*)item->ob_type)->tp_name,  // NOLINT
            i));
      }
    }
    attrs[key] = value;
  } else {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "%s(): argument (position %d) must be "
        "list or tuple, but got %s",
        op_type, arg_pos + 1,
        ((PyTypeObject*)obj->ob_type)->tp_name));  // NOLINT
  }
}

static inline void CastPyArg2AttrStrings(
    PyObject* obj,
    paddle::framework::AttributeMap& attrs,  // NOLINT
    const std::string& key, const std::string& op_type, ssize_t arg_pos) {
  if (PyList_Check(obj)) {
    Py_ssize_t len = PyList_Size(obj);
    PyObject* item = nullptr;
    std::vector<std::string> value;
    for (Py_ssize_t i = 0; i < len; i++) {
      item = PyList_GetItem(obj, i);
      if (PyObject_CheckString(item)) {
        Py_ssize_t size;
        const char* data;
        data = PyUnicode_AsUTF8AndSize(item, &size);
        value.emplace_back(std::string(data, (size_t)size));  // NOLINT
      } else {
        PADDLE_THROW(platform::errors::InvalidArgument(
            "%s(): argument (position %d) must be "
            "list of str, but got %s at pos %d",
            op_type, arg_pos + 1,
            ((PyTypeObject*)item->ob_type)->tp_name,  // NOLINT
            i));
      }
    }
    attrs[key] = value;
  } else if (PyTuple_Check(obj)) {
    Py_ssize_t len = PyTuple_Size(obj);
    PyObject* item = nullptr;
    std::vector<std::string> value;
    for (Py_ssize_t i = 0; i < len; i++) {
      item = PyTuple_GetItem(obj, i);
      if (PyObject_CheckString(item)) {
        Py_ssize_t size;
        const char* data;
        data = PyUnicode_AsUTF8AndSize(item, &size);
        value.emplace_back(std::string(data, (size_t)size));
      } else {
        PADDLE_THROW(platform::errors::InvalidArgument(
            "%s(): argument (position %d) must be "
            "list of str, but got %s at pos %d",
            op_type, arg_pos + 1,
            ((PyTypeObject*)item->ob_type)->tp_name,  // NOLINT
            i));
      }
    }
    attrs[key] = value;
  } else {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "%s(): argument (position %d) must be "
        "list or tuple, but got %s",
        op_type, arg_pos + 1,
        ((PyTypeObject*)obj->ob_type)->tp_name));  // NOLINT
  }
}

static inline void CastPyArg2AttrBlock(
    PyObject* obj, paddle::framework::AttributeMap& attrs,  // NOLINT
    const std::string& key, const std::string& op_type, ssize_t arg_pos) {
  ::pybind11::detail::instance* inst =
      (::pybind11::detail::instance*)obj;  // NOLINT

  if (!PyObject_IsInstance((PyObject*)inst,                   // NOLINT
                           (PyObject*)g_blockdesc_pytype)) {  // NOLINT
    PADDLE_THROW(platform::errors::InvalidArgument(
        "%s(): argument (position %d) must be "
        "BlockDesc, but got %s",
        op_type, arg_pos + 1,
        ((PyTypeObject*)obj->ob_type)->tp_name));  // NOLINT
  }
  void** vh = inst->simple_layout ? inst->simple_value_holder
                                  : &inst->nonsimple.values_and_holders[0];
  attrs[key] = reinterpret_cast<paddle::framework::BlockDesc*&>(vh[0]);
}

static inline void ConstructAttrMapFromPyArgs(
    const std::string& op_type, PyObject* args, ssize_t attr_start,
    ssize_t attr_end, paddle::framework::AttributeMap& attrs) {  // NOLINT
  PADDLE_ENFORCE_EQ(
      (attr_end - attr_start) % 2, 0,
      platform::errors::InvalidArgument(
          "The number of arguments for attributes should be even."));

  auto attr_type_map = &(OpAttrTypeMap::Instance().Map()[op_type]);

  PyObject* obj = nullptr;
  for (ssize_t arg_pos = attr_start; arg_pos < attr_end; arg_pos += 2) {
    Py_ssize_t key_len;
    const char* key_ptr;
    obj = PyTuple_GET_ITEM(args, arg_pos);
    if (PyObject_CheckString(obj)) {
      key_ptr = PyUnicode_AsUTF8AndSize(obj, &key_len);
    } else {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "%s(): argument (position %d) must be str, but got "
          "%s",
          op_type, arg_pos, ((PyTypeObject*)obj->ob_type)->tp_name));  // NOLINT
    }

    std::string key(key_ptr, (size_t)key_len);
    auto iter = attr_type_map->find(key);
    if (iter == attr_type_map->end()) {
      continue;
    }

    obj = PyTuple_GET_ITEM(args, arg_pos + 1);

    switch (iter->second) {
      case paddle::framework::proto::AttrType::INT:
        CastPyArg2AttrInt(obj, attrs, key, op_type, arg_pos);
        break;
      case paddle::framework::proto::AttrType::FLOAT:
        CastPyArg2AttrFloat(obj, attrs, key, op_type, arg_pos);
        break;
      case paddle::framework::proto::AttrType::STRING:
        CastPyArg2AttrString(obj, attrs, key, op_type, arg_pos);
        break;
      case paddle::framework::proto::AttrType::INTS:
        CastPyArg2AttrInts(obj, attrs, key, op_type, arg_pos);
        break;
      case paddle::framework::proto::AttrType::FLOATS:
        CastPyArg2AttrFloats(obj, attrs, key, op_type, arg_pos);
        break;
      case paddle::framework::proto::AttrType::STRINGS:
        CastPyArg2AttrStrings(obj, attrs, key, op_type, arg_pos);
        break;
      case paddle::framework::proto::AttrType::BOOLEAN:
        CastPyArg2AttrBoolean(obj, attrs, key, op_type, arg_pos);
        break;
      case paddle::framework::proto::AttrType::BOOLEANS:
        CastPyArg2AttrBooleans(obj, attrs, key, op_type, arg_pos);
        break;
      case paddle::framework::proto::AttrType::LONG:
        CastPyArg2AttrLong(obj, attrs, key, op_type, arg_pos);
        break;
      case paddle::framework::proto::AttrType::LONGS:
        CastPyArg2AttrLongs(obj, attrs, key, op_type, arg_pos);
        break;
      case paddle::framework::proto::AttrType::FLOAT64S:
        CastPyArg2AttrFloat64s(obj, attrs, key, op_type, arg_pos);
        break;
      case paddle::framework::proto::AttrType::BLOCK:
        CastPyArg2AttrBlock(obj, attrs, key, op_type, arg_pos);
        break;
      default:
        break;
    }
  }
}

static inline std::shared_ptr<imperative::VarBase> GetVarBaseFromArgs(
    const std::string& op_type, const std::string& arg_name, PyObject* args,
    ssize_t arg_idx, bool dispensable = false) {
  ::pybind11::detail::instance* inst =
      (::pybind11::detail::instance*)PyTuple_GET_ITEM(args, arg_idx);

  if (PyTuple_Check((PyObject*)inst)) {  // NOLINT
    inst = (::pybind11::detail::instance*)PyTuple_GET_ITEM(inst, 0);
  }

  if (inst == nullptr || (PyObject*)inst == Py_None) {  // NOLINT
    if (!dispensable) {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "%s(): argument '%s' (position %d) must be Tensor, but got None",
          op_type, arg_name, arg_idx));
    }
    return nullptr;
  }

  if (!PyObject_IsInstance((PyObject*)inst,                 // NOLINT
                           (PyObject*)g_varbase_pytype)) {  // NOLINT
    PADDLE_THROW(platform::errors::InvalidArgument(
        "%s(): argument '%s' (position %d) must be Tensor, but got "
        "%s",
        op_type, arg_name, arg_idx,
        ((PyTypeObject*)((PyObject*)inst)->ob_type)->tp_name));  // NOLINT
  }

  void** vh = inst->simple_layout ? inst->simple_value_holder
                                  : &inst->nonsimple.values_and_holders[0];
  return reinterpret_cast<std::shared_ptr<paddle::imperative::VarBase>&>(vh[1]);
}

static inline std::vector<std::shared_ptr<imperative::VarBase>>
GetVarBaseListFromArgs(const std::string& op_type, const std::string& arg_name,
                       PyObject* args, ssize_t arg_idx,
                       bool dispensable = false) {
  PyObject* list = PyTuple_GET_ITEM(args, arg_idx);

  if (list == nullptr) {
    if (!dispensable) {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "%s(): argument '%s' (position %d) must be list of Tensor, but got "
          "None",
          op_type, arg_name, arg_idx));  // NOLINT
    }
    return {};
  }

  std::vector<std::shared_ptr<imperative::VarBase>> result;

  if (PyList_Check(list)) {
    Py_ssize_t len = PyList_Size(list);
    if (len == 0) {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "%s(): argument '%s' (position %d) must be list of Tensors, but got "
          "empty list",
          op_type, arg_name, arg_idx));
    }
    ::pybind11::detail::instance* item = nullptr;
    for (Py_ssize_t i = 0; i < len; i++) {
      item = (::pybind11::detail::instance*)PyList_GetItem(list, i);
      if (!PyObject_IsInstance((PyObject*)item,                 // NOLINT
                               (PyObject*)g_varbase_pytype)) {  // NOLINT
        PADDLE_THROW(platform::errors::InvalidArgument(
            "%s(): argument '%s' (position %d) must be list of Tensors, but "
            "got list of "
            "%s",
            op_type, arg_name, arg_idx,
            ((PyTypeObject*)((PyObject*)item)->ob_type)->tp_name));  // NOLINT
      }
      void** vh = item->simple_layout ? item->simple_value_holder
                                      : &item->nonsimple.values_and_holders[0];
      result.emplace_back(
          reinterpret_cast<std::shared_ptr<paddle::imperative::VarBase>&>(
              vh[1]));
    }
  } else if (PyTuple_Check(list)) {
    Py_ssize_t len = PyTuple_Size(list);
    if (len == 0) {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "%s(): argument '%s' (position %d) must be list of Tensors, but got "
          "empty list",
          op_type, arg_name, arg_idx));
    }
    ::pybind11::detail::instance* item = nullptr;
    for (Py_ssize_t i = 0; i < len; i++) {
      item = (::pybind11::detail::instance*)PyTuple_GetItem(list, i);  // NOLINT
      if (!PyObject_IsInstance((PyObject*)item,                        // NOLINT
                               (PyObject*)g_varbase_pytype)) {         // NOLINT
        PADDLE_THROW(platform::errors::InvalidArgument(
            "%s(): argument '%s' (position %d) must be list of Tensors, but "
            "got list of "
            "%s",
            op_type, arg_name, arg_idx,
            ((PyTypeObject*)((PyObject*)item)->ob_type)->tp_name));  // NOLINT
      }
      void** vh = item->simple_layout ? item->simple_value_holder
                                      : &item->nonsimple.values_and_holders[0];
      result.emplace_back(
          reinterpret_cast<std::shared_ptr<paddle::imperative::VarBase>&>(
              vh[1]));
    }
  } else {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "%s(): argument '%s' (position %d) must be list of Tensors, but got "
        "%s",
        op_type, arg_name, arg_idx,
        ((PyTypeObject*)list->ob_type)->tp_name));  // NOLINT
  }

  return result;
}

static inline unsigned long GetUnsignedLongFromArgs(  // NOLINT
    const std::string& op_type, const std::string& arg_name, PyObject* args,
    ssize_t arg_idx, bool dispensable = false) {
  PyObject* item = PyTuple_GET_ITEM(args, arg_idx);

  if (item == nullptr) {
    if (!dispensable) {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "%s(): argument '%s' (position %d) must be long, but got None",
          op_type, arg_name, arg_idx));
    }
    return 0;
  }

  if (PyObject_CheckLongOrToLong(&item)) {
    return PyLong_AsUnsignedLong(item);
  } else {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "%s(): argument '%s' (position %d) must be "
        "long, but got %s",
        op_type, arg_name, arg_idx,
        ((PyTypeObject*)item->ob_type)->tp_name));  // NOLINT
  }
}

static inline PyObject* MakeReturnPyObject(
    const std::shared_ptr<paddle::imperative::VarBase>& out) {
  return ::pybind11::detail::type_caster_base<imperative::VarBase>::cast_holder(
             ::pybind11::detail::holder_helper<
                 std::shared_ptr<imperative::VarBase>>::get(out),
             &out)
      .ptr();
}

static inline PyObject* MakeReturnPyObject(
    const std::vector<std::shared_ptr<imperative::VarBase>>& out) {
  PyObject* result = PyList_New((Py_ssize_t)out.size());

  for (size_t i = 0; i < out.size(); i++) {
    PyList_SET_ITEM(
        result, (Py_ssize_t)i,
        ::pybind11::detail::type_caster_base<imperative::VarBase>::cast_holder(
            ::pybind11::detail::holder_helper<
                std::shared_ptr<imperative::VarBase>>::get(out[i]),
            &out[i])
            .ptr());  // NOLINT
  }

  return result;
}

template <typename Tuple, size_t N>
struct TupleVarBasesResult {
  static void Run(const Tuple& out, PyObject* result) {
    TupleVarBasesResult<Tuple, N - 1>::Run(out, result);
    PyTuple_SET_ITEM(result, N - 1, MakeReturnPyObject(std::get<N - 1>(out)));
  }
};

template <typename Tuple>
struct TupleVarBasesResult<Tuple, 1> {
  static void Run(const Tuple& out, PyObject* result) {
    PyTuple_SET_ITEM(result, 0, MakeReturnPyObject(std::get<0>(out)));
  }
};

template <typename... Args>
static inline PyObject* MakeReturnPyObject(const std::tuple<Args...>& out) {
  auto len = sizeof...(Args);
  PyObject* result = PyTuple_New(len);

  TupleVarBasesResult<decltype(out), sizeof...(Args)>::Run(out, result);

  return result;
}

void InitOpsAttrTypeMap() {
  auto op_info_map = paddle::framework::OpInfoMap::Instance().map();
  for (auto iter = op_info_map.begin(); iter != op_info_map.end(); ++iter) {
    auto op_proto = iter->second.proto_;
    if (op_proto == nullptr) {
      continue;
    }
    auto attrs_proto = op_proto->attrs();
    for (auto& attr : attrs_proto) {
      OpAttrTypeMap::Instance().Map()[iter->first][attr.name()] = attr.type();
    }
  }
}

void ThrowExceptionToPython(std::exception_ptr p) {
  static PyObject* EOFExceptionException =
      PyErr_NewException("paddle.EOFException", PyExc_Exception, NULL);
  static PyObject* EnforceNotMetException =
      PyErr_NewException("paddle.EnforceNotMet", PyExc_Exception, NULL);
  try {
    if (p) std::rethrow_exception(p);
  } catch (const platform::EOFException& e) {
    PyErr_SetString(EOFExceptionException, e.what());
  } catch (const platform::EnforceNotMet& e) {
    switch (e.code()) {
      case paddle::platform::error::INVALID_ARGUMENT:
        PyErr_SetString(PyExc_ValueError, e.what());
        break;
      case paddle::platform::error::NOT_FOUND:
      case paddle::platform::error::ALREADY_EXISTS:
      case paddle::platform::error::PRECONDITION_NOT_MET:
      case paddle::platform::error::PERMISSION_DENIED:
      case paddle::platform::error::EXECUTION_TIMEOUT:
      case paddle::platform::error::UNAVAILABLE:
        PyErr_SetString(PyExc_RuntimeError, e.what());
        break;
      case paddle::platform::error::OUT_OF_RANGE:
        PyErr_SetString(PyExc_IndexError, e.what());
        break;
      case paddle::platform::error::RESOURCE_EXHAUSTED:
        PyErr_SetString(PyExc_MemoryError, e.what());
        break;
      case paddle::platform::error::UNIMPLEMENTED:
        PyErr_SetString(PyExc_NotImplementedError, e.what());
        break;
      case paddle::platform::error::FATAL:
        PyErr_SetString(PyExc_SystemError, e.what());
        break;
      case paddle::platform::error::EXTERNAL:
        PyErr_SetString(PyExc_OSError, e.what());
        break;
      default:
        PyErr_SetString(EnforceNotMetException, e.what());
        break;
    }
  }
}

}  // namespace pybind
}  // namespace paddle

// This include must be the last line
#include "paddle/fluid/pybind/op_function_impl_new.h"
