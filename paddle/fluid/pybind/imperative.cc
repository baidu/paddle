/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/pybind/imperative.h"

#include <Python.h>
#include <pybind11/chrono.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/stl.h>

#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "paddle/fluid/imperative/all_reduce.h"
#include "paddle/fluid/imperative/amp_auto_cast.h"
#include "paddle/fluid/imperative/basic_engine.h"
#include "paddle/fluid/imperative/data_loader.h"
#include "paddle/fluid/imperative/layer.h"
#include "paddle/fluid/imperative/nccl_context.h"
#include "paddle/fluid/imperative/partial_grad_engine.h"
#include "paddle/fluid/imperative/profiler.h"
#include "paddle/fluid/imperative/tracer.h"
#include "paddle/fluid/imperative/type_defs.h"
#include "paddle/fluid/memory/allocation/mmap_allocator.h"
#include "paddle/fluid/pybind/op_function.h"
#include "paddle/fluid/pybind/pybind_boost_headers.h"
#include "paddle/fluid/pybind/tensor_py.h"

namespace paddle {
namespace pybind {

namespace py = ::pybind11;

class Layer : public imperative::Layer {
 public:
  using imperative::Layer::Layer;  // Inherit constructors

  std::vector<std::shared_ptr<imperative::VarBase>> Forward(
      const std::vector<std::shared_ptr<imperative::VarBase>> &inputs)
      override {
    PYBIND11_OVERLOAD(std::vector<std::shared_ptr<imperative::VarBase>>, Layer,
                      Forward, inputs);  // NOLINT
  }
};

static const platform::Place PyObjectToPlace(const py::object &place_obj) {
  if (py::isinstance<platform::CPUPlace>(place_obj)) {
    return place_obj.cast<platform::CPUPlace>();
  } else if (py::isinstance<platform::CUDAPlace>(place_obj)) {
    return place_obj.cast<platform::CUDAPlace>();
  } else if (py::isinstance<platform::XPUPlace>(place_obj)) {
    return place_obj.cast<platform::XPUPlace>();
  } else if (py::isinstance<platform::CUDAPinnedPlace>(place_obj)) {
    return place_obj.cast<platform::CUDAPinnedPlace>();
  } else {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "Place should be one of CPUPlace/XPUPlace/CUDAPlace/CUDAPinnedPlace"));
  }
}

static void InitTensorForVarBase(imperative::VarBase *self,
                                 const py::array &array,
                                 const platform::Place place,
                                 bool persistable = false,
                                 bool zero_copy = false, std::string name = "",
                                 int stop_gradient = -1) {
  if (name == "") {
    name =
        imperative::GetCurrentTracer()->GenerateUniqueName("generated_tensor");
  }
  VLOG(5) << "Init Tensor as: / name: " << name
          << " / persistable: " << persistable << " / zero_copy: " << zero_copy
          << " / stop_gradient: " << stop_gradient;
  new (self) imperative::VarBase(name);
  auto *tensor = self->MutableVar()->GetMutable<framework::LoDTensor>();
  if (platform::is_cpu_place(place)) {
    SetTensorFromPyArray<platform::CPUPlace>(
        tensor, array, BOOST_GET_CONST(platform::CPUPlace, place), zero_copy);
  } else if (platform::is_xpu_place(place)) {
    SetTensorFromPyArray<platform::XPUPlace>(
        tensor, array, BOOST_GET_CONST(platform::XPUPlace, place), zero_copy);
  } else if (platform::is_gpu_place(place)) {
    SetTensorFromPyArray<platform::CUDAPlace>(
        tensor, array, BOOST_GET_CONST(platform::CUDAPlace, place), zero_copy);
  } else if (platform::is_cuda_pinned_place(place)) {
    SetTensorFromPyArray<platform::CUDAPinnedPlace>(
        tensor, array, BOOST_GET_CONST(platform::CUDAPinnedPlace, place),
        zero_copy);
  } else {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "Place should be one of CPUPlace/XPUPlace/CUDAPlace/CUDAPinnedPlace"));
  }
  if (stop_gradient != -1) {
    self->SetOverridedStopGradient(stop_gradient);
  }
  self->SetPersistable(persistable);
  self->SetType(framework::proto::VarType::LOD_TENSOR);
  self->SetDataType(tensor->type());
}

static void InitVarBaseFromNumpyWithKwargs(imperative::VarBase *self,
                                           const py::kwargs &kwargs) {
  VLOG(4) << "Init VarBase from kwargs: ";
  PADDLE_ENFORCE_EQ(
      kwargs.contains("value"), true,
      platform::errors::NotFound(
          "The kwargs used to create Varbase misses argument: value"));
  auto persistable = kwargs.contains("persistable")
                         ? kwargs["persistable"].cast<bool>()
                         : false;
  auto array = kwargs.contains("value") ? kwargs["value"].cast<py::array>()
                                        : py::array();
  auto zero_copy =
      kwargs.contains("zero_copy") ? kwargs["zero_copy"].cast<bool>() : false;
  auto name = kwargs.contains("name") ? kwargs["name"].cast<std::string>() : "";
  auto stop_gradient = kwargs.contains("stop_gradient")
                           ? kwargs["stop_gradient"].cast<int>()
                           : -1;
  auto default_place = imperative::GetCurrentTracer()->ExpectedPlace();
  auto place = kwargs.contains("place") ? PyObjectToPlace(kwargs["place"])
                                        : default_place;
  InitTensorForVarBase(self, array, place, persistable, zero_copy, name,
                       stop_gradient);
}

template <typename P>
static void InitVarBaseFromNumpyWithArg(imperative::VarBase *self,
                                        const py::array &array, const P &place,
                                        bool persistable = false,
                                        bool zero_copy = false,
                                        std::string name = "",
                                        int stop_gradient = -1) {
  VLOG(4) << "Init VarBase from Arg: ";
  // 0: self, 1: value, 2: place, 3: persistable, 4: zero_copy, 5: name , 6:
  // stop_gradient
  if (name == "") {
    name =
        imperative::GetCurrentTracer()->GenerateUniqueName("generated_tensor");
  }
  VLOG(5) << "Init Tensor as: / name: " << name
          << " / persistable: " << persistable << " / zero_copy: " << zero_copy
          << " / stop_gradient: " << stop_gradient;
  new (self) imperative::VarBase(name);
  self->SetPersistable(persistable);
  auto *tensor = self->MutableVar()->GetMutable<framework::LoDTensor>();
  if (stop_gradient != -1) {
    self->SetOverridedStopGradient(stop_gradient);
  }
  SetTensorFromPyArray<P>(tensor, array, place, zero_copy);
  self->SetType(framework::proto::VarType::LOD_TENSOR);
  self->SetDataType(tensor->type());
}

static void InitVarBaseFromNumpyWithArgDefault(imperative::VarBase *self,
                                               const py::array &array) {
  VLOG(4) << "Init VarBase from numpy: ";
  auto place = imperative::GetCurrentTracer()->ExpectedPlace();
  InitTensorForVarBase(self, array, place);
}

static void InitVarBaseFromTensorWithArgDefault(
    imperative::VarBase *self, const framework::LoDTensor &tensor) {
  VLOG(4) << "Init VarBase";
  auto place = imperative::GetCurrentTracer()->ExpectedPlace();
  new (self) imperative::VarBase(
      imperative::GetCurrentTracer()->GenerateUniqueName("generated_tensor"));
  self->SetPersistable(false);
  self->SetType(framework::proto::VarType::LOD_TENSOR);
  self->SetDataType(tensor.type());
  auto *new_tensor = self->MutableVar()->GetMutable<framework::LoDTensor>();
  // Same place，share data directly
  if (place == tensor.place()) {
    new_tensor->ShareDataWith(tensor);
    VLOG(4) << "Same place, do ShareDataWith";
  } else {
    framework::TensorCopy(tensor, place, new_tensor);
    VLOG(4) << "Different place, do TensorCopy";
  }
}

static std::string GetTypeName(const imperative::VarBase &var) {
  if (var.Type() == framework::proto::VarType::RAW) {
    return "RAW";
  } else if (!var.Var().IsInitialized()) {
    return "nullptr";
  } else {
    return framework::ToTypeName(var.Var().Type());
  }
}

using PyNameVarBaseMap = std::unordered_map<std::string, py::handle>;

template <typename T>
static T PyObjectCast(PyObject *obj) {
  try {
    return py::cast<T>(py::handle(obj));
  } catch (py::cast_error &) {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "Python object is not type of %s", typeid(T).name()));
  }
}

// NOTE(zjl): py::handle is a very light wrapper of PyObject *.
// Unlike py::object, py::handle does not change reference count of PyObject *.
static std::vector<std::shared_ptr<imperative::VarBase>>
GetVarBaseListFromPyHandle(const py::handle &handle) {
  PyObject *py_obj = handle.ptr();  // get underlying PyObject
  // Python None is not nullptr in C++!
  if (!py_obj || py_obj == Py_None) {
    return {};
  }

  std::vector<std::shared_ptr<imperative::VarBase>> result;

  if (PyList_Check(py_obj)) {  // List of VarBase
    size_t len = PyList_GET_SIZE(py_obj);
    result.reserve(len);
    for (size_t i = 0; i < len; ++i) {
      PyObject *py_ivar = PyList_GET_ITEM(py_obj, i);
      PADDLE_ENFORCE_NOT_NULL(
          py_ivar, platform::errors::InvalidArgument("Python Object is NULL"));
      result.emplace_back(
          PyObjectCast<std::shared_ptr<imperative::VarBase>>(py_ivar));
    }
  } else if (PyTuple_Check(py_obj)) {  // Tuple of VarBase
    size_t len = PyTuple_GET_SIZE(py_obj);
    result.reserve(len);
    for (size_t i = 0; i < len; ++i) {
      PyObject *py_ivar = PyTuple_GET_ITEM(py_obj, i);
      PADDLE_ENFORCE_NOT_NULL(
          py_ivar, platform::errors::InvalidArgument("Python Object is NULL"));
      result.emplace_back(
          PyObjectCast<std::shared_ptr<imperative::VarBase>>(py_ivar));
    }
  } else {  // VarBase
    result.emplace_back(
        PyObjectCast<std::shared_ptr<imperative::VarBase>>(py_obj));
  }

  return result;
}

static imperative::NameVarBaseMap ConvertToNameVarBaseMap(
    const PyNameVarBaseMap &map) {
  imperative::NameVarBaseMap result;
  for (auto &pair : map) {
    auto var_vec = GetVarBaseListFromPyHandle(pair.second);
    if (!var_vec.empty()) {
      result.emplace(pair.first, std::move(var_vec));
    }
  }

  PADDLE_ENFORCE_EQ(
      PyErr_Occurred(), nullptr,
      platform::errors::InvalidArgument(py::str(py::handle(PyErr_Occurred()))));
  return result;
}

static bool PyCheckInteger(PyObject *obj) {
#if PY_VERSION_HEX < 0x03000000
  return (PyLong_Check(obj) || PyInt_Check(obj)) && !PyBool_Check(obj);
#else
  return PyLong_Check(obj) && !PyBool_Check(obj);
#endif
}

// NOTE(zhiqiu): Revised version of PySlice_GetIndices. From:
// https://github.com/python/cpython/blob/8d21aa21f2cbc6d50aab3f420bb23be1d081dac4/Objects/sliceobject.c#L103
// Original PySlice_GetIndices return wrong result when
// slice_item contains long int, such as arr[:180L].
// NOT sure why this happens !!!
// Besides, PySlice_GetIndices cannot raise error when float in slice item.
// So, I make a revised version of PySlice_GetIndices, named to
// _PySlice_GetIndices. Try to use _PySlice_Unpack which is more robust than
// PySlice_GetIndices in the future.
static int _PySlice_GetIndices(PySliceObject *r, Py_ssize_t length,
                               Py_ssize_t *start, Py_ssize_t *stop,
                               Py_ssize_t *step) {
  /* XXX support long ints */
  if (r->step == Py_None) {
    *step = 1;
  } else {
    if (PyCheckInteger(r->step)) {
      *step = PyLong_AsLong(r->step);
    } else {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "Currently, VarBase.__getitem__() only allows None or integers in "
          "slice item, but received %s.",
          std::string(Py_TYPE(r->step)->tp_name)));
    }
  }
  if (r->start == Py_None) {
    *start = *step < 0 ? length - 1 : 0;
  } else {
    if (PyCheckInteger(r->start)) {
      *start = PyLong_AsLong(r->start);
    } else {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "Currently, VarBase.__getitem__() only allows None or integers in "
          "slice item, but received %s.",
          std::string(Py_TYPE(r->start)->tp_name)));
    }
    if (*start < 0) *start += length;
  }
  if (r->stop == Py_None) {
    *stop = *step < 0 ? -1 : length;
  } else {
    if (PyCheckInteger(r->stop)) {
      *stop = PyLong_AsLong(r->stop);
    } else {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "Currently, VarBase.__getitem__() only allows None or integers in "
          "slice item, but received %s.",
          std::string(Py_TYPE(r->stop)->tp_name)));
    }
    if (*stop < 0) *stop += length;
  }
  if (*stop > length) return -1;
  if (*start >= length) return -1;
  if (*step == 0) return -1;
  return 0;
}

static void ParseIndexingSlice(framework::LoDTensor *tensor, PyObject *_index,
                               std::vector<int> *slice_axes,
                               std::vector<int> *slice_starts,
                               std::vector<int> *slice_ends,
                               std::vector<int> *slice_strides,
                               std::vector<int> *decrease_axis,
                               std::vector<int> *infer_flags) {
  // We allow indexing by Integers, Slices, and tuples of those
  // types.
  // Ellipsis and None are not supported yet.
  // wrap to tuple
  PyObject *index = !PyTuple_Check(_index) ? PyTuple_Pack(1, _index) : _index;
  PADDLE_ENFORCE_EQ(
      tensor->IsInitialized(), true,
      platform::errors::InvalidArgument("tensor has not been initialized"));
  const auto &shape = tensor->dims();
  const int rank = shape.size();
  const int size = PyTuple_GET_SIZE(index);
  PADDLE_ENFORCE_EQ(
      size <= rank, true,
      platform::errors::InvalidArgument(
          "too many indices (%d) for tensor of dimension %d", size, rank));
  for (int dim = 0; dim < size; ++dim) {
    PyObject *slice_item = PyTuple_GetItem(index, dim);
    PADDLE_ENFORCE_EQ(PyCheckInteger(slice_item) || PySlice_Check(slice_item),
                      true,
                      platform::errors::InvalidArgument(
                          "Currently, VarBase.__getitem__() only allows "
                          "indexing by Integers, Slices, and tuples of "
                          "these types, but received %s in %dth slice item",
                          std::string(Py_TYPE(slice_item)->tp_name), dim + 1));
    infer_flags->push_back(1);
    int dim_len = shape[dim];
    if (PyCheckInteger(slice_item)) {
      // integer, PyLong_AsLong supports both int and long
      int start = static_cast<int>(PyLong_AsLong(slice_item));
      auto s_t = start;
      start = start < 0 ? start + dim_len : start;
      if (start >= dim_len) {
        std::string str_error_message =
            "The starting index " + std::to_string(s_t) +
            " of slice is out of bounds in tensor " + std::to_string(dim) +
            "-th axis, it shound be in the range of [" +
            std::to_string(-dim_len) + ", " + std::to_string(dim_len) + ")";
        // py::index_error is corresponding to IndexError in Python
        // Used to indicate out of bounds access in __getitem__, __setitem__
        throw py::index_error(str_error_message);
      }
      slice_axes->push_back(dim);
      slice_starts->push_back(start);
      slice_ends->push_back(start + 1);
      slice_strides->push_back(1);
      decrease_axis->push_back(dim);
    } else {
      // slice item
      Py_ssize_t start, end, step;
      PySliceObject *p = reinterpret_cast<PySliceObject *>(slice_item);
      _PySlice_GetIndices(p, dim_len, &start, &end, &step);

      // :: or : or 0:dim_len:1
      if (start == 0 && end == dim_len && step == 1) {
        continue;
      }
      slice_axes->push_back(dim);
      slice_starts->push_back(start);
      slice_ends->push_back(end);
      slice_strides->push_back(step);
    }
  }
  if (!PyTuple_Check(_index)) Py_DecRef(index);
}

// Bind Methods
void BindImperative(py::module *m_ptr) {
  auto &m = *m_ptr;

  BindOpFunctions(&m);

#ifndef _WIN32
  // Dygraph DataLoader signal handler
  m.def("_set_process_pids", [](int64_t key, py::object &obj) {
    PADDLE_ENFORCE_EQ(
        py::isinstance<py::tuple>(obj) || py::isinstance<py::list>(obj), true,
        platform::errors::InvalidArgument(
            "The subprocess ids set in DataLoader is illegal."
            "Expected data type is tuple or list, but received %s",
            obj.get_type()));
    py::list pids = py::cast<py::list>(obj);
    std::set<pid_t> pids_set = {};
    for (size_t i = 0; i < pids.size(); i++) {
      pids_set.insert(pids[i].cast<pid_t>());
    }
    imperative::SetLoadProcessPIDs(key, pids_set);
  });
  m.def("_erase_process_pids",
        [](int64_t key) { imperative::EraseLoadProcessPIDs(key); });
  m.def("_set_process_signal_handler",
        []() { imperative::SetLoadProcessSignalHandler(); });
  m.def("_throw_error_if_process_failed",
        []() { imperative::ThrowErrorIfLoadProcessFailed(); });

  // Dygraph DataLoader reader process & thread related functions
  m.def(
      "_convert_to_tensor_list",
      [](py::object &obj) -> py::list {
        // 0. input data check
        PADDLE_ENFORCE(
            py::isinstance<py::tuple>(obj) || py::isinstance<py::list>(obj),
            platform::errors::InvalidArgument(
                "The batch data read into DataLoader is illegal."
                "Expected data type is tuple or list, but received %s",
                obj.get_type()));
        py::list batch = py::cast<py::list>(obj);
        py::list tensors;
        for (size_t i = 0; i < batch.size(); ++i) {
          // 1. cast to python array
          auto array = batch[i].cast<py::array>();
          PADDLE_ENFORCE_NE(
              string::Sprintf("%s", array.dtype()).compare("object"), 0,
              platform::errors::InvalidArgument(
                  "Faild to convert input data to a regular ndarray.\n  * "
                  "Usually this means the input data contains nested "
                  "lists with different lengths.\n  * Check the reader "
                  "function passed to 'set_(sample/sample_list/batch)"
                  "_generator' to locate the data causes this issue."));
          // 2. construcct LoDTensor
          framework::LoDTensor t;
          SetTensorFromPyArray<platform::CPUPlace>(&t, array,
                                                   platform::CPUPlace(), true);
          // 3. allocate shared memory
          void *data_ptr = t.data<void>();
          size_t data_size = t.numel() * framework::SizeOfType(t.type());
          auto shared_writer_holder =
              memory::allocation::AllocateMemoryMapWriterAllocation(data_size);
          // 4. maintain mmap fd set & backup ipc_name
          const std::string &ipc_name = shared_writer_holder->ipc_name();
          memory::allocation::MemoryMapFdSet::Instance().Insert(ipc_name);
          // 5. copy data & reset holder
          memory::Copy(platform::CPUPlace(), shared_writer_holder->ptr(),
                       platform::CPUPlace(), data_ptr, data_size);
          t.ResetHolder(shared_writer_holder);
          // 6. append to result list
          tensors.append(t);
        }
        return tensors;
      },
      py::return_value_policy::take_ownership);

  m.def("_remove_tensor_list_mmap_fds", [](py::list &tensor_list) {
    for (size_t i = 0; i < tensor_list.size(); ++i) {
      auto t = tensor_list[i].cast<framework::LoDTensor>();
      auto *mmap_writer_allocation =
          dynamic_cast<memory::allocation::MemoryMapWriterAllocation *>(
              t.Holder().get());
      PADDLE_ENFORCE_NOT_NULL(
          mmap_writer_allocation,
          platform::errors::NotFound("The shared memory of LoDTensor in "
                                     "DataLoader's child process has been "
                                     "released."));
      memory::allocation::MemoryMapFdSet::Instance().Remove(
          mmap_writer_allocation->ipc_name());
    }
  });

  m.def("_cleanup_mmap_fds",
        []() { memory::allocation::MemoryMapFdSet::Instance().Clear(); });
#endif

  m.def("start_imperative_gperf_profiler",
        []() { imperative::StartProfile(); });

  m.def("stop_imperative_gperf_profiler", []() { imperative::StopProfile(); });

  m.def("_is_dygraph_debug_enabled",
        []() { return imperative::IsDebugEnabled(); });
  m.def("_dygraph_debug_level", []() { return imperative::GetDebugLevel(); });
  m.def("_switch_tracer",
        [](const std::shared_ptr<imperative::Tracer> &tracer) {
          imperative::SetCurrentTracer(tracer);
        });

  py::class_<imperative::VarBase, std::shared_ptr<imperative::VarBase>>(
      m, "VarBase", R"DOC()DOC")
      .def_static("_alive_vars", &imperative::VarBase::AliveVarNames)
      .def("__init__",
           [](imperative::VarBase &self, framework::proto::VarType::Type dtype,
              const std::vector<int> &dims, const py::handle &name,
              framework::proto::VarType::Type type, bool persistable) {
             VLOG(4) << "Init VarBase";
             std::string act_name = "";
             if (!name.ptr() || name.ptr() == Py_None) {
               act_name = imperative::GetCurrentTracer()->GenerateUniqueName(
                   "generated_tensor");
             } else {
               act_name = name.cast<std::string>();
             }
             new (&self) imperative::VarBase(act_name);
             self.SetPersistable(persistable);
             self.SetType(type);
             self.SetDataType(dtype);
             if (type == framework::proto::VarType::LOD_TENSOR) {
               auto *tensor =
                   self.MutableVar()->GetMutable<framework::LoDTensor>();
               tensor->Resize(framework::make_ddim(dims));
             }
           })
      .def("__init__", &InitVarBaseFromNumpyWithArg<platform::CPUPlace>,
           py::arg("value"), py::arg("place"), py::arg("persistable") = false,
           py::arg("zero_copy") = false, py::arg("name") = "",
           py::arg("stop_gradient") = -1)
      .def("__init__", &InitVarBaseFromNumpyWithArg<platform::XPUPlace>,
           py::arg("value"), py::arg("place"), py::arg("persistable") = false,
           py::arg("zero_copy") = false, py::arg("name") = "",
           py::arg("stop_gradient") = -1)
      .def("__init__", &InitVarBaseFromNumpyWithArg<platform::CUDAPlace>,
           py::arg("value"), py::arg("place"), py::arg("persistable") = false,
           py::arg("zero_copy") = false, py::arg("name") = "",
           py::arg("stop_gradient") = -1)
      .def("__init__", &InitVarBaseFromNumpyWithArg<platform::CUDAPinnedPlace>,
           py::arg("value"), py::arg("place"), py::arg("persistable") = false,
           py::arg("zero_copy") = false, py::arg("name") = "",
           py::arg("stop_gradient") = -1)
      .def("__init__", &InitVarBaseFromNumpyWithArgDefault, py::arg("value"))
      .def("__init__", &InitVarBaseFromTensorWithArgDefault, py::arg("tensor"))
      .def("__init__", &InitVarBaseFromNumpyWithKwargs)
      .def("__setitem__",
           [](std::shared_ptr<imperative::VarBase> &self, py::handle _index,
              py::object &value_obj) {
             auto self_tensor =
                 self->MutableVar()->GetMutable<framework::LoDTensor>();
             auto self_numpy = TensorToPyArray(*self_tensor);

             if (py::isinstance<py::array>(value_obj) ||
                 py::isinstance<py::int_>(value_obj) ||
                 py::isinstance<py::float_>(value_obj)) {
               auto value_numpy = value_obj;
               self_numpy[_index] = value_numpy;
               SetTensorFromPyArray(self_tensor, self_numpy,
                                    self_tensor->place(), true);

             } else {
               auto value =
                   value_obj.cast<std::shared_ptr<imperative::VarBase>>();
               auto value_tensor =
                   value->MutableVar()->GetMutable<framework::LoDTensor>();
               auto value_numpy = TensorToPyArray(*value_tensor);

               self_numpy[_index] = value_numpy;
               SetTensorFromPyArray(self_tensor, self_numpy,
                                    self_tensor->place(), true);
             }
           })
      .def("__getitem__",
           [](std::shared_ptr<imperative::VarBase> &self, py::handle _index) {
             std::vector<int> slice_axes, slice_starts, slice_ends,
                 slice_strides, decrease_axis, infer_flags;
             auto tensor =
                 self->MutableVar()->GetMutable<framework::LoDTensor>();
             ParseIndexingSlice(tensor, _index.ptr(), &slice_axes,
                                &slice_starts, &slice_ends, &slice_strides,
                                &decrease_axis, &infer_flags);
             // release gil and do tracing
             py::gil_scoped_release release;
             const auto &tracer = imperative::GetCurrentTracer();
             if (slice_axes.empty()) {
               return self;
             } else {
               imperative::NameVarBaseMap ins = {{"Input", {self}}};
               framework::AttributeMap attrs = {
                   {"axes", slice_axes},
                   {"starts", slice_starts},
                   {"ends", slice_ends},
                   {"infer_flags", infer_flags},
                   {"decrease_axis", decrease_axis}};
               auto out = std::shared_ptr<imperative::VarBase>(
                   new imperative::VarBase(tracer->GenerateUniqueName()));
               imperative::NameVarBaseMap outs = {{"Out", {out}}};
               std::string op_type = "slice";
               for (auto stride : slice_strides) {
                 if (stride != 1) {
                   op_type = "strided_slice";
                   attrs.insert({"strides", slice_strides});
                   attrs.erase("decrease_axis");
                   break;
                 }
               }
               tracer->TraceOp(op_type, ins, outs, std::move(attrs));
               return out;
             }
           })
      .def("numpy",
           [](imperative::VarBase &self) -> py::array {
             const auto &tensor =
                 self.MutableVar()->Get<framework::LoDTensor>();
             PADDLE_ENFORCE_EQ(
                 tensor.IsInitialized(), true,
                 platform::errors::InvalidArgument(
                     "Tensor of %s is Empty, please check if it has no data.",
                     self.Name()));
             return TensorToPyArray(tensor, true);
           },
           R"DOC(
        **Notes**:
            **This API is ONLY available in Dygraph mode**

        Returns a numpy array shows the value of current :ref:`api_guide_Variable_en`

        Returns:
            ndarray: The numpy value of current Variable.

        Returns type:
            ndarray: dtype is same as current Variable

        Examples:
            .. code-block:: python

                import paddle.fluid as fluid
                from paddle.fluid.dygraph.base import to_variable
                from paddle.fluid.dygraph import Linear
                import numpy as np

                data = np.random.uniform(-1, 1, [30, 10, 32]).astype('float32')
                with fluid.dygraph.guard():
                    linear = Linear(32, 64)
                    data = to_variable(data)
                    x = linear(data)
                    print(x.numpy())

       )DOC")
      .def("detach",
           [](const imperative::VarBase &self) {
             const auto &tensor = self.Var().Get<framework::LoDTensor>();
             PADDLE_ENFORCE_EQ(tensor.IsInitialized(), true,
                               platform::errors::InvalidArgument(
                                   "%s has not been initialized", self.Name()));
             return self.NewVarBase(tensor.place(), false);
           },
           py::return_value_policy::copy, R"DOC(

        Returns a new Tensor, detached from the current graph.

        Returns: The detached Tensor.

        Examples:
            .. code-block:: python

                import paddle
                paddle.disable_static()

                linear = Linear(32, 64)
                data = paddle.uniform(shape=[30, 10, 32], -1, 1)
                x = linear(data)
                y = x.detach()
       )DOC")
      .def("clear_gradient", &imperative::VarBase::ClearGradient, R"DOC(

        Only for Tensor that has gradient, normally we use this for Parameters since other temporary Tensor doesen't has gradient.

        The Gradient of current Tensor will be set to ``0`` .

        Returns:  None

        Examples:
             .. code-block:: python

                import paddle
                paddle.disable_static()

                inputs = []
                for _ in range(10):
                    tmp = paddle.ones([2, 2])
                    tmp.stop_gradient=False
                    inputs.append(tmp)
                ret = paddle.sums(inputs2)
                loss = paddle.fluid.layers.reduce_sum(ret)
                loss.backward()
                print("Before clear_gradient {}".format(loss.grad))
                loss.clear_gradient()
                print("After clear_gradient {}".format(loss.grad))
      )DOC")
      .def("clone",
           [](std::shared_ptr<imperative::VarBase> &self) {
             const auto &tensor = self->Var().Get<framework::LoDTensor>();
             PADDLE_ENFORCE_EQ(
                 tensor.IsInitialized(), true,
                 platform::errors::InvalidArgument(
                     "%s has not been initialized", self->Name()));
             auto tracer = imperative::GetCurrentTracer();
             auto new_var = std::make_shared<imperative::VarBase>(
                 true, tracer->GenerateUniqueName(self->Name() + "_clone"));
             framework::AttributeMap attrs;
             imperative::NameVarBaseMap ins = {{"X", {self}}};
             imperative::NameVarBaseMap outs = {{"Out", {new_var}}};
             tracer->TraceOp("assign", ins, outs, attrs);
             return new_var;
           },
           py::return_value_policy::copy, R"DOC(

        Returns a new Tensor, which is clone of origin Tensor, and it remains in the current graph.
        It will always have a Tensor copy.
        Tn addition, the cloned Tensor provides gradient propagation.

        Returns: The cloned Tensor.

        Examples:
            .. code-block:: python

              import paddle

              x = paddle.to_tensor(1.0, stop_gradient=False)
              clone_x = x.clone()
              y = clone_x**2
              y.backward()
              print(clone_x.stop_gradient) # False
              print(clone_x.grad)          # [2.0], support gradient propagation
              print(x.stop_gradient)       # False
              print(x.grad)                # [2.0], clone_x support gradient propagation for x

              x = paddle.to_tensor(1.0)
              clone_x = x.clone()
              clone_x.stop_gradient = False
              z = clone_x**3
              z.backward()
              print(clone_x.stop_gradient) # False
              print(clone_x.grad)          # [3.0], support gradient propagation
              print(x.stop_gradient) # True
              print(x.grad)          # None
       )DOC")
      .def("_run_backward",
           [](imperative::VarBase &self, const imperative::Tracer &tracer,
              bool retain_graph) {
             // TODO(jiabin): when we impl more backward execution we can
             // select them
             auto *engine = tracer.GetEngine();
             engine->Init(&self, retain_graph);
             VLOG(3) << "Start backward";
             engine->Execute();
             VLOG(3) << "Finish backward";
           },
           py::call_guard<py::gil_scoped_release>())
      .def("_grad_name", &imperative::VarBase::GradVarName)
      .def("_grad_value",
           [](imperative::VarBase &self) {
             return self.MutableGradVar()->Get<framework::LoDTensor>();
           },
           py::return_value_policy::reference)
      .def("_set_grad_type",
           [](imperative::VarBase &self, framework::proto::VarType::Type type) {
             self.MutableGradVarBase()->SetType(type);
           })
      .def("_grad_ivar",
           [](const imperative::VarBase &self) {
             auto &grad_var = self.GradVarBase();
             if (grad_var && grad_var->Var().IsInitialized()) {
               auto *tensor =
                   grad_var->MutableVar()->IsType<framework::LoDTensor>()
                       ? grad_var->MutableVar()
                             ->GetMutable<framework::LoDTensor>()
                       : grad_var->MutableVar()
                             ->GetMutable<framework::SelectedRows>()
                             ->mutable_value();
               if (tensor->IsInitialized()) {
                 return grad_var;
               }
             }
             return std::shared_ptr<imperative::VarBase>(nullptr);
           },
           py::return_value_policy::copy)
      .def("_is_sparse",
           [](imperative::VarBase &self) {
             return self.Var().IsType<framework::SelectedRows>();
           })
      .def("_allreduce",
           [](imperative::VarBase &self,
              const imperative::ParallelStrategy &strategy) {
             if (strategy.nranks_ > 1) {
#ifdef PADDLE_WITH_NCCL
#if NCCL_VERSION_CODE >= 2212
               imperative::AllReduce(self.Var(), self.MutableVar(), strategy);
#else
               if (!self.Var().IsType<framework::SelectedRows>()) {
                 imperative::AllReduce(self.Var(), self.MutableVar(), strategy);
               } else {
                 PADDLE_THROW(platform::errors::Unimplemented(
                     "Imperative SelectedRows allreduce is not supported when "
                     "paddle is compiled with NCCL verison lower than v2.2.12. "
                     "You can set is_sparse=False for the Layer containing "
                     "this argument, such as Embedding(is_sparse=False)."));
               }
#endif  // NCCL_VERSION_CODE
#else
               PADDLE_THROW(platform::errors::Unimplemented(
                   "Imperative allreduce is not supported when paddle is "
                   "not compiled with NCCL."));
#endif  // PADDLE_WITH_NCCL
             }
           },
           py::call_guard<py::gil_scoped_release>())
      .def("_copy_to",
           [](const imperative::VarBase &self, const platform::CPUPlace &place,
              bool blocking) { return self.NewVarBase(place, blocking); },
           py::return_value_policy::copy)
      .def("_copy_to",
           [](const imperative::VarBase &self,
              const platform::CUDAPinnedPlace &place,
              bool blocking) { return self.NewVarBase(place, blocking); },
           py::return_value_policy::copy)
      .def("_copy_to",
           [](const imperative::VarBase &self, const platform::XPUPlace &place,
              bool blocking) { return self.NewVarBase(place, blocking); },
           py::return_value_policy::copy)
      .def("_copy_to",
           [](const imperative::VarBase &self, const platform::CUDAPlace &place,
              bool blocking) { return self.NewVarBase(place, blocking); },
           py::return_value_policy::copy)
      .def("value", [](imperative::VarBase &self) { return self.MutableVar(); },
           py::return_value_policy::reference)
      .def_property("name", &imperative::VarBase::Name,
                    &imperative::VarBase::SetName)
      .def_property("stop_gradient",
                    &imperative::VarBase::OverridedStopGradient,
                    &imperative::VarBase::SetOverridedStopGradient)
      .def_property("persistable", &imperative::VarBase::Persistable,
                    &imperative::VarBase::SetPersistable)
      .def_property_readonly(
          "shape",
          [](imperative::VarBase &self) {
            if (self.Var().IsType<framework::LoDTensor>()) {
              return framework::vectorize<int>(
                  self.Var().Get<framework::LoDTensor>().dims());
            } else if (self.Var().IsType<framework::SelectedRows>()) {
              return framework::vectorize<int>(
                  self.Var().Get<framework::SelectedRows>().value().dims());
            } else {
              VLOG(2) << "It is meaningless to get shape of "
                         "variable type "
                      << GetTypeName(self);
              return std::vector<int>();
            }
          })
      .def_property_readonly(
          "place", [](imperative::VarBase &self) { return self.Place(); },
          py::return_value_policy::copy)
      .def_property_readonly("_place_str",
                             [](imperative::VarBase &self) {
                               std::stringstream ostr;
                               ostr << self.Place();
                               return ostr.str();
                             })
      .def_property_readonly("type", &imperative::VarBase::Type)
      .def_property_readonly("dtype", &imperative::VarBase::DataType);

  py::class_<imperative::Layer, Layer /* <--- trampoline*/> layer(m, "Layer");
  layer.def(py::init<>())
      .def("forward",
           [](imperative::Layer &self,
              const std::vector<std::shared_ptr<imperative::VarBase>> &inputs) {
             return self.Forward(inputs);
           });

  py::class_<imperative::jit::ProgramDescTracer>(m, "ProgramDescTracer", "")
      .def("create_program_desc",
           &imperative::jit::ProgramDescTracer::CreateProgramDesc)
      .def("reset", &imperative::jit::ProgramDescTracer::Reset);

  py::class_<imperative::Tracer, std::shared_ptr<imperative::Tracer>>(
      m, "Tracer", R"DOC()DOC")
      .def("__init__",
           [](imperative::Tracer &self) { new (&self) imperative::Tracer(); })
      .def_property("_enable_program_desc_tracing",
                    &imperative::Tracer::IsProgramDescTracingEnabled,
                    &imperative::Tracer::SetEnableProgramDescTracing)
      .def_property("_enable_autocast", &imperative::Tracer::IsAutoCastEnabled,
                    &imperative::Tracer::SetEnableAutoCast)
      .def_property("_train_mode", &imperative::Tracer::HasGrad,
                    &imperative::Tracer::SetHasGrad)
      .def_property(
          "_expected_place",
          [](const imperative::Tracer &self) -> py::object {
            return py::cast(self.ExpectedPlace());
          },
          [](imperative::Tracer &self, const py::object &obj) {
            if (py::isinstance<platform::CUDAPlace>(obj)) {
              auto p = obj.cast<platform::CUDAPlace *>();
              self.SetExpectedPlace(*p);
            } else if (py::isinstance<platform::XPUPlace>(obj)) {
              auto p = obj.cast<platform::XPUPlace *>();
              self.SetExpectedPlace(*p);
            } else if (py::isinstance<platform::CPUPlace>(obj)) {
              auto p = obj.cast<platform::CPUPlace *>();
              self.SetExpectedPlace(*p);
            } else if (py::isinstance<platform::CUDAPinnedPlace>(obj)) {
              auto p = obj.cast<platform::CUDAPinnedPlace *>();
              self.SetExpectedPlace(*p);
            } else {
              PADDLE_THROW(platform::errors::InvalidArgument(
                  "Incompatible Place Type: supports XPUPlace, CUDAPlace, "
                  "CPUPlace, "
                  "and CUDAPinnedPlace, "
                  "but got Unknown Type!"));
            }
          })
      .def("_get_program_desc_tracer",
           &imperative::Tracer::GetProgramDescTracer,
           py::return_value_policy::reference)
      .def("_generate_unique_name", &imperative::Tracer::GenerateUniqueName,
           py::arg("key") = "dygraph_tmp")
      .def(
          "_set_amp_op_list",
          [](imperative::Tracer &self,
             std::unordered_set<std::string> &allow_ops,
             std::unordered_set<std::string> &block_ops) {
            // NOTE(zhiqiu): The automatic conversion in pybind11 between c++
            // STL and python set/list/dict involve a copy operation that
            // prevents pass-by-reference semantics, so it is ok to swap.
            // The reaseon why not directly pass
            // std::shared_ptr<std::unordered_set<std::string>>
            // is that pybind11 forbid shared_ptr<T> where T is not custom type.
            imperative::AmpOperators::Instance().GetAllowOps()->swap(allow_ops);
            imperative::AmpOperators::Instance().GetBlockOps()->swap(block_ops);
          })
      .def("_get_amp_op_list",
           [](imperative::Tracer &self) {
             return std::make_tuple(
                 *(imperative::AmpOperators::Instance().GetAllowOps()),
                 *(imperative::AmpOperators::Instance().GetBlockOps()));
           })
      .def("trace",
           [](imperative::Tracer &self, const std::string &type,
              const PyNameVarBaseMap &ins, const PyNameVarBaseMap &outs,
              framework::AttributeMap attrs, const platform::XPUPlace &place,
              bool trace_backward) {
             auto ins_map = ConvertToNameVarBaseMap(ins);
             auto outs_map = ConvertToNameVarBaseMap(outs);
             {
               py::gil_scoped_release release;
               self.TraceOp(type, std::move(ins_map), std::move(outs_map),
                            std::move(attrs), place, trace_backward);
             }
           })
      .def("trace",
           [](imperative::Tracer &self, const std::string &type,
              const PyNameVarBaseMap &ins, const PyNameVarBaseMap &outs,
              framework::AttributeMap attrs, const platform::CUDAPlace &place,
              bool trace_backward) {
             auto ins_map = ConvertToNameVarBaseMap(ins);
             auto outs_map = ConvertToNameVarBaseMap(outs);
             {
               py::gil_scoped_release release;
               self.TraceOp(type, std::move(ins_map), std::move(outs_map),
                            std::move(attrs), place, trace_backward);
             }
           })
      .def("trace",
           [](imperative::Tracer &self, const std::string &type,
              const PyNameVarBaseMap &ins, const PyNameVarBaseMap &outs,
              framework::AttributeMap attrs, const platform::CPUPlace &place,
              bool trace_backward) {
             auto ins_map = ConvertToNameVarBaseMap(ins);
             auto outs_map = ConvertToNameVarBaseMap(outs);
             {
               py::gil_scoped_release release;
               self.TraceOp(type, std::move(ins_map), std::move(outs_map),
                            std::move(attrs), place, trace_backward);
             }
           });

  // define parallel context
  py::class_<imperative::ParallelStrategy> parallel_strategy(
      m, "ParallelStrategy", "");
  parallel_strategy.def(py::init())
      .def_property(
          "nranks",
          [](const imperative::ParallelStrategy &self) { return self.nranks_; },
          [](imperative::ParallelStrategy &self, int nranks) {
            self.nranks_ = nranks;
          })
      .def_property("local_rank",
                    [](const imperative::ParallelStrategy &self) {
                      return self.local_rank_;
                    },
                    [](imperative::ParallelStrategy &self, int local_rank) {
                      self.local_rank_ = local_rank;
                    })
      .def_property(
          "trainer_endpoints",
          [](const imperative::ParallelStrategy &self) {
            return self.trainer_endpoints_;
          },
          [](imperative::ParallelStrategy &self, std::vector<std::string> eps) {
            self.trainer_endpoints_ = eps;
          })
      .def_property("current_endpoint",
                    [](const imperative::ParallelStrategy &self) {
                      return self.current_endpoint_;
                    },
                    [](imperative::ParallelStrategy &self,
                       const std::string &ep) { self.current_endpoint_ = ep; });

  m.def(
      "dygraph_partial_grad",
      [](const std::vector<std::shared_ptr<imperative::VarBase>> &input_targets,
         const std::vector<std::shared_ptr<imperative::VarBase>>
             &output_targets,
         const std::vector<std::shared_ptr<imperative::VarBase>> &output_grads,
         const std::vector<std::shared_ptr<imperative::VarBase>> &no_grad_vars,
         const platform::Place &place, bool create_graph, bool retain_graph,
         bool allow_unused, bool only_inputs) {
        imperative::PartialGradEngine engine(
            input_targets, output_targets, output_grads, no_grad_vars, place,
            create_graph, retain_graph, allow_unused, only_inputs);
        engine.Execute();
        return engine.GetResult();
      },
      py::call_guard<py::gil_scoped_release>());

#if defined(PADDLE_WITH_NCCL)
  py::class_<imperative::NCCLParallelContext> nccl_ctx(m,
                                                       "NCCLParallelContext");

  nccl_ctx
      .def(py::init<const imperative::ParallelStrategy &,
                    const platform::CUDAPlace &>())
      .def("init", [](imperative::NCCLParallelContext &self) { self.Init(); });
#endif
}

}  // namespace pybind
}  // namespace paddle
