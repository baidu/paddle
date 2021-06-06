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

#include <algorithm>
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
#include "paddle/fluid/imperative/bkcl_context.h"
#include "paddle/fluid/imperative/data_loader.h"
#include "paddle/fluid/imperative/hooks.h"
#include "paddle/fluid/imperative/layer.h"
#include "paddle/fluid/imperative/nccl_context.h"
#include "paddle/fluid/imperative/partial_grad_engine.h"
#include "paddle/fluid/imperative/profiler.h"
#include "paddle/fluid/imperative/py_layer_fwd.h"
#include "paddle/fluid/imperative/reducer.h"
#include "paddle/fluid/imperative/tracer.h"
#include "paddle/fluid/imperative/type_defs.h"
#include "paddle/fluid/memory/allocation/mmap_allocator.h"
#include "paddle/fluid/pybind/op_function.h"
#include "paddle/fluid/pybind/pybind_boost_headers.h"
#include "paddle/fluid/pybind/tensor_py.h"

extern uint64_t start_time_python;
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

template <typename T>
static T PyObjectCast(PyObject *obj) {
  try {
    return py::cast<T>(py::handle(obj));
  } catch (py::cast_error &) {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "Python object is not type of %s", typeid(T).name()));
  }
}

class PyVariableWrapperHook : public imperative::VariableWrapperHook {
 public:
  explicit PyVariableWrapperHook(PyObject *func) : py_func_(func) {
    Py_INCREF(py_func_);
  }

  ~PyVariableWrapperHook() {
    py::gil_scoped_acquire gil;
    Py_DECREF(py_func_);
  }

  std::shared_ptr<imperative::VariableWrapper> operator()(
      const std::shared_ptr<imperative::VariableWrapper> &var) override {
    py::gil_scoped_acquire gil;
    VLOG(3) << "Call PyVariableWrapperHook for var " << var->Name();

    // 1. unpack temp VarBase from VariableWrapper
    std::shared_ptr<imperative::VarBase> tmp_varbase =
        std::make_shared<imperative::VarBase>(var);

    // 2. call hook and return
    PyObject *res = nullptr;
    try {
      res = PyObject_CallFunctionObjArgs(py_func_, py::cast(tmp_varbase).ptr(),
                                         nullptr);
    } catch (platform::EnforceNotMet &e) {
      throw std::move(e);
    } catch (std::exception &e) {
      PADDLE_THROW(platform::errors::Unavailable(
          "Hook function of Tensor raises an exception: %s.", e.what()));
    } catch (...) {
      PADDLE_THROW(platform::errors::Fatal(
          "Hook function of Tensor raises an unknown exception."));
    }

    PADDLE_ENFORCE_NOT_NULL(res,
                            platform::errors::Unavailable(
                                "Hook function of Tensor return a nullptr."));
    if (res == Py_None) {
      return var;
    }

    return PyObjectCast<std::shared_ptr<imperative::VarBase>>(res)->SharedVar();
  }

 private:
  PyObject *py_func_;
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
  } else if (py::isinstance<platform::Place>(place_obj)) {
    return place_obj.cast<platform::Place>();
  } else {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "Place should be one of "
        "Place/CPUPlace/XPUPlace/CUDAPlace/CUDAPinnedPlace"));
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
          << " / stop_gradient: " << stop_gradient << " / at " << place;
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
  auto place = imperative::GetCurrentTracer()->ExpectedPlace();
  VLOG(4) << "Init VarBase from numpy at " << place;
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
    *start = std::max(*start, static_cast<Py_ssize_t>(0));
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
    *stop = std::min(*stop, length);
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
      if (start >= dim_len || start < 0) {
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

template <typename P>
static void VarBaseCopy(std::shared_ptr<imperative::VarBase> &src,
                        imperative::VarBase &dst, const P &dst_device,
                        const bool blocking) {
  if (dst.SharedVar()->IsEmpty()) {
    VLOG(3) << "deep copy Variable from " << src->Name() << " to "
            << dst.Name();
    dst.SetPersistable(src->Persistable());
    dst.SetDataType(src->DataType());
    dst.SetType(src->Type());
    dst.SetOverridedStopGradient(src->OverridedStopGradient());
    if (!src->SharedVar()->IsEmpty()) {
      if (src->Var().IsType<framework::LoDTensor>()) {
        auto &src_tensor = src->Var().Get<framework::LoDTensor>();
        auto *dst_tensor = dst.MutableVar()->GetMutable<framework::LoDTensor>();
        dst_tensor->set_lod(src_tensor.lod());
        framework::TensorCopy(src_tensor, dst_device, dst_tensor);
        if (blocking) {
          platform::DeviceContextPool::Instance().Get(dst_device)->Wait();
          auto src_device = src_tensor.place();
          if (!(src_device == dst_device)) {
            platform::DeviceContextPool::Instance().Get(src_device)->Wait();
          }
        }
      } else if (src->Var().IsType<framework::SelectedRows>()) {
        auto &src_selected_rows = src->Var().Get<framework::SelectedRows>();
        auto *dst_selected_rows =
            dst.MutableVar()->GetMutable<framework::SelectedRows>();
        dst_selected_rows->set_height(src_selected_rows.height());
        dst_selected_rows->set_rows(src_selected_rows.rows());
        framework::TensorCopy(src_selected_rows.value(), dst_device,
                              dst_selected_rows->mutable_value());
        if (blocking) {
          platform::DeviceContextPool::Instance().Get(dst_device)->Wait();
          auto src_device = src_selected_rows.value().place();
          if (!(src_device == dst_device)) {
            platform::DeviceContextPool::Instance().Get(src_device)->Wait();
          }
        }
      }

      if (!blocking) {
        IncreaseVarbaseReferenceCountUntilCopyComplete(src, dst_device);
      }

    } else {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "The source Tensor(%s) can not copy when it is empty.", src->Name()));
    }
  } else {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "The destion Tensor(%s) can not copy when it is not empty.",
        dst.Name()));
  }
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

  m.def("_array_to_share_memory_tensor",
        [](py::object &obj) {
          // 1. cast to python array
          auto array = obj.cast<py::array>();
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

          return t;
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
           [](imperative::VarBase &self) {
             std::string name =
                 imperative::GetCurrentTracer()->GenerateUniqueName(
                     "generated_tensor");
             new (&self) imperative::VarBase(name);
           })
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
             PyObject *index_ptr = !PyTuple_Check(_index.ptr())
                                       ? PyTuple_Pack(1, _index.ptr())
                                       : _index.ptr();
             // 1. Check argumnets
             // 1.1 Check whether value obj is a tensor.
             bool value_is_tensor = true;
             bool parse_index = true;
             if (py::isinstance<py::array>(value_obj) ||
                 py::isinstance<py::int_>(value_obj) ||
                 py::isinstance<py::float_>(value_obj)) {
               value_is_tensor = false;
             }

             // 1.2 Check whether _index can be parsed.
             const int size = PyTuple_GET_SIZE(index_ptr);
             for (int dim = 0; dim < size; ++dim) {
               PyObject *slice_item = PyTuple_GetItem(index_ptr, dim);
               if (!(PyCheckInteger(slice_item) || PySlice_Check(slice_item))) {
                 parse_index = false;
                 break;
               }
             }

             // 2. Call op set_value to speed up if the condition is met,
             // otherwise call TensorToPyArray.
             // TODO(liym27): Try not to call TensorToPyArray because it always
             // copys data to cpu place, which reduces performance.
             if (parse_index && value_is_tensor) {
               std::vector<int> axes, starts, ends, steps, decrease_axes,
                   infer_flags;
               ParseIndexingSlice(self_tensor, index_ptr, &axes, &starts, &ends,
                                  &steps, &decrease_axes, &infer_flags);

               framework::AttributeMap attrs = {
                   {"axes", axes},
                   {"starts", starts},
                   {"ends", ends},
                   {"steps", steps},
                   {"decrease_axes", decrease_axes}};

               imperative::NameVarBaseMap ins = {{"Input", {self}}};
               imperative::NameVarBaseMap outs = {{"Out", {self}}};

               PADDLE_ENFORCE_EQ(
                   self->IsLeaf() && !self->OverridedStopGradient(), false,
                   platform::errors::InvalidArgument(
                       "Leaf Tensor (%s) that doesn't stop gradient can't use "
                       "inplace strategy.",
                       self->Name()));

               auto value_tensor =
                   value_obj.cast<std::shared_ptr<imperative::VarBase>>();
               ins.insert({"ValueTensor", {value_tensor}});

               const auto &tracer = imperative::GetCurrentTracer();
               {
                 // Release gil and do tracing
                 py::gil_scoped_release release;
                 tracer->TraceOp("set_value", ins, outs, std::move(attrs),
                                 {{"Input", "Out"}});
               }
             } else {
               auto self_numpy = TensorToPyArray(*self_tensor);

               if (value_is_tensor) {
                 auto value =
                     value_obj.cast<std::shared_ptr<imperative::VarBase>>();
                 auto value_tensor =
                     value->MutableVar()->GetMutable<framework::LoDTensor>();
                 auto value_numpy = TensorToPyArray(*value_tensor);

                 self_numpy[_index] = value_numpy;
                 SetTensorFromPyArray(self_tensor, self_numpy,
                                      self_tensor->place(), true);
               } else {
                 auto value_numpy = value_obj;
                 self_numpy[_index] = value_numpy;
                 SetTensorFromPyArray(self_tensor, self_numpy,
                                      self_tensor->place(), true);
               }
             }
             // NOTE(liym27):
             // Increase the version of VarBase self because __setitem__ is an
             // inplace operator for the VarBase self.
             self->BumpInplaceVersion();
           })
      .def("_getitem_index_not_tensor",
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
      .def(
          "_getitem_from_offset",
          [](std::shared_ptr<imperative::VarBase> &self, const py::args &args) {
            const auto &tensor = self->Var().Get<framework::LoDTensor>();
            PADDLE_ENFORCE_EQ(
                tensor.IsInitialized(), true,
                platform::errors::InvalidArgument(
                    "Tensor of %s is Empty, please check if it has no data.",
                    self->Name()));

            const auto &tensor_dims = tensor.dims();

            std::vector<size_t> dims(tensor_dims.size());
            std::vector<size_t> strides(tensor_dims.size());

            size_t numel = 1;
            for (int i = tensor_dims.size() - 1; i >= 0; --i) {
              strides[i] = numel;
              dims[i] = static_cast<size_t>(tensor_dims[i]);
              numel *= dims[i];
            }
            size_t offset = 0;
            if (args.empty()) {
              PADDLE_ENFORCE_EQ(
                  numel, 1,
                  platform::errors::InvalidArgument(
                      "only one element tensors can be converted to Python "
                      "scalars when no input coordinates"));
            } else if (args.size() == 1) {
              offset = args[0].cast<size_t>();
              PADDLE_ENFORCE_LT(
                  offset, numel,
                  platform::errors::InvalidArgument(
                      "index %d is out of bounds for size %d", offset, numel));
            } else {
              PADDLE_ENFORCE_EQ(args.size(), dims.size(),
                                platform::errors::InvalidArgument(
                                    "incorrect number of indices for Tensor"));

              for (size_t i = 0; i < args.size(); ++i) {
                size_t index = args[i].cast<size_t>();
                PADDLE_ENFORCE_LT(
                    index, dims[i],
                    platform::errors::InvalidArgument(
                        "index %d is out fo bounds for axis %d with size %d",
                        index, i, dims[i]));
                offset += index * strides[i];
              }
            }
#define TENSOR_TO_PY_SCALAR(T, proto_type)                                   \
  if (tensor.type() == proto_type) {                                         \
    std::string py_dtype_str = details::TensorDTypeToPyDTypeStr(proto_type); \
    T b = TensorGetElement<T>(tensor, offset);                               \
    return py::array(py::dtype(py_dtype_str.c_str()), {}, {},                \
                     static_cast<void *>(&b));                               \
  }

            _ForEachDataType_(TENSOR_TO_PY_SCALAR);
#undef TENSOR_TO_PY_SCALAR
            PADDLE_THROW(platform::errors::Unimplemented(
                "Unsupported tensor data type: %s",
                framework::DataTypeToString(tensor.type())));
          },
          py::return_value_policy::copy)
      .def("_inplace_version",
           [](imperative::VarBase &self) -> uint32_t {
             const auto &var = self.MutableVar();
             PADDLE_ENFORCE_EQ(
                 var->IsInitialized(), true,
                 platform::errors::InvalidArgument(
                     "Tensor of %s is Empty, please check if it has no data.",
                     self.Name()));
             return var->CurrentInplaceVersion();
           })
      .def("_bump_inplace_version",
           [](std::shared_ptr<imperative::VarBase> &self) {
             // NOTE(liym27): _bump_inplace_version is only used for inplace
             // operation
             self->BumpInplaceVersion();
           },
           R"DOC(
        **Notes**:
            **This API is ONLY available in Dygraph mode.**
            **This is a very low level API. Users should not use it directly. **
         Bump the version whenever the Tensor is modified through an inplace operation.
            )DOC")
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
        Returns a numpy array shows the value of current Tensor.
        
        Returns:
            ndarray: The numpy value of current Tensor.

        Returns type:
            ndarray: dtype is same as current Tensor

        Examples:
            .. code-block:: python

                import paddle
                import numpy as np
                data = np.random.uniform(-1, 1, [30, 10, 32]).astype('float32')
                linear = paddle.nn.Linear(32, 64)
                data = paddle.to_tensor(data)
                x = linear(data)
                print(x.numpy())
       )DOC")
      .def("detach",
           [](const imperative::VarBase
                  &self) -> std::shared_ptr<imperative::VarBase> {
             PADDLE_ENFORCE_EQ(
                 self.Var().IsInitialized(), true,
                 platform::errors::InvalidArgument(
                     "Tensor %s has not been initialized!", self.Name()));

             PADDLE_ENFORCE_EQ(
                 self.Var().IsType<framework::LoDTensor>() ||
                     self.Var().IsType<framework::SelectedRows>(),
                 true,
                 platform::errors::InvalidArgument(
                     "Type of Tensor[%s] must be LoDTensor or SelectedRows!",
                     self.Name()));

             auto detach_var = std::make_shared<imperative::VarBase>(
                 true, "detach_" + self.Name());

             detach_var->SetPersistable(self.Persistable());
             detach_var->SetType(self.Type());
             detach_var->SetDataType(self.DataType());

             if (self.Var().IsType<framework::LoDTensor>()) {
               const auto &origin_tensor =
                   self.Var().Get<framework::LoDTensor>();
               PADDLE_ENFORCE_EQ(
                   origin_tensor.IsInitialized(), true,
                   platform::errors::InvalidArgument(
                       "Tensor %s has not been initialized!", self.Name()));

               auto *detach_tensor =
                   detach_var->MutableVar()->GetMutable<framework::LoDTensor>();
               detach_tensor->ShareDataWith(origin_tensor);
               // NOTE(liym27): Call ShareInplaceVersionCounterWith to share the
               // same TensorInplaceVersion, which is used to check whether
               // inplace
               // operations are correct.
               detach_tensor->ShareInplaceVersionCounterWith(origin_tensor);
             } else {
               const auto &origin_selected_rows =
                   self.Var().Get<framework::SelectedRows>();
               PADDLE_ENFORCE_EQ(
                   origin_selected_rows.value().IsInitialized(), true,
                   platform::errors::InvalidArgument(
                       "Tensor %s has not been initialized!", self.Name()));

               auto *detach_selected_rows =
                   detach_var->MutableVar()
                       ->GetMutable<framework::SelectedRows>();
               detach_selected_rows->set_height(origin_selected_rows.height());
               detach_selected_rows->set_rows(origin_selected_rows.rows());
               detach_selected_rows->mutable_value()->ShareDataWith(
                   origin_selected_rows.value());
               detach_selected_rows->mutable_value()
                   ->ShareInplaceVersionCounterWith(
                       origin_selected_rows.value());
             }
             VLOG(3) << "The detached Tensor(" << detach_var->Name()
                     << ") share data with " << self.Name();
             return detach_var;
           },
           py::return_value_policy::take_ownership, R"DOC(

        Returns a new Tensor, detached from the current graph.
        It will share data with origin Tensor and always doesn't have a Tensor copy.
        In addition, the detached Tensor doesn't provide gradient propagation.

        Returns: The detached Tensor.

        Examples:
            .. code-block:: python

                import paddle

                x = paddle.to_tensor(1.0, stop_gradient=False)
                detach_x = x.detach()
                detach_x[:] = 10.0
                print(x)  # Tensor(shape=[1], dtype=float32, place=CPUPlace, stop_gradient=False,
                          #        [10.])
                y = x**2
                y.backward()
                print(x.grad)         # [20.0]
                print(detach_x.grad)  # None, 'stop_gradient=True' by default

                detach_x.stop_gradient = False # Set stop_gradient to be False, supported auto-grad
                z = detach_x**3
                z.backward()

                print(x.grad)         # [20.0], detach_x is detached from x's graph, not affect each other
                print(detach_x.grad)  # [300.0], detach_x has its own graph

                # Due to sharing of data with origin Tensor, There are some unsafe operations:
                y = 2 * x
                detach_x[:] = 5.0
                y.backward() 
                # It will raise Error:
                #   one of the variables needed for gradient computation has been modified by an inplace operation.
             
       )DOC")
      .def("clear_gradient", &imperative::VarBase::ClearGradient, R"DOC(

        Only for Tensor that has gradient, normally we use this for Parameters since other temporary Tensor doesen't has gradient.

        The Gradient of current Tensor will be set to ``0`` .

        Returns:  None

        Examples:
             .. code-block:: python

                import paddle
                input = paddle.uniform([10, 2])
                linear = paddle.nn.Linear(2, 3)
                out = linear(input)
                out.backward()
                print("Before clear_gradient, linear.weight.grad: {}".format(linear.weight.grad))
                linear.weight.clear_gradient()
                print("After clear_gradient, linear.weight.grad: {}".format(linear.weight.grad))
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
      .def("_set_grad_ivar",
           [](imperative::VarBase &self, imperative::VarBase &grad) {
             self.SetGradVarBase(grad);
           })
      .def("_is_sparse",
           [](imperative::VarBase &self) {
             return self.Var().IsType<framework::SelectedRows>();
           })
      .def("_allreduce",
           [](imperative::VarBase &self,
              const imperative::ParallelStrategy &strategy) {
             if (strategy.nranks_ > 1) {
#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
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
#endif  // PADDLE_WITH_NCCL or PADDLE_WITH_RCCL
             }
           },
           py::call_guard<py::gil_scoped_release>())
      .def("_register_grad_hook",
           [](imperative::VarBase &self, const py::handle &hook) {
             PADDLE_ENFORCE_EQ(
                 !self.OverridedStopGradient() && self.HasGradVar(), true,
                 platform::errors::InvalidArgument(
                     "Cannot register gradient hook on a Tensor that stop "
                     "gradient or without gradient."));
             return self.GradVarBase()->AddVariableWrapperHook(
                 std::make_shared<PyVariableWrapperHook>(hook.ptr()));
           })
      .def("_remove_grad_hook",
           [](imperative::VarBase &self, int64_t hook_id) {
             PADDLE_ENFORCE_EQ(
                 !self.OverridedStopGradient() && self.HasGradVar(), true,
                 platform::errors::InvalidArgument(
                     "Cannot remove gradient hook on a Tensor that stop "
                     "gradient or without gradient."));
             return self.GradVarBase()->RemoveVariableWrapperHook(hook_id);
           })
      .def("_register_backward_hook",
           [](imperative::VarBase &self, const py::handle &hook) {
             PADDLE_ENFORCE_EQ(
                 self.IsLeaf(), true,
                 platform::errors::InvalidArgument(
                     "Only can register backward hook for leaf Tensor."));
             PADDLE_ENFORCE_EQ(
                 !self.OverridedStopGradient() && self.HasGradVar(), true,
                 platform::errors::InvalidArgument(
                     "Cannot register backward hook on a Tensor that stop "
                     "gradient or without gradient."));
             auto py_func = PyObjectCast<std::function<void()>>(hook.ptr());
             self.GradVarBase()->AddVoidHook(
                 std::make_shared<std::function<void()>>(py_func));
           },
           R"DOC(
             Registers a backward hook for current Tensor.

             This hook will be called every time the gradient of current Tensor has been fully calculated.

             There are two differences with `_register_grad_hook`:
             1. This backward hook will be executed after the gradient accumulation completed across batchs,
                but the hook registered by `_register_grad_hook` will be executed the gradient accumulation
                completed in current batch.
             2. This backward hook function should have the following signature:

                  hook() -> None

                It requires no input and no return value.

             Args:
                 hook(function): A backward hook to be registered for Tensor.gradient

             Returns:
                 None
           )DOC")
      .def("cpu",
           [](const std::shared_ptr<imperative::VarBase> &self) {
             if (platform::is_cpu_place(self->Place())) {
               return self;
             } else {
               auto new_var = self->NewVarBase(platform::CPUPlace(), true);
               new_var->SetOverridedStopGradient(self->OverridedStopGradient());
               return new_var;
             }
           },
           R"DOC(
        Returns a copy of this Tensor in CPU memory.

        If this Tensor is already in CPU memory, then no copy is performed and the original Tensor is returned.

        Examples:
            .. code-block:: python

              import paddle
              x = paddle.to_tensor(1.0, place=paddle.CUDAPlace(0))
              print(x.place)    # CUDAPlace(0)
              
              y = x.cpu()
              print(y.place)    # CPUPlace

              )DOC")
      .def("pin_memory",
           [](const std::shared_ptr<imperative::VarBase> &self) {
#if !defined(PADDLE_WITH_CUDA) && !defined(PADDLE_WITH_HIP)
             PADDLE_THROW(platform::errors::PermissionDenied(
                 "Cannot copy this Tensor to pinned memory in CPU version "
                 "Paddle, "
                 "Please recompile or reinstall Paddle with CUDA support."));
#endif
             if (platform::is_cuda_pinned_place(self->Place())) {
               return self;
             } else {
               auto new_var =
                   self->NewVarBase(platform::CUDAPinnedPlace(), true);
               new_var->SetOverridedStopGradient(self->OverridedStopGradient());
               return new_var;
             }
           },
           R"DOC(
        Returns a copy of this Tensor in pin memory.

        If this Tensor is already in pin memory, then no copy is performed and the original Tensor is returned.

        Examples:
            .. code-block:: python

              import paddle
              x = paddle.to_tensor(1.0, place=paddle.CUDAPlace(0))
              print(x.place)      # CUDAPlace(0)

              y = x.pin_memory()
              print(y.place)      # CUDAPinnedPlace

      )DOC")
      .def("cuda",
           [](const std::shared_ptr<imperative::VarBase> &self, int device_id,
              bool blocking) {
#if !defined(PADDLE_WITH_CUDA) && !defined(PADDLE_WITH_HIP)
             PADDLE_THROW(platform::errors::PermissionDenied(
                 "Cannot copy this Tensor to GPU in CPU version Paddle, "
                 "Please recompile or reinstall Paddle with CUDA support."));
#else
             int device_count = platform::GetCUDADeviceCount();
             if (device_id == -1) {
               if (platform::is_gpu_place(self->Place())) {
                 return self;
               } else {
                 device_id = 0;
               }
             }
             PADDLE_ENFORCE_GE(
                 device_id, 0,
                 platform::errors::InvalidArgument(
                     "Can not copy Tensor to Invalid CUDAPlace(%d), device id "
                     "must inside [0, %d)",
                     device_id, device_count));
             PADDLE_ENFORCE_LT(
                 device_id, device_count,
                 platform::errors::InvalidArgument(
                     "Can not copy Tensor to Invalid CUDAPlace(%d), device id "
                     "must inside [0, %d)",
                     device_id, device_count));
             platform::CUDAPlace place = platform::CUDAPlace(device_id);
             if (platform::is_same_place(self->Place(), place)) {
               return self;
             } else {
               auto new_var = self->NewVarBase(place, blocking);
               new_var->SetOverridedStopGradient(self->OverridedStopGradient());
               return new_var;
             }
#endif
           },
           py::arg("device_id") = -1, py::arg("blocking") = true, R"DOC(
        Returns a copy of this Tensor in GPU memory.

        If this Tensor is already in GPU memory and device_id is default, 
        then no copy is performed and the original Tensor is returned.
        
        Args:
            device_id(int, optional): The destination GPU device id. Defaults to the current device.
            blocking(bool, optional): If False and the source is in pinned memory, the copy will be 
              asynchronous with respect to the host. Otherwise, the argument has no effect. Default: False.

        Examples:
            .. code-block:: python

              import paddle
              x = paddle.to_tensor(1.0, place=paddle.CPUPlace())
              print(x.place)        # CPUPlace

              y = x.cuda()
              print(y.place)        # CUDAPlace(0)

              y = x.cuda(1)
              print(y.place)        # CUDAPlace(1)
       )DOC")
      .def("_share_memory",
           [](const std::shared_ptr<imperative::VarBase> &self) {
#ifndef _WIN32
             PADDLE_ENFORCE_EQ(
                 platform::is_cpu_place(self->Place()), true,
                 platform::errors::InvalidArgument(
                     "Sharing memory only support CPU Tensor currently"));
             // 1. get LoDTensor
             auto *t = self->MutableVar()->GetMutable<framework::LoDTensor>();
             // 2. allocate shared memory
             void *data_ptr = t->data<void>();
             size_t data_size = t->numel() * framework::SizeOfType(t->type());
             auto shared_writer_holder =
                 memory::allocation::AllocateMemoryMapWriterAllocation(
                     data_size);
             // 3. maintain mmap fd set & backup ipc_name
             const std::string &ipc_name = shared_writer_holder->ipc_name();
             memory::allocation::MemoryMapFdSet::Instance().Insert(ipc_name);
             // 4. copy data & reset holder
             memory::Copy(platform::CPUPlace(), shared_writer_holder->ptr(),
                          platform::CPUPlace(), data_ptr, data_size);
             t->ResetHolder(shared_writer_holder);
             return *t;
#else
             PADDLE_THROW(platform::errors::PermissionDenied(
                 "Sharing memory in Windows OS is not supported currently"));
#endif
           },
           py::return_value_policy::reference)
      .def("copy_", &imperative::VarBase::CopyFrom)
      .def("_copy_to",
           [](const std::shared_ptr<imperative::VarBase> &self,
              const platform::CPUPlace &place, bool blocking) {
             auto new_var = self->NewVarBase(place, blocking);
             // Note(zhiqiu): Since NewVarBase may use GpuCopyAsync to
             // copy data from the tensor of self to the tensor of new varbase,
             // we need to ensure that the varbase self is not destructed until
             // the GpuCopyAsync is completed. Otherwise, the memory may be
             // freed
             // when varbase self is destructed.
             // To do that, we increase the reference count of self by 1 and
             // add a cuda event to wait the GpuCopyAsync's completion.
             if (!blocking) {
               IncreaseVarbaseReferenceCountUntilCopyComplete(self, place);
             }
             return new_var;
           },
           py::return_value_policy::copy)
      .def("_copy_to",
           [](const std::shared_ptr<imperative::VarBase> &self,
              const platform::CUDAPinnedPlace &place, bool blocking) {
             auto new_var = self->NewVarBase(place, blocking);
             if (!blocking) {
               IncreaseVarbaseReferenceCountUntilCopyComplete(self, place);
             }
             return new_var;
           },
           py::return_value_policy::copy)
      .def("_copy_to",
           [](const std::shared_ptr<imperative::VarBase> &self,
              const platform::XPUPlace &place, bool blocking) {
             auto new_var = self->NewVarBase(place, blocking);
             if (!blocking) {
               IncreaseVarbaseReferenceCountUntilCopyComplete(self, place);
             }
             return new_var;
           },
           py::return_value_policy::copy)
      .def("_copy_to",
           [](const std::shared_ptr<imperative::VarBase> &self,
              const platform::CUDAPlace &place, bool blocking) {
             auto new_var = self->NewVarBase(place, blocking);
             if (!blocking) {
               IncreaseVarbaseReferenceCountUntilCopyComplete(self, place);
             }
             return new_var;
           },
           py::return_value_policy::copy)
      .def("_copy_to",
           [](const std::shared_ptr<imperative::VarBase> &self,
              const platform::Place &place, bool blocking) {
             auto new_var = self->NewVarBase(place, blocking);
             if (!blocking) {
               IncreaseVarbaseReferenceCountUntilCopyComplete(self, place);
             }
             return new_var;
           },
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
      .def_property_readonly("shape",
                             [](imperative::VarBase &self) {
                               if (self.Var().IsType<framework::LoDTensor>()) {
                                 return framework::vectorize<int>(
                                     self.Var()
                                         .Get<framework::LoDTensor>()
                                         .dims());
                               } else if (self.Var()
                                              .IsType<
                                                  framework::SelectedRows>()) {
                                 return framework::vectorize<int>(
                                     self.Var()
                                         .Get<framework::SelectedRows>()
                                         .value()
                                         .dims());
                               } else {
                                 VLOG(2) << "It is meaningless to get shape of "
                                            "variable type "
                                         << GetTypeName(self);
                                 return std::vector<int>();
                               }
                             })
      .def_property_readonly("is_leaf", &imperative::VarBase::IsLeaf,
                             R"DOC(
      Whether a Tensor is leaf Tensor.

      For the Tensor whose stop_gradient is ``True`` , it will be leaf Tensor. 
      
      For the Tensor whose stop_gradient is ``False`` , it will be leaf Tensor too if it is created by user.

      Returns:
          bool: Whether a Tensor is leaf Tensor.

      Examples:
          .. code-block:: python

              import paddle

              x = paddle.to_tensor(1.)
              print(x.is_leaf) # True

              x = paddle.to_tensor(1., stop_gradient=True)
              y = x + 1
              print(x.is_leaf) # True
              print(y.is_leaf) # True

              x = paddle.to_tensor(1., stop_gradient=False)
              y = x + 1
              print(x.is_leaf) # True
              print(y.is_leaf) # False
       )DOC")
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
      .def_property("_has_grad", &imperative::Tracer::HasGrad,
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
              VLOG(4) << "Tracer(" << &self << ")"
                      << " set expected place " << *p;
            } else if (py::isinstance<platform::XPUPlace>(obj)) {
              auto p = obj.cast<platform::XPUPlace *>();
              self.SetExpectedPlace(*p);
              VLOG(4) << "Tracer(" << &self << ")"
                      << " set expected place " << *p;
            } else if (py::isinstance<platform::CPUPlace>(obj)) {
              auto p = obj.cast<platform::CPUPlace *>();
              self.SetExpectedPlace(*p);
              VLOG(4) << "Tracer(" << &self << ")"
                      << " set expected place " << *p;
            } else if (py::isinstance<platform::CUDAPinnedPlace>(obj)) {
              auto p = obj.cast<platform::CUDAPinnedPlace *>();
              self.SetExpectedPlace(*p);
              VLOG(4) << "Tracer(" << &self << ")"
                      << " set expected place " << *p;
            } else if (py::isinstance<platform::Place>(obj)) {
              auto p = obj.cast<platform::Place *>();
              self.SetExpectedPlace(*p);
              VLOG(4) << "Tracer(" << &self << ")"
                      << " set expected place " << *p;
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
      .def("_set_amp_op_list",
           [](imperative::Tracer &self,
              std::unordered_set<std::string> &allow_ops,
              std::unordered_set<std::string> &block_ops) {
             // NOTE(zhiqiu): The automatic conversion in pybind11 between
             // c++
             // STL and python set/list/dict involve a copy operation that
             // prevents pass-by-reference semantics, so it is ok to swap.
             // The reaseon why not directly pass
             // std::shared_ptr<std::unordered_set<std::string>>
             // is that pybind11 forbid shared_ptr<T> where T is not custom
             // type.
             imperative::AmpOperators::Instance().GetMutableAllowOps()->swap(
                 allow_ops);
             imperative::AmpOperators::Instance().GetMutableBlockOps()->swap(
                 block_ops);
             VLOG(5) << "AMP operators changed, "
                     << imperative::AmpOperators::Instance();
           })
      .def("_get_amp_op_list",
           [](imperative::Tracer &self) {
             return std::make_tuple(
                 *(imperative::AmpOperators::Instance().GetMutableAllowOps()),
                 *(imperative::AmpOperators::Instance().GetMutableBlockOps()));
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
                       const std::string &ep) { self.current_endpoint_ = ep; })
      .def_property(
          "nrings",
          [](const imperative::ParallelStrategy &self) { return self.nrings_; },
          [](imperative::ParallelStrategy &self, int nrings) {
            self.nrings_ = nrings;
          });

  m.def("varbase_copy", &VarBaseCopy<platform::Place>);
  m.def("varbase_copy", &VarBaseCopy<platform::CPUPlace>);
  m.def("varbase_copy", &VarBaseCopy<platform::CUDAPlace>);
  m.def("varbase_copy", &VarBaseCopy<platform::XPUPlace>);
  m.def("varbase_copy", &VarBaseCopy<platform::CUDAPinnedPlace>);

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

  m.def(
      "dygraph_run_backward",
      [](const std::vector<std::shared_ptr<imperative::VarBase>> &tensors,
         const std::vector<std::shared_ptr<imperative::VarBase>> &grad_tensors,
         bool retain_graph, const imperative::Tracer &tracer) {
        auto *engine = tracer.GetEngine();
        engine->Init(tensors, grad_tensors, retain_graph);
        start_time_python = 0;
        VLOG(3) << "Start backward";
        engine->Execute();
        VLOG(3) << "Finish backward";
      },
      py::call_guard<py::gil_scoped_release>());

#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL) || \
    defined(PADDLE_WITH_XPU_BKCL)
  py::class_<imperative::ParallelContext,
             std::shared_ptr<imperative::ParallelContext>>(m,
                                                           "ParallelContext");

  py::class_<imperative::Reducer, std::shared_ptr<imperative::Reducer>>(
      m, "Reducer", R"DOC()DOC")
      .def(py::init<const std::vector<std::shared_ptr<imperative::VarBase>> &,
                    const std::vector<std::vector<size_t>> &,
                    const std::vector<bool> &,
                    std::shared_ptr<imperative::ParallelContext>,
                    const std::vector<size_t> &, bool>())
      .def("prepare_for_backward", &imperative::Reducer::PrepareForBackward,
           py::arg("vars"), py::call_guard<py::gil_scoped_release>());

  m.def("assign_group_by_size", &imperative::AssignGroupBySize, py::arg("vars"),
        py::arg("is_sparse_gradient"),
        py::arg("group_size_limits") = std::vector<size_t>{25 * 1024 * 1024},
        py::arg("tensor_indices") = std::vector<int64_t>{},
        py::call_guard<py::gil_scoped_release>());
#endif

#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
  py::class_<imperative::NCCLParallelContext, imperative::ParallelContext,
             std::shared_ptr<imperative::NCCLParallelContext>>(
      m, "NCCLParallelContext")
      .def(py::init<const imperative::ParallelStrategy &,
                    const platform::CUDAPlace &>())
      .def("init", [](imperative::NCCLParallelContext &self) { self.Init(); })
      .def("init_with_ring_id",
           &imperative::NCCLParallelContext::InitWithRingID,
           py::arg("ring_id"));
#endif

#if defined(PADDLE_WITH_XPU_BKCL)
  py::class_<imperative::BKCLParallelContext, imperative::ParallelContext,
             std::shared_ptr<imperative::BKCLParallelContext>>(
      m, "BKCLParallelContext")
      .def(py::init<const imperative::ParallelStrategy &,
                    const platform::XPUPlace &>())
      .def("init", [](imperative::BKCLParallelContext &self) { self.Init(); })
      .def("init_with_ring_id",
           &imperative::BKCLParallelContext::InitWithRingID,
           py::arg("ring_id"));
#endif
  m.def("pylayer_apply",
        [](const platform::CPUPlace &place, const py::object &cls,
           const py::args args, const py::kwargs kwargs) {
          return imperative::PyLayerApply(place, cls, args, kwargs);
        });

  m.def("pylayer_apply",
        [](const platform::CUDAPlace &place, const py::object &cls,
           const py::args args, const py::kwargs kwargs) {
          return imperative::PyLayerApply(place, cls, args, kwargs);
        });

  m.def("pylayer_apply",
        [](const platform::XPUPlace &place, const py::object &cls,
           const py::args args, const py::kwargs kwargs) {
          return imperative::PyLayerApply(place, cls, args, kwargs);
        });

  m.def("pylayer_apply",
        [](const platform::CUDAPinnedPlace &place, const py::object &cls,
           const py::args args, const py::kwargs kwargs) {
          return imperative::PyLayerApply(place, cls, args, kwargs);
        });
}

}  // namespace pybind
}  // namespace paddle
