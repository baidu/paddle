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

#include "paddle/fluid/pybind/dygraph.h"
#include <Python.h>

namespace paddle {
namespace pybind {

namespace py = ::pybind11;

class TestTensor {
 public:
  int value;
};

PyTypeObject* pTestTensorType;

typedef struct { PyObject_HEAD TestTensor testtensor; } TestTensorObject;

PyObject* TestTensor_new(PyTypeObject* type, PyObject* args, PyObject* kwargs) {
  PyObject* obj = type->tp_alloc(type, 0);
  if (obj) {
    auto v = (TestTensorObject*)obj;  // NOLINT
    v->testtensor.value = 100;
  }
  return obj;
}

static void TestTensor_dealloc(TestTensorObject* self) {
  //   PyObject_GC_UnTrack(self);
  Py_TYPE(self)->tp_free((PyObject*)self);  // NOLINT
}

static int TestTensor_init(TestTensorObject* self, PyObject* args,
                           PyObject* kwargs) {
  return 0;
}

static PyObject* add_method(PyObject* self_, PyObject* args, PyObject* kwargs) {
  auto& self = reinterpret_cast<TestTensorObject*>(self_)->testtensor;
  auto& b = reinterpret_cast<TestTensorObject*>(PyTuple_GET_ITEM(args, 0))
                ->testtensor;

  PyObject* obj = pTestTensorType->tp_alloc(pTestTensorType, 0);
  if (obj) {
    auto v = (TestTensorObject*)obj;  // NOLINT
    v->testtensor.value = self.value + b.value;
  }
  return obj;
}

static PyObject* set_value_method(PyObject* self_, PyObject* args,
                                  PyObject* kwargs) {
  auto& self = reinterpret_cast<TestTensorObject*>(self_)->testtensor;
  self.value = PyLong_AsLong(PyTuple_GET_ITEM(args, 0));

  Py_INCREF(Py_None);
  return Py_None;
}

static PyObject* add_function(PyObject* self_, PyObject* args,
                              PyObject* kwargs) {
  auto& a = reinterpret_cast<TestTensorObject*>(PyTuple_GET_ITEM(args, 0))
                ->testtensor;
  auto& b = reinterpret_cast<TestTensorObject*>(PyTuple_GET_ITEM(args, 1))
                ->testtensor;

  PyObject* obj = pTestTensorType->tp_alloc(pTestTensorType, 0);
  if (obj) {
    auto v = (TestTensorObject*)obj;  // NOLINT
    v->testtensor.value = a.value + b.value;
  }
  return obj;
}

static PyObject* print_function(PyObject* self_, PyObject* args,
                                PyObject* kwargs) {
  auto& a = reinterpret_cast<TestTensorObject*>(PyTuple_GET_ITEM(args, 0))
                ->testtensor;

  std::cout << "value = " << a.value << std::endl;

  Py_INCREF(Py_None);
  return Py_None;
}

PyMethodDef variable_methods[] = {
    {"add", (PyCFunction)(void (*)(void))add_method,
     METH_VARARGS | METH_KEYWORDS, NULL},
    {"set_value", (PyCFunction)(void (*)(void))set_value_method,
     METH_VARARGS | METH_KEYWORDS, NULL},
    {NULL, NULL, 0, NULL}};

PyMethodDef variable_functions[] = {
    {"print_function", (PyCFunction)(void (*)(void))print_function,
     METH_VARARGS | METH_KEYWORDS, NULL},
    {"add_function", (PyCFunction)(void (*)(void))add_function,
     METH_VARARGS | METH_KEYWORDS, NULL},
    {NULL, NULL, 0, NULL}};

static PyTypeObject TestTensorType = {
    PyVarObject_HEAD_INIT(NULL, 0) "core_avx.TestTensor", /* tp_name */
    sizeof(TestTensorObject),                             /* tp_basicsize */
    0,                                                    /* tp_itemsize */
    (destructor)TestTensor_dealloc,                       /* tp_dealloc */
    0,                                        /* tp_vectorcall_offset */
    0,                                        /* tp_getattr */
    0,                                        /* tp_setattr */
    0,                                        /* tp_reserved */
    0,                                        /* tp_repr */
    0,                                        /* tp_as_number */
    0,                                        /* tp_as_sequence */
    0,                                        /* tp_as_mapping */
    0,                                        /* tp_hash  */
    0,                                        /* tp_call */
    0,                                        /* tp_str */
    0,                                        /* tp_getattro */
    0,                                        /* tp_setattro */
    0,                                        /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /* tp_flags */
    0,                                        /* tp_doc */
    0,                                        /* tp_traverse */
    0,                                        /* tp_clear */
    0,                                        /* tp_richcompare */
    0,                                        /* tp_weaklistoffset */
    0,                                        /* tp_iter */
    0,                                        /* tp_iternext */
    variable_methods,                         /* tp_methods */
    0,                                        /* tp_members */
    0,                                        /* tp_getset */
    0,                                        /* tp_base */
    0,                                        /* tp_dict */
    0,                                        /* tp_descr_get */
    0,                                        /* tp_descr_set */
    0,                                        /* tp_dictoffset */
    (initproc)TestTensor_init,                /* tp_init */
    0,                                        /* tp_alloc */
    TestTensor_new,                           /* tp_new */
};

void BindDygraph(pybind11::module* m) {
  pTestTensorType = &TestTensorType;
  if (PyType_Ready(&TestTensorType) < 0) {
    return;
  }

  Py_INCREF(&TestTensorType);
  if (PyModule_AddObject(m->ptr(), "TestTensor",
                         (PyObject*)&TestTensorType) <  // NOLINT
      0) {
    Py_DECREF(&TestTensorType);
    Py_DECREF(m->ptr());
    return;
  }

  if (PyModule_AddFunctions(m->ptr(), variable_functions) < 0) {
    return;
  }
}

}  // namespace pybind
}  // namespace paddle
