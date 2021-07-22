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

#include "paddle/fluid/pybind/exception.h"

namespace paddle {
namespace pybind {

/* Paddle Exception mapping rules:
 *   - InvalidArgumentError -> ValueError
 *   - NotFoundError -> RuntimeError
 *   - OutOfRangeError -> IndexError
 *   - AlreadyExistsError -> RuntimeError
 *   - ResourceExhaustedError -> MemoryError
 *   - PreconditionNotMetError -> RuntimeError
 *   - PermissionDeniedError -> RuntimeError
 *   - ExecutionTimeoutError -> RuntimeError
 *   - UnimplementedError -> NotImplementedError
 *   - UnavailableError -> RuntimeError
 *   - FatalError -> SystemError
 *   - ExternalError -> OSError
 */

void BindException(pybind11::module* m) {
  static pybind11::exception<platform::EOFException> eof(*m, "EOFException");
  static pybind11::exception<platform::EnforceNotMet> exc(*m, "EnforceNotMet");
  pybind11::register_exception_translator([](std::exception_ptr p) {
    try {
      if (p) {
        VLOG(0) << "register_exception_translator catch a exception.";
        std::rethrow_exception(p);
      }
    } catch (const platform::EOFException& e) {
      eof(e.what());
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
          exc(e.what());
          break;
      }
    } catch (const std::exception& e) {
      VLOG(0) << "std::exception caught: " << e.what();
      std::rethrow_exception(std::current_exception());
    }
  });

  m->def("__unittest_throw_exception__", [] {
    PADDLE_THROW(
        platform::errors::PermissionDenied("This is a test of exception"));
  });
}

}  // namespace pybind
}  // namespace paddle
