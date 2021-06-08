/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/platform/errors.h"

#include <stdexcept>

namespace paddle {
namespace platform {

using Code = ::paddle::platform::error::Code;

std::string error_name(Code code) {
  switch (code) {
    case paddle::platform::error::LEGACY:
      return "Error";
      break;
    case paddle::platform::error::INVALID_ARGUMENT:
      return "InvalidArgumentError";
      break;
    case paddle::platform::error::NOT_FOUND:
      return "NotFoundError";
      break;
    case paddle::platform::error::OUT_OF_RANGE:
      return "OutOfRangeError";
      break;
    case paddle::platform::error::ALREADY_EXISTS:
      return "AlreadyExistsError";
      break;
    case paddle::platform::error::RESOURCE_EXHAUSTED:
      return "ResourceExhaustedError";
      break;
    case paddle::platform::error::PRECONDITION_NOT_MET:
      return "PreconditionNotMetError";
      break;
    case paddle::platform::error::PERMISSION_DENIED:
      return "PermissionDeniedError";
      break;
    case paddle::platform::error::EXECUTION_TIMEOUT:
      return "ExecutionTimeoutError";
      break;
    case paddle::platform::error::UNIMPLEMENTED:
      return "UnimplementedError";
      break;
    case paddle::platform::error::UNAVAILABLE:
      return "UnavailableError";
      break;
    case paddle::platform::error::FATAL:
      return "FatalError";
      break;
    case paddle::platform::error::EXTERNAL:
      return "ExternalError";
      break;
    default:
      throw std::invalid_argument("The error type is undefined.");
      break;
  }
}

std::string ErrorSummary::to_string() const {
  std::string result(error_name(code()));
  result += ": ";
  result += error_message();
  return result;
}

}  // namespace platform
}  // namespace paddle
