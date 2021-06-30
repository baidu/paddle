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

#pragma once

#include <cctype>
#include <ostream>
#include <string>

#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace framework {

enum class DataLayout {
  kNHWC = 0,
  kNCHW = 1,
  kAnyLayout = 2,
  kMKLDNN = 3,     // all layouts supported by MKLDNN internally
  kFractalNZ = 4,  // only for NPU device
};

inline DataLayout StringToDataLayout(const std::string& str) {
  std::string s(str);
  for (size_t i = 0; i < s.size(); ++i) {
    s[i] = toupper(s[i]);
  }

  if (s == "NHWC") {
    return DataLayout::kNHWC;
  } else if (s == "NCHW") {
    return DataLayout::kNCHW;
  } else if (s == "ANYLAYOUT") {
    return DataLayout::kAnyLayout;
  } else if (s == "MKLDNNLAYOUT") {
    return DataLayout::kMKLDNN;
  } else if (s == "FRACTAL_NZ") {
    return DataLayout::kFractalNZ;
  } else {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "Unknown data layout type string: %s.", s));
  }
}

inline std::string DataLayoutToString(const DataLayout& data_layout) {
  switch (data_layout) {
    case DataLayout::kNHWC:
      return "NHWC";
    case DataLayout::kNCHW:
      return "NCHW";
    case DataLayout::kAnyLayout:
      return "ANY_LAYOUT";
    case DataLayout::kMKLDNN:
      return "MKLDNNLAYOUT";
    case DataLayout::kFractalNZ:
      return "FRACTAL_NZ";
    default:
      PADDLE_THROW(platform::errors::InvalidArgument(
          "Unknown Data Layout type %d.", data_layout));
  }
}

inline std::ostream& operator<<(std::ostream& out, const DataLayout& l) {
  out << DataLayoutToString(l);
  return out;
}

}  // namespace framework
}  // namespace paddle
