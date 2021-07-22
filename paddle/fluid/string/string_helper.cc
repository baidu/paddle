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

#include "paddle/fluid/string/string_helper.h"

#include <ctype.h>
#include <stdio.h>
#include <cstring>
#include <string>

#include "glog/logging.h"

namespace paddle {
namespace string {

inline size_t count_spaces(const char* s) {
  size_t count = 0;

  while (*s != 0 && isspace(*s++)) {
    count++;
  }

  return count;
}

inline size_t count_nonspaces(const char* s) {
  size_t count = 0;

  while (*s != 0 && !isspace(*s++)) {
    count++;
  }

  return count;
}

// remove leading and tailing spaces
std::string trim_spaces(const std::string& str) {
  const char* p = str.c_str();

  while (*p != 0 && isspace(*p)) {
    p++;
  }

  size_t len = strlen(p);

  while (len > 0 && isspace(p[len - 1])) {
    len--;
  }

  return std::string(p, len);
}

std::string erase_spaces(const std::string& str) {
  std::string result;
  result.reserve(str.size());
  const char* p = str.c_str();
  while (*p != 0) {
    if (!isspace(*p)) {
      result.append(p, 1);
    }
    ++p;
  }
  return result;
}

inline int str_to_float(const char* str, float* v) {
  const char* head = str;
  char* cursor = NULL;
  int index = 0;
  while (*(head += count_spaces(head)) != 0) {
    v[index++] = std::strtof(head, &cursor);
    if (head == cursor) {
      break;
    }
    head = cursor;
  }
  return index;
}

bool ends_with(std::string const& input, std::string const& test) {
  if (test.size() > input.size()) return false;
  return std::equal(test.rbegin(), test.rend(), input.rbegin());
}

// A helper class for reading lines from file.
// A line buffer is maintained. It
// doesn't need to know the maximum possible length of a line.
char* LineFileReader::getdelim(FILE* f, char delim) {
#ifndef _WIN32
  int32_t ret = ::getdelim(&_buffer, &_buf_size, delim, f);

  if (ret >= 0) {
    if (ret >= 1 && _buffer[ret - 1] == delim) {
      _buffer[--ret] = 0;
    }

    _length = static_cast<size_t>(ret);
    return _buffer;
  } else {
    _length = 0;
    CHECK(feof(f));
    return NULL;
  }
#else
  return NULL;
#endif
}

}  // end namespace string
}  // end namespace paddle
