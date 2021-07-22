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

#include <ctype.h>
#include <stdio.h>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include "boost/lexical_cast.hpp"
#include "glog/logging.h"

namespace paddle {
namespace string {

inline size_t count_spaces(const char* s);

inline size_t count_nonspaces(const char* s);

template <class... ARGS>
void format_string_append(std::string& str, const char* fmt,  // NOLINT
                          ARGS&&... args) {
  int len = snprintf(NULL, 0, fmt, args...);
  CHECK_GE(len, 0);
  size_t oldlen = str.length();
  str.resize(oldlen + len + 1);
  CHECK(snprintf(&str[oldlen], static_cast<size_t>(len) + 1, fmt, args...) ==
        len);
  str.resize(oldlen + len);
}

template <class... ARGS>
void format_string_append(std::string& str, const std::string& fmt,  // NOLINT
                          ARGS&&... args) {
  format_string_append(str, fmt.c_str(), args...);
}

template <class... ARGS>
std::string format_string(const char* fmt, ARGS&&... args) {
  std::string str;
  format_string_append(str, fmt, args...);
  return std::move(str);
}

template <class... ARGS>
std::string format_string(const std::string& fmt, ARGS&&... args) {
  return format_string(fmt.c_str(), args...);
}

// remove leading and tailing spaces
std::string trim_spaces(const std::string& str);

// erase all spaces in str
std::string erase_spaces(const std::string& str);

int str_to_float(const char* str, float* v);

// checks whether the test string is a suffix of the input string.
bool ends_with(std::string const& input, std::string const& test);

// split string by delim
template <class T = std::string>
std::vector<T> split_string(const std::string& str, const std::string& delim) {
  size_t pre_pos = 0;
  size_t pos = 0;
  std::string tmp_str;
  std::vector<T> res_list;
  res_list.clear();
  if (str.empty()) {
    return res_list;
  }
  while ((pos = str.find(delim, pre_pos)) != std::string::npos) {
    tmp_str.assign(str, pre_pos, pos - pre_pos);
    res_list.push_back(tmp_str);
    pre_pos = pos + 1;
  }
  tmp_str.assign(str, pre_pos, str.length() - pre_pos);
  if (!tmp_str.empty()) {
    res_list.push_back(tmp_str);
  }
  return res_list;
}

// split string by spaces. Leading and tailing spaces are ignored. Consecutive
// spaces are treated as one delim.
template <class T = std::string>
std::vector<T> split_string(const std::string& str) {
  std::vector<T> list;
  const char* p;
  int pre_pos = 0;
  int pos = 0;
  std::string tmp_str;
  if (str.empty()) {
    return list;
  }
  for (p = str.c_str(); *p != 0;) {
    if (!isspace(*p)) {
      pos = pre_pos;
      p++;

      while (*p != 0 && !isspace(*p)) {
        pos++;
        p++;
      }
      tmp_str.assign(str, pre_pos, pos - pre_pos + 1);
      list.push_back(tmp_str);
      pre_pos = pos + 1;
    } else {
      pre_pos++;
      p++;
    }
  }
  return list;
}

template <class Container>
std::string join_strings(const Container& strs, char delim) {
  std::string str;

  int i = 0;
  for (auto& elem : strs) {
    if (i > 0) {
      str += delim;
    }

    str += boost::lexical_cast<std::string>(elem);
    ++i;
  }

  return str;
}

// A helper class for reading lines from file. A line buffer is maintained. It
// doesn't need to know the maximum possible length of a line.

class LineFileReader {
 public:
  LineFileReader() {}
  LineFileReader(LineFileReader&&) = delete;
  LineFileReader(const LineFileReader&) = delete;
  ~LineFileReader() { ::free(_buffer); }
  char* getline(FILE* f) { return this->getdelim(f, '\n'); }
  char* getdelim(FILE* f, char delim);
  char* get() { return _buffer; }
  size_t length() { return _length; }

 private:
  char* _buffer = NULL;
  size_t _buf_size = 0;
  size_t _length = 0;
};
}  // end namespace string
}  // end namespace paddle
