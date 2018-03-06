//   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
#include <string>
#include <vector>

#include "paddle/fluid/platform/macros.h"
#include "paddle/fluid/recordio/header.h"

namespace paddle {
namespace recordio {

// A Chunk contains the Header and optionally compressed records.
class Chunk {
public:
  Chunk() : num_bytes_(0) {}
  void Add(std::string buf) {
    records_.push_back(buf);
    num_bytes_ += buf.size();
  }
  // dump the chunk into w, and clears the chunk and makes it ready for
  // the next add invocation.
  bool Write(std::ostream& fo, Compressor ct) const;
  void Clear() {
    records_.clear();
    num_bytes_ = 0;
  }
  void Parse(std::istream& sin);
  size_t NumBytes() { return num_bytes_; }
  const std::string& Record(int i) const { return records_[i]; }

private:
  std::vector<std::string> records_;
  // sum of record lengths in bytes.
  size_t num_bytes_;
  DISABLE_COPY_AND_ASSIGN(Chunk);
};

size_t CompressData(const char* in, size_t in_length, Compressor ct, char* out);

void DeflateData(const char* in, size_t in_length, Compressor ct, char* out);

}  // namespace recordio
}  // namespace paddle
