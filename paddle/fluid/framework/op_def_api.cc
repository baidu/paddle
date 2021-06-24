// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#if defined _WIN32 || defined __APPLE__
#else
#define _LINUX
#endif
#include "paddle/fluid/framework/op_def_api.h"
#include <fstream>
#include <mutex>
#include <string>
#include <unordered_map>
#ifdef _LINUX
#include <stdio_ext.h>
#include <sys/mman.h>
#include <sys/stat.h>
#endif
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>
#include "glog/logging.h"
#include "io/fs.h"
#include "paddle/fluid/framework/op_def.pb.h"

/*
// op_def.pbtxt
namespace {
 const std::unordered_map<std::string, std::std::string> op_def_map = {...};
}
*/
#include "paddle/fluid/framework/op_def.pbtxt"  //NOLINT

namespace paddle {
namespace framework {

const proto::OpDef& GetOpDef(const std::string& op_name) {
  static std::unordered_map<std::string, proto::OpDef> ops_definition;
  static std::mutex mtx;
  if (ops_definition.find(op_name) == ops_definition.end()) {
    std::lock_guard<std::mutex> lk(mtx);
    if (ops_definition.find(op_name) == ops_definition.end()) {
      proto::OpDef op_def;
      if (op_def_map.find(op_name) == op_def_map.end()) {
        LOG(WARNING) << op_name << ".pbtxt not exist!";
      } else {
        if (!::google::protobuf::TextFormat::ParseFromString(
                op_def_map.at(op_name), &op_def)) {
          LOG(WARNING) << "Failed to parse " << op_name;
        }
      }
      if (op_def.type() != op_name) {
        LOG(WARNING) << op_name << ".pbtxt has error type :" << op_def.type();
        ops_definition.emplace(std::make_pair(op_name, proto::OpDef()));
      } else {
        ops_definition.emplace(std::make_pair(op_name, std::move(op_def)));
      }
    }
  }
  return ops_definition.at(op_name);
}

bool HasOpDef(const std::string& op_name) {
  return op_def_map.find(op_name) != op_def_map.end();
}
}  // namespace framework
}  // namespace paddle
