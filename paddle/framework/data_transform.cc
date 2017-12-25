/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/framework/data_transform.h"

namespace paddle {
namespace framework {

static DataTransformFnMap* data_transform_map = nullptr;

DataTransformFnMap& DataTransformFnMap::Instance() {
  if (data_transform_map == nullptr) {
    //    data_transform_map = new DataTransformFnMap();
    new DataTransformFnMap();
  }

  return *data_transform_map;
}

}  // namespace framework
}  // namespace paddle
