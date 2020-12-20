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

#pragma once
#include <vector>
#include "paddle/fluid/framework/fleet/heter_box/hashtable/feature_value.h"

#ifdef PADDLE_WITH_PSLIB

namespace paddle {
namespace framework {

class HeterBoxBase {
 public:
  HeterBoxBase() {};
  HeterBoxBase(size_t capacity, std::shared_ptr<HeterBoxResource> resource) {};
  virtual ~HeterBoxBase() {};
  HeterBoxBase(const HeterBoxBase&) = delete;
  HeterBoxBase& operator=(const HeterBoxBase&) = delete;

  virtual void pull_sparse(int num, FeatureKey* d_keys, FeatureValue* d_vals, size_t len) = 0;
  virtual void build_ps(int num, FeatureKey* h_keys, FeatureValue* h_vals, size_t len, size_t chunk_size, int stream_num) = 0;
  virtual int get_index_by_devid(int devid) = 0;
  virtual void dump() = 0;
  virtual void show_one_table(int gpu_num) = 0;
  virtual void push_sparse(int num, FeatureKey* d_keys, FeaturePushValue* d_grads, size_t len) = 0;
  static HeterBoxBase* get_instance(size_t capacity, std::shared_ptr<HeterBoxResource> resource);
};

}  // end namespace framework
}  // end namespace paddle
#endif
