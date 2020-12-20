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

#include <vector>
#include "paddle/fluid/framework/fleet/heter_box/heter_box.h"

#ifdef PADDLE_WITH_PSLIB

namespace paddle {
namespace framework {

HeterBoxBase* HeterBoxBase::get_instance(size_t capacity, std::shared_ptr<HeterBoxResource> resource) {
  return new HeterBox(capacity, resource);
}

HeterBox::HeterBox(size_t capacity, std::shared_ptr<HeterBoxResource> resource) {
  gpu_ps_ = std::make_shared<GpuPs<FeatureKey, FeatureValue, FeaturePushValue> >(capacity, resource);
  opt_ = Optimizer<FeatureValue, FeaturePushValue>();
}

HeterBox::~HeterBox() {}

void HeterBox::pull_sparse(int num, FeatureKey* d_keys, FeatureValue* d_vals, size_t len) {
  gpu_ps_->pull_sparse(num, d_keys, d_vals, len);
}

void HeterBox::build_ps(int num, FeatureKey* h_keys, FeatureValue* h_vals, size_t len, size_t chunk_size, int stream_num) {
    gpu_ps_->build_ps(num, h_keys, h_vals, len, chunk_size, stream_num);
}

int HeterBox::get_index_by_devid(int devid) {
  return gpu_ps_->get_index_by_devid(devid);
}

void HeterBox::dump() {
}

void HeterBox::show_one_table(int gpu_num) {
  gpu_ps_->show_one_table(gpu_num);
}

void HeterBox::push_sparse(int num, FeatureKey* d_keys, FeaturePushValue* d_grads, size_t len) {
  gpu_ps_->push_sparse(num, d_keys, d_grads, len, opt_);
}

}  // end namespace framework
}  // end namespace paddle
#endif
