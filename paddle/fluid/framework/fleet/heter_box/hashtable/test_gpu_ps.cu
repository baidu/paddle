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

#include <gtest/gtest.h>
#include <vector>
#include "paddle/fluid/framework/fleet/heter_box/optimizer/optimizer.cuh"
#include "paddle/fluid/platform/cuda_device_guard.h"
#include "paddle/fluid/framework/fleet/heter_box/hashtable/gpu_resource.h"
#include "paddle/fluid/framework/fleet/heter_box/hashtable/gpu_ps.h"
#include "paddle/fluid/framework/fleet/heter_box/hashtable/feature_value.h"

using namespace paddle::framework;

TEST(TEST_FLEET, gpu_ps) {
  int gpu_count = 3;
  std::vector<int> dev_ids;
  dev_ids.push_back(0);
  dev_ids.push_back(1);
  dev_ids.push_back(2);
  std::shared_ptr<HeterBoxResource> resource = std::make_shared<HeterBoxResource>(dev_ids);
  resource->enable_p2p(); 
  std::vector<size_t> count;
  std::vector<std::vector<FeatureKey>> keys;
  std::vector<std::vector<FeatureValue>> vals;
  count.resize(dev_ids.size(), 0);
  keys.resize(dev_ids.size());
  vals.resize(dev_ids.size());


  for (int i = 0; i < 10; i++) {
    FeatureKey key;
    FeatureValue val;
    int gpu_num = i % gpu_count;
    key = i;
    val.lr = i;
    val.lr_g2sum = val.mf_size = val.show = val.clk = val.slot = 0;
    keys[gpu_num].push_back(key);
    vals[gpu_num].push_back(val);
    count[gpu_num] += 1;
  }
  
  size_t size = 0;
  for (size_t i = 0; i < count.size(); ++i) {
    size = std::max(size, count[i]);
  }
  
  auto gpu_ps = std::make_shared<GpuPs<FeatureKey, FeatureValue, FeaturePushValue> >(size, resource);
  for (int i = 0; i < gpu_count; ++i) {
    std::cout << "building table: " << i << std::endl;
    gpu_ps->build_ps(i, keys[i].data(), vals[i].data(), count[i], 10, 1);
    gpu_ps->show_one_table(i);
  }
  
  std::cout << "testing pull sparse:" << std::endl;
  paddle::platform::CUDADeviceGuard guard(0);
  FeatureKey* pull_keys;
  FeatureValue* pull_vals;
  cudaMallocManaged(&pull_keys, 5 * sizeof(FeatureKey));
  cudaMallocManaged(&pull_vals, 5 * sizeof(FeatureValue));
  
  pull_keys[0] = 2;
  pull_keys[1] = 3;
  pull_keys[2] = 9;
  pull_keys[3] = 1;
  pull_keys[4] = 6;
  
  gpu_ps->pull_sparse(0, pull_keys, pull_vals, 5);
  for (int i = 0; i < 5; i++) {
    std::cout << pull_keys[i] << ": " << pull_vals[i] << std::endl;
  }
  cudaFree(pull_keys);
  cudaFree(pull_vals);

  std::cout << "testing push sparse:" << std::endl;
  Optimizer<FeatureValue, FeaturePushValue> opt;
  FeatureKey* push_keys;
  FeaturePushValue* push_vals;
  cudaMallocManaged(&push_keys, 5 * sizeof(FeatureKey));
  cudaMallocManaged(&push_vals, 5 * sizeof(FeaturePushValue));
  push_keys[0] = 2;
  push_keys[1] = 3;
  push_keys[2] = 9;
  push_keys[3] = 1;
  push_keys[4] = 3;
  for (int i = 0; i < 5; ++i) {
    push_vals[i].lr_g = push_keys[i] * 100;
    push_vals[i].slot = push_keys[i];
    push_vals[i].show = push_keys[i];
    push_vals[i].clk = push_keys[i];
  }
  gpu_ps->push_sparse(0, push_keys, push_vals, 5, opt);
  for (int i = 0; i < gpu_count; ++i) {
    std::cout << "table " << i << ";" << std::endl;
    gpu_ps->show_one_table(i);
  }

  cudaFree(push_keys);
  cudaFree(push_vals);
}
