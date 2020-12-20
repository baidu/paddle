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
#include <cstddef>
#include <memory>
#include <map>
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/cuda_device_guard.h"


#ifdef PADDLE_WITH_PSLIB

namespace paddle {
namespace framework {

class GPUResource {
 public:
  GPUResource(int device_id, int index);
  virtual ~GPUResource();
  GPUResource(const GPUResource&) = delete;
  GPUResource& operator=(const GPUResource&) = delete;

  int dev_id() const { return dev_id_; }
  int index() const { return index_; }
  cudaStream_t stream() { return stream_; }
  cudaStream_t copy_stream() { return copy_stream_; }
  
  int dev_id_;
  int index_;
  cudaStream_t stream_;
  cudaStream_t copy_stream_;

};

class HeterBoxResource {
 public:
  HeterBoxResource(const std::vector<int>& dev_ids);
  HeterBoxResource(const HeterBoxResource&) = delete;
  HeterBoxResource& operator=(const HeterBoxResource&) = delete;
  virtual ~HeterBoxResource() {}
  void enable_p2p();
  int total_gpu();
  int get_index_by_devid(int devid);
  cudaStream_t stream(int num);
  cudaStream_t copy_stream(int num);
  int dev_id(int num);

  std::vector<std::shared_ptr<GPUResource>> resources_;
  std::vector<int> dev_ids_;
  std::map<int, int> devid_2_index_;
};

}  // end namespace framework
}  // end namespace paddle
#endif
