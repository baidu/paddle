/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

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
#include <cstddef>
#include <map>
#include <memory>
#include <vector>
#include "paddle/fluid/platform/cuda_device_guard.h"
#include "paddle/fluid/platform/enforce.h"

#ifdef PADDLE_WITH_HETERPS

namespace paddle {
namespace framework {

class GPUResource {
 public:
  GPUResource(std::vector<int>& device_id, int index);
  virtual ~GPUResource();
  GPUResource(const GPUResource&) = delete;
  GPUResource& operator=(const GPUResource&) = delete;

  int dev_id() const { return dev_id_; }
  int index() const { return index_; }
  gpuStream_t local_stream(int num) { return local_streams_[num]; }
  gpuStream_t remote_stream(int num) { return remote_streams_[num]; }
  gpuStream_t comm_stream(int num) { return comm_streams_[num]; }

  int dev_id_;
  int index_;
  std::vector<int> dev_ids_;
  std::vector<gpuStream_t> remote_streams_;
  std::vector<gpuStream_t> local_streams_;
  std::vector<gpuStream_t> comm_streams_;
};

class HeterPsResource {
 public:
  HeterPsResource(const std::vector<int>& dev_ids);
  HeterPsResource(const HeterPsResource&) = delete;
  HeterPsResource& operator=(const HeterPsResource&) = delete;
  virtual ~HeterPsResource() {}
  void enable_p2p();
  int total_gpu();
  int get_index_by_devid(int devid);
  int dev_id(int num);
  gpuStream_t local_stream(int gpu_num, int stream_num);
  gpuStream_t remote_stream(int gpu_num, int stream_num);
  gpuStream_t comm_stream(int gpu_num, int stream_num);

  std::vector<std::shared_ptr<GPUResource>> resources_;
  std::vector<int> dev_ids_;
  std::map<int, int> devid_2_index_;
};

}  // end namespace framework
}  // end namespace paddle
#endif
