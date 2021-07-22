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

#ifdef PADDLE_WITH_HETERPS
#include "heter_resource.h"
#include "paddle/fluid/platform/cuda_device_guard.h"

namespace paddle {
namespace framework {

GPUResource::GPUResource(std::vector<int>& dev_ids, int index) {
  index_ = index;
  dev_ids_ = dev_ids;
  dev_id_ = dev_ids_[index];

  platform::CUDADeviceGuard guard(dev_id_);
  local_streams_.resize(dev_ids_.size());
  comm_streams_.resize(dev_ids_.size());
  remote_streams_.resize(dev_ids_.size());

  for (size_t i = 0; i < dev_ids_.size(); ++i) {
    PADDLE_ENFORCE_CUDA_SUCCESS(
        cudaStreamCreateWithFlags(&local_streams_[i], cudaStreamNonBlocking));
    PADDLE_ENFORCE_CUDA_SUCCESS(
        cudaStreamCreateWithFlags(&comm_streams_[i], cudaStreamNonBlocking));
    PADDLE_ENFORCE_CUDA_SUCCESS(
        cudaStreamCreateWithFlags(&remote_streams_[i], cudaStreamNonBlocking));
  }
}

GPUResource::~GPUResource() {
  platform::CUDADeviceGuard guard(dev_id_);
  for (size_t i = 0; i < local_streams_.size(); ++i) {
    PADDLE_ENFORCE_CUDA_SUCCESS(cudaStreamDestroy(local_streams_[i]));
  }
  for (size_t i = 0; i < comm_streams_.size(); ++i) {
    PADDLE_ENFORCE_CUDA_SUCCESS(cudaStreamDestroy(comm_streams_[i]));
  }
  for (size_t i = 0; i < remote_streams_.size(); ++i) {
    PADDLE_ENFORCE_CUDA_SUCCESS(cudaStreamDestroy(remote_streams_[i]));
  }
}

void HeterPsResource::enable_p2p() {
  for (size_t i = 0; i < dev_ids_.size(); ++i) {
    platform::CUDADeviceGuard guard(dev_ids_[i]);
    for (size_t j = 0; j < dev_ids_.size(); ++j) {
      if (i != j) {
        int p2p_flag;
        PADDLE_ENFORCE_CUDA_SUCCESS(
            cudaDeviceCanAccessPeer(&p2p_flag, dev_ids_[i], dev_ids_[j]));
        if (p2p_flag == 1) {
          cudaError_t ret = cudaDeviceEnablePeerAccess(dev_ids_[j], 0);
          if (ret != cudaSuccess && ret != cudaErrorPeerAccessAlreadyEnabled) {
            VLOG(0) << " Cuda error(" << ret << "), " << cudaGetErrorString(ret)
                    << ".";
          } else {
            cudaGetLastError();
          }
        }
      }
    }
  }
}

HeterPsResource::HeterPsResource(const std::vector<int>& dev_ids) {
  dev_ids_ = dev_ids;
  for (size_t i = 0; i < dev_ids_.size(); ++i) {
    std::shared_ptr<GPUResource> resource =
        std::make_shared<GPUResource>(dev_ids_, i);
    resources_.push_back(resource);
    devid_2_index_[dev_ids_[i]] = i;
  }
}

cudaStream_t HeterPsResource::comm_stream(int gpu_num, int stream_num) {
  return resources_[gpu_num]->comm_stream(stream_num);
}

cudaStream_t HeterPsResource::local_stream(int gpu_num, int stream_num) {
  return resources_[gpu_num]->local_stream(stream_num);
}

cudaStream_t HeterPsResource::remote_stream(int gpu_num, int stream_num) {
  return resources_[gpu_num]->remote_stream(stream_num);
}

int HeterPsResource::dev_id(int num) { return dev_ids_[num]; }

int HeterPsResource::get_index_by_devid(int devid) {
  return devid_2_index_[devid];
}

int HeterPsResource::total_gpu() { return dev_ids_.size(); }

}  // end namespace framework
}  // end namespace paddle
#endif
