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

#ifdef PADDLE_WITH_PSLIB
#include "gpu_resource.h"
#include "paddle/fluid/platform/cuda_device_guard.h"

namespace paddle {
namespace framework {

GPUResource::GPUResource(int dev_id, int index) {
  index_ = index;
  dev_id_ = dev_id;
  
  platform::CUDADeviceGuard guard(dev_id_);
  
  PADDLE_ENFORCE_CUDA_SUCCESS(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
  PADDLE_ENFORCE_CUDA_SUCCESS(cudaStreamCreateWithFlags(&copy_stream_, cudaStreamNonBlocking));
}

GPUResource::~GPUResource() {
  platform::CUDADeviceGuard guard(dev_id_);
  
  PADDLE_ENFORCE_CUDA_SUCCESS(cudaStreamDestroy(stream_));
  PADDLE_ENFORCE_CUDA_SUCCESS(cudaStreamDestroy(copy_stream_));
}

void HeterBoxResource::enable_p2p() {
  for (size_t i = 0; i < dev_ids_.size(); ++i) {
    platform::CUDADeviceGuard guard(dev_ids_[i]);
    for (size_t j = 0; j < dev_ids_.size(); ++j) {
      if (i != j) {
        int p2p_flag;
        PADDLE_ENFORCE_CUDA_SUCCESS(cudaDeviceCanAccessPeer(&p2p_flag, dev_ids_[i], dev_ids_[j]));
        if (p2p_flag == 1) {
          cudaError_t ret = cudaDeviceEnablePeerAccess(dev_ids_[j], 0);
          if (ret != cudaSuccess && ret != cudaErrorPeerAccessAlreadyEnabled) {
            VLOG(0) << " Cuda error(" << ret << "), " << cudaGetErrorString(ret) << ".";
          } else {
            cudaGetLastError();
          }
        }

      }
    }
  }
}

HeterBoxResource::HeterBoxResource(const std::vector<int>& dev_ids) {
  dev_ids_ = dev_ids;
  for (size_t i = 0; i < dev_ids_.size(); ++i) {
    std::shared_ptr<GPUResource> resource = std::make_shared<GPUResource>(dev_ids_[i], i);
    resources_.push_back(resource);
    devid_2_index_[dev_ids_[i]] = i;
  }
}

cudaStream_t HeterBoxResource::copy_stream(int num) {
  return resources_[num]->copy_stream();
}

cudaStream_t HeterBoxResource::stream(int num) {
  return resources_[num]->stream();
}

int HeterBoxResource::dev_id(int num) {
  return dev_ids_[num];
}

int HeterBoxResource::get_index_by_devid(int devid) {
  return devid_2_index_[devid];
}

int HeterBoxResource::total_gpu() {
  return dev_ids_.size();
}

}  // end namespace framework
}  // end namespace paddle
#endif
