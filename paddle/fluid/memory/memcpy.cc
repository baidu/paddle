/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/memory/memcpy.h"

#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/profiler.h"

#ifdef PADDLE_WITH_XPU
#include "paddle/fluid/platform/xpu_header.h"
#endif

namespace paddle {
namespace memory {

template <>
void Copy<platform::CPUPlace, platform::CPUPlace>(platform::CPUPlace, void* dst,
                                                  platform::CPUPlace,
                                                  const void* src, size_t num) {
  if (UNLIKELY(num == 0)) return;
  VLOG(4) << "src: " << src << ", dst: " << dst << ", num: " << num;
  std::memcpy(dst, src, num);
}

#ifdef PADDLE_WITH_XPU
template <>
void Copy<platform::XPUPlace, platform::CPUPlace>(platform::XPUPlace dst_place,
                                                  void* dst,
                                                  platform::CPUPlace src_place,
                                                  const void* src, size_t num) {
  if (num <= 0) {
    VLOG(1) << "memcpy XPU_HOST_TO_DEVICE size <= 0 (" << num << ")";
    return;
  }
  int dev_id = -1;
  int ret = xpu_current_device(&dev_id);
  PADDLE_ENFORCE_EQ(ret, XPU_SUCCESS,
                    platform::errors::External(
                        "XPU API return wrong value[%d], please check whether "
                        "Baidu Kunlun Card is properly installed.",
                        ret));
  if (dev_id >= 64) {
    // if dev_id >= 64, the device is a simulator device, -64 to get real dev_id
    dev_id -= 64;
  }
  if (dev_id != dst_place.device) {
    ret = xpu_set_device(dst_place.device);
    PADDLE_ENFORCE_EQ(
        ret, XPU_SUCCESS,
        platform::errors::External(
            "XPU API return wrong value[%d], please check whether "
            "Baidu Kunlun Card is properly installed.",
            ret));
  }
  ret = xpu_memcpy(dst, src, num, XPUMemcpyKind::XPU_HOST_TO_DEVICE);
  PADDLE_ENFORCE_EQ(ret, XPU_SUCCESS,
                    platform::errors::External(
                        "XPU API return wrong value[%d], please check whether "
                        "Baidu Kunlun Card is properly installed.",
                        ret));
  if (dev_id != dst_place.device) {
    ret = xpu_set_device(dev_id);
    PADDLE_ENFORCE_EQ(
        ret, XPU_SUCCESS,
        platform::errors::External(
            "XPU API return wrong value[%d], please check whether "
            "Baidu Kunlun Card is properly installed.",
            ret));
  }
}

template <>
void Copy<platform::CPUPlace, platform::XPUPlace>(platform::CPUPlace dst_place,
                                                  void* dst,
                                                  platform::XPUPlace src_place,
                                                  const void* src, size_t num) {
  if (num <= 0) {
    VLOG(1) << "memcpy XPU_DEVICE_TO_HOST size <= 0 (" << num << ")";
    return;
  }
  int dev_id = -1;
  int ret = xpu_current_device(&dev_id);
  PADDLE_ENFORCE_EQ(ret, XPU_SUCCESS,
                    platform::errors::External(
                        "XPU API return wrong value[%d], please check whether "
                        "Baidu Kunlun Card is properly installed.",
                        ret));
  if (dev_id >= 64) {
    // if dev_id >= 64, the device is a simulator device, -64 to get real dev_id
    dev_id -= 64;
  }
  if (dev_id != src_place.device) {
    ret = xpu_set_device(src_place.device);
    PADDLE_ENFORCE_EQ(
        ret, XPU_SUCCESS,
        platform::errors::External(
            "XPU API return wrong value[%d], please check whether "
            "Baidu Kunlun Card is properly installed.",
            ret));
  }
  ret = xpu_memcpy(dst, src, num, XPUMemcpyKind::XPU_DEVICE_TO_HOST);
  PADDLE_ENFORCE_EQ(ret, XPU_SUCCESS,
                    platform::errors::External(
                        "XPU API return wrong value[%d], please check whether "
                        "Baidu Kunlun Card is properly installed.",
                        ret));
  if (dev_id != src_place.device) {
    ret = xpu_set_device(dev_id);
    PADDLE_ENFORCE_EQ(
        ret, XPU_SUCCESS,
        platform::errors::External(
            "XPU API return wrong value[%d], please check whether "
            "Baidu Kunlun Card is properly installed.",
            ret));
  }
}

template <>
void Copy<platform::XPUPlace, platform::XPUPlace>(platform::XPUPlace dst_place,
                                                  void* dst,
                                                  platform::XPUPlace src_place,
                                                  const void* src, size_t num) {
  if (num <= 0) {
    VLOG(1) << "memcpy XPU_DEVICE_TO_DEVICE size <= 0 (" << num << ")";
    return;
  }
  int dev_id = -1;
  int ret = xpu_current_device(&dev_id);
  PADDLE_ENFORCE_EQ(ret, XPU_SUCCESS,
                    platform::errors::External(
                        "XPU API return wrong value[%d], please check whether "
                        "Baidu Kunlun Card is properly installed.",
                        ret));
  if (dev_id >= 64) {
    // if dev_id >= 64, the device is a simulator device, -64 to get real dev_id
    dev_id -= 64;
  }
  if (dev_id != src_place.device || dev_id != dst_place.device) {
    ret = xpu_set_device(src_place.device);
    PADDLE_ENFORCE_EQ(
        ret, XPU_SUCCESS,
        platform::errors::External(
            "XPU API return wrong value[%d], please check whether "
            "Baidu Kunlun Card is properly installed.",
            ret));
    void* tmp = malloc(num);
    ret = xpu_memcpy(tmp, src, num, XPUMemcpyKind::XPU_DEVICE_TO_HOST);
    PADDLE_ENFORCE_EQ(
        ret, XPU_SUCCESS,
        platform::errors::External(
            "XPU API return wrong value[%d], please check whether "
            "Baidu Kunlun Card is properly installed.",
            ret));
    ret = xpu_set_device(dst_place.device);
    PADDLE_ENFORCE_EQ(
        ret, XPU_SUCCESS,
        platform::errors::External(
            "XPU API return wrong value[%d], please check whether "
            "Baidu Kunlun Card is properly installed.",
            ret));
    ret = xpu_memcpy(dst, tmp, num, XPUMemcpyKind::XPU_HOST_TO_DEVICE);
    PADDLE_ENFORCE_EQ(
        ret, XPU_SUCCESS,
        platform::errors::External(
            "XPU API return wrong value[%d], please check whether "
            "Baidu Kunlun Card is properly installed.",
            ret));
    ret = xpu_set_device(dev_id);
    PADDLE_ENFORCE_EQ(
        ret, XPU_SUCCESS,
        platform::errors::External(
            "XPU API return wrong value[%d], please check whether "
            "Baidu Kunlun Card is properly installed.",
            ret));
    free(tmp);
  } else {
    platform::DeviceContextPool& pool = platform::DeviceContextPool::Instance();
    auto* dev_ctx = pool.GetByPlace(src_place);
    dev_ctx->Wait();
    int ret = xpu::memcpy_device(dev_ctx->x_context(), dst, src, num);
    PADDLE_ENFORCE_EQ(ret, XPU_SUCCESS, platform::errors::External(
                                            "XPU API return wrong value[%d %s]",
                                            ret, XPUAPIErrorMsg[ret]));
  }
}
#endif

#ifdef PADDLE_WITH_ASCEND_CL
template <>
void Copy<platform::NPUPlace, platform::CPUPlace>(platform::NPUPlace dst_place,
                                                  void* dst,
                                                  platform::CPUPlace src_place,
                                                  const void* src, size_t num,
                                                  aclrtStream stream) {
  if (UNLIKELY(num == 0)) return;

  platform::SetNPUDeviceId(dst_place.device);

  VLOG(4) << "memory::Copy " << num << " Bytes from " << src_place << " to "
          << dst_place << " by thream(" << stream << ")";

  if (stream) {
    platform::RecordEvent record_event("NpuMemcpyAsync:CPU->NPU");
    platform::NPUMemcpyAsync(dst, src, num, ACL_MEMCPY_HOST_TO_DEVICE, stream);
  } else {
    // On NPU, async operation after sync operation is ok, while sync operation
    // after async is not ok, since the async operation may not done.
    // So, its needed to do wait before sync operation.
    platform::DeviceContextPool& pool = platform::DeviceContextPool::Instance();
    static_cast<platform::NPUDeviceContext*>(pool.Get(dst_place))->Wait();

    platform::RecordEvent record_event("NpuMemcpySync:CPU->NPU");
    platform::NPUMemcpySync(dst, src, num, ACL_MEMCPY_HOST_TO_DEVICE);
  }
}

template <>
void Copy<platform::CPUPlace, platform::NPUPlace>(platform::CPUPlace dst_place,
                                                  void* dst,
                                                  platform::NPUPlace src_place,
                                                  const void* src, size_t num,
                                                  aclrtStream stream) {
  if (UNLIKELY(num == 0)) return;

  platform::SetNPUDeviceId(src_place.device);

  VLOG(4) << "memory::Copy " << num << " Bytes from " << src_place << " to "
          << dst_place << " by thream(" << stream << ")";

  if (stream) {
    platform::RecordEvent record_event("NpuMemcpyAsync:NPU->CPU");
    platform::NPUMemcpyAsync(dst, src, num, ACL_MEMCPY_DEVICE_TO_HOST, stream);
  } else {
    platform::DeviceContextPool& pool = platform::DeviceContextPool::Instance();
    static_cast<platform::NPUDeviceContext*>(pool.Get(src_place))->Wait();

    platform::RecordEvent record_event("NpuMemcpySync:NPU->CPU");
    platform::NPUMemcpySync(dst, src, num, ACL_MEMCPY_DEVICE_TO_HOST);
  }
}

template <>
void Copy<platform::NPUPlace, platform::NPUPlace>(platform::NPUPlace dst_place,
                                                  void* dst,
                                                  platform::NPUPlace src_place,
                                                  const void* src, size_t num,
                                                  aclrtStream stream) {
  if (UNLIKELY(num == 0)) return;

  VLOG(4) << "memory::Copy " << num << " Bytes from " << src_place << " to "
          << dst_place << " by stream(" << stream << ")";
  if (dst_place == src_place) {
    platform::SetNPUDeviceId(src_place.device);
    if (stream) {
      platform::RecordEvent record_event("NpuMemcpyAsync(same_npu):NPU->NPU");
      platform::NPUMemcpyAsync(dst, src, num, ACL_MEMCPY_DEVICE_TO_DEVICE,
                               stream);
    } else {
      platform::DeviceContextPool& pool =
          platform::DeviceContextPool::Instance();
      static_cast<platform::NPUDeviceContext*>(pool.Get(dst_place))->Wait();

      platform::RecordEvent record_event("NpuMemcpySync(same_npu):NPU->NPU");
      platform::NPUMemcpySync(dst, src, num, ACL_MEMCPY_DEVICE_TO_DEVICE);
    }
  } else {
    if (!platform::NPUCanAccessPeer(dst_place.device, dst_place.device)) {
      PADDLE_THROW(platform::errors::Unavailable(
          "Peer access between NPU places is not allowed."));
    }
    if (stream) {
      // TODO(zhiqiu): support peer access?
      platform::RecordEvent record_event("NpuMemcpyPeerAsync:NPU->NPU");
      platform::NPUMemcpyAsync(dst, src, num, ACL_MEMCPY_DEVICE_TO_DEVICE,
                               stream);
    } else {
      platform::DeviceContextPool& pool =
          platform::DeviceContextPool::Instance();
      static_cast<platform::NPUDeviceContext*>(pool.Get(dst_place))->Wait();

      platform::RecordEvent record_event("NpuMemcpyPeerSync:NPU->NPU");
      platform::NPUMemcpySync(dst, src, num, ACL_MEMCPY_DEVICE_TO_DEVICE);
    }
  }
}

template <>
void Copy<platform::CPUPlace, platform::NPUPinnedPlace>(
    platform::CPUPlace dst_place, void* dst, platform::NPUPinnedPlace src_place,
    const void* src, size_t num) {
  VLOG(4) << "memory::Copy " << num << " Bytes from " << src_place << " to "
          << dst_place;
  if (UNLIKELY(num == 0)) return;
  std::memcpy(dst, src, num);
}

template <>
void Copy<platform::NPUPinnedPlace, platform::CPUPlace>(
    platform::NPUPinnedPlace dst_place, void* dst, platform::CPUPlace src_place,
    const void* src, size_t num) {
  VLOG(4) << "memory::Copy " << num << " Bytes from " << src_place << " to "
          << dst_place;
  if (UNLIKELY(num == 0)) return;
  std::memcpy(dst, src, num);
}

template <>
void Copy<platform::NPUPinnedPlace, platform::NPUPinnedPlace>(
    platform::NPUPinnedPlace dst_place, void* dst,
    platform::NPUPinnedPlace src_place, const void* src, size_t num) {
  VLOG(4) << "memory::Copy " << num << " Bytes from " << src_place << " to "
          << dst_place;
  if (UNLIKELY(num == 0)) return;
  std::memcpy(dst, src, num);
}

template <>
void Copy<platform::NPUPinnedPlace, platform::NPUPlace>(
    platform::NPUPinnedPlace dst_place, void* dst, platform::NPUPlace src_place,
    const void* src, size_t num, aclrtStream stream) {
  if (UNLIKELY(num == 0)) return;

  platform::SetNPUDeviceId(src_place.device);

  VLOG(4) << "memory::Copy " << num << " Bytes from " << src_place << " to "
          << dst_place << " by thream(" << stream << ")";

  if (stream) {
    platform::RecordEvent record_event("NpuMemcpyAsync:NPU->NPUPinned");
    platform::NPUMemcpyAsync(dst, src, num, ACL_MEMCPY_DEVICE_TO_HOST, stream);
  } else {
    platform::DeviceContextPool& pool = platform::DeviceContextPool::Instance();
    static_cast<platform::NPUDeviceContext*>(pool.Get(src_place))->Wait();

    platform::RecordEvent record_event("NpuMemcpySync:NPU->NPUPinned");
    platform::NPUMemcpySync(dst, src, num, ACL_MEMCPY_DEVICE_TO_HOST);
  }
}

template <>
void Copy<platform::NPUPlace, platform::NPUPinnedPlace>(
    platform::NPUPlace dst_place, void* dst, platform::NPUPinnedPlace src_place,
    const void* src, size_t num, aclrtStream stream) {
  if (UNLIKELY(num == 0)) return;

  platform::SetNPUDeviceId(dst_place.device);

  VLOG(4) << "memory::Copy " << num << " Bytes from " << src_place << " to "
          << dst_place << " by thream(" << stream << ")";

  if (stream) {
    platform::RecordEvent record_event("NpuMemcpyAsync:NPUPinned->NPU");
    platform::NPUMemcpyAsync(dst, src, num, ACL_MEMCPY_HOST_TO_DEVICE, stream);
  } else {
    // On NPU, async operation after sync operation is ok, while sync operation
    // after async is not ok, since the async operation may not done.
    // So, its needed to do wait before sync operation.
    platform::DeviceContextPool& pool = platform::DeviceContextPool::Instance();
    static_cast<platform::NPUDeviceContext*>(pool.Get(dst_place))->Wait();

    platform::RecordEvent record_event("NpuMemcpySync:NPUPinned->NPU");
    platform::NPUMemcpySync(dst, src, num, ACL_MEMCPY_HOST_TO_DEVICE);
  }
}

#endif

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
static constexpr size_t kMaxGpuAsyncCopyBytes = 64 * 1024;  // 64K

#ifdef PADDLE_WITH_HIP
inline void SyncCUDAStream() {
#if !defined(_WIN32)
  hipStreamSynchronize(0);
#else
  hipError_t e_sync = hipSuccess;
  while (e_sync = hipStreamQuery(0)) {
    if (e_sync == hipErrorNotReady) continue;
    break;
  }
#endif
}
#else
inline void SyncCUDAStream() {
#if !defined(_WIN32)
  cudaStreamSynchronize(0);
#else
  cudaError_t e_sync = cudaSuccess;
  while (e_sync = cudaStreamQuery(0)) {
    if (e_sync == cudaErrorNotReady) continue;
    break;
  }
#endif
}
#endif

// NOTE(zcd): Do not use GpuMemcpySync as much as possible.
// because GpuMemcpySync issues the copying command to the default stream,
// which will make two commands from different streams cannot run concurrently.
// Reference:
// https://devblogs.nvidia.com/gpu-pro-tip-cuda-7-streams-simplify-concurrency/

template <>
void Copy<platform::CPUPlace, platform::CUDAPlace>(
    platform::CPUPlace dst_place, void* dst, platform::CUDAPlace src_place,
    const void* src, size_t num, gpuStream_t stream) {
  if (UNLIKELY(num == 0)) return;

  platform::SetDeviceId(src_place.device);
  VLOG(4) << "memory::Copy " << num << " Bytes from " << src_place << " to "
          << dst_place << " by stream(" << stream << ")";
  if (stream) {
    platform::RecordEvent record_event("GpuMemcpyAsync:GPU->CPU");
#ifdef PADDLE_WITH_HIP
    platform::GpuMemcpyAsync(dst, src, num, hipMemcpyDeviceToHost, stream);
#else
    platform::GpuMemcpyAsync(dst, src, num, cudaMemcpyDeviceToHost, stream);
#endif
  } else {
    platform::RecordEvent record_event("GpuMemcpySync:GPU->CPU");
#ifdef PADDLE_WITH_HIP
    platform::GpuMemcpySync(dst, src, num, hipMemcpyDeviceToHost);
#else
    platform::GpuMemcpySync(dst, src, num, cudaMemcpyDeviceToHost);
#endif
    // FIXME(zjl): do we really need it?
    if (num <= kMaxGpuAsyncCopyBytes) {
      SyncCUDAStream();
    }
  }
}

template <>
void Copy<platform::CUDAPlace, platform::CPUPlace>(
    platform::CUDAPlace dst_place, void* dst, platform::CPUPlace src_place,
    const void* src, size_t num, gpuStream_t stream) {
  if (UNLIKELY(num == 0)) return;

  platform::SetDeviceId(dst_place.device);
  VLOG(4) << "memory::Copy " << num << " Bytes from " << src_place << " to "
          << dst_place << " by thream(" << stream << ")";
  if (stream) {
    platform::RecordEvent record_event("GpuMemcpyAsync:CPU->GPU");
#ifdef PADDLE_WITH_HIP
    platform::GpuMemcpyAsync(dst, src, num, hipMemcpyHostToDevice, stream);
#else
    platform::GpuMemcpyAsync(dst, src, num, cudaMemcpyHostToDevice, stream);
#endif
  } else {
    platform::RecordEvent record_event("GpuMemcpySync:CPU->GPU");
#ifdef PADDLE_WITH_HIP
    platform::GpuMemcpySync(dst, src, num, hipMemcpyHostToDevice);
#else
    platform::GpuMemcpySync(dst, src, num, cudaMemcpyHostToDevice);
#endif
    // FIXME(zjl): do we really need it?
    if (num <= kMaxGpuAsyncCopyBytes) {
      SyncCUDAStream();
    }
  }
}

template <>
void Copy<platform::CUDAPlace, platform::CUDAPlace>(
    platform::CUDAPlace dst_place, void* dst, platform::CUDAPlace src_place,
    const void* src, size_t num, gpuStream_t stream) {
  if (UNLIKELY(num == 0)) return;

  VLOG(4) << "memory::Copy " << num << " Bytes from " << src_place << " to "
          << dst_place << " by stream(" << stream << ")";
  if (dst_place == src_place) {
    platform::SetDeviceId(src_place.device);
    if (stream) {
      platform::RecordEvent record_event("GpuMemcpyAsync(same_gpu):GPU->GPU");
#ifdef PADDLE_WITH_HIP
      platform::GpuMemcpyAsync(dst, src, num, hipMemcpyDeviceToDevice, stream);
#else
      platform::GpuMemcpyAsync(dst, src, num, cudaMemcpyDeviceToDevice, stream);
#endif
    } else {
      platform::RecordEvent record_event("GpuMemcpySync(same_gpu):GPU->GPU");
#ifdef PADDLE_WITH_HIP
      platform::GpuMemcpySync(dst, src, num, hipMemcpyDeviceToDevice);
#else
      platform::GpuMemcpySync(dst, src, num, cudaMemcpyDeviceToDevice);
#endif
    }
  } else {
    if (stream) {
      platform::RecordEvent record_event("GpuMemcpyPeerAsync:GPU->GPU");
      platform::GpuMemcpyPeerAsync(dst, dst_place.device, src, src_place.device,
                                   num, stream);
    } else {
      platform::RecordEvent record_event("GpuMemcpyPeerSync:GPU->GPU");
      platform::GpuMemcpyPeerSync(dst, dst_place.device, src, src_place.device,
                                  num);
    }
  }
}

template <>
void Copy<platform::CPUPlace, platform::CUDAPinnedPlace>(
    platform::CPUPlace dst_place, void* dst,
    platform::CUDAPinnedPlace src_place, const void* src, size_t num) {
  VLOG(4) << "memory::Copy " << num << " Bytes from " << src_place << " to "
          << dst_place;
  if (UNLIKELY(num == 0)) return;
  std::memcpy(dst, src, num);
}

template <>
void Copy<platform::CUDAPinnedPlace, platform::CPUPlace>(
    platform::CUDAPinnedPlace dst_place, void* dst,
    platform::CPUPlace src_place, const void* src, size_t num) {
  VLOG(4) << "memory::Copy " << num << " Bytes from " << src_place << " to "
          << dst_place;
  if (UNLIKELY(num == 0)) return;
  std::memcpy(dst, src, num);
}

template <>
void Copy<platform::CUDAPinnedPlace, platform::CUDAPinnedPlace>(
    platform::CUDAPinnedPlace dst_place, void* dst,
    platform::CUDAPinnedPlace src_place, const void* src, size_t num) {
  VLOG(4) << "memory::Copy " << num << " Bytes from " << src_place << " to "
          << dst_place;
  if (UNLIKELY(num == 0)) return;
  std::memcpy(dst, src, num);
}

template <>
void Copy<platform::CUDAPinnedPlace, platform::CUDAPlace>(
    platform::CUDAPinnedPlace dst_place, void* dst,
    platform::CUDAPlace src_place, const void* src, size_t num,
    gpuStream_t stream) {
  if (UNLIKELY(num == 0)) return;
  platform::SetDeviceId(src_place.device);
  VLOG(4) << "memory::Copy " << num << " Bytes from " << src_place << " to "
          << dst_place << " by thream(" << stream << ")";
  if (stream) {
    platform::RecordEvent record_event("GpuMemcpyAsync:GPU->CUDAPinned");
#ifdef PADDLE_WITH_HIP
    platform::GpuMemcpyAsync(dst, src, num, hipMemcpyDeviceToHost, stream);
#else
    platform::GpuMemcpyAsync(dst, src, num, cudaMemcpyDeviceToHost, stream);
#endif
  } else {
    platform::RecordEvent record_event("GpuMemcpySync:GPU->CUDAPinned");
#ifdef PADDLE_WITH_HIP
    platform::GpuMemcpySync(dst, src, num, hipMemcpyDeviceToHost);
#else
    platform::GpuMemcpySync(dst, src, num, cudaMemcpyDeviceToHost);
#endif
  }
}

template <>
void Copy<platform::CUDAPlace, platform::CUDAPinnedPlace>(
    platform::CUDAPlace dst_place, void* dst,
    platform::CUDAPinnedPlace src_place, const void* src, size_t num,
    gpuStream_t stream) {
  if (UNLIKELY(num == 0)) return;

  platform::SetDeviceId(dst_place.device);
  VLOG(4) << "memory::Copy " << num << " Bytes from " << src_place << " to "
          << dst_place << " by thream(" << stream << ")";
  if (stream) {
    platform::RecordEvent record_event("GpuMemcpyAsync:CUDAPinned->GPU");
#ifdef PADDLE_WITH_HIP
    platform::GpuMemcpyAsync(dst, src, num, hipMemcpyHostToDevice, stream);
#else
    platform::GpuMemcpyAsync(dst, src, num, cudaMemcpyHostToDevice, stream);
#endif
  } else {
    platform::RecordEvent record_event("GpuMemcpySync:CUDAPinned->GPU");
#ifdef PADDLE_WITH_HIP
    platform::GpuMemcpySync(dst, src, num, hipMemcpyHostToDevice);
#else
    platform::GpuMemcpySync(dst, src, num, cudaMemcpyHostToDevice);
#endif
  }
}

#endif

}  // namespace memory
}  // namespace paddle
