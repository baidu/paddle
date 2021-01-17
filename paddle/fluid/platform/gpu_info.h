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

#pragma once

#if defined PADDLE_WITH_CUDA || defined PADDLE_WITH_HIP
#ifdef PADDLE_WITH_CUDA
#include <cuda_runtime.h>
#endif
#ifdef PADDLE_WITH_HIP
#include <hip/hip_runtime.h>
#endif

#include <stddef.h>
#include <string>
#include <vector>

namespace paddle {
namespace platform {
//! Get the version of cudnn
int CudnnVersion();

//! Get the total number of GPU devices in system.
int GetCUDADeviceCount();

//! Get the compute capability of the ith GPU (format: major * 10 + minor)
int GetCUDAComputeCapability(int i);

//! Get the runtime version of the ith GPU
int GetCUDARuntimeVersion(int id);

//! Get the driver version of the ith GPU
int GetCUDADriverVersion(int id);

//! Wheter the current device support TensorCore
bool TensorCoreAvailable();

//! Get the MultiProcessors of the ith GPU.
int GetCUDAMultiProcessors(int i);

//! Get the MaxThreads of each MultiProcessor of the ith GPU.
int GetCUDAMaxThreadsPerMultiProcessor(int i);

//! Get the MaxThreads of each block of the ith GPU.
int GetCUDAMaxThreadsPerBlock(int i);

//! Get the current GPU device id in system.
int GetCurrentDeviceId();

//! Get the maximum GridDim size for GPU buddy allocator.
dim3 GetGpuMaxGridDimSize(int);

//! Get a list of device ids from environment variable or use all.
std::vector<int> GetSelectedDevices();

//! Set the GPU device id for next execution.
void SetDeviceId(int device_id);

//! Get the memory usage of current GPU device.
void GpuMemoryUsage(size_t *available, size_t *total);

//! Get the available memory to allocate, which is the size of available gpu
//! minus reserving.
size_t GpuAvailableMemToAlloc();

//! Get the maximum allocation size of current GPU device.
size_t GpuMaxAllocSize();

//! Get the initial allocation size of current GPU device.
size_t GpuInitAllocSize();

//! Get the re-allocation size of current GPU device.
size_t GpuReallocSize();

//! Get the minimum chunk size for GPU buddy allocator.
size_t GpuMinChunkSize();

//! Get the maximum chunk size for GPU buddy allocator.
size_t GpuMaxChunkSize();
#ifdef PADDLE_WITH_CUDA
//! Copy memory from address src to dst asynchronously.
void GpuMemcpyAsync(void *dst, const void *src, size_t count,
                    enum cudaMemcpyKind kind, cudaStream_t stream);

//! Copy memory from address src to dst synchronously.
void GpuMemcpySync(void *dst, const void *src, size_t count,
                   enum cudaMemcpyKind kind);

//! Copy memory from one device to another device asynchronously.
void GpuMemcpyPeerAsync(void *dst, int dst_device, const void *src,
                        int src_device, size_t count, cudaStream_t stream);

//! Copy memory from one device to another device synchronously.
void GpuMemcpyPeerSync(void *dst, int dst_device, const void *src,
                       int src_device, size_t count);

//! Set memory dst with value count size asynchronously
void GpuMemsetAsync(void *dst, int value, size_t count, cudaStream_t stream);

//! Blocks until stream has completed all operations.
void GpuStreamSync(cudaStream_t stream);

//! CudaMalloc with recorded info
cudaError_t RecordedCudaMalloc(void **ptr, size_t size, int dev_id);

//! CudaFree with recorded info
void RecordedCudaFree(void *p, size_t size, int dev_id);

//! Get available and total gpu memory with considering limitation
bool RecordedCudaMemGetInfo(size_t *avail, size_t *total, size_t *actual_avail,
                            size_t *actual_total, int dev_id);

//! Get recorded cudaMalloc size. If record is disabled, return 0.
uint64_t RecordedCudaMallocSize(int dev_id);

bool IsCudaMallocRecorded(int dev_id);
#endif  // PADDLE_WITH_CUDA

#ifdef PADDLE_WITH_HIP
//! Copy memory from address src to dst asynchronously.
void GpuMemcpyAsync(void *dst, const void *src, size_t count,
                    enum hipMemcpyKind kind, hipStream_t stream);

//! Copy memory from address src to dst synchronously.
void GpuMemcpySync(void *dst, const void *src, size_t count,
                   enum hipMemcpyKind kind);

//! Copy memory from one device to another device asynchronously.
void GpuMemcpyPeerAsync(void *dst, int dst_device, const void *src,
                        int src_device, size_t count, hipStream_t stream);

//! Copy memory from one device to another device synchronously.
void GpuMemcpyPeerSync(void *dst, int dst_device, const void *src,
                       int src_device, size_t count);

//! Set memory dst with value count size asynchronously
void GpuMemsetAsync(void *dst, int value, size_t count, hipStream_t stream);

//! Blocks until stream has completed all operations.
void GpuStreamSync(hipStream_t stream);

//! CudaMalloc with recorded info
hipError_t RecordedCudaMalloc(void **ptr, size_t size, int dev_id);

//! CudaFree with recorded info
void RecordedCudaFree(void *p, size_t size, int dev_id);

//! Get available and total gpu memory with considering limitation
bool RecordedCudaMemGetInfo(size_t *avail, size_t *total, size_t *actual_avail,
                            size_t *actual_total, int dev_id);

//! Get recorded hipMalloc size. If record is disabled, return 0.
uint64_t RecordedCudaMallocSize(int dev_id);

bool IsCudaMallocRecorded(int dev_id);
#endif  // PADDLE_WITH_HIP

}  // namespace platform
}  // namespace paddle
#endif
