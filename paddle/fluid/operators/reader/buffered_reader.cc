// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/operators/reader/buffered_reader.h"

#include <utility>

#include "paddle/fluid/platform/profiler.h"

namespace paddle {
namespace operators {
namespace reader {
BufferedReader::~BufferedReader() {
  VLOG(1) << "~BufferedReader";
  reader_->Shutdown();
  while (!position_.empty()) {
    auto &front = position_.front();
    if (front.valid()) {
      front.wait();
    }
    position_.pop();
  }
}

BufferedReader::BufferedReader(
    const std::shared_ptr<framework::ReaderBase> &reader, platform::Place place,
    size_t buffer_size, bool pin_memory)
    : framework::DecoratedReader(reader),
      thread_pool_(1),
      place_(std::move(place)),
      buffer_size_(buffer_size),
      pin_memory_(pin_memory) {
  VLOG(1) << "BufferedReader";
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  if (platform::is_gpu_place(place_) && !pin_memory) {
    int dev_idx = BOOST_GET_CONST(platform::CUDAPlace, place_).device;
    compute_stream_ =
        ((platform::CUDADeviceContext *)(platform::DeviceContextPool::Instance()
                                             .Get(place_)))
            ->stream();
    events_.resize(buffer_size);
    for (auto &event : events_) {
      event = platform::CudaEventResourcePool::Instance().New(dev_idx);
    }
    stream_ = platform::CudaStreamResourcePool::Instance().New(dev_idx);
  }
#endif

#ifdef PADDLE_WITH_ASCEND_CL
  if (platform::is_npu_place(place_)) {
    int dev_idx = BOOST_GET_CONST(platform::NPUPlace, place_).device;
    compute_stream_ =
        ((platform::NPUDeviceContext *)(platform::DeviceContextPool::Instance()
                                            .Get(place_)))
            ->stream();
    events_.resize(buffer_size);
    for (auto &event : events_) {
      event = platform::NpuEventResourcePool::Instance().New(dev_idx);
    }
    stream_ = platform::NpuStreamResourcePool::Instance().New(dev_idx);
  }
#endif
  is_same_place_ = false;
  cpu_buffer_.resize(buffer_size);
  cuda_buffer_.resize(buffer_size);
  npu_buffer_.resize(buffer_size);
  ReadTillBufferFullAsync();
}

void BufferedReader::ReadTillBufferFullAsync() {
  for (size_t i = 0; i < buffer_size_; ++i) {
    ReadAsync(i);
  }
}

void BufferedReader::ReadAsync(size_t i) {
  position_.emplace(thread_pool_.enqueue([this, i]() -> size_t {
    TensorVec &cpu = cpu_buffer_[i];
    reader_->ReadNext(&cpu);

    if (cpu.empty()) {
      return -1UL;
    }

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)  // @{ Group GPU Place
    if (platform::is_gpu_place(place_)) {
      TensorVec &cuda = cuda_buffer_[i];
      if (cuda.empty()) {
        cuda.resize(cpu.size());
      } else {
        PADDLE_ENFORCE_EQ(
            cuda.size(), cpu.size(),
            platform::errors::InvalidArgument(
                "Input tensor number on GPU and CPU devices are not matched."));
      }
      if (pin_memory_) {
        // NOTE: [Copy processing of different input devices]
        // We may accept input tensor in three different devices:
        //   - CPUPlace
        //   - CUDAPinnedPlace
        //   - CUDAPlace
        // CUDA Stream Synchronizing is slow, in order to avoid Synchronizing
        // in BufferedReader thread, we do data copy as follows:
        //   - If src Tensor on CPU memory, we copy it to CUDAPinned memory
        //   - IF src Tensor on CUDAPinned memory, we use it directly
        //   - IF src Tensor on CUDA memory, we use it directly
        platform::CUDAPinnedPlace cuda_pinned_place;
        std::vector<void *> cuda_pinned_ptrs;
        cuda_pinned_ptrs.reserve(cpu.size());
        platform::RecordEvent record_event("BufferedReader:MemoryCopy");
        // NODE(chenwehiang): When we use CUDAPinned Memory, we need call
        // cudaHostAlloc, that is a CUDA API, calling CUDA API need load
        // cuda lib into device, it will cost hundreds of MB of GPU memory.
        // If we don't set Device here, which will use CUDAPlace(0) default.
        platform::SetDeviceId(
            BOOST_GET_CONST(platform::CUDAPlace, place_).device);
        for (size_t i = 0; i < cpu.size(); ++i) {
          if (platform::is_cpu_place(cpu[i].place())) {
            cuda[i].Resize(cpu[i].dims());
            cuda[i].set_layout(cpu[i].layout());
            cuda_pinned_ptrs.emplace_back(
                cuda[i].mutable_data(cuda_pinned_place, cpu[i].type()));
            auto size =
                cpu[i].numel() * paddle::framework::SizeOfType(cpu[i].type());

            memory::Copy(cuda_pinned_place, cuda_pinned_ptrs[i],
                         BOOST_GET_CONST(platform::CPUPlace, cpu[i].place()),
                         cpu[i].data<void>(), size);
            cuda[i].set_lod(cpu[i].lod());
          } else {
            // we set same place flag & use cpu[i] directly
            is_same_place_ = true;
          }
        }
      } else {
        // NOTE(liangdun): using async copy instead of TensorCopySync
        // TensorCopySync would block other stream, because TensorCopySync
        // issues the copying command to the default stream, it will make two
        // commands from different streams cannot run concurrently.
        std::vector<void *> gpu_ptrs;
        gpu_ptrs.reserve(cpu.size());
        for (size_t i = 0; i < cpu.size(); ++i) {
          cuda[i].Resize(cpu[i].dims());
          cuda[i].set_layout(cpu[i].layout());
          gpu_ptrs.emplace_back(cuda[i].mutable_data(place_, cpu[i].type()));
        }

        // NOTE(zjl): cudaStreamWaitEvent() must be called after all
        // cuda[i].mutable_data() is called, since some ops release
        // cuda memory immediately without waiting cuda kernel ends
        platform::SetDeviceId(
            BOOST_GET_CONST(platform::CUDAPlace, place_).device);
#ifdef PADDLE_WITH_HIP
        PADDLE_ENFORCE_CUDA_SUCCESS(
            hipEventRecord(events_[i].get(), compute_stream_));
        PADDLE_ENFORCE_CUDA_SUCCESS(
            hipStreamWaitEvent(stream_.get(), events_[i].get(), 0));
#else
        PADDLE_ENFORCE_CUDA_SUCCESS(
            cudaEventRecord(events_[i].get(), compute_stream_));
        PADDLE_ENFORCE_CUDA_SUCCESS(
            cudaStreamWaitEvent(stream_.get(), events_[i].get(), 0));
#endif

        platform::RecordEvent record_event("BufferedReader:MemoryCopy");
        for (size_t i = 0; i < cpu.size(); ++i) {
          auto cpu_place = cpu[i].place();
          auto cpu_ptr = cpu[i].data<void>();
          auto gpu_ptr = gpu_ptrs[i];
          auto size =
              cpu[i].numel() * paddle::framework::SizeOfType(cpu[i].type());
          if (platform::is_cuda_pinned_place(cpu_place)) {
            memory::Copy(BOOST_GET_CONST(platform::CUDAPlace, place_), gpu_ptr,
                         BOOST_GET_CONST(platform::CUDAPinnedPlace, cpu_place),
                         cpu_ptr, size, stream_.get());
          } else if ((platform::is_gpu_place(cpu_place))) {
            memory::Copy(BOOST_GET_CONST(platform::CUDAPlace, place_), gpu_ptr,
                         BOOST_GET_CONST(platform::CUDAPlace, cpu_place),
                         cpu_ptr, size, stream_.get());
          } else {
            platform::CUDAPinnedPlace cuda_pinned_place;
            framework::LoDTensor cuda_pinned_tensor;
            cuda_pinned_tensor.Resize(cpu[i].dims());
            auto cuda_pinned_ptr = cuda_pinned_tensor.mutable_data(
                cuda_pinned_place, cpu[i].type());
            memory::Copy(cuda_pinned_place, cuda_pinned_ptr,
                         BOOST_GET_CONST(platform::CPUPlace, cpu_place),
                         cpu_ptr, size);
            memory::Copy(BOOST_GET_CONST(platform::CUDAPlace, place_), gpu_ptr,
                         cuda_pinned_place, cuda_pinned_ptr, size,
                         stream_.get());
#ifdef PADDLE_WITH_HIP
            PADDLE_ENFORCE_CUDA_SUCCESS(hipStreamSynchronize(stream_.get()));
#else
            PADDLE_ENFORCE_CUDA_SUCCESS(cudaStreamSynchronize(stream_.get()));
#endif
          }
          cuda[i].set_lod(cpu[i].lod());
        }
#ifdef PADDLE_WITH_HIP
        PADDLE_ENFORCE_CUDA_SUCCESS(hipStreamSynchronize(stream_.get()));
#else
        PADDLE_ENFORCE_CUDA_SUCCESS(cudaStreamSynchronize(stream_.get()));
#endif
      }
    }
#endif

#ifdef PADDLE_WITH_ASCEND_CL
    if (platform::is_npu_place(place_)) {
      TensorVec &npu = npu_buffer_[i];
      if (npu.empty()) {
        npu.resize(cpu.size());
      } else {
        PADDLE_ENFORCE_EQ(
            npu.size(), cpu.size(),
            platform::errors::InvalidArgument(
                "Input tensor number on NPU and CPU devices are not matched. "
                "The number on NPU is %d, on CPU is %d",
                npu.size(), cpu.size()));
      }

      std::vector<void *> npu_ptrs;
      npu_ptrs.reserve(cpu.size());
      for (size_t i = 0; i < cpu.size(); ++i) {
        npu[i].Resize(cpu[i].dims());
        npu[i].set_layout(cpu[i].layout());
        npu_ptrs.emplace_back(npu[i].mutable_data(place_, cpu[i].type()));
      }

      platform::SetNPUDeviceId(
          BOOST_GET_CONST(platform::NPUPlace, place_).device);
      PADDLE_ENFORCE_NPU_SUCCESS(
          aclrtRecordEvent(events_[i].get(), compute_stream_));
      PADDLE_ENFORCE_NPU_SUCCESS(
          aclrtStreamWaitEvent(stream_.get(), events_[i].get()));

      platform::RecordEvent record_event("BufferedReader:MemoryCopy");
      for (size_t i = 0; i < cpu.size(); ++i) {
        auto cpu_place = cpu[i].place();
        auto cpu_ptr = cpu[i].data<void>();
        auto npu_ptr = npu_ptrs[i];
        auto size =
            cpu[i].numel() * paddle::framework::SizeOfType(cpu[i].type());
        if ((platform::is_npu_place(cpu_place))) {
          memory::Copy(BOOST_GET_CONST(platform::NPUPlace, place_), npu_ptr,
                       BOOST_GET_CONST(platform::NPUPlace, cpu_place), cpu_ptr,
                       size, stream_.get());
        } else {
          memory::Copy(BOOST_GET_CONST(platform::NPUPlace, place_), npu_ptr,
                       BOOST_GET_CONST(platform::CPUPlace, cpu_place), cpu_ptr,
                       size, stream_.get());
          PADDLE_ENFORCE_NPU_SUCCESS(aclrtSynchronizeStream(stream_.get()));
        }
        npu[i].set_lod(cpu[i].lod());
      }
      PADDLE_ENFORCE_NPU_SUCCESS(aclrtSynchronizeStream(stream_.get()));
    }
#endif
    return i;
  }));
}

void BufferedReader::ShutdownImpl() {
  VLOG(1) << "ShutdownImpl";
  reader_->Shutdown();
  while (!position_.empty()) {
    position_.pop();
  }
  prev_pos_ = -1UL;
}

void BufferedReader::StartImpl() {
  reader_->Start();
  ReadTillBufferFullAsync();
}

void BufferedReader::ReadNextImpl(std::vector<framework::LoDTensor> *out) {
  if (position_.empty()) {
    out->clear();
    return;
  }
  size_t i = position_.front().get();
  position_.pop();

  if (i == -1UL) {
    ReadNextImpl(out);
    return;
  }

  if (platform::is_gpu_place(place_) && !is_same_place_) {
    *out = std::move(cuda_buffer_[i]);
  } else if (platform::is_npu_place(place_) && !is_same_place_) {
    *out = std::move(npu_buffer_[i]);
  } else {
    *out = std::move(cpu_buffer_[i]);
  }

  // Do not push current position into ReadAsync. Push the previous position
  // Since all computation in fluid are async, change the data of
  // current position may cause data error.
  if (prev_pos_ != -1Ul) {
    ReadAsync(prev_pos_);
  }
  prev_pos_ = i;
}

}  // namespace reader
}  // namespace operators
}  // namespace paddle
