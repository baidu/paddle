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

#pragma once

#include <list>
#include <memory>
#include <queue>
#include <vector>
#include "ThreadPool.h"
#include "paddle/fluid/framework/reader.h"
#ifdef PADDLE_WITH_CUDA
#include "paddle/fluid/platform/cuda_resource_pool.h"
#include "paddle/fluid/platform/gpu_info.h"
#endif

namespace paddle {
namespace operators {
namespace reader {

class BufferedReader : public framework::DecoratedReader {
  using TensorVec = std::vector<framework::LoDTensor>;
  using VecFuture = std::future<TensorVec>;

 public:
  BufferedReader(const std::shared_ptr<framework::ReaderBase>& reader,
                 const platform::Place& place, size_t buffer_size);

  ~BufferedReader() override;

 private:
  void ReadTillBufferFullAsync();

  void ReadAsync(size_t i);

 protected:
  void ShutdownImpl() override;
  void StartImpl() override;
  void ReadNextImpl(std::vector<framework::LoDTensor>* out) override;

 private:
  ThreadPool thread_pool_;
  platform::Place place_;
  const size_t buffer_size_;

  std::queue<std::future<size_t>> position_;

  // The buffer for reading data.
  // NOTE: the simplest way to implement buffered reader is do not use any
  // buffer, just read async and create futures as buffer size. However, to
  // malloc tensors every time is extremely slow. Here we store all data in
  // buffers and prevent alloc every time.
  std::vector<TensorVec> cpu_buffer_;
  std::vector<TensorVec> gpu_buffer_;
  size_t prev_pos_{-1UL};
#ifdef PADDLE_WITH_CUDA
  cudaStream_t compute_stream_;
  std::shared_ptr<platform::CudaStreamObject> stream_;
  std::vector<std::shared_ptr<platform::CudaEventObject>> events_;
#endif
};

class SharedBufferedReader : public framework::DecoratedReader {
  using TensorVec = std::vector<framework::LoDTensor>;
  using VecFuture = std::future<TensorVec>;

 public:
  SharedBufferedReader(const std::shared_ptr<framework::ReaderBase>& reader,
                       const platform::Place& place, size_t buffer_size);

  ~SharedBufferedReader() override;

 private:
  void ReadTillBufferFullAsync();

  void ReadAsync(size_t i);

 protected:
  void ShutdownImpl() override;
  void StartImpl() override;
  std::shared_ptr<std::vector<framework::LoDTensor>> ReadNextSharedImpl()
      override;

 private:
  ThreadPool thread_pool_;
  platform::Place place_;
  const size_t buffer_size_;

  std::queue<std::future<int64_t>> position_;

  // The buffer for reading data.
  // NOTE: the simplest way to implement buffered reader is do not use any
  // buffer, just read async and create futures as buffer size. However, to
  // malloc tensors every time is extremely slow. Here we store all data in
  // buffers and prevent alloc every time.
  std::vector<std::shared_ptr<TensorVec>> cpu_buffer_;
  std::vector<std::shared_ptr<TensorVec>> gpu_buffer_;
  int64_t prev_pos_{-1L};
#ifdef PADDLE_WITH_CUDA
  cudaStream_t compute_stream_;
  std::shared_ptr<platform::CudaStreamObject> stream_;
  std::vector<std::shared_ptr<platform::CudaEventObject>> events_;
#endif
};

}  // namespace reader
}  // namespace operators
}  // namespace paddle
