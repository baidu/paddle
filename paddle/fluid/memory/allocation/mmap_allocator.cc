// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#ifndef _WIN32

#include "paddle/fluid/memory/allocation/mmap_allocator.h"

#include <fcntl.h>
#include <sys/mman.h>

#include <cstdlib>
#include <random>
#include <string>

#include "glog/logging.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace memory {
namespace allocation {

MemoryMapWriterAllocation::~MemoryMapWriterAllocation() {
  PADDLE_ENFORCE_NE(
      munmap(this->ptr(), this->size()), -1,
      platform::errors::Unavailable("could not unmap the shared memory file %s",
                                    this->ipc_name()));
}

MemoryMapReaderAllocation::~MemoryMapReaderAllocation() {
  PADDLE_ENFORCE_NE(
      munmap(this->ptr(), this->size()), -1,
      platform::errors::Unavailable("could not unmap the shared memory file %s",
                                    this->ipc_name()));
  /* Here we do not pay attention to the result of shm_unlink,
     because the memory mapped file may have been cleared due to the
     MemoryMapFdSet::Clear() */
  shm_unlink(this->ipc_name().c_str());
  MemoryMapFdSet::Instance().Remove(this->ipc_name());
  VLOG(3) << "~MemoryMapReaderAllocation: " << this->ipc_name();
}

std::string GetIPCName() {
  static std::random_device rd;
  std::string handle = "/paddle_";
#ifdef _WIN32
  handle += std::to_string(GetCurrentProcessId());
#else
  handle += std::to_string(getpid());
#endif
  handle += "_";
  handle += std::to_string(rd());
  return std::move(handle);
}

std::shared_ptr<MemoryMapWriterAllocation> AllocateMemoryMapWriterAllocation(
    size_t size) {
  const std::string &ipc_name = GetIPCName();
  int flags = O_RDWR | O_CREAT;

  int fd = shm_open(ipc_name.c_str(), flags, 0644);
  PADDLE_ENFORCE_NE(fd, -1,
                    platform::errors::Unavailable(
                        "File descriptor %s open failed", ipc_name.c_str()));
  PADDLE_ENFORCE_EQ(ftruncate(fd, size), 0,
                    platform::errors::Unavailable(
                        "Fruncate a file to a specified length failed!"));

  void *ptr = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  PADDLE_ENFORCE_NE(ptr, MAP_FAILED,
                    platform::errors::Unavailable(
                        "Memory map failed when create shared memory."));
  close(fd);

  return std::make_shared<MemoryMapWriterAllocation>(ptr, size, ipc_name);
}

std::shared_ptr<MemoryMapReaderAllocation> RebuildMemoryMapReaderAllocation(
    const std::string &ipc_name, size_t size) {
  int fd = shm_open(ipc_name.c_str(), O_RDONLY, 0644);
  PADDLE_ENFORCE_NE(fd, -1,
                    platform::errors::Unavailable(
                        "File descriptor %s open failed", ipc_name.c_str()));

  void *ptr = mmap(nullptr, size, PROT_READ, MAP_SHARED, fd, 0);
  PADDLE_ENFORCE_NE(ptr, MAP_FAILED,
                    platform::errors::Unavailable(
                        "Memory map failed when rebuild shared memory."));
  close(fd);
  return std::make_shared<MemoryMapReaderAllocation>(ptr, size, ipc_name);
}

MemoryMapFdSet &MemoryMapFdSet::Instance() {  // NOLINT
  static MemoryMapFdSet set;
  return set;
}

void MemoryMapFdSet::Insert(const std::string &ipc_name) {
  std::lock_guard<std::mutex> guard(mtx_);
  fd_set_.emplace(ipc_name);
  VLOG(3) << "PID: " << getpid() << ", MemoryMapFdSet: insert " << ipc_name
          << ", set size: " << fd_set_.size();
}

void MemoryMapFdSet::Remove(const std::string &ipc_name) {
  std::lock_guard<std::mutex> guard(mtx_);
  fd_set_.erase(ipc_name);
  VLOG(3) << "PID: " << getpid() << ", MemoryMapFdSet: erase " << ipc_name
          << ", set size: " << fd_set_.size();
}

void MemoryMapFdSet::Clear() {
  VLOG(3) << "PID: " << getpid() << ", MemoryMapFdSet: set size - "
          << fd_set_.size();
  std::lock_guard<std::mutex> guard(mtx_);
  for (auto fd : fd_set_) {
    int rlt = shm_unlink(fd.c_str());
    if (rlt == 0) {
      VLOG(3) << "PID: " << getpid() << ", MemoryMapFdSet: clear " << fd;
    }
  }
  fd_set_.clear();
}

MemoryMapFdSet::~MemoryMapFdSet() { Clear(); }

}  // namespace allocation
}  // namespace memory
}  // namespace paddle

#endif
