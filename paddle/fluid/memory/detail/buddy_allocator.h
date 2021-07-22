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

#include <stdint.h>
#include <memory>
#include <mutex>  // NOLINT
#include <set>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "paddle/fluid/memory/detail/memory_block.h"
#include "paddle/fluid/memory/detail/system_allocator.h"
#include "paddle/fluid/platform/cpu_info.h"
#include "paddle/fluid/platform/gpu_info.h"
#include "paddle/fluid/platform/npu_info.h"
#include "paddle/fluid/platform/xpu_info.h"

namespace paddle {
namespace memory {
namespace detail {

class BuddyAllocator {
 public:
  BuddyAllocator(std::unique_ptr<SystemAllocator> system_allocator,
                 size_t min_chunk_size, size_t max_chunk_size);

  ~BuddyAllocator();

 public:
  void* Alloc(size_t unaligned_size);
  void Free(void* ptr);
  // Release the unused memory pool, a real free operation for the OS.
  uint64_t Release();
  size_t Used();
  size_t GetMinChunkSize();
  size_t GetMaxChunkSize();

 public:
  // Disable copy and assignment
  BuddyAllocator(const BuddyAllocator&) = delete;
  BuddyAllocator& operator=(const BuddyAllocator&) = delete;

 private:
  // Tuple (allocator index, memory size, memory address)
  using IndexSizeAddress = std::tuple<size_t, size_t, void*>;
  // Each element in PoolSet is a free allocation
  using PoolSet = std::set<IndexSizeAddress>;

  /*! \brief Allocate fixed-size memory from system */
  void* SystemAlloc(size_t size);

  /*! \brief If existing chunks are not suitable, refill pool */
  PoolSet::iterator RefillPool(size_t request_bytes);

  /**
   *  \brief   Find the suitable chunk from existing pool and split
   *           it to left and right buddies
   *
   *  \param   it     the iterator of pool list
   *  \param   size   the size of allocation
   *
   *  \return  the left buddy address
   */
  void* SplitToAlloc(PoolSet::iterator it, size_t size);

  /*! \brief Find the existing chunk which used to allocation */
  PoolSet::iterator FindExistChunk(size_t size);

 private:
  size_t total_used_ = 0;  // the total size of used memory
  size_t total_free_ = 0;  // the total size of free memory

  size_t min_chunk_size_;  // the minimum size of each chunk
  size_t max_chunk_size_;  // the maximum size of each chunk

  size_t realloc_size_ = 0;  // the size of re-allocated chunk

 private:
  /**
   * \brief A list of free allocation
   *
   * \note  Only store free chunk memory in pool
   */
  PoolSet pool_;

  /**
   * \brief Record the allocated chunks when Refill pool.
   */
  PoolSet chunks_;

 private:
  /*! Unify the metadata format between GPU and CPU allocations */
  MetadataCache cache_;

 private:
  /*! Allocate CPU/GPU memory from system */
  std::unique_ptr<SystemAllocator> system_allocator_;
  std::mutex mutex_;
};

}  // namespace detail
}  // namespace memory
}  // namespace paddle
