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

#include "paddle/fluid/memory/detail/buddy_allocator.h"
#include <algorithm>

#include "gflags/gflags.h"
#include "glog/logging.h"

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
DECLARE_uint64(reallocate_gpu_memory_in_mb);
#endif
#ifdef PADDLE_WITH_ASCEND_CL
DECLARE_uint64(reallocate_gpu_memory_in_mb);
#endif
#ifdef PADDLE_WITH_XPU
DECLARE_uint64(reallocate_xpu_memory_in_mb);
#endif

namespace paddle {
namespace memory {
namespace detail {

BuddyAllocator::BuddyAllocator(
    std::unique_ptr<SystemAllocator> system_allocator, size_t min_chunk_size,
    size_t max_chunk_size)
    : min_chunk_size_(min_chunk_size),
      max_chunk_size_(max_chunk_size),
#ifdef PADDLE_WITH_XPU
      cache_(system_allocator->UseXPU()),
#else
      cache_(system_allocator->UseGpu()),
#endif
      system_allocator_(std::move(system_allocator)) {
}

BuddyAllocator::~BuddyAllocator() {
  VLOG(10) << "BuddyAllocator Disconstructor makes sure that all of these "
              "have actually been freed";
  while (!pool_.empty()) {
    auto block = static_cast<MemoryBlock*>(std::get<2>(*pool_.begin()));
    auto desc = cache_.LoadDesc(block);
    VLOG(10) << "Free from block (" << block << ", " << desc->get_total_size()
             << ")";

    system_allocator_->Free(block, desc->get_total_size(), desc->get_index());
    cache_.Invalidate(block);
    pool_.erase(pool_.begin());
  }
}

inline size_t align(size_t size, size_t alignment) {
  size_t remaining = size % alignment;
  return remaining == 0 ? size : size + (alignment - remaining);
}

void* BuddyAllocator::Alloc(size_t unaligned_size) {
  // adjust allocation alignment
  size_t size =
      align(unaligned_size + sizeof(MemoryBlock::Desc), min_chunk_size_);

  // acquire the allocator lock
  std::lock_guard<std::mutex> lock(mutex_);

  VLOG(10) << "Allocate " << unaligned_size << " bytes from chunk size "
           << size;

  // if the allocation is huge, send directly to the system allocator
  if (size > max_chunk_size_) {
    VLOG(10) << "Allocate from system allocator.";
    return SystemAlloc(size);
  }

  // query and allocate from the existing chunk
  auto it = FindExistChunk(size);

  // refill the pool if failure
  if (it == pool_.end()) {
    it = RefillPool(size);
    // if still failure, fail fatally
    if (it == pool_.end()) {
      return nullptr;
    }
  } else {
    VLOG(10) << "Allocation from existing memory block " << std::get<2>(*it)
             << " at address "
             << reinterpret_cast<MemoryBlock*>(std::get<2>(*it))->Data();
  }

  total_used_ += size;
  total_free_ -= size;

  // split the allocation and return data for use
  return reinterpret_cast<MemoryBlock*>(SplitToAlloc(it, size))->Data();
}

void BuddyAllocator::Free(void* p) {
  // Point back to metadata
  auto block = static_cast<MemoryBlock*>(p)->Metadata();

  // Acquire the allocator lock
  std::lock_guard<std::mutex> lock(mutex_);

  VLOG(10) << "Free from address " << block;

  auto* desc = cache_.LoadDesc(block);
  if (desc->get_type() == MemoryBlock::HUGE_CHUNK) {
    VLOG(10) << "Free directly from system allocator";
    system_allocator_->Free(block, desc->get_total_size(), desc->get_index());

    // Invalidate GPU allocation from cache
    cache_.Invalidate(block);

    return;
  }

  block->MarkAsFree(&cache_);

  total_used_ -= desc->get_total_size();
  total_free_ += desc->get_total_size();

  // Trying to merge the right buddy
  MemoryBlock* right_buddy = block->GetRightBuddy(&cache_);
  if (right_buddy) {
    VLOG(10) << "Merging this block " << block << " with its right buddy "
             << right_buddy;

    auto rb_desc = cache_.LoadDesc(right_buddy);
    if (rb_desc->get_type() == MemoryBlock::FREE_CHUNK) {
      // Take away right buddy from pool
      pool_.erase(IndexSizeAddress(rb_desc->get_index(),
                                   rb_desc->get_total_size(), right_buddy));

      // merge its right buddy to the block
      block->Merge(&cache_, right_buddy);
    }
  }

  // Trying to merge the left buddy
  MemoryBlock* left_buddy = block->GetLeftBuddy(&cache_);
  if (left_buddy) {
    VLOG(10) << "Merging this block " << block << " with its left buddy "
             << left_buddy;

    // auto left_buddy = block->left_buddy(cache_);
    auto* lb_desc = cache_.LoadDesc(left_buddy);
    if (lb_desc->get_type() == MemoryBlock::FREE_CHUNK) {
      // Take away right buddy from pool
      pool_.erase(IndexSizeAddress(lb_desc->get_index(),
                                   lb_desc->get_total_size(), left_buddy));

      // merge the block to its left buddy
      left_buddy->Merge(&cache_, block);
      block = left_buddy;
      desc = lb_desc;
    }
  }

  // Dumping this block into pool
  VLOG(10) << "Inserting free block (" << block << ", "
           << desc->get_total_size() << ")";
  pool_.insert(
      IndexSizeAddress(desc->get_index(), desc->get_total_size(), block));
}

uint64_t BuddyAllocator::Release() {
  std::lock_guard<std::mutex> lock(mutex_);
  int num = 0;
  uint64_t bytes = 0;
  bool del_flag = false;
  for (auto iter = pool_.begin(); iter != pool_.end();) {
    auto remain_size = std::get<1>(*iter);
    auto remain_ptr = std::get<2>(*iter);
    for (auto& chunk : chunks_) {
      auto init_size = std::get<1>(chunk);
      auto init_ptr = std::get<2>(chunk);

      if (init_size == remain_size && init_ptr == remain_ptr) {
        ++num;
        bytes += init_size;
        total_free_ -= init_size;
        auto block = static_cast<MemoryBlock*>(std::get<2>(chunk));
        system_allocator_->Free(init_ptr, init_size, std::get<0>(chunk));
        cache_.Invalidate(block);
        del_flag = true;
        break;
      }
    }

    if (del_flag) {
      iter = pool_.erase(iter);
    } else {
      iter++;
    }
  }
  VLOG(10) << "Release " << num << " chunk, Free " << bytes << " bytes.";
  return bytes;
}

size_t BuddyAllocator::Used() { return total_used_; }
size_t BuddyAllocator::GetMinChunkSize() { return min_chunk_size_; }
size_t BuddyAllocator::GetMaxChunkSize() { return max_chunk_size_; }

void* BuddyAllocator::SystemAlloc(size_t size) {
  size_t index = 0;
  void* p = system_allocator_->Alloc(&index, size);

  VLOG(10) << "Allocated " << p << " from system allocator.";

  if (p == nullptr) return nullptr;

  static_cast<MemoryBlock*>(p)->Init(&cache_, MemoryBlock::HUGE_CHUNK, index,
                                     size, nullptr, nullptr);

  return static_cast<MemoryBlock*>(p)->Data();
}

BuddyAllocator::PoolSet::iterator BuddyAllocator::RefillPool(
    size_t request_bytes) {
  size_t allocate_bytes = max_chunk_size_;
  size_t index = 0;

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  if (system_allocator_->UseGpu()) {
    if ((total_used_ + total_free_) == 0) {
      // Compute the allocation size for gpu for the first allocation.
      allocate_bytes = std::max(platform::GpuInitAllocSize(), request_bytes);
    } else {
      // Compute the re-allocation size, we store the re-allocation size when
      // user set FLAGS_reallocate_gpu_memory_in_mb to fix value.
      if (realloc_size_ == 0 || FLAGS_reallocate_gpu_memory_in_mb == 0ul) {
        realloc_size_ = platform::GpuReallocSize();
      }
      allocate_bytes = std::max(realloc_size_, request_bytes);
    }
  }
#endif
#ifdef PADDLE_WITH_ASCEND_CL
  if (system_allocator_->UseGpu()) {
    if ((total_used_ + total_free_) == 0) {
      // Compute the allocation size for gpu for the first allocation.
      allocate_bytes = std::max(platform::NPUInitAllocSize(), request_bytes);
    } else {
      // Compute the re-allocation size, we store the re-allocation size when
      // user set FLAGS_reallocate_gpu_memory_in_mb to fix value.
      if (realloc_size_ == 0 || FLAGS_reallocate_gpu_memory_in_mb == 0ul) {
        realloc_size_ = platform::NPUReallocSize();
      }
      allocate_bytes = std::max(realloc_size_, request_bytes);
    }
  }
#endif
#ifdef PADDLE_WITH_XPU
  if (system_allocator_->UseXPU()) {
    if ((total_used_ + total_free_) == 0) {
      // Compute the allocation size for xpu for the first allocation.
      allocate_bytes = std::max(platform::XPUInitAllocSize(), request_bytes);
    } else {
      // Compute the re-allocation size, we store the re-allocation size when
      // user set FLAGS_reallocate_xpu_memory_in_mb to fix value.
      if (realloc_size_ == 0 || FLAGS_reallocate_xpu_memory_in_mb == 0ul) {
        realloc_size_ = platform::XPUReallocSize();
      }
      allocate_bytes = std::max(realloc_size_, request_bytes);
    }
  }
#endif

  // Allocate a new block
  void* p = system_allocator_->Alloc(&index, allocate_bytes);

  if (p == nullptr) return pool_.end();

  VLOG(10) << "Creating and inserting new block " << p
           << " from system allocator";

  static_cast<MemoryBlock*>(p)->Init(&cache_, MemoryBlock::FREE_CHUNK, index,
                                     allocate_bytes, nullptr, nullptr);

  total_free_ += allocate_bytes;

  // record the chunk.
  chunks_.insert(IndexSizeAddress(index, allocate_bytes, p));

  // dump the block into pool
  return pool_.insert(IndexSizeAddress(index, allocate_bytes, p)).first;
}

BuddyAllocator::PoolSet::iterator BuddyAllocator::FindExistChunk(size_t size) {
  size_t index = 0;

  while (1) {
    auto it = pool_.lower_bound(IndexSizeAddress(index, size, nullptr));

    // no match chunk memory
    if (it == pool_.end()) return it;

    if (std::get<0>(*it) > index) {
      // find suitable one
      if (std::get<1>(*it) >= size) {
        return it;
      }
      // update and continue
      index = std::get<0>(*it);
      continue;
    }
    return it;
  }
}

void* BuddyAllocator::SplitToAlloc(BuddyAllocator::PoolSet::iterator it,
                                   size_t size) {
  auto block = static_cast<MemoryBlock*>(std::get<2>(*it));
  auto desc = cache_.LoadDesc(block);
  pool_.erase(it);

  VLOG(10) << "Split block (" << block << ", " << desc->get_total_size()
           << ") into";
  block->Split(&cache_, size);

  VLOG(10) << "Left block (" << block << ", " << desc->get_total_size() << ")";
  desc->set_type(MemoryBlock::ARENA_CHUNK);

  // the rest of memory if exist
  MemoryBlock* right_buddy = block->GetRightBuddy(&cache_);
  if (right_buddy) {
    auto* rb_desc = cache_.LoadDesc(right_buddy);
    if (rb_desc->get_type() == MemoryBlock::FREE_CHUNK) {
      VLOG(10) << "Insert right block (" << right_buddy << ", "
               << rb_desc->get_total_size() << ")";

      pool_.insert(IndexSizeAddress(rb_desc->get_index(),
                                    rb_desc->get_total_size(), right_buddy));
    }
  }

  return block;
}

}  // namespace detail
}  // namespace memory
}  // namespace paddle
