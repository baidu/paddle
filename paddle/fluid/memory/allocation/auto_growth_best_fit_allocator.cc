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

#include "paddle/fluid/memory/allocation/auto_growth_best_fit_allocator.h"

#include <algorithm>
#include <mutex>  // NOLINT

#include "paddle/fluid/memory/allocation/aligned_allocator.h"

DEFINE_bool(free_idle_chunk, false,
            "Whether to free idle chunk when each allocation is freed. "
            "If false, all freed allocation would be cached to speed up next "
            "allocation request. If true, no allocation would be cached. This "
            "flag only works when FLAGS_allocator_strategy=auto_growth.");

DEFINE_bool(free_when_no_cache_hit, false,
            "Whether to free idle chunks when no cache hit. If true, idle "
            "chunk would be freed when no cache hit; if false, idle "
            "chunk would be freed when out of memory occurs. This flag "
            "only works when FLAGS_allocator_strategy=auto_growth.");

namespace paddle {
namespace memory {
namespace allocation {

AutoGrowthBestFitAllocator::AutoGrowthBestFitAllocator(
    const std::shared_ptr<Allocator> &underlying_allocator, size_t alignment,
    size_t chunk_size)
    : underlying_allocator_(
          std::make_shared<AlignedAllocator>(underlying_allocator, alignment)),
      alignment_(alignment),
      chunk_size_(std::max(AlignedSize(chunk_size, alignment), alignment)) {}

Allocation *AutoGrowthBestFitAllocator::AllocateImpl(size_t size) {
  size = AlignedSize(size, alignment_);

  std::lock_guard<std::mutex> guard(mtx_);
  auto iter = free_blocks_.lower_bound(std::make_pair(size, nullptr));
  BlockIt block_it;
  if (iter != free_blocks_.end()) {
    block_it = iter->second;
    free_blocks_.erase(iter);
    auto *chunk = block_it->chunk_;
    size_t remaining_size = block_it->size_ - size;
    if (remaining_size == 0) {
      block_it->is_free_ = false;
    } else {
      auto remaining_free_block = chunk->blocks_.insert(
          block_it, Block(block_it->ptr_, remaining_size, true, chunk));
      free_blocks_.emplace(std::make_pair(remaining_size, block_it->ptr_),
                           remaining_free_block);
      block_it->ptr_ =
          reinterpret_cast<uint8_t *>(block_it->ptr_) + remaining_size;
      block_it->size_ = size;
      block_it->is_free_ = false;
    }
  } else {
    if (FLAGS_free_when_no_cache_hit) {
      FreeIdleChunks();
    }
    size_t realloc_size = std::max(size, chunk_size_);

    try {
      chunks_.emplace_back(underlying_allocator_->Allocate(realloc_size));
    } catch (BadAlloc &ex) {
      if (FLAGS_free_when_no_cache_hit) throw ex;
      FreeIdleChunks();
      chunks_.emplace_back(underlying_allocator_->Allocate(realloc_size));
    }

    auto *chunk = &(*chunks_.rbegin());
    realloc_size = chunk->allocation_->size();
    auto *p = reinterpret_cast<uint8_t *>(chunk->allocation_->ptr());
    auto &blocks = chunk->blocks_;

    size_t remaining_size = realloc_size - size;
    if (remaining_size > 0) {
      blocks.emplace_back(p, remaining_size, true, chunk);
      free_blocks_.emplace(std::make_pair(remaining_size, p), --(blocks.end()));
    }
    blocks.emplace_back(p + remaining_size, size, false, chunk);
    block_it = --(blocks.end());
    VLOG(2) << "Not found and reallocate " << realloc_size << ", and remaining "
            << remaining_size;
  }
  return new BlockAllocation(block_it);
}

void AutoGrowthBestFitAllocator::FreeImpl(Allocation *allocation) {
  std::lock_guard<std::mutex> guard(mtx_);
  auto block_it = static_cast<BlockAllocation *>(allocation)->block_it_;
  auto &blocks = block_it->chunk_->blocks_;

  block_it->is_free_ = true;

  if (block_it != blocks.begin()) {
    auto prev_it = block_it;
    --prev_it;

    if (prev_it->is_free_) {
      free_blocks_.erase(std::make_pair(prev_it->size_, prev_it->ptr_));
      prev_it->size_ += block_it->size_;
      blocks.erase(block_it);
      block_it = prev_it;
    }
  }

  auto next_it = block_it;
  ++next_it;

  if (next_it != blocks.end() && next_it->is_free_) {
    free_blocks_.erase(std::make_pair(next_it->size_, next_it->ptr_));
    block_it->size_ += next_it->size_;
    blocks.erase(next_it);
  }

  free_blocks_.emplace(std::make_pair(block_it->size_, block_it->ptr_),
                       block_it);

  delete allocation;

  if (FLAGS_free_idle_chunk) {
    FreeIdleChunks();
  }
}

uint64_t AutoGrowthBestFitAllocator::FreeIdleChunks() {
  uint64_t bytes = 0;
  for (auto chunk_it = chunks_.begin(); chunk_it != chunks_.end();) {
    auto &blocks = chunk_it->blocks_;
    if (blocks.size() == 1 && blocks.begin()->is_free_) {
      auto &block = *blocks.begin();
      VLOG(2) << "Free chunk with size " << block.size_;
      bytes += block.size_;
      free_blocks_.erase(std::make_pair(block.size_, block.ptr_));
      chunk_it = chunks_.erase(chunk_it);
    } else {
      ++chunk_it;
    }
  }
  return bytes;
}

}  // namespace allocation
}  // namespace memory
}  // namespace paddle
