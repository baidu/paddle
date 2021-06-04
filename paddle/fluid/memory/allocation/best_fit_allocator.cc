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

#include "paddle/fluid/memory/allocation/best_fit_allocator.h"

#include <cmath>

#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace memory {
namespace allocation {

static int HighestBitPos(size_t N) {
  if (UNLIKELY(N == 0)) {
    return 0;
  } else {
#ifdef __GNUCC__
    return sizeof(unsigned int) * 8 - __builtin_clz(N);
#else
    return static_cast<int>(std::log2(N) + 1);
#endif
  }
}

BestFitAllocator::BestFitAllocator(Allocation* allocation)
    : allocation_(allocation) {
  details::Chunk chunk;
  chunk.size_ = allocation_->size();
  chunk.offset_ = 0;
  chunk.is_free = true;
  chunks_.emplace_back(chunk);
  free_chunks_[HighestBitPos(chunk.size_)].insert(
      {chunk.size_, chunks_.begin()});
}

size_t BestFitAllocator::FreeSize() const {
  size_t acc = 0;
  for (auto& array_item : free_chunks_) {
    for (auto& pair : array_item) {
      acc += pair.second->size_;
    }
  }
  return acc;
}

BestFitAllocator::ListIt BestFitAllocator::SplitChunk(size_t request_size,
                                                      size_t free_chunk_offset,
                                                      MapIt bin_iterator) {
  auto to_split_it = bin_iterator->second;
  free_chunks_[free_chunk_offset].erase(bin_iterator);

  PADDLE_ENFORCE_EQ(to_split_it->is_free, true,
                    platform::errors::PreconditionNotMet(
                        "The memory chunk to split is not free"));
  PADDLE_ENFORCE_GE(to_split_it->size_, request_size,
                    platform::errors::PreconditionNotMet(
                        "The size of memory chunk to split is "
                        "not larger than size of request memory"));

  auto remaining_size = to_split_it->size_ - request_size;
  details::Chunk to_use;
  details::Chunk remaining;
  to_use.size_ = request_size;
  to_use.is_free = false;
  remaining.size_ = remaining_size;
  remaining.is_free = true;

  // calc offsets
  to_use.offset_ = to_split_it->offset_;
  remaining.offset_ = to_use.offset_ + to_use.size_;

  // insert to chunk list
  auto to_use_it = chunks_.insert(to_split_it, to_use);
  if (remaining.size_ != 0) {
    auto bit_size = static_cast<size_t>(HighestBitPos(remaining.size_));
    free_chunks_[bit_size].insert(
        {remaining.size_, chunks_.insert(to_split_it, remaining)});
  }
  chunks_.erase(to_split_it);
  return to_use_it;
}

void BestFitAllocator::InsertFreeNode(const ListIt& it) {
  auto pos = static_cast<size_t>(HighestBitPos(it->size_));
  auto& free_map = free_chunks_[pos];
  free_map.insert({it->size_, it});
}
void BestFitAllocator::EraseFreeNode(const ListIt& it) {
  auto pos = static_cast<size_t>(HighestBitPos(it->size_));
  auto& free_map = free_chunks_[pos];
  auto map_it = free_map.find(it->size_);
  while (map_it->second != it && map_it != free_map.end()) {
    ++map_it;
  }
  PADDLE_ENFORCE_NE(
      map_it, free_map.end(),
      platform::errors::NotFound("The node to erase is not found in map"));
  free_map.erase(map_it);
}
size_t BestFitAllocator::NumFreeChunks() const {
  size_t num = 0;
  for (auto& array_item : free_chunks_) {
    num += array_item.size();
  }
  return num;
}
void BestFitAllocator::FreeImpl(Allocation* allocation) {
  auto* bf_allocation = dynamic_cast<BestFitAllocation*>(allocation);
  PADDLE_ENFORCE_NOT_NULL(
      bf_allocation,
      platform::errors::InvalidArgument(
          "The input allocation is not type of BestFitAllocation."));
  auto chunk_it = bf_allocation->ChunkIterator();
  PADDLE_ENFORCE_EQ(chunk_it->is_free, false,
                    platform::errors::PreconditionNotMet(
                        "The chunk of allocation to free is freed already"));
  chunk_it->is_free = true;
  if (chunk_it != chunks_.begin()) {
    auto prev_it = chunk_it;
    --prev_it;

    if (prev_it->is_free) {
      // Merge Left.
      EraseFreeNode(prev_it);
      prev_it->size_ += chunk_it->size_;
      chunks_.erase(chunk_it);
      chunk_it = prev_it;
    }
  }

  auto next_it = chunk_it;
  ++next_it;
  if (next_it != chunks_.end() && next_it->is_free) {
    EraseFreeNode(next_it);
    chunk_it->size_ += next_it->size_;
    chunks_.erase(next_it);
  }

  InsertFreeNode(chunk_it);
  delete allocation;
}
Allocation* BestFitAllocator::AllocateImpl(size_t size) {
  auto highest_set_bit = static_cast<size_t>(HighestBitPos(size));
  MapIt map_it;
  for (; highest_set_bit < free_chunks_.size(); ++highest_set_bit) {
    map_it = free_chunks_[highest_set_bit].lower_bound(size);
    if (map_it != free_chunks_[highest_set_bit].end()) {
      break;
    }
  }
  if (UNLIKELY(highest_set_bit == free_chunks_.size())) {
    PADDLE_THROW_BAD_ALLOC(platform::errors::ResourceExhausted(
        "Cannot allocate %d, All fragments size is %d.", size, FreeSize()));
  }
  auto chunk_it = SplitChunk(size, highest_set_bit, map_it);
  return new BestFitAllocation(this, chunk_it);
}

BestFitAllocation::BestFitAllocation(
    paddle::memory::allocation::BestFitAllocator* allocator,
    typename details::ChunkList::iterator chunk_it)
    : Allocation(reinterpret_cast<void*>(
                     reinterpret_cast<uintptr_t>(allocator->BasePtr()) +
                     chunk_it->offset_),
                 chunk_it->size_, allocator->Place()),
      chunk_it_(chunk_it) {}
}  // namespace allocation
}  // namespace memory
}  // namespace paddle
