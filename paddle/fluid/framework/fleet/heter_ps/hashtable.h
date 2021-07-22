/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

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
#include <glog/logging.h>
#include <limits>
#include <memory>
#include <vector>
#ifdef PADDLE_WITH_PSLIB
#include "common_value.h"  // NOLINT
#endif
#ifdef PADDLE_WITH_PSCORE
#include "paddle/fluid/distributed/table/depends/large_scale_kv.h"
#endif
#include "paddle/fluid/framework/rw_lock.h"
#include "thrust/pair.h"
// #include "cudf/concurrent_unordered_map.cuh.h"
#include "paddle/fluid/framework/fleet/heter_ps/cudf/concurrent_unordered_map.cuh.h"
#ifdef PADDLE_WITH_HETERPS
#include "paddle/fluid/platform/type_defs.h"

namespace paddle {
namespace framework {

template <typename KeyType, typename ValType>
class TableContainer
    : public concurrent_unordered_map<KeyType, ValType,
                                      std::numeric_limits<KeyType>::max()> {
 public:
  TableContainer(size_t capacity)
      : concurrent_unordered_map<KeyType, ValType,
                                 std::numeric_limits<KeyType>::max()>(
            capacity, ValType()) {}
};

template <typename KeyType, typename ValType>
class HashTable {
 public:
  HashTable(size_t capacity);
  virtual ~HashTable();
  HashTable(const HashTable&) = delete;
  HashTable& operator=(const HashTable&) = delete;
  void insert(const KeyType* d_keys, const ValType* d_vals, size_t len,
              gpuStream_t stream);
  void get(const KeyType* d_keys, ValType* d_vals, size_t len,
           gpuStream_t stream);
  void show();
  void dump_to_cpu(int devid, cudaStream_t stream);

  template <typename GradType, typename Sgd>
  void update(const KeyType* d_keys, const GradType* d_grads, size_t len,
              Sgd sgd, gpuStream_t stream);

  int size() { return container_->size(); }

  std::unique_ptr<RWLock> rwlock_{nullptr};

 private:
  TableContainer<KeyType, ValType>* container_;
  int BLOCK_SIZE_{256};
  float LOAD_FACTOR{0.75f};
  size_t capacity_;
};
}  // end namespace framework
}  // end namespace paddle
#include "hashtable_inl.h"
#endif
