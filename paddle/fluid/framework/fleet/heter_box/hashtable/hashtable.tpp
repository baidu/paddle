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

/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#ifdef PADDLE_WITH_PSLIB

namespace paddle {
namespace framework {

template <typename value_type>
struct ReplaceOp {
  __host__ __device__ value_type operator()(value_type new_value, value_type old_value) {
    return new_value;
  }
};

template <typename Table>
__global__ void insert_kernel(Table* table, const typename Table::key_type* const keys,
                              const typename Table::mapped_type* const vals, size_t len) {
  ReplaceOp<typename Table::mapped_type> op;
  thrust::pair<typename Table::key_type, typename Table::mapped_type> kv;

  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len) {
    kv.first = keys[i];
    kv.second = vals[i];
    auto it = table->insert(kv, op);
    assert(it != table->end() && "error: insert fails: table is full");
  }
}

template <typename Table>
__global__ void search_kernel(Table* table, const typename Table::key_type* const keys,
                              typename Table::mapped_type* const vals, size_t len) {
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len) {
    auto it = table->find(keys[i]);
    assert(it != table->end() && "error: can't find key");
    vals[i] = it->second;
  }
}

template <typename Table, typename GradType, typename Sgd>
__global__ void update_kernel(Table* table,
                              const typename Table::key_type* const keys,
                              const GradType* const grads,
                              size_t len,
                              Sgd sgd){

    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < len) {
        auto it = table->find(keys[i]);
        sgd.update_value((it.getter())->second, grads[i]);
    }
}


template<typename KeyType, typename ValType>
HashTable<KeyType, ValType>::HashTable(size_t capacity) {
  container_ = new TableContainer<KeyType, ValType>(capacity);
}

template<typename KeyType, typename ValType>
HashTable<KeyType, ValType>::~HashTable() {
  delete container_;
}

template<typename KeyType, typename ValType>
void HashTable<KeyType, ValType>::show() {
  container_->print();
}

template<typename KeyType, typename ValType>
void HashTable<KeyType, ValType>::get(const KeyType* d_keys, ValType* d_vals, size_t len, cudaStream_t stream) {
  if (len == 0) {
    return;
  }
  const int grid_size = (len - 1) / BLOCK_SIZE_ + 1;
  search_kernel<<<grid_size, BLOCK_SIZE_, 0, stream>>>(container_, d_keys, d_vals, len);
}

template <typename KeyType, typename ValType>
void HashTable<KeyType, ValType>::insert(const KeyType* d_keys, const ValType* d_vals, size_t len,
                                         cudaStream_t stream) {
  if (len == 0) {
    return;
  }
  const int grid_size = (len - 1) / BLOCK_SIZE_ + 1;
  insert_kernel<<<grid_size, BLOCK_SIZE_, 0, stream>>>(container_, d_keys, d_vals, len);
}

template <typename KeyType, typename ValType>
template <typename GradType, typename Sgd>
void HashTable<KeyType, ValType>::update(const KeyType* d_keys, const GradType* d_grads, size_t len, Sgd sgd, cudaStream_t stream) {
  if (len == 0) {
    return;
  }
  const int grid_size = (len - 1) / BLOCK_SIZE_ + 1;
  update_kernel<<<grid_size, BLOCK_SIZE_, 0, stream>>>(container_, d_keys, d_grads, len, sgd);
}

}  // end namespace framework
}  // end namespace paddle
#endif
