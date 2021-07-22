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
#ifdef PADDLE_WITH_HETERPS
#include <queue>

namespace paddle {
namespace framework {

template <typename T>
__global__ void fill_idx(T* idx, size_t len) {
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len) {
    idx[i] = i;
  }
}

template <typename T>
void show_tensor(T* input, size_t len, gpuStream_t stream, std::string name) {
  T tmp[len];
  cudaMemcpyAsync(&tmp, input, sizeof(T) * len, cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);
  std::cout << name;
  for (int i = 0; i < len; ++i) {
    std::cout << ":" << tmp[i];
  }
  std::cout << std::endl;
}

template <typename T>
__global__ void calc_shard_offset(T* idx, T* left, T* right, size_t len) {
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len - 1) {
    if (idx[i] != idx[i + 1]) {
      right[idx[i]] = i;
      left[idx[i + 1]] = i + 1;
    }
  }
  if (i == 0) {
    left[idx[i]] = i;
  }
  if (i == (len - 1)) {
    right[idx[i]] = i;
  }
}

template <typename KeyType, typename T>
__global__ void calc_shard_index(KeyType* d_keys, size_t len, T* shard_index,
                                 int total_gpu) {
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len) {
    shard_index[i] = d_keys[i] % total_gpu;
  }
}

template <typename KeyType, typename T>
__global__ void fill_shard_key(KeyType* d_shard_keys, KeyType* d_keys, T* idx,
                               size_t len) {
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len) {
    d_shard_keys[i] = d_keys[idx[i]];
  }
}

template <typename KeyType, typename GradType, typename T>
__global__ void fill_shard_grads(KeyType* d_shard_keys, KeyType* d_keys,
                                 GradType* d_shard_grads, GradType* d_grads,
                                 T* idx, size_t len) {
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len) {
    d_shard_keys[i] = d_keys[idx[i]];
    d_shard_grads[i] = d_grads[idx[i]];
  }
}

template <typename ValType, typename T>
__global__ void fill_dvals(ValType* d_shard_vals, ValType* d_vals, T* idx,
                           size_t len) {
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len) {
    d_vals[idx[i]] = d_shard_vals[i];
  }
}

template <typename KeyType, typename ValType, typename GradType>
HeterComm<KeyType, ValType, GradType>::HeterComm(
    size_t capacity, std::shared_ptr<HeterPsResource> resource) {
  resource_ = resource;
  storage_.resize(resource_->total_gpu());
  for (int i = 0; i < resource_->total_gpu(); ++i) {
    platform::CUDADeviceGuard guard(resource_->dev_id(i));
    auto table = new Table(capacity / load_factor_);
    tables_.push_back(table);
    if (multi_node_) {
      storage_[i].init(feanum_, resource_->dev_id(i));
    }
  }
  init_path();
}

template <typename KeyType, typename ValType, typename GradType>
void HeterComm<KeyType, ValType, GradType>::init_path() {
  int total_gpu = resource_->total_gpu();
  path_.resize(total_gpu);

  if (!topo_aware_) {
    VLOG(3) << "init path without topo aware";
    for (int i = 0; i < total_gpu; ++i) {
      path_[i].resize(total_gpu);
      for (int j = 0; j < total_gpu; ++j) {
        auto& nodes = path_[i][j].nodes_;
        nodes.resize(1);
        nodes[0].in_stream = resource_->comm_stream(i, j);
        nodes[0].out_stream = resource_->comm_stream(j, i);
        nodes[0].key_storage = NULL;
        nodes[0].val_storage = NULL;
        nodes[0].sync = 0;
        nodes[0].gpu_num = j;
      }
    }
  } else {
    VLOG(3) << "init path with topo aware";
    for (int i = 0; i < total_gpu; ++i) {
      path_[i].resize(total_gpu);
      for (int j = 0; j < total_gpu; ++j) {
        auto& nodes = path_[i][j].nodes_;
        int from = resource_->dev_id(i);
        int to = resource_->dev_id(j);
        int transfer_id = i;
        if (need_transfer(from, to)) {
          transfer_id = resource_->get_index_by_devid(get_transfer_devid(from));
          nodes.push_back(Node());
          Node& node = nodes.back();
          node.in_stream = resource_->comm_stream(i, transfer_id);
          node.out_stream = resource_->comm_stream(transfer_id, i);
          node.key_storage = NULL;
          node.val_storage = NULL;
          node.sync = 1;
          node.gpu_num = transfer_id;
        }
        nodes.push_back(Node());
        Node& node = nodes.back();
        node.in_stream = resource_->comm_stream(i, transfer_id);
        node.out_stream = resource_->comm_stream(transfer_id, i);
        node.key_storage = NULL;
        node.val_storage = NULL;
        node.sync = 0;
        node.gpu_num = j;
      }
    }
  }
}

template <typename KeyType, typename ValType, typename GradType>
void HeterComm<KeyType, ValType, GradType>::create_storage(
    int start_index, int end_index, int keylen, int vallen,
    std::vector<std::shared_ptr<memory::Allocation>>& local_storage) {
  auto& nodes = path_[start_index][end_index].nodes_;
  for (size_t i = 0; i < nodes.size(); ++i) {
    platform::CUDADeviceGuard guard(resource_->dev_id(nodes[i].gpu_num));
    platform::CUDAPlace remote_place =
        platform::CUDAPlace(resource_->dev_id(nodes[i].gpu_num));
    auto key_mem = memory::AllocShared(remote_place, keylen);
    local_storage.push_back(key_mem);
    nodes[i].key_storage = reinterpret_cast<char*>(key_mem->ptr());

    auto val_mem = memory::AllocShared(remote_place, vallen);
    local_storage.push_back(val_mem);
    nodes[i].val_storage = reinterpret_cast<char*>(val_mem->ptr());
    nodes[i].key_bytes_len = keylen;
    nodes[i].val_bytes_len = vallen;
  }
}

template <typename KeyType, typename ValType, typename GradType>
void HeterComm<KeyType, ValType, GradType>::walk_to_dest(
    int start_index, int gpu_num, int* h_left, int* h_right, KeyType* src_key,
    GradType* src_val) {
  int need_copy_val = 0;
  if (src_val) {
    need_copy_val = 1;
  }
  std::queue<CopyTask> que;
  for (int i = 0; i < gpu_num; i++) {
    if (h_left[i] == -1 || h_right[i] == -1) {
      continue;
    }
    int size = path_[start_index][i].nodes_.size();
    auto& node = path_[start_index][i].nodes_[0];
    CopyTask t(&path_[start_index][i], 0);
    que.push(t);
    cudaMemcpyAsync(node.key_storage,
                    reinterpret_cast<char*>(src_key + h_left[i]),
                    node.key_bytes_len, cudaMemcpyDefault, node.in_stream);
    if (need_copy_val) {
      cudaMemcpyAsync(node.val_storage,
                      reinterpret_cast<char*>(src_val + h_left[i]),
                      node.val_bytes_len, cudaMemcpyDefault, node.in_stream);
    }
  }
  while (!que.empty()) {
    CopyTask& cur_task = que.front();
    que.pop();
    if (cur_task.path->nodes_[cur_task.step].sync) {
      cudaStreamSynchronize(cur_task.path->nodes_[cur_task.step].in_stream);
    }
    if (cur_task.step != cur_task.path->nodes_.size() - 1) {
      int cur_step = cur_task.step;
      CopyTask c(cur_task.path, cur_step + 1);
      que.push(c);
      cudaMemcpyAsync(cur_task.path->nodes_[cur_step + 1].key_storage,
                      cur_task.path->nodes_[cur_step].key_storage,
                      cur_task.path->nodes_[cur_step + 1].key_bytes_len,
                      cudaMemcpyDefault,
                      cur_task.path->nodes_[cur_step + 1].in_stream);
      if (need_copy_val) {
        cudaMemcpyAsync(cur_task.path->nodes_[cur_step + 1].val_storage,
                        cur_task.path->nodes_[cur_step].val_storage,
                        cur_task.path->nodes_[cur_step + 1].val_bytes_len,
                        cudaMemcpyDefault,
                        cur_task.path->nodes_[cur_step + 1].in_stream);
      }
    }
  }
}

template <typename KeyType, typename ValType, typename GradType>
void HeterComm<KeyType, ValType, GradType>::walk_to_src(
    int start_index, int gpu_num, int* h_left, int* h_right, ValType* src_val) {
  std::queue<CopyTask> que;
  for (int i = 0; i < gpu_num; i++) {
    if (h_left[i] == -1 || h_right[i] == -1) {
      continue;
    }
    int cur_step = path_[start_index][i].nodes_.size() - 1;
    auto& node = path_[start_index][i].nodes_[cur_step];
    if (cur_step == 0) {
      cudaMemcpyAsync(reinterpret_cast<char*>(src_val + h_left[i]),
                      node.val_storage, node.val_bytes_len, cudaMemcpyDefault,
                      node.out_stream);
    } else {
      CopyTask t(&path_[start_index][i], cur_step - 1);
      que.push(t);
      cudaMemcpyAsync(path_[start_index][i].nodes_[cur_step - 1].val_storage,
                      node.val_storage,
                      path_[start_index][i].nodes_[cur_step - 1].val_bytes_len,
                      cudaMemcpyDefault,
                      path_[start_index][i].nodes_[cur_step - 1].out_stream);
    }
  }
  while (!que.empty()) {
    CopyTask& cur_task = que.front();
    que.pop();
    int cur_step = cur_task.step;
    if (cur_task.path->nodes_[cur_step].sync) {
      cudaStreamSynchronize(cur_task.path->nodes_[cur_step].out_stream);
    }
    if (cur_step > 0) {
      CopyTask c(cur_task.path, cur_step - 1);
      que.push(c);
      cudaMemcpyAsync(cur_task.path->nodes_[cur_step - 1].val_storage,
                      cur_task.path->nodes_[cur_step].val_storage,
                      cur_task.path->nodes_[cur_step - 1].val_bytes_len,
                      cudaMemcpyDefault,
                      cur_task.path->nodes_[cur_step - 1].out_stream);
    } else if (cur_step == 0) {
      int end_index = cur_task.path->nodes_.back().gpu_num;
      cudaMemcpyAsync(reinterpret_cast<char*>(src_val + h_left[end_index]),
                      cur_task.path->nodes_[cur_step].val_storage,
                      cur_task.path->nodes_[cur_step].val_bytes_len,
                      cudaMemcpyDefault,
                      cur_task.path->nodes_[cur_step].out_stream);
    }
  }
}

template <typename KeyType, typename ValType, typename GradType>
HeterComm<KeyType, ValType, GradType>::~HeterComm() {
  for (auto& table : tables_) {
    delete table;
    table = nullptr;
  }
}

template <typename KeyType, typename ValType, typename GradType>
void HeterComm<KeyType, ValType, GradType>::show_one_table(int gpu_num) {
  tables_[gpu_num]->show();
}

template <typename KeyType, typename ValType, typename GradType>
int HeterComm<KeyType, ValType, GradType>::log2i(int x) {
  unsigned res = 0;
  while (x >>= 1) {
    ++res;
  }
  return res;
}

template <typename KeyType, typename ValType, typename GradType>
int HeterComm<KeyType, ValType, GradType>::get_index_by_devid(int devid) {
  return resource_->get_index_by_devid(devid);
}

template <typename KeyType, typename ValType, typename GradType>
void HeterComm<KeyType, ValType, GradType>::build_ps(int num, KeyType* h_keys,
                                                     ValType* h_vals,
                                                     size_t len,
                                                     size_t chunk_size,
                                                     int stream_num) {
  if (len <= 0) {
    return;
  }
  int dev_id = resource_->dev_id(num);
  platform::CUDAPlace place = platform::CUDAPlace(dev_id);
  platform::CUDADeviceGuard guard(dev_id);

  std::vector<std::shared_ptr<memory::Allocation>> d_key_bufs;
  std::vector<std::shared_ptr<memory::Allocation>> d_val_bufs;

  gpuStream_t streams[stream_num];
  for (int i = 0; i < stream_num; ++i) {
    PADDLE_ENFORCE_CUDA_SUCCESS(cudaStreamCreate(&(streams[i])));
    auto d_k_buf = memory::AllocShared(place, chunk_size * sizeof(KeyType));
    auto d_v_buf = memory::AllocShared(place, chunk_size * sizeof(ValType));
    d_key_bufs.push_back(d_k_buf);
    d_val_bufs.push_back(d_v_buf);
  }

  int cur_len = 0;
  int cur_stream = 0;

  while (cur_len < len) {
    cur_stream = cur_stream % stream_num;
    int tmp_len = cur_len + chunk_size > len ? len - cur_len : chunk_size;
    PADDLE_ENFORCE_CUDA_SUCCESS(
        cudaMemcpyAsync(d_key_bufs[cur_stream]->ptr(), h_keys + cur_len,
                        sizeof(KeyType) * tmp_len, cudaMemcpyHostToDevice,
                        streams[cur_stream]));
    PADDLE_ENFORCE_CUDA_SUCCESS(
        cudaMemcpyAsync(d_val_bufs[cur_stream]->ptr(), h_vals + cur_len,
                        sizeof(ValType) * tmp_len, cudaMemcpyHostToDevice,
                        streams[cur_stream]));
    tables_[num]->insert(
        reinterpret_cast<KeyType*>(d_key_bufs[cur_stream]->ptr()),
        reinterpret_cast<ValType*>(d_val_bufs[cur_stream]->ptr()), tmp_len,
        streams[cur_stream]);
    cur_stream += 1;
    cur_len += tmp_len;
  }

  for (int i = 0; i < stream_num; ++i) {
    cudaStreamSynchronize(streams[i]);
    PADDLE_ENFORCE_CUDA_SUCCESS(cudaStreamDestroy(streams[i]));
  }
}

template <typename KeyType, typename ValType, typename GradType>
void HeterComm<KeyType, ValType, GradType>::merge_grad(int gpu_num,
                                                       KeyType* d_keys,
                                                       GradType* d_grads,
                                                       size_t len,
                                                       int& uniq_len) {
  int dev_id = resource_->dev_id(gpu_num);
  platform::CUDAPlace place = platform::CUDAPlace(dev_id);
  platform::CUDADeviceGuard guard(dev_id);
  auto stream = resource_->local_stream(gpu_num, 0);

  size_t temp_storage_bytes;

  auto d_merge_keys = memory::AllocShared(place, len * sizeof(KeyType));
  KeyType* d_merge_keys_ptr = reinterpret_cast<KeyType*>(d_merge_keys->ptr());

  auto d_merge_grads = memory::AllocShared(place, len * sizeof(GradType));
  GradType* d_merge_grads_ptr =
      reinterpret_cast<GradType*>(d_merge_grads->ptr());

  PADDLE_ENFORCE_CUDA_SUCCESS(cub::DeviceRadixSort::SortPairs(
      NULL, temp_storage_bytes, d_keys, d_merge_keys_ptr, d_grads,
      d_merge_grads_ptr, len, 0, 8 * sizeof(KeyType), stream, false));

  void* d_buff = NULL;
  auto d_temp_storage = memory::AllocShared(place, temp_storage_bytes);

  PADDLE_ENFORCE_CUDA_SUCCESS(cub::DeviceRadixSort::SortPairs(
      d_temp_storage->ptr(), temp_storage_bytes, d_keys, d_merge_keys_ptr,
      d_grads, d_merge_grads_ptr, len, 0, 8 * sizeof(KeyType), stream, false));
  temp_storage_bytes = 0;

  auto d_num_runs_out_mem = memory::AllocShared(place, sizeof(int));
  int* d_num_runs_out = reinterpret_cast<int*>(d_num_runs_out_mem->ptr());

  PADDLE_ENFORCE_CUDA_SUCCESS(cub::DeviceReduce::ReduceByKey(
      NULL, temp_storage_bytes, d_merge_keys_ptr, d_keys, d_merge_grads_ptr,
      d_grads, d_num_runs_out, merger_, len, stream, false));

  if (d_temp_storage->size() < temp_storage_bytes) {
    d_temp_storage = NULL;
    d_temp_storage = memory::AllocShared(place, temp_storage_bytes);
  }

  PADDLE_ENFORCE_CUDA_SUCCESS(cub::DeviceReduce::ReduceByKey(
      d_temp_storage->ptr(), temp_storage_bytes, d_merge_keys_ptr, d_keys,
      d_merge_grads_ptr, d_grads, d_num_runs_out, merger_, len, stream, false));

  cudaMemcpyAsync(&uniq_len, d_num_runs_out, sizeof(int),
                  cudaMemcpyDeviceToHost, stream);
  PADDLE_ENFORCE_CUDA_SUCCESS(cudaStreamSynchronize(stream));
}

template <typename KeyType, typename ValType, typename GradType>
void HeterComm<KeyType, ValType, GradType>::split_input_to_shard(
    KeyType* d_keys, int* d_idx_ptr, size_t len, int* left, int* right,
    int gpu_num) {
  int total_gpu = resource_->total_gpu();
  int dev_id = resource_->dev_id(gpu_num);
  platform::CUDAPlace place = platform::CUDAPlace(dev_id);
  platform::CUDADeviceGuard guard(dev_id);
  auto stream = resource_->local_stream(gpu_num, 0);

  auto d_idx_tmp = memory::AllocShared(place, len * sizeof(int));
  int* d_idx_tmp_ptr = reinterpret_cast<int*>(d_idx_tmp->ptr());

  auto d_shard_index = memory::AllocShared(place, len * sizeof(int));
  int* d_shard_index_ptr = reinterpret_cast<int*>(d_shard_index->ptr());

  auto d_shard_index_tmp = memory::AllocShared(place, len * sizeof(int));
  int* d_shard_index_tmp_ptr = reinterpret_cast<int*>(d_shard_index_tmp->ptr());

  int grid_size = (len - 1) / block_size_ + 1;
  fill_idx<<<grid_size, block_size_, 0, stream>>>(d_idx_tmp_ptr, len);
  calc_shard_index<<<grid_size, block_size_, 0, stream>>>(
      d_keys, len, d_shard_index_tmp_ptr, total_gpu);

  size_t temp_storage_bytes;
  const int num_bits = 1 + log2i(total_gpu);
  PADDLE_ENFORCE_CUDA_SUCCESS(cub::DeviceRadixSort::SortPairs(
      NULL, temp_storage_bytes, d_shard_index_tmp_ptr, d_shard_index_ptr,
      d_idx_tmp_ptr, d_idx_ptr, len, 0, num_bits, stream));

  auto d_temp_storage = memory::AllocShared(place, temp_storage_bytes);
  PADDLE_ENFORCE_CUDA_SUCCESS(cub::DeviceRadixSort::SortPairs(
      d_temp_storage->ptr(), temp_storage_bytes, d_shard_index_tmp_ptr,
      d_shard_index_ptr, d_idx_tmp_ptr, d_idx_ptr, len, 0, num_bits, stream));
  calc_shard_offset<<<grid_size, block_size_, 0, stream>>>(d_shard_index_ptr,
                                                           left, right, len);
  cudaStreamSynchronize(stream);
}

template <typename KeyType, typename ValType, typename GradType>
void HeterComm<KeyType, ValType, GradType>::pull_sparse(int num,
                                                        KeyType* d_keys,
                                                        ValType* d_vals,
                                                        size_t len) {
  if (len == 0) {
    return;
  }

  int total_gpu = resource_->total_gpu();
  int dev_id = resource_->dev_id(num);
  platform::CUDAPlace place = platform::CUDAPlace(dev_id);
  platform::CUDADeviceGuard guard(dev_id);
  auto stream = resource_->local_stream(num, 0);

  int grid_size = (len - 1) / block_size_ + 1;

  int h_left[total_gpu];
  int h_right[total_gpu];

  auto d_left = memory::AllocShared(place, total_gpu * sizeof(int));
  auto d_right = memory::AllocShared(place, total_gpu * sizeof(int));
  int* d_left_ptr = reinterpret_cast<int*>(d_left->ptr());
  int* d_right_ptr = reinterpret_cast<int*>(d_right->ptr());

  cudaMemset(d_left_ptr, -1, total_gpu * sizeof(int));
  cudaMemset(d_right_ptr, -1, total_gpu * sizeof(int));
  //
  auto d_idx = memory::AllocShared(place, len * sizeof(int));
  int* d_idx_ptr = reinterpret_cast<int*>(d_idx->ptr());

  auto d_shard_keys = memory::AllocShared(place, len * sizeof(KeyType));
  KeyType* d_shard_keys_ptr = reinterpret_cast<KeyType*>(d_shard_keys->ptr());
  auto d_shard_vals = memory::AllocShared(place, len * sizeof(ValType));
  ValType* d_shard_vals_ptr = reinterpret_cast<ValType*>(d_shard_vals->ptr());

  split_input_to_shard(d_keys, d_idx_ptr, len, d_left_ptr, d_right_ptr, num);

  fill_shard_key<<<grid_size, block_size_, 0, stream>>>(d_shard_keys_ptr,
                                                        d_keys, d_idx_ptr, len);

  cudaStreamSynchronize(stream);

  cudaMemcpy(h_left, d_left_ptr, total_gpu * sizeof(int),
             cudaMemcpyDeviceToHost);
  cudaMemcpy(h_right, d_right_ptr, total_gpu * sizeof(int),
             cudaMemcpyDeviceToHost);

  std::vector<std::shared_ptr<memory::Allocation>> local_storage;

  for (int i = 0; i < total_gpu; ++i) {
    int shard_len = h_right[i] - h_left[i] + 1;
    if (shard_len == 0) {
      continue;
    }
    create_storage(num, i, shard_len * sizeof(KeyType),
                   shard_len * sizeof(ValType), local_storage);
  }

  walk_to_dest(num, total_gpu, h_left, h_right, d_shard_keys_ptr, NULL);

  for (int i = 0; i < total_gpu; ++i) {
    if (h_left[i] == -1) {
      continue;
    }
    auto& node = path_[num][i].nodes_.back();
    cudaStreamSynchronize(node.in_stream);
    platform::CUDADeviceGuard guard(resource_->dev_id(i));
    tables_[i]->rwlock_->RDLock();
    tables_[i]->get(reinterpret_cast<KeyType*>(node.key_storage),
                    reinterpret_cast<ValType*>(node.val_storage),
                    h_right[i] - h_left[i] + 1,
                    resource_->remote_stream(i, num));
  }
  for (int i = 0; i < total_gpu; ++i) {
    cudaStreamSynchronize(resource_->remote_stream(i, num));
    tables_[i]->rwlock_->UNLock();
  }

  walk_to_src(num, total_gpu, h_left, h_right, d_shard_vals_ptr);

  for (int i = 0; i < total_gpu; ++i) {
    auto& node = path_[num][i].nodes_.front();
    cudaStreamSynchronize(node.out_stream);
  }

  fill_dvals<<<grid_size, block_size_, 0, stream>>>(d_shard_vals_ptr, d_vals,
                                                    d_idx_ptr, len);
  cudaStreamSynchronize(stream);
}

template <typename KeyType, typename ValType, typename GradType>
template <typename Sgd>
void HeterComm<KeyType, ValType, GradType>::push_sparse(int gpu_num,
                                                        KeyType* d_keys,
                                                        GradType* d_grads,
                                                        size_t len, Sgd& sgd) {
  if (len == 0) {
    return;
  }

  int total_gpu = resource_->total_gpu();
  int dev_id = resource_->dev_id(gpu_num);
  platform::CUDAPlace place = platform::CUDAPlace(dev_id);
  platform::CUDADeviceGuard guard(dev_id);
  auto stream = resource_->local_stream(gpu_num, 0);

  int h_left[total_gpu];
  int h_right[total_gpu];

  auto d_left = memory::AllocShared(place, total_gpu * sizeof(int));
  auto d_right = memory::AllocShared(place, total_gpu * sizeof(int));
  int* d_left_ptr = reinterpret_cast<int*>(d_left->ptr());
  int* d_right_ptr = reinterpret_cast<int*>(d_right->ptr());

  cudaMemset(d_left_ptr, -1, total_gpu * sizeof(int));
  cudaMemset(d_right_ptr, -1, total_gpu * sizeof(int));
  //
  auto d_idx = memory::AllocShared(place, len * sizeof(int));
  int* d_idx_ptr = reinterpret_cast<int*>(d_idx->ptr());

  auto d_shard_keys = memory::AllocShared(place, len * sizeof(KeyType));
  KeyType* d_shard_keys_ptr = reinterpret_cast<KeyType*>(d_shard_keys->ptr());
  auto d_shard_grads = memory::AllocShared(place, len * sizeof(GradType));
  GradType* d_shard_grads_ptr =
      reinterpret_cast<GradType*>(d_shard_grads->ptr());

  int uniq_len = len;
  merge_grad(gpu_num, d_keys, d_grads, len, uniq_len);

  int grid_size = (uniq_len - 1) / block_size_ + 1;

  split_input_to_shard(d_keys, d_idx_ptr, uniq_len, d_left_ptr, d_right_ptr,
                       gpu_num);

  fill_shard_grads<<<grid_size, block_size_, 0, stream>>>(
      d_shard_keys_ptr, d_keys, d_shard_grads_ptr, d_grads, d_idx_ptr,
      uniq_len);

  cudaStreamSynchronize(stream);

  cudaMemcpy(h_left, d_left_ptr, total_gpu * sizeof(int),
             cudaMemcpyDeviceToHost);
  cudaMemcpy(h_right, d_right_ptr, total_gpu * sizeof(int),
             cudaMemcpyDeviceToHost);

  std::vector<std::shared_ptr<memory::Allocation>> local_storage;
  for (int i = 0; i < total_gpu; ++i) {
    int shard_len = h_right[i] - h_left[i] + 1;
    if (h_left[i] == -1 || h_right[i] == -1) {
      continue;
    }
    create_storage(gpu_num, i, shard_len * sizeof(KeyType),
                   shard_len * sizeof(GradType), local_storage);
  }

  walk_to_dest(gpu_num, total_gpu, h_left, h_right, d_shard_keys_ptr,
               d_shard_grads_ptr);

  for (int i = 0; i < total_gpu; ++i) {
    if (h_left[i] == -1 || h_right[i] == -1) {
      continue;
    }
    auto& node = path_[gpu_num][i].nodes_.back();
    cudaStreamSynchronize(node.in_stream);

    platform::CUDADeviceGuard guard(resource_->dev_id(i));
    tables_[i]->rwlock_->WRLock();
    tables_[i]->update(reinterpret_cast<KeyType*>(node.key_storage),
                       reinterpret_cast<GradType*>(node.val_storage),
                       h_right[i] - h_left[i] + 1, sgd,
                       resource_->remote_stream(i, gpu_num));
  }
  for (int i = 0; i < total_gpu; ++i) {
    cudaStreamSynchronize(resource_->remote_stream(i, gpu_num));
    tables_[i]->rwlock_->UNLock();
  }
}

template <typename KeyType, typename ValType, typename GradType>
template <typename Sgd>
void HeterComm<KeyType, ValType, GradType>::update_one_table(
    int gpu_num, KeyType* d_keys, GradType* d_grads, size_t len, Sgd& sgd) {
  if (len == 0) {
    return;
  }

  int dev_id = resource_->dev_id(gpu_num);
  platform::CUDADeviceGuard guard(dev_id);
  tables_[gpu_num]->rwlock_->WRLock();
  tables_[gpu_num]->update(d_keys, d_grads, len, sgd,
                           resource_->remote_stream(gpu_num, gpu_num));
  tables_[gpu_num]->rwlock_->UNLock();
  cudaStreamSynchronize(resource_->remote_stream(gpu_num, gpu_num));
}

template <typename KeyType, typename ValType, typename GradType>
template <typename Sgd>
void HeterComm<KeyType, ValType, GradType>::push_sparse_multi_node(
    int gpu_num, KeyType* d_keys, GradType* d_grads, size_t len, Sgd& sgd) {
  if (len == 0) {
    return;
  }

  int uniq_len = len;
  merge_grad(gpu_num, d_keys, d_grads, len, uniq_len);

  uniq_len = gather_one_node_grad(gpu_num, d_keys, d_grads, uniq_len);

  uniq_len = gather_multi_node_grad(gpu_num, storage_[gpu_num].local_keys,
                                    storage_[gpu_num].local_grads, uniq_len);

  update_one_table(gpu_num, storage_[gpu_num].local_keys,
                   storage_[gpu_num].local_grads, uniq_len, sgd);
}

template <typename KeyType, typename ValType, typename GradType>
int HeterComm<KeyType, ValType, GradType>::gather_one_node_grad(
    int gpu_num, KeyType* d_keys, GradType* d_grads, int len) {
  int total_gpu = resource_->total_gpu();
  int dev_id = resource_->dev_id(gpu_num);
  auto& storage = storage_[gpu_num];
  platform::CUDAPlace place = platform::CUDAPlace(dev_id);
  platform::CUDADeviceGuard guard(dev_id);
  auto stream = resource_->local_stream(gpu_num, 0);
  int max_size = 0;

  ncclComm_t nccl_inner_comm = nccl_inner_comms_[gpu_num];
  // alloc for size
  int h_node_len[total_gpu];
  auto d_node_len_mem = memory::AllocShared(place, total_gpu * sizeof(int));
  int* d_node_len = reinterpret_cast<int*>(d_node_len_mem->ptr());
  h_node_len[gpu_num] = len;

  cudaMemcpy(d_node_len + gpu_num, h_node_len + gpu_num, sizeof(int),
             cudaMemcpyHostToDevice);

  // allgather grad len
  PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::ncclGroupStart());
  PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::ncclAllGather(
      (const void*)(d_node_len + gpu_num), (void*)d_node_len, 1, ncclInt,
      nccl_inner_comm, stream));
  PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::ncclGroupEnd());
  PADDLE_ENFORCE_CUDA_SUCCESS(cudaStreamSynchronize(stream));
  cudaMemcpy(h_node_len, d_node_len, sizeof(int) * total_gpu,
             cudaMemcpyDeviceToHost);

  for (int i = 0; i < total_gpu; ++i) {
    if (h_node_len[i] > max_size) {
      max_size = h_node_len[i];
    }
  }
  storage.alloc(max_size * total_gpu);

  // allgather keys and grads
  PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::ncclGroupStart());
  PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::ncclAllGather(
      d_keys, storage.all_keys, max_size, ncclUint64, nccl_inner_comm, stream));

  PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::ncclAllGather(
      d_grads, storage.all_grads, max_size * sizeof(GradType), ncclUint8,
      nccl_inner_comm, stream));
  PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::ncclGroupEnd());
  PADDLE_ENFORCE_CUDA_SUCCESS(cudaStreamSynchronize(stream));

  int h_left[total_gpu];
  int h_right[total_gpu];
  auto d_left = memory::AllocShared(place, total_gpu * sizeof(int));
  auto d_right = memory::AllocShared(place, total_gpu * sizeof(int));
  int* d_left_ptr = reinterpret_cast<int*>(d_left->ptr());
  int* d_right_ptr = reinterpret_cast<int*>(d_right->ptr());

  int merge_num = 0;
  for (int i = 0; i < total_gpu; ++i) {
    int index = i * max_size;
    auto d_idx = memory::AllocShared(place, h_node_len[i] * sizeof(int));
    int* d_idx_ptr = reinterpret_cast<int*>(d_idx->ptr());

    cudaMemset(d_left_ptr, -1, total_gpu * sizeof(int));
    cudaMemset(d_right_ptr, -1, total_gpu * sizeof(int));

    split_input_to_shard(storage.all_keys + index, d_idx_ptr, h_node_len[i],
                         d_left_ptr, d_right_ptr, gpu_num);
    cudaMemcpy(h_left, d_left_ptr, total_gpu * sizeof(int),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(h_right, d_right_ptr, total_gpu * sizeof(int),
               cudaMemcpyDeviceToHost);

    int grid_size = (h_node_len[i] - 1) / block_size_ + 1;
    fill_shard_grads<<<grid_size, block_size_, 0, stream>>>(
        storage.local_keys + merge_num, storage.all_keys + index,
        storage.local_grads + merge_num, storage.all_grads + index,
        d_idx_ptr + h_left[gpu_num], h_right[gpu_num] - h_left[gpu_num] + 1);
    merge_num = merge_num + h_right[gpu_num] - h_left[gpu_num] + 1;
  }

  int ret = merge_num;
  merge_grad(gpu_num, storage.local_keys, storage.local_grads, merge_num, ret);
  return ret;
}

template <typename KeyType, typename ValType, typename GradType>
int HeterComm<KeyType, ValType, GradType>::gather_multi_node_grad(
    int gpu_num, KeyType* d_keys, GradType* d_grads, int len) {
  int dev_id = resource_->dev_id(gpu_num);
  auto& storage = storage_[gpu_num];
  platform::CUDAPlace place = platform::CUDAPlace(dev_id);
  platform::CUDADeviceGuard guard(dev_id);
  auto stream = resource_->local_stream(gpu_num, 0);
  int max_size = 0;
  ncclComm_t nccl_inter_comm = nccl_inter_comms_[gpu_num];
  // alloc for size
  int h_node_len[node_size_];
  auto d_node_len_mem = memory::AllocShared(place, node_size_ * sizeof(int));
  int* d_node_len = reinterpret_cast<int*>(d_node_len_mem->ptr());
  h_node_len[0] = len;

  cudaMemcpy(d_node_len, h_node_len, sizeof(int), cudaMemcpyHostToDevice);

  // allgather grad len
  PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::ncclGroupStart());
  PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::ncclAllGather(
      d_node_len, d_node_len, 1, ncclInt, nccl_inter_comm, stream));
  PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::ncclGroupEnd());
  PADDLE_ENFORCE_CUDA_SUCCESS(cudaStreamSynchronize(stream));
  cudaMemcpy(h_node_len, d_node_len, sizeof(int) * node_size_,
             cudaMemcpyDeviceToHost);

  for (int i = 0; i < node_size_; ++i) {
    if (h_node_len[i] > max_size) {
      max_size = h_node_len[i];
    }
  }
  storage.alloc(max_size * node_size_);

  // allgather keys and grads
  PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::ncclGroupStart());
  PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::ncclAllGather(
      d_keys, storage.all_keys, max_size, ncclUint64, nccl_inter_comm, stream));

  PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::ncclAllGather(
      d_grads, storage.all_grads, max_size * sizeof(GradType), ncclUint8,
      nccl_inter_comm, stream));
  PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::ncclGroupEnd());
  PADDLE_ENFORCE_CUDA_SUCCESS(cudaStreamSynchronize(stream));

  int merge_num = 0;
  for (int i = 0; i < node_size_; ++i) {
    int index = i * max_size;
    cudaMemcpyAsync(storage.local_keys + merge_num, storage.all_keys + index,
                    h_node_len[i], cudaMemcpyDefault, stream);
    cudaMemcpyAsync(storage.local_grads + merge_num, storage.all_grads + index,
                    h_node_len[i], cudaMemcpyDefault, stream);
    merge_num += h_node_len[i];
  }

  int ret = merge_num;
  merge_grad(gpu_num, storage.local_keys, storage.local_grads, merge_num, ret);
  return ret;
}

template <typename KeyType, typename ValType, typename GradType>
void HeterComm<KeyType, ValType, GradType>::end_pass() {
  int total_gpu = resource_->total_gpu();
  std::vector<std::thread> threads;

  auto dump_to_cpu_func = [this](int index) {
    auto stream = resource_->local_stream(index, 0);
    int dev_id = resource_->dev_id(index);
    platform::CUDADeviceGuard guard(dev_id);
    tables_[index]->dump_to_cpu(dev_id, stream);
  };

  for (int i = 0; i < total_gpu; ++i) {
    threads.push_back(std::thread(dump_to_cpu_func, i));
  }
  for (auto& t : threads) {
    t.join();
  }
}

// template <typename KeyType, typename ValType, typename GradType>
// void HeterComm<KeyType, ValType, GradType>::dump_to_cpu(int index) {
//  auto stream = resource_->local_stream(index, 0);
//  int dev_id = resource_->dev_id(index);
//  platform::CUDADeviceGuard guard(dev_id);
//  tables_[index]->dump_to_cpu(dev_id, stream);
//}

}  // end namespace framework
}  // end namespace paddle
#endif
