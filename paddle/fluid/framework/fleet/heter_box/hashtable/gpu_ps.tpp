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
#pragma once
#ifdef PADDLE_WITH_PSLIB
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
void show_tensor(T* input, size_t len, cudaStream_t stream, std::string name) {
  T tmp[len];
  cudaMemcpyAsync(&tmp, input, sizeof(T) * len, cudaMemcpyDeviceToHost, stream);
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
__global__ void calc_shard_index(KeyType* d_keys, size_t len, T* shard_index, int total_gpu) {

  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len) {
    shard_index[i] = d_keys[i] % total_gpu;
  }
}

template <typename KeyType, typename T>
__global__ void fill_shard_key(KeyType* d_shard_keys, KeyType* d_keys, T* idx, size_t len) {
 
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len) {
    d_shard_keys[i] = d_keys[idx[i]];
  }
}

template <typename KeyType, typename GradType, typename T>
__global__ void fill_shard_grads(KeyType* d_shard_keys, KeyType* d_keys, GradType* d_shard_grads, GradType* d_grads, T* idx, size_t len) {

  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len) {
    d_shard_keys[i] = d_keys[idx[i]];
    d_shard_grads[i] = d_grads[idx[i]];
  }
}

template <typename ValType, typename T>
__global__ void fill_dvals(ValType* d_shard_vals, ValType* d_vals, T* idx, size_t len) {

  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len) {
    d_vals[idx[i]] = d_shard_vals[i];
  }
}

template<typename KeyType, typename ValType, typename GradType>
GpuPs<KeyType, ValType, GradType>::GpuPs(size_t capacity, std::shared_ptr<HeterBoxResource> resource) {
  resource_ = resource;
  for (int i = 0; i < resource_->total_gpu(); ++i) {
    platform::CUDADeviceGuard guard(resource_->dev_id(i));
    auto table = new Table(capacity / load_factor_);
    tables_.push_back(table);
  }
}

template<typename KeyType, typename ValType, typename GradType>
GpuPs<KeyType, ValType, GradType>::~GpuPs() {
  for (auto& table : tables_) {
    delete table;
    table = nullptr;
  }
}

template<typename KeyType, typename ValType, typename GradType>
void GpuPs<KeyType, ValType, GradType>::show_one_table(int gpu_num) {
  tables_[gpu_num]->show();
}

template<typename KeyType, typename ValType, typename GradType>
int GpuPs<KeyType, ValType, GradType>::log2i(int x) {
  unsigned res = 0;
  while (x >>= 1) {
      ++res;
  }   
  return res;
}

template<typename KeyType, typename ValType, typename GradType>
int GpuPs<KeyType, ValType, GradType>::get_index_by_devid(int devid) {
  return resource_->get_index_by_devid(devid);
}

template<typename KeyType, typename ValType, typename GradType>
void GpuPs<KeyType, ValType, GradType>::build_ps(int num, KeyType* h_keys, ValType* h_vals, size_t len, size_t chunk_size, int stream_num) {
  if (len <= 0) {
    return;
  }
  int dev_id = resource_->dev_id(num);
  platform::CUDAPlace place = platform::CUDAPlace(dev_id);
  platform::CUDADeviceGuard guard(dev_id);
  
  std::vector<std::shared_ptr<memory::Allocation>> d_key_bufs;
  std::vector<std::shared_ptr<memory::Allocation>> d_val_bufs;
  
  cudaStream_t streams[stream_num];
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
    PADDLE_ENFORCE_CUDA_SUCCESS(cudaMemcpyAsync(d_key_bufs[cur_stream]->ptr(), h_keys + cur_len, sizeof(KeyType) * tmp_len, cudaMemcpyHostToDevice, streams[cur_stream]));
    PADDLE_ENFORCE_CUDA_SUCCESS(cudaMemcpyAsync(d_val_bufs[cur_stream]->ptr(), h_vals + cur_len, sizeof(ValType) * tmp_len, cudaMemcpyHostToDevice, streams[cur_stream]));
    tables_[num]->insert(
                         reinterpret_cast<KeyType*>(d_key_bufs[cur_stream]->ptr()),
                         reinterpret_cast<ValType*>(d_val_bufs[cur_stream]->ptr()),
                         tmp_len, streams[cur_stream]);
    cur_stream += 1;
    cur_len += tmp_len;
  }
  
  for (int i = 0; i < stream_num; ++i) {
    cudaStreamSynchronize(streams[i]);
    PADDLE_ENFORCE_CUDA_SUCCESS(cudaStreamDestroy(streams[i]));
  }
}

template<typename KeyType, typename ValType, typename GradType>
void GpuPs<KeyType, ValType, GradType>::merge_grad(int gpu_num, KeyType* d_keys, GradType* d_grads, size_t len, size_t& uniq_len) {
  
  int dev_id = resource_->dev_id(gpu_num);
  platform::CUDAPlace place = platform::CUDAPlace(dev_id);
  platform::CUDADeviceGuard guard(dev_id);
  auto stream = resource_->stream(gpu_num);
  
  size_t temp_storage_bytes;
  
  auto d_merge_keys = memory::AllocShared(place, len * sizeof(KeyType));
  KeyType* d_merge_keys_ptr = reinterpret_cast<KeyType*>(d_merge_keys->ptr());
  
  auto d_merge_grads = memory::AllocShared(place, len * sizeof(GradType));
  GradType* d_merge_grads_ptr = reinterpret_cast<GradType*>(d_merge_grads->ptr());

  PADDLE_ENFORCE_CUDA_SUCCESS(cub::DeviceRadixSort::SortPairs(
          NULL,
          temp_storage_bytes,
          d_keys,
          d_merge_keys_ptr,
          d_grads,
          d_merge_grads_ptr,
          len,
          0,
          8 * sizeof(KeyType),
          stream,
          false));
  
  void *d_buff = NULL;
  auto d_temp_storage = memory::AllocShared(place, temp_storage_bytes);

  PADDLE_ENFORCE_CUDA_SUCCESS(cub::DeviceRadixSort::SortPairs(
          d_temp_storage->ptr(),
          temp_storage_bytes,
          d_keys,
          d_merge_keys_ptr,
          d_grads,
          d_merge_grads_ptr,
          len,
          0,
          8 * sizeof(KeyType),
          stream,
          false));
  temp_storage_bytes = 0;
  int* d_num_runs_out = NULL;
  
  cudaMallocManaged(&d_num_runs_out, sizeof(int));
  
  PADDLE_ENFORCE_CUDA_SUCCESS(cub::DeviceReduce::ReduceByKey(
          NULL,
          temp_storage_bytes,
          d_merge_keys_ptr,
          d_keys,
          d_merge_grads_ptr,
          d_grads,
          d_num_runs_out,
          merger_,
          len,
          stream,
          false));
  
  if (d_temp_storage->size() < temp_storage_bytes) {
      d_temp_storage = NULL;
      d_temp_storage = memory::AllocShared(place, temp_storage_bytes);
  }
  
  PADDLE_ENFORCE_CUDA_SUCCESS(cub::DeviceReduce::ReduceByKey(
          d_temp_storage->ptr(),
          temp_storage_bytes,
          d_merge_keys_ptr,
          d_keys,
          d_merge_grads_ptr,
          d_grads,
          d_num_runs_out,
          merger_,
          len,
          stream,
          false));
  PADDLE_ENFORCE_CUDA_SUCCESS(cudaStreamSynchronize(stream));
  cudaFree(d_num_runs_out);
  uniq_len = (*d_num_runs_out);
}

template<typename KeyType, typename ValType, typename GradType>
void GpuPs<KeyType, ValType, GradType>::split_input_to_shard(KeyType* d_keys, int* d_idx_ptr, size_t len, int* left, int* right, int gpu_num) {
  int total_gpu = resource_->total_gpu();
  int dev_id = resource_->dev_id(gpu_num);
  platform::CUDAPlace place = platform::CUDAPlace(dev_id);
  platform::CUDADeviceGuard guard(dev_id);
  auto stream = resource_->stream(gpu_num);
  
  auto d_idx_tmp = memory::AllocShared(place, len * sizeof(int));
  int* d_idx_tmp_ptr = reinterpret_cast<int*>(d_idx_tmp->ptr());
  
  auto d_shard_index = memory::AllocShared(place, len * sizeof(int));
  int* d_shard_index_ptr = reinterpret_cast<int*>(d_shard_index->ptr());
  
  auto d_shard_index_tmp = memory::AllocShared(place, len * sizeof(int));
  int* d_shard_index_tmp_ptr = reinterpret_cast<int*>(d_shard_index_tmp->ptr());
  
  int grid_size = (len - 1) / block_size_ + 1;
  fill_idx<<<grid_size, block_size_, 0, stream>>>(d_idx_tmp_ptr, len);
  calc_shard_index<<<grid_size, block_size_, 0, stream>>>(d_keys, len, d_shard_index_tmp_ptr, total_gpu);
    
  size_t temp_storage_bytes;
  const int num_bits = 1 + log2i(total_gpu);
  PADDLE_ENFORCE_CUDA_SUCCESS(cub::DeviceRadixSort::SortPairs(NULL, temp_storage_bytes,
                                             d_shard_index_tmp_ptr, d_shard_index_ptr,
                                             d_idx_tmp_ptr, d_idx_ptr,
                                             len,
                                             0,
                                             num_bits,
                                             stream));

  auto d_temp_storage = memory::AllocShared(place, temp_storage_bytes);
  PADDLE_ENFORCE_CUDA_SUCCESS(cub::DeviceRadixSort::SortPairs(d_temp_storage->ptr(), temp_storage_bytes,
                                             d_shard_index_tmp_ptr, d_shard_index_ptr,
                                             d_idx_tmp_ptr, d_idx_ptr,
                                             len,
                                             0,
                                             num_bits,
                                             stream));
  calc_shard_offset<<<grid_size, block_size_, 0, stream>>>(d_shard_index_ptr, left, right, len);
  
}

template<typename KeyType, typename ValType, typename GradType>
void GpuPs<KeyType, ValType, GradType>::pull_sparse(int num, KeyType* d_keys, ValType* d_vals, size_t len) {
  if (len == 0) {
    return ;
  }
  
  int total_gpu = resource_->total_gpu();
  int dev_id = resource_->dev_id(num);
  platform::CUDAPlace place = platform::CUDAPlace(dev_id);
  platform::CUDADeviceGuard guard(dev_id);
  auto stream = resource_->stream(num);
  
  int grid_size = (len - 1) / block_size_ + 1;

  int* left = NULL;
  int* right = NULL;
  cudaMallocManaged(&left, total_gpu * sizeof(int));
  cudaMallocManaged(&right, total_gpu * sizeof(int));
  cudaMemPrefetchAsync(left, total_gpu * sizeof(int), dev_id, stream);
  cudaMemPrefetchAsync(right, total_gpu * sizeof(int), dev_id, stream);
  cudaMemset(left, -1, total_gpu * sizeof(int));
  cudaMemset(right, -1, total_gpu * sizeof(int));
  // 
  auto d_idx = memory::AllocShared(place, len * sizeof(int));
  int* d_idx_ptr = reinterpret_cast<int*>(d_idx->ptr());
  
  auto d_shard_keys = memory::AllocShared(place, len * sizeof(KeyType));
  KeyType* d_shard_keys_ptr = reinterpret_cast<KeyType*>(d_shard_keys->ptr());
  auto d_shard_vals = memory::AllocShared(place, len * sizeof(ValType));
  ValType* d_shard_vals_ptr = reinterpret_cast<ValType*>(d_shard_vals->ptr());
  
  
  split_input_to_shard(d_keys, d_idx_ptr, len, left, right, num);
  
  fill_shard_key<<<grid_size, block_size_, 0, stream>>>(d_shard_keys_ptr, d_keys, d_idx_ptr, len);

  cudaStreamSynchronize(stream); 

  for (int i = 0; i < total_gpu; ++i) {
    if (left[i] == -1) {
      continue;
    }
    platform::CUDADeviceGuard guard(resource_->dev_id(i));
    tables_[i]->get(d_shard_keys_ptr + left[i], d_shard_vals_ptr + left[i], right[i] - left[i] + 1, resource_->stream(i));
  }
  for (int i = 0; i < total_gpu; ++i) {
    cudaStreamSynchronize(resource_->stream(i));
  }
  
  fill_dvals<<<grid_size, block_size_, 0, stream>>>(d_shard_vals_ptr, d_vals, d_idx_ptr, len);
  cudaStreamSynchronize(stream);
  cudaFree(left);
  cudaFree(right);
}

template <typename KeyType, typename ValType, typename GradType>
template <typename Sgd>
void GpuPs<KeyType, ValType, GradType>::push_sparse(int gpu_num, KeyType* d_keys, GradType* d_grads, size_t len, Sgd& sgd) {
  if (len == 0) {
    return ;
  }
  
  int total_gpu = resource_->total_gpu();
  int dev_id = resource_->dev_id(gpu_num);
  platform::CUDAPlace place = platform::CUDAPlace(dev_id);
  platform::CUDADeviceGuard guard(dev_id);
  auto stream = resource_->stream(gpu_num);
  
  int* left = NULL;
  int* right = NULL;
  cudaMallocManaged(&left, total_gpu * sizeof(int));
  cudaMallocManaged(&right, total_gpu * sizeof(int));
  cudaMemPrefetchAsync(left, total_gpu * sizeof(int), dev_id, stream);
  cudaMemPrefetchAsync(right, total_gpu * sizeof(int), dev_id, stream);

  cudaMemset(left, -1, total_gpu * sizeof(int));
  cudaMemset(right, -1, total_gpu * sizeof(int));
  // 
  auto d_idx = memory::AllocShared(place, len * sizeof(int));
  int* d_idx_ptr = reinterpret_cast<int*>(d_idx->ptr());
  
  auto d_shard_keys = memory::AllocShared(place, len * sizeof(KeyType));
  KeyType* d_shard_keys_ptr = reinterpret_cast<KeyType*>(d_shard_keys->ptr());
  auto d_shard_grads = memory::AllocShared(place, len * sizeof(GradType));
  GradType* d_shard_grads_ptr = reinterpret_cast<GradType*>(d_shard_grads->ptr());

  int grid_size = (len - 1) / block_size_ + 1;
  size_t uniq_len = len;
  merge_grad(gpu_num, d_keys, d_grads, len, uniq_len);
  split_input_to_shard(d_keys, d_idx_ptr, uniq_len, left, right, gpu_num);
  
  fill_shard_grads<<<grid_size, block_size_, 0, stream>>>(d_shard_keys_ptr, d_keys, d_shard_grads_ptr, d_grads, d_idx_ptr, uniq_len);

  cudaStreamSynchronize(stream);
  for (int i = 0; i < total_gpu; ++i) {
    if (left[i] == -1) {
      continue;
    }
    platform::CUDADeviceGuard guard(resource_->dev_id(i));
    tables_[i]->update(d_shard_keys_ptr + left[i], d_shard_grads_ptr + left[i], right[i] - left[i] + 1, sgd, resource_->stream(i));
  }
  for (int i = 0; i < total_gpu; ++i) {
    cudaStreamSynchronize(resource_->stream(i));
  }
  
  cudaFree(left);
  cudaFree(right);
}

}  // end namespace framework
}  // end namespace paddle
#endif
