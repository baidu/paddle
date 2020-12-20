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
#include <vector>
#include "thrust/pair.h"
#include "paddle/fluid/platform/cuda_device_guard.h"
#include "paddle/fluid/platform/place.h"
#include "paddle/fluid/memory/memory.h"
#include "cub/cub.cuh"
#include "hashtable.h"
#include "gpu_resource.h"
#include "paddle/fluid/framework/fleet/heter_box/optimizer/optimizer.cuh"

#ifdef PADDLE_WITH_PSLIB

namespace paddle {
namespace framework {

struct CustomGradMerger
{
  template <typename T>
    CUB_RUNTIME_FUNCTION __forceinline__ __device__
    T operator()(const T &a, const T &b) const {
      T out;
      out.slot = a.slot;
      out.show = a.show + b.show;
      out.clk = a.clk + b.clk;
      out.lr_g = a.lr_g + b.lr_g;
      for (int i = 0; i < MF_DIM; ++i) {
        out.mf_g[i] = a.mf_g[i] + b.mf_g[i];
      }
      return out;
    }
};

template <typename KeyType, typename ValType, typename GradType>
class GpuPs {
 public:
  GpuPs(size_t capacity, std::shared_ptr<HeterBoxResource> resource);
  virtual ~GpuPs();
  GpuPs(const GpuPs&) = delete;
  GpuPs& operator=(const GpuPs&) = delete;

  void split_input_to_shard(KeyType* d_keys, int* d_idx_ptr, size_t len, int* left, int* right, int gpu_num);
  void merge_grad(int gpu_num, KeyType* d_keys, GradType* d_grads, size_t len, size_t& uniq_len);
  void pull_sparse(int num, KeyType* d_keys, ValType* d_vals, size_t len);
  void build_ps(int num, KeyType* h_keys, ValType* h_vals, size_t len, size_t chunk_size, int stream_num);
  void dump();
  void show_one_table(int gpu_num);
  int get_index_by_devid(int devid);
  
  template <typename Sgd>
  void push_sparse(int num, KeyType* d_keys, GradType* d_grads, size_t len, Sgd& sgd);
  
  int log2i(int x);

 private:
  
  using Table = HashTable<KeyType, ValType>;
  int block_size_{256};
  float load_factor_{0.75};
  std::vector<Table*> tables_;
  std::shared_ptr<HeterBoxResource> resource_;
  CustomGradMerger merger_;
};

}  // end namespace framework
}  // end namespace paddle
#include "paddle/fluid/framework/fleet/heter_box/hashtable/gpu_ps.tpp"
#endif
