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

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator_kernel_configs.h"
#include "paddle/fluid/platform/cudnn_helper.h"

namespace paddle {
namespace framework {

template <typename AlgoT>
struct CudnnConvAlgorithm {
  AlgoT algo;
  size_t workspace_size;

  CudnnConvAlgorithm() { workspace_size = 0; }
  CudnnConvAlgorithm(AlgoT a, size_t w) : algo(a), workspace_size(w) {}
};

// ConvSearchCache using framework::AlgorithmsCache to search
// cudnnConvolutionFwdAlgo_t, cudnnConvolutionBwdDataAlgo_t or
// cudnnConvolutionBwdFilterAlgo_t
class ConvSearchCache {
 public:
  static ConvSearchCache& Instance() {
    static ConvSearchCache instance;
    return instance;
  }

  AlgorithmsCache<CudnnConvAlgorithm<cudnnConvolutionFwdAlgo_t>>* GetForward() {
    return &forward_cache_;
  }
  AlgorithmsCache<CudnnConvAlgorithm<cudnnConvolutionBwdDataAlgo_t>>*
  GetBackwardData() {
    return &backward_data_cache_;
  }
  AlgorithmsCache<CudnnConvAlgorithm<cudnnConvolutionBwdFilterAlgo_t>>*
  GetBackwardFilter() {
    return &backward_filter_cache_;
  }
  AlgorithmsCache<cudnnConvolutionFwdAlgo_t>* GetConvFusion() {
    return &fusion_forward_cache_;
  }

 private:
  ConvSearchCache() {}
  ~ConvSearchCache() {}
  ConvSearchCache(const ConvSearchCache&) {}
  ConvSearchCache& operator=(const ConvSearchCache&) {}

  AlgorithmsCache<CudnnConvAlgorithm<cudnnConvolutionFwdAlgo_t>> forward_cache_;
  AlgorithmsCache<CudnnConvAlgorithm<cudnnConvolutionBwdDataAlgo_t>>
      backward_data_cache_;
  AlgorithmsCache<CudnnConvAlgorithm<cudnnConvolutionBwdFilterAlgo_t>>
      backward_filter_cache_;
  AlgorithmsCache<cudnnConvolutionFwdAlgo_t> fusion_forward_cache_;
};

}  // namespace framework
}  // namespace paddle
