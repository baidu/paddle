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
#include <string>
#include <vector>
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/detection/bbox_util.h"
#include "paddle/fluid/operators/math/math_function.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using LoDTensor = framework::LoDTensor;

inline std::vector<size_t> GetBoxLodFromRoisNum(const Tensor* rois_num) {
  std::vector<size_t> rois_lod;
  auto* rois_num_data = rois_num->data<int>();
  Tensor cpu_tensor;
  if (platform::is_gpu_place(rois_num->place())) {
    TensorCopySync(*rois_num, platform::CPUPlace(), &cpu_tensor);
    rois_num_data = cpu_tensor.data<int>();
  }
  rois_lod.push_back(static_cast<size_t>(0));
  for (int i = 0; i < rois_num->numel(); ++i) {
    rois_lod.push_back(static_cast<size_t>(rois_lod.back() + rois_num_data[i]));
  }
  return rois_lod;
}

template <typename DeviceContext, typename T>
class BoxClipKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* input_box = context.Input<LoDTensor>("Input");
    auto* im_shape = context.Input<LoDTensor>("ImShape");
    auto* output_box = context.Output<LoDTensor>("Output");
    auto& dev_ctx =
        context.template device_context<platform::CPUDeviceContext>();
    output_box->mutable_data<T>(context.GetPlace());
    if (!context.HasInput("RoisNum") && input_box->lod().size()) {
      PADDLE_ENFORCE_EQ(input_box->lod().size(), 1UL,
                        platform::errors::InvalidArgument(
                            "Input(Input) of BoxClip only supports 1 level "
                            "of LoD. But received the "
                            "level = %d",
                            input_box->lod().size()));
    }
    std::vector<size_t> box_lod;
    if (context.HasInput("RoisNum")) {
      auto* rois_num = context.Input<Tensor>("RoisNum");
      box_lod = GetBoxLodFromRoisNum(rois_num);
    } else {
      box_lod = input_box->lod().back();
    }
    int64_t n = static_cast<int64_t>(box_lod.size() - 1);
    for (int i = 0; i < n; ++i) {
      Tensor im_shape_slice = im_shape->Slice(i, i + 1);
      Tensor box_slice = input_box->Slice(box_lod[i], box_lod[i + 1]);
      Tensor output_slice = output_box->Slice(box_lod[i], box_lod[i + 1]);
      bool is_scale = true;
      ClipTiledBoxes<T>(dev_ctx, im_shape_slice, box_slice, &output_slice,
                        is_scale);
    }
  }
};

}  // namespace operators
}  // namespace paddle
