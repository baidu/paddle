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

#include <algorithm>
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/detection/box_clip_op.h"
#include "paddle/fluid/operators/math/math_function.h"
#include "paddle/fluid/platform/cuda_primitives.h"
#include "paddle/fluid/platform/hostdevice.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using LoDTenso = framework::LoDTensor;

static constexpr int ImShapeSize = 2;

template <typename T, int BlockSize>
static __global__ void GPUBoxClip(const T *input, const size_t *lod,
                                  const size_t width, const T *im_shape,
                                  T *output) {
  T im_w = im_shape[blockIdx.x * ImShapeSize + 1];
  T im_h = im_shape[blockIdx.x * ImShapeSize];
  for (size_t i = threadIdx.x;
       i < (lod[blockIdx.x + 1] - lod[blockIdx.x]) * width; i += BlockSize) {
    size_t idx = lod[blockIdx.x] * width + i;
    T im_size = (idx % 2 == 0) ? im_w : im_h;
    output[idx] = max(min(input[idx], im_size - 1), T(0.));
  }
}

template <typename DeviceContext, typename T>
class GPUBoxClipKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &context) const override {
    auto *input = context.Input<LoDTensor>("Input");
    auto *im_shape = context.Input<Tensor>("ImShape");
    auto *output = context.Output<LoDTensor>("Output");
    const int64_t num = input->dims()[0];
    const int64_t bbox_width = input->numel() / num;

    auto &dev_ctx = context.template device_context<DeviceContext>();
    auto stream = dev_ctx.stream();
    T *output_data = output->mutable_data<T>(input->dims(), dev_ctx.GetPlace());

    framework::LoD lod;
    if (context.HasInput("RoisNum")) {
      auto *rois_num = context.Input<Tensor>("RoisNum");
      std::vector<size_t> offset_v(1, 0);
      const int *rois_num_data = rois_num->data<int>();

      size_t offset = 0;
      auto place = BOOST_GET_CONST(platform::CUDAPlace, dev_ctx.GetPlace());
      auto cpu_place = platform::CPUPlace();
      auto temp_val_ptr =
          memory::Alloc(platform::CPUPlace(), rois_num->numel() * sizeof(int));
      int *cpu_data = reinterpret_cast<int *>(temp_val_ptr->ptr());

      memory::Copy(cpu_place, cpu_data, place, rois_num_data,
                   sizeof(int) * rois_num->numel(), dev_ctx.stream());
      dev_ctx.Wait();

      for (int i = 0; i < rois_num->numel(); ++i) {
        offset += cpu_data[i];
        offset_v.emplace_back(offset);
      }
      lod.emplace_back(offset_v);
    } else {
      lod = input->lod();
    }
    framework::LoD abs_offset_lod = framework::ToAbsOffset(lod);
    const size_t batch_size = lod.back().size() - 1;
    GPUBoxClip<T, 512><<<batch_size, 512, 0, stream>>>(
        input->data<T>(), abs_offset_lod[0].CUDAMutableData(dev_ctx.GetPlace()),
        bbox_width, im_shape->data<T>(), output_data);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(
    box_clip, ops::GPUBoxClipKernel<paddle::platform::CUDADeviceContext, float>,
    ops::GPUBoxClipKernel<paddle::platform::CUDADeviceContext, double>);
