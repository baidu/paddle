/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */
#include <vector>
#include "paddle/fluid/memory/memory.h"
#include "paddle/fluid/operators/roi_pool_op.h"
#include "paddle/fluid/platform/cuda_primitives.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using LoDTensor = framework::LoDTensor;

static constexpr int kNumCUDAThreads = 512;
static constexpr int kNumMaxinumNumBlocks = 4096;

static inline int NumBlocks(const int N) {
  return std::min((N + kNumCUDAThreads - 1) / kNumCUDAThreads,
                  kNumMaxinumNumBlocks);
}

template <typename T>
__global__ void GPUROIPoolForward(
    const int nthreads, const T* input_data, const T* input_rois,
    const float spatial_scale, const int channels, const int height,
    const int width, const int pooled_height, const int pooled_width,
    int* roi_batch_id_data, T* output_data, int64_t* argmax_data) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int offset = blockDim.x * gridDim.x;
  for (size_t i = index; i < nthreads; i += offset) {
    int pw = i % pooled_width;
    int ph = (i / pooled_width) % pooled_height;
    int c = (i / pooled_width / pooled_height) % channels;
    int n = i / pooled_width / pooled_height / channels;

    const T* offset_input_rois = input_rois + n * kROISize;
    int roi_batch_ind = roi_batch_id_data[n];
    int roi_start_w = round(offset_input_rois[0] * spatial_scale);
    int roi_start_h = round(offset_input_rois[1] * spatial_scale);
    int roi_end_w = round(offset_input_rois[2] * spatial_scale);
    int roi_end_h = round(offset_input_rois[3] * spatial_scale);

    int roi_width = max(roi_end_w - roi_start_w + 1, 1);
    int roi_height = max(roi_end_h - roi_start_h + 1, 1);

    int hstart = static_cast<int>(floor(static_cast<double>(ph) *
                                        static_cast<double>(roi_height) /
                                        static_cast<double>(pooled_height)));
    int wstart = static_cast<int>(floor(static_cast<double>(pw) *
                                        static_cast<double>(roi_width) /
                                        static_cast<double>(pooled_width)));
    int hend = static_cast<int>(ceil(static_cast<double>(ph + 1) *
                                     static_cast<double>(roi_height) /
                                     static_cast<double>(pooled_height)));
    int wend = static_cast<int>(ceil(static_cast<double>(pw + 1) *
                                     static_cast<double>(roi_width) /
                                     static_cast<double>(pooled_width)));
    hstart = min(max(hstart + roi_start_h, 0), height);
    hend = min(max(hend + roi_start_h, 0), height);
    wstart = min(max(wstart + roi_start_w, 0), width);
    wend = min(max(wend + roi_start_w, 0), width);
    bool is_empty = (hend <= hstart) || (wend <= wstart);

    T maxval = is_empty ? 0 : -std::numeric_limits<T>::max();
    int maxidx = -1;
    const T* offset_input_data =
        input_data + (roi_batch_ind * channels + c) * height * width;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        int input_data_index = h * width + w;
        if (offset_input_data[input_data_index] > maxval) {
          maxval = offset_input_data[input_data_index];
          maxidx = input_data_index;
        }
      }
    }
    output_data[i] = maxval;
    if (argmax_data) {
      argmax_data[i] = maxidx;
    }
  }
}

template <typename T>
__global__ void GPUROIPoolBackward(
    const int nthreads, const T* input_rois, const T* output_grad,
    const int64_t* argmax_data, const int num_rois, const float spatial_scale,
    const int channels, const int height, const int width,
    const int pooled_height, const int pooled_width, int* roi_batch_id_data,
    T* input_grad) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int offset = blockDim.x * gridDim.x;
  for (int i = index; i < nthreads; i += offset) {
    int pw = i % pooled_width;
    int ph = (i / pooled_width) % pooled_height;
    int c = (i / pooled_width / pooled_height) % channels;
    int n = i / pooled_width / pooled_height / channels;

    int roi_batch_ind = roi_batch_id_data[n];
    int input_offset = (roi_batch_ind * channels + c) * height * width;
    int output_offset = (n * channels + c) * pooled_height * pooled_width;
    const T* offset_output_grad = output_grad + output_offset;
    T* offset_input_grad = input_grad + input_offset;
    const int64_t* offset_argmax_data = argmax_data + output_offset;

    int argmax = offset_argmax_data[ph * pooled_width + pw];
    if (argmax != -1) {
      platform::CudaAtomicAdd(
          offset_input_grad + argmax,
          static_cast<T>(offset_output_grad[ph * pooled_width + pw]));
    }
  }
}

template <typename Place, typename T>
class GPUROIPoolOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* in = ctx.Input<Tensor>("X");
    auto* rois = ctx.Input<LoDTensor>("ROIs");
    auto* out = ctx.Output<Tensor>("Out");
    auto* argmax = ctx.Output<Tensor>("Argmax");

    auto pooled_height = ctx.Attr<int>("pooled_height");
    auto pooled_width = ctx.Attr<int>("pooled_width");
    auto spatial_scale = ctx.Attr<float>("spatial_scale");

    auto in_dims = in->dims();
    int batch_size = in_dims[0];
    auto in_stride = framework::stride(in_dims);
    int channels = in_dims[1];
    int height = in_dims[2];
    int width = in_dims[3];

    int rois_num = rois->dims()[0];

    if (rois_num == 0) return;

    int output_size = out->numel();
    int blocks = NumBlocks(output_size);
    int threads = kNumCUDAThreads;

    framework::Tensor roi_batch_id_list;
    roi_batch_id_list.Resize({rois_num});
    auto cplace = platform::CPUPlace();
    int* roi_batch_id_data = roi_batch_id_list.mutable_data<int>(cplace);
    auto& dev_ctx = ctx.cuda_device_context();
    auto gplace = BOOST_GET_CONST(platform::CUDAPlace, ctx.GetPlace());
    if (ctx.HasInput("RoisLod")) {
      auto* rois_lod = ctx.Input<Tensor>("RoisLod");
      int rois_batch_size = rois_lod->numel();
      PADDLE_ENFORCE_EQ(
          rois_batch_size - 1, batch_size,
          platform::errors::InvalidArgument(
              "The batch size of input(ROIs) and input(X) must be the same but "
              "received batch size of input(ROIs) and input(X) is %d and %d "
              "respectively.",
              rois_batch_size, batch_size));
      std::vector<int64_t> rois_lod_(rois_batch_size);
      memory::Copy(cplace, rois_lod_.data(), gplace, rois_lod->data<int64_t>(),
                   sizeof(int64_t) * rois_batch_size, 0);
      for (int n = 0; n < rois_batch_size - 1; ++n) {
        for (size_t i = rois_lod_[n]; i < rois_lod_[n + 1]; ++i) {
          roi_batch_id_data[i] = n;
        }
      }
    } else {
      auto rois_lod = rois->lod().back();
      int rois_batch_size = rois_lod.size() - 1;
      PADDLE_ENFORCE_EQ(
          rois_batch_size, batch_size,
          platform::errors::InvalidArgument(
              "The batch size of input(ROIs) and input(X) must be the same but "
              "received batch size of input(ROIs) and input(X) is %d and %d "
              "respectively.",
              rois_batch_size, batch_size));

      int rois_num_with_lod = rois_lod[rois_batch_size];
      PADDLE_ENFORCE_EQ(rois_num, rois_num_with_lod,
                        platform::errors::InvalidArgument(
                            "The number of rois from input(ROIs) and its LOD "
                            "must be the same. Received rois %d of input(ROIs) "
                            "but the number of rois %d from its LOD is %d",
                            rois_num, rois_num_with_lod));
      for (int n = 0; n < rois_batch_size; ++n) {
        for (size_t i = rois_lod[n]; i < rois_lod[n + 1]; ++i) {
          roi_batch_id_data[i] = n;
        }
      }
    }
    int bytes = roi_batch_id_list.numel() * sizeof(int);
    auto roi_ptr = memory::Alloc(dev_ctx, bytes);
    int* roi_id_data = reinterpret_cast<int*>(roi_ptr->ptr());
    memory::Copy(gplace, roi_id_data, cplace, roi_batch_id_data, bytes,
                 dev_ctx.stream());

    GPUROIPoolForward<T><<<blocks, threads, 0, dev_ctx.stream()>>>(
        output_size, in->data<T>(), rois->data<T>(), spatial_scale, channels,
        height, width, pooled_height, pooled_width, roi_id_data,
        out->mutable_data<T>(ctx.GetPlace()),
        argmax->mutable_data<int64_t>(ctx.GetPlace()));
  }
};

template <typename Place, typename T>
class GPUROIPoolGradOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* in = ctx.Input<Tensor>("X");
    auto* rois = ctx.Input<LoDTensor>("ROIs");
    auto* rois_lod = ctx.Input<Tensor>("RoisLod");
    auto* argmax = ctx.Input<Tensor>("Argmax");

    auto* out_grad = ctx.Input<Tensor>(framework::GradVarName("Out"));
    auto* x_grad = ctx.Output<Tensor>(framework::GradVarName("X"));

    auto pooled_height = ctx.Attr<int>("pooled_height");
    auto pooled_width = ctx.Attr<int>("pooled_width");
    auto spatial_scale = ctx.Attr<float>("spatial_scale");

    int rois_num = rois->dims()[0];
    int channels = in->dims()[1];
    int height = in->dims()[2];
    int width = in->dims()[3];

    if (x_grad) {
      framework::Tensor roi_batch_id_list;
      roi_batch_id_list.Resize({rois_num});
      auto cplace = platform::CPUPlace();
      int* roi_batch_id_data = roi_batch_id_list.mutable_data<int>(cplace);

      auto& dev_ctx = ctx.cuda_device_context();
      auto gplace = BOOST_GET_CONST(platform::CUDAPlace, ctx.GetPlace());
      if (ctx.HasInput("RoisLod")) {
        auto* rois_lod = ctx.Input<Tensor>("RoisLod");
        int rois_batch_size = rois_lod->numel();
        std::vector<int64_t> rois_lod_(rois_batch_size);
        memory::Copy(cplace, rois_lod_.data(), gplace,
                     rois_lod->data<int64_t>(),
                     sizeof(int64_t) * rois_batch_size, 0);
        for (int n = 0; n < rois_batch_size - 1; ++n) {
          for (size_t i = rois_lod_[n]; i < rois_lod_[n + 1]; ++i) {
            roi_batch_id_data[i] = n;
          }
        }
      } else {
        auto rois_lod = rois->lod().back();
        int rois_batch_size = rois_lod.size() - 1;
        for (int n = 0; n < rois_batch_size; ++n) {
          for (size_t i = rois_lod[n]; i < rois_lod[n + 1]; ++i) {
            roi_batch_id_data[i] = n;
          }
        }
      }
      int bytes = roi_batch_id_list.numel() * sizeof(int);
      auto roi_ptr = memory::Alloc(dev_ctx, bytes);
      int* roi_id_data = reinterpret_cast<int*>(roi_ptr->ptr());
      memory::Copy(gplace, roi_id_data, cplace, roi_batch_id_data, bytes,
                   dev_ctx.stream());

      x_grad->mutable_data<T>(ctx.GetPlace());
      math::SetConstant<Place, T> set_zero;
      set_zero(dev_ctx, x_grad, static_cast<T>(0));

      int output_grad_size = out_grad->numel();
      int blocks = NumBlocks(output_grad_size);
      int threads = kNumCUDAThreads;

      if (output_grad_size > 0) {
        GPUROIPoolBackward<T><<<blocks, threads, 0, dev_ctx.stream()>>>(
            output_grad_size, rois->data<T>(), out_grad->data<T>(),
            argmax->data<int64_t>(), rois_num, spatial_scale, channels, height,
            width, pooled_height, pooled_width, roi_id_data,
            x_grad->mutable_data<T>(ctx.GetPlace()));
      }
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(
    roi_pool,
    ops::GPUROIPoolOpKernel<paddle::platform::CUDADeviceContext, float>,
    ops::GPUROIPoolOpKernel<paddle::platform::CUDADeviceContext, double>);
REGISTER_OP_CUDA_KERNEL(
    roi_pool_grad,
    ops::GPUROIPoolGradOpKernel<paddle::platform::CUDADeviceContext, float>,
    ops::GPUROIPoolGradOpKernel<paddle::platform::CUDADeviceContext, double>);
