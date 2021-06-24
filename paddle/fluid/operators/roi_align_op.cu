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
#include "paddle/fluid/operators/roi_align_op.h"
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

template <class T>
__device__ T BilinearInterpolate(const T* input_data, const int height,
                                 const int width, T y, T x) {
  if (y < -1.0 || y > height || x < -1.0 || x > width) {
    return 0;
  }
  y = y <= 0 ? 0 : y;
  x = x <= 0 ? 0 : x;
  int y_low = static_cast<int>(y);
  int x_low = static_cast<int>(x);
  int y_high;
  int x_high;
  if (y_low >= height - 1) {
    y_high = y_low = height - 1;
    y = static_cast<T>(y_low);
  } else {
    y_high = y_low + 1;
  }
  if (x_low >= width - 1) {
    x_high = x_low = width - 1;
    x = static_cast<T>(x_low);
  } else {
    x_high = x_low + 1;
  }
  T ly = y - y_low, lx = x - x_low;
  T hy = 1. - ly, hx = 1. - lx;

  T v1 = input_data[y_low * width + x_low];
  T v2 = input_data[y_low * width + x_high];
  T v3 = input_data[y_high * width + x_low];
  T v4 = input_data[y_high * width + x_high];
  T w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;

  T val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
  return val;
}

template <class T>
__device__ void BilinearInterpolateGradient(const int height, const int width,
                                            T y, T x, T* w1, T* w2, T* w3,
                                            T* w4, int* x_low, int* x_high,
                                            int* y_low, int* y_high) {
  if (y < -1.0 || y > height || x < -1.0 || x > width) {
    return;
  }

  y = y <= 0 ? 0 : y;
  x = x <= 0 ? 0 : x;
  *y_low = static_cast<int>(y);
  *x_low = static_cast<int>(x);
  if (*y_low >= height - 1) {
    *y_high = *y_low = height - 1;
    y = static_cast<T>(*y_low);
  } else {
    *y_high = *y_low + 1;
  }
  if (*x_low >= width - 1) {
    *x_high = *x_low = width - 1;
    x = static_cast<T>(*x_low);
  } else {
    *x_high = *x_low + 1;
  }
  T ly = y - *y_low, lx = x - *x_low;
  T hy = 1. - ly, hx = 1. - lx;
  *w1 = hy * hx, *w2 = hy * lx, *w3 = ly * hx, *w4 = ly * lx;

  return;
}

template <class T>
__global__ void GPUROIAlignForward(
    const int nthreads, const T* input_data, const T* input_rois,
    const float spatial_scale, const int channels, const int height,
    const int width, const int pooled_height, const int pooled_width,
    const int sampling_ratio, int* roi_batch_id_data, T* output_data,
    const bool continuous_coordinate) {
  CUDA_KERNEL_LOOP(i, nthreads) {
    int pw = i % pooled_width;
    int ph = (i / pooled_width) % pooled_height;
    int c = (i / pooled_width / pooled_height) % channels;
    int n = i / pooled_width / pooled_height / channels;

    const T* offset_input_rois = input_rois + n * kROISize;
    int roi_batch_ind = roi_batch_id_data[n];

    T roi_offset = continuous_coordinate ? static_cast<T>(0.5) : 0;
    T roi_xmin = offset_input_rois[0] * spatial_scale - roi_offset;
    T roi_ymin = offset_input_rois[1] * spatial_scale - roi_offset;
    T roi_xmax = offset_input_rois[2] * spatial_scale - roi_offset;
    T roi_ymax = offset_input_rois[3] * spatial_scale - roi_offset;

    T roi_width = roi_xmax - roi_xmin;
    T roi_height = roi_ymax - roi_ymin;
    if (!continuous_coordinate) {
      roi_width = max(roi_width, static_cast<T>(1.));
      roi_height = max(roi_height, static_cast<T>(1.));
    }

    T bin_size_h = static_cast<T>(roi_height) / static_cast<T>(pooled_height);
    T bin_size_w = static_cast<T>(roi_width) / static_cast<T>(pooled_width);

    const T* offset_input_data =
        input_data + (roi_batch_ind * channels + c) * height * width;

    int roi_bin_grid_h = (sampling_ratio > 0)
                             ? sampling_ratio
                             : ceil(roi_height / pooled_height);
    int roi_bin_grid_w =
        (sampling_ratio > 0) ? sampling_ratio : ceil(roi_width / pooled_width);
    const T count = max(roi_bin_grid_h * roi_bin_grid_w, 1);
    T output_val = 0;
    for (int iy = 0; iy < roi_bin_grid_h; iy++) {
      const T y = roi_ymin + ph * bin_size_h +
                  static_cast<T>(iy + .5f) * bin_size_h /
                      static_cast<T>(roi_bin_grid_h);
      for (int ix = 0; ix < roi_bin_grid_w; ix++) {
        const T x = roi_xmin + pw * bin_size_w +
                    static_cast<T>(ix + .5f) * bin_size_w /
                        static_cast<T>(roi_bin_grid_w);
        T val = BilinearInterpolate(offset_input_data, height, width, y, x);
        output_val += val;
      }
    }
    output_val /= count;
    output_data[i] = output_val;
  }
}

template <typename T>
__global__ void GPUROIAlignBackward(
    const int nthreads, const T* input_rois, const T* out_grad,
    const int num_rois, const float spatial_scale, const int channels,
    const int height, const int width, const int pooled_height,
    const int pooled_width, const int sampling_ratio, int* roi_batch_id_data,
    T* input_grad, const bool continuous_coordinate) {
  CUDA_KERNEL_LOOP(i, nthreads) {
    int pw = i % pooled_width;
    int ph = (i / pooled_width) % pooled_height;
    int c = (i / pooled_width / pooled_height) % channels;
    int n = i / pooled_width / pooled_height / channels;
    const T* offset_input_rois = input_rois + n * kROISize;
    int roi_batch_ind = roi_batch_id_data[n];

    T roi_offset = continuous_coordinate ? T(0.5) : 0;
    T roi_xmin = offset_input_rois[0] * spatial_scale - roi_offset;
    T roi_ymin = offset_input_rois[1] * spatial_scale - roi_offset;
    T roi_xmax = offset_input_rois[2] * spatial_scale - roi_offset;
    T roi_ymax = offset_input_rois[3] * spatial_scale - roi_offset;

    T roi_width = roi_xmax - roi_xmin;
    T roi_height = roi_ymax - roi_ymin;
    if (!continuous_coordinate) {
      roi_width = max(roi_width, static_cast<T>(1.));
      roi_height = max(roi_height, static_cast<T>(1.));
    }
    T bin_size_h = static_cast<T>(roi_height) / static_cast<T>(pooled_height);
    T bin_size_w = static_cast<T>(roi_width) / static_cast<T>(pooled_width);

    T* offset_input_grad =
        input_grad + (roi_batch_ind * channels + c) * height * width;

    const T* offset_out_grad =
        out_grad + (n * channels + c) * pooled_height * pooled_width;
    const T out_grad_this_bin = offset_out_grad[ph * pooled_width + pw];

    int roi_bin_grid_h = (sampling_ratio > 0)
                             ? sampling_ratio
                             : ceil(roi_height / pooled_height);
    int roi_bin_grid_w =
        (sampling_ratio > 0) ? sampling_ratio : ceil(roi_width / pooled_width);

    const T count = roi_bin_grid_h * roi_bin_grid_w;
    for (int iy = 0; iy < roi_bin_grid_h; iy++) {
      const T y = roi_ymin + ph * bin_size_h +
                  static_cast<T>(iy + .5f) * bin_size_h /
                      static_cast<T>(roi_bin_grid_h);
      for (int ix = 0; ix < roi_bin_grid_w; ix++) {
        const T x = roi_xmin + pw * bin_size_w +
                    static_cast<T>(ix + .5f) * bin_size_w /
                        static_cast<T>(roi_bin_grid_w);
        T w1 = 0, w2 = 0, w3 = 0, w4 = 0;
        int x_low = -1, x_high = -1, y_low = -1, y_high = -1;
        BilinearInterpolateGradient(height, width, y, x, &w1, &w2, &w3, &w4,
                                    &x_low, &x_high, &y_low, &y_high);
        T diff1 = out_grad_this_bin * w1 / count;
        T diff2 = out_grad_this_bin * w2 / count;
        T diff3 = out_grad_this_bin * w3 / count;
        T diff4 = out_grad_this_bin * w4 / count;
        if (x_low >= 0 && x_high >= 0 && y_low >= 0 && y_high >= 0) {
          platform::CudaAtomicAdd(offset_input_grad + y_low * width + x_low,
                                  diff1);
          platform::CudaAtomicAdd(offset_input_grad + y_low * width + x_high,
                                  diff2);
          platform::CudaAtomicAdd(offset_input_grad + y_high * width + x_low,
                                  diff3);
          platform::CudaAtomicAdd(offset_input_grad + y_high * width + x_high,
                                  diff4);
        }
      }
    }
  }
}

template <typename Place, typename T>
class GPUROIAlignOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* in = ctx.Input<Tensor>("X");
    auto* rois = ctx.Input<LoDTensor>("ROIs");
    auto* out = ctx.Output<Tensor>("Out");

    auto pooled_height = ctx.Attr<int>("pooled_height");
    auto pooled_width = ctx.Attr<int>("pooled_width");
    auto spatial_scale = ctx.Attr<float>("spatial_scale");
    auto sampling_ratio = ctx.Attr<int>("sampling_ratio");
    auto aligned = ctx.Attr<bool>("aligned");

    auto in_dims = in->dims();
    int batch_size = in_dims[0];
    int channels = in_dims[1];
    int height = in_dims[2];
    int width = in_dims[3];

    int rois_num = rois->dims()[0];

    if (rois_num == 0) return;

    int output_size = out->numel();
    int blocks = NumBlocks(output_size);
    int threads = kNumCUDAThreads;

    Tensor roi_batch_id_list;
    roi_batch_id_list.Resize({rois_num});
    auto cplace = platform::CPUPlace();
    int* roi_batch_id_data = roi_batch_id_list.mutable_data<int>(cplace);
    auto& dev_ctx = ctx.cuda_device_context();
    auto gplace = BOOST_GET_CONST(platform::CUDAPlace, ctx.GetPlace());
    if (ctx.HasInput("RoisNum")) {
      auto* rois_num_t = ctx.Input<Tensor>("RoisNum");
      int rois_batch_size = rois_num_t->numel();
      PADDLE_ENFORCE_EQ(
          rois_batch_size, batch_size,
          platform::errors::InvalidArgument(
              "The rois_batch_size and imgs "
              "batch_size must be the same. But received rois_batch_size = %d, "
              "batch_size = %d",
              rois_batch_size, batch_size));

      std::vector<int> rois_num_list(rois_batch_size);
      memory::Copy(cplace, rois_num_list.data(), gplace,
                   rois_num_t->data<int>(), sizeof(int) * rois_batch_size, 0);
      int start = 0;
      for (int n = 0; n < rois_batch_size; ++n) {
        for (int i = start; i < start + rois_num_list[n]; ++i) {
          roi_batch_id_data[i] = n;
        }
        start += rois_num_list[n];
      }
    } else {
      auto lod = rois->lod();
      PADDLE_ENFORCE_EQ(
          lod.empty(), false,
          platform::errors::InvalidArgument("Input(ROIs) in ROIAlignOp does "
                                            "not contain LoD information."));
      auto rois_lod = lod.back();
      int rois_batch_size = rois_lod.size() - 1;
      PADDLE_ENFORCE_EQ(
          rois_batch_size, batch_size,
          platform::errors::InvalidArgument(
              "The batch size of rois and batch size "
              "of images must be the same. But received rois batch size = %d, "
              "and images batch size = %d",
              rois_batch_size, batch_size));
      int rois_num_with_lod = rois_lod[rois_batch_size];
      PADDLE_ENFORCE_EQ(
          rois_num, rois_num_with_lod,
          platform::errors::InvalidArgument(
              "The actual number of rois and the number of rois "
              "provided from Input(RoIsLoD) in RoIAlign must be the same."
              " But received actual number of rois is %d, and the number "
              "of rois from RoIsLoD is %d",
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
    GPUROIAlignForward<T><<<blocks, threads, 0, dev_ctx.stream()>>>(
        output_size, in->data<T>(), rois->data<T>(), spatial_scale, channels,
        height, width, pooled_height, pooled_width, sampling_ratio, roi_id_data,
        out->mutable_data<T>(ctx.GetPlace()), aligned);
  }
};

template <typename Place, typename T>
class GPUROIAlignGradOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* in = ctx.Input<Tensor>("X");
    auto* rois = ctx.Input<LoDTensor>("ROIs");

    auto* out_grad = ctx.Input<Tensor>(framework::GradVarName("Out"));
    auto* in_grad = ctx.Output<Tensor>(framework::GradVarName("X"));

    auto pooled_height = ctx.Attr<int>("pooled_height");
    auto pooled_width = ctx.Attr<int>("pooled_width");
    auto spatial_scale = ctx.Attr<float>("spatial_scale");
    auto sampling_ratio = ctx.Attr<int>("sampling_ratio");
    auto aligned = ctx.Attr<bool>("aligned");

    int rois_num = rois->dims()[0];
    int channels = in->dims()[1];
    int height = in->dims()[2];
    int width = in->dims()[3];

    if (!in_grad) {
      return;
    }
    Tensor roi_batch_id_list;
    roi_batch_id_list.Resize({rois_num});
    auto cplace = platform::CPUPlace();
    int* roi_batch_id_data = roi_batch_id_list.mutable_data<int>(cplace);

    auto& dev_ctx = ctx.cuda_device_context();
    auto gplace = BOOST_GET_CONST(platform::CUDAPlace, ctx.GetPlace());
    if (ctx.HasInput("RoisNum")) {
      auto* rois_num_t = ctx.Input<Tensor>("RoisNum");
      int rois_batch_size = rois_num_t->numel();
      std::vector<int> rois_num_list(rois_batch_size);
      memory::Copy(cplace, rois_num_list.data(), gplace,
                   rois_num_t->data<int>(), sizeof(int) * rois_batch_size, 0);
      int start = 0;
      for (int n = 0; n < rois_batch_size; ++n) {
        for (size_t i = start; i < start + rois_num_list[n]; ++i) {
          roi_batch_id_data[i] = n;
        }
        start += rois_num_list[n];
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
    auto roi_ptr =
        memory::Alloc(dev_ctx, roi_batch_id_list.numel() * sizeof(int));
    int* roi_id_data = reinterpret_cast<int*>(roi_ptr->ptr());
    int bytes = roi_batch_id_list.numel() * sizeof(int);
    memory::Copy(gplace, roi_id_data, cplace, roi_batch_id_data, bytes,
                 dev_ctx.stream());
    in_grad->mutable_data<T>(ctx.GetPlace());
    math::SetConstant<Place, T> set_zero;
    set_zero(dev_ctx, in_grad, static_cast<T>(0));

    int output_grad_size = out_grad->numel();
    int blocks = NumBlocks(output_grad_size);
    int threads = kNumCUDAThreads;

    if (output_grad_size > 0) {
      GPUROIAlignBackward<T><<<blocks, threads, 0, dev_ctx.stream()>>>(
          output_grad_size, rois->data<T>(), out_grad->data<T>(), rois_num,
          spatial_scale, channels, height, width, pooled_height, pooled_width,
          sampling_ratio, roi_id_data, in_grad->mutable_data<T>(ctx.GetPlace()),
          aligned);
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(
    roi_align,
    ops::GPUROIAlignOpKernel<paddle::platform::CUDADeviceContext, float>,
    ops::GPUROIAlignOpKernel<paddle::platform::CUDADeviceContext, double>);
REGISTER_OP_CUDA_KERNEL(
    roi_align_grad,
    ops::GPUROIAlignGradOpKernel<paddle::platform::CUDADeviceContext, float>,
    ops::GPUROIAlignGradOpKernel<paddle::platform::CUDADeviceContext, double>);
