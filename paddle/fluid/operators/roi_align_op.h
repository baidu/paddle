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
#include <algorithm>
#include <limits>
#include <vector>
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/math/math_function.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using LoDTensor = framework::LoDTensor;

static constexpr int kROISize = 4;

template <class T>
void PreCalcForBilinearInterpolate(
    const platform::DeviceContext& ctx, const int height, const int width,
    const int pooled_height, const int pooled_width, const int iy_upper,
    const int ix_upper, T roi_ymin, T roi_xmin, T bin_size_h, T bin_size_w,
    int roi_bin_grid_h, int roi_bin_grid_w, Tensor* pre_pos, Tensor* pre_w) {
  int pre_calc_index = 0;
  int* pre_pos_data = pre_pos->mutable_data<int>(ctx.GetPlace());
  T* pre_w_data = pre_w->mutable_data<T>(ctx.GetPlace());
  for (int ph = 0; ph < pooled_height; ph++) {
    for (int pw = 0; pw < pooled_width; pw++) {
      for (int iy = 0; iy < iy_upper; iy++) {
        // calculate y of sample points
        T y = roi_ymin + ph * bin_size_h +
              static_cast<T>(iy + .5f) * bin_size_h /
                  static_cast<T>(roi_bin_grid_h);
        // calculate x of samle points
        for (int ix = 0; ix < ix_upper; ix++) {
          T x = roi_xmin + pw * bin_size_w +
                static_cast<T>(ix + .5f) * bin_size_w /
                    static_cast<T>(roi_bin_grid_w);
          // deal with elements out of map
          if (y < -1.0 || y > height || x < -1.0 || x > width) {
            for (int i = 0; i < kROISize; ++i) {
              pre_pos_data[i + pre_calc_index * kROISize] = 0;
              pre_w_data[i + pre_calc_index * kROISize] = 0;
            }
            pre_calc_index += 1;
            continue;
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
          pre_pos_data[pre_calc_index * kROISize] = y_low * width + x_low;
          pre_pos_data[pre_calc_index * kROISize + 1] = y_low * width + x_high;
          pre_pos_data[pre_calc_index * kROISize + 2] = y_high * width + x_low;
          pre_pos_data[pre_calc_index * kROISize + 3] = y_high * width + x_high;
          pre_w_data[pre_calc_index * kROISize] = hy * hx;
          pre_w_data[pre_calc_index * kROISize + 1] = hy * lx;
          pre_w_data[pre_calc_index * kROISize + 2] = ly * hx;
          pre_w_data[pre_calc_index * kROISize + 3] = ly * lx;
          pre_calc_index += 1;
        }
      }
    }
  }
}

template <class T>
void bilinear_interpolate_gradient(const int height, const int width, T y, T x,
                                   const T out_grad_this_bin, const T count,
                                   T* batch_grad_data) {
  int x_low, y_low, x_high, y_high;
  T w1, w2, w3, w4;
  if (y < -1.0 || y > height || x < -1.0 || x > width) {
    w1 = w2 = w3 = w4 = 0;
    x_low = x_high = y_low = y_high = -1;
    return;
  }
  y = y <= 0 ? 0 : y;
  x = x <= 0 ? 0 : x;
  y_low = static_cast<int>(y);
  x_low = static_cast<int>(x);
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
  w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;
  T diff1 = out_grad_this_bin * w1 / count;
  T diff2 = out_grad_this_bin * w2 / count;
  T diff3 = out_grad_this_bin * w3 / count;
  T diff4 = out_grad_this_bin * w4 / count;
  if (x_low >= 0 && x_high >= 0 && y_low >= 0 && y_high >= 0) {
    *(batch_grad_data + y_low * width + x_low) += diff1;
    *(batch_grad_data + y_low * width + x_high) += diff2;
    *(batch_grad_data + y_high * width + x_low) += diff3;
    *(batch_grad_data + y_high * width + x_high) += diff4;
  }
}

template <typename DeviceContext, typename T>
class CPUROIAlignOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* in = ctx.Input<framework::Tensor>("X");
    auto* rois = ctx.Input<framework::LoDTensor>("ROIs");
    auto* out = ctx.Output<framework::Tensor>("Out");
    auto pooled_height = ctx.Attr<int>("pooled_height");
    auto pooled_width = ctx.Attr<int>("pooled_width");
    auto spatial_scale = ctx.Attr<float>("spatial_scale");
    auto sampling_ratio = ctx.Attr<int>("sampling_ratio");
    auto aligned = ctx.Attr<bool>("aligned");

    auto& dev_ctx = ctx.template device_context<DeviceContext>();

    auto in_dims = in->dims();
    int batch_size = in_dims[0];
    int channels = in_dims[1];
    int height = in_dims[2];
    int width = in_dims[3];
    int rois_num = rois->dims()[0];

    auto in_stride = framework::stride(in_dims);
    auto roi_stride = framework::stride(rois->dims());
    auto out_stride = framework::stride(out->dims());

    const T* input_data = in->data<T>();
    framework::Tensor roi_batch_id_list;
    roi_batch_id_list.Resize({rois_num});
    int* roi_batch_id_data =
        roi_batch_id_list.mutable_data<int>(ctx.GetPlace());
    int rois_batch_size;
    if (ctx.HasInput("RoisNum")) {
      auto* rois_num_t = ctx.Input<framework::Tensor>("RoisNum");
      rois_batch_size = rois_num_t->numel();
      PADDLE_ENFORCE_EQ(
          rois_batch_size, batch_size,
          platform::errors::InvalidArgument(
              "The batch size of rois and the batch size of images "
              " must be the same. But received the batch size of rois is %d, "
              "and the batch size of images is %d",
              rois_batch_size, batch_size));
      auto* rois_num_data = rois_num_t->data<int>();
      int start = 0;
      for (int n = 0; n < rois_batch_size; ++n) {
        for (int i = start; i < start + rois_num_data[n]; ++i) {
          roi_batch_id_data[i] = n;
        }
        start += rois_num_data[n];
      }
    } else {
      auto lod = rois->lod();
      PADDLE_ENFORCE_EQ(lod.empty(), false,
                        platform::errors::InvalidArgument(
                            "Input(ROIs) Tensor of ROIAlignOp "
                            "does not contain LoD information."));
      auto rois_lod = lod.back();
      int rois_batch_size = rois_lod.size() - 1;
      PADDLE_ENFORCE_EQ(
          rois_batch_size, batch_size,
          platform::errors::InvalidArgument(
              "The rois_batch_size and imgs "
              "batch_size must be the same. But received rois_batch_size = %d, "
              "batch_size = %d",
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
    T* output_data = out->mutable_data<T>(ctx.GetPlace());
    const T* rois_data = rois->data<T>();
    T roi_offset = aligned ? T(0.5) : 0;
    for (int n = 0; n < rois_num; ++n) {
      int roi_batch_id = roi_batch_id_data[n];
      T roi_xmin = rois_data[0] * spatial_scale - roi_offset;
      T roi_ymin = rois_data[1] * spatial_scale - roi_offset;
      T roi_xmax = rois_data[2] * spatial_scale - roi_offset;
      T roi_ymax = rois_data[3] * spatial_scale - roi_offset;

      T roi_width = roi_xmax - roi_xmin;
      T roi_height = roi_ymax - roi_ymin;
      if (!aligned) {
        roi_width = std::max(roi_width, static_cast<T>(1.));
        roi_height = std::max(roi_height, static_cast<T>(1.));
      }

      T bin_size_h = static_cast<T>(roi_height) / static_cast<T>(pooled_height);
      T bin_size_w = static_cast<T>(roi_width) / static_cast<T>(pooled_width);
      const T* batch_data = input_data + roi_batch_id * in_stride[0];

      int roi_bin_grid_h = (sampling_ratio > 0)
                               ? sampling_ratio
                               : ceil(roi_height / pooled_height);
      int roi_bin_grid_w = (sampling_ratio > 0)
                               ? sampling_ratio
                               : ceil(roi_width / pooled_width);
      const T count = std::max(roi_bin_grid_h * roi_bin_grid_w, 1);
      Tensor pre_pos;
      Tensor pre_w;
      int pre_size = count * out_stride[1];
      pre_pos.Resize({pre_size, kROISize});
      pre_w.Resize({pre_size, kROISize});

      PreCalcForBilinearInterpolate(
          dev_ctx, height, width, pooled_height, pooled_width, roi_bin_grid_h,
          roi_bin_grid_w, roi_ymin, roi_xmin, bin_size_h, bin_size_w,
          roi_bin_grid_h, roi_bin_grid_w, &pre_pos, &pre_w);
      const int* pre_pos_data = pre_pos.data<int>();
      const T* pre_w_data = pre_w.data<T>();
      for (int c = 0; c < channels; c++) {
        int pre_calc_index = 0;
        for (int ph = 0; ph < pooled_height; ph++) {
          for (int pw = 0; pw < pooled_width; pw++) {
            const int pool_index = ph * pooled_width + pw;
            T output_val = 0;
            for (int iy = 0; iy < roi_bin_grid_h; iy++) {
              for (int ix = 0; ix < roi_bin_grid_w; ix++) {
                for (int i = 0; i < kROISize; i++) {
                  int pos = pre_pos_data[pre_calc_index * kROISize + i];
                  T w = pre_w_data[pre_calc_index * kROISize + i];
                  output_val += w * batch_data[pos];
                }
                pre_calc_index += 1;
              }
            }
            output_val /= count;
            output_data[pool_index] = output_val;
          }
        }
        batch_data += in_stride[1];
        output_data += out_stride[1];
      }
      rois_data += roi_stride[0];
    }
  }
};

template <typename DeviceContext, typename T>
class CPUROIAlignGradOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* in = ctx.Input<framework::Tensor>("X");
    auto* rois = ctx.Input<framework::LoDTensor>("ROIs");
    auto* out_grad =
        ctx.Input<framework::Tensor>(framework::GradVarName("Out"));
    auto* in_grad = ctx.Output<framework::Tensor>(framework::GradVarName("X"));

    auto pooled_height = ctx.Attr<int>("pooled_height");
    auto pooled_width = ctx.Attr<int>("pooled_width");
    auto spatial_scale = ctx.Attr<float>("spatial_scale");
    auto sampling_ratio = ctx.Attr<int>("sampling_ratio");
    auto in_dims = in->dims();
    auto aligned = ctx.Attr<bool>("aligned");

    int channels = in_dims[1];
    int height = in_dims[2];
    int width = in_dims[3];
    int rois_num = rois->dims()[0];

    if (!in_grad) {
      return;
    }
    Tensor roi_batch_id_list;
    roi_batch_id_list.Resize({rois_num});
    int* roi_batch_id_data =
        roi_batch_id_list.mutable_data<int>(ctx.GetPlace());

    int rois_batch_size;
    if (ctx.HasInput("RoisNum")) {
      auto* rois_num_t = ctx.Input<framework::Tensor>("RoisNum");
      rois_batch_size = rois_num_t->numel();
      auto* rois_num_data = rois_num_t->data<int>();
      int start = 0;
      for (int n = 0; n < rois_batch_size; ++n) {
        for (int i = start; i < start + rois_num_data[n]; ++i) {
          roi_batch_id_data[i] = n;
        }
        start += rois_num_data[n];
      }
    } else {
      auto rois_lod = rois->lod().back();
      rois_batch_size = rois_lod.size() - 1;
      for (int n = 0; n < rois_batch_size; ++n) {
        for (size_t i = rois_lod[n]; i < rois_lod[n + 1]; ++i) {
          roi_batch_id_data[i] = n;
        }
      }
    }
    in_grad->mutable_data<T>(ctx.GetPlace());
    auto& dev_ctx = ctx.template device_context<DeviceContext>();
    math::SetConstant<DeviceContext, T> set_zero;
    set_zero(dev_ctx, in_grad, static_cast<T>(0));

    int output_grad_size = out_grad->numel();

    if ((!out_grad->IsInitialized()) || (output_grad_size <= 0)) {
      return;
    }

    const T* rois_data = rois->data<T>();
    const T* out_grad_data = out_grad->data<T>();
    T* in_grad_data = in_grad->mutable_data<T>(ctx.GetPlace());

    auto in_stride = framework::stride(in->dims());
    auto roi_stride = framework::stride(rois->dims());
    auto out_stride = framework::stride(out_grad->dims());

    T roi_offset = aligned ? T(0.5) : 0;
    for (int n = 0; n < rois_num; ++n) {
      int roi_batch_idx = roi_batch_id_data[n];
      T roi_xmin = rois_data[0] * spatial_scale - roi_offset;
      T roi_ymin = rois_data[1] * spatial_scale - roi_offset;
      T roi_xmax = rois_data[2] * spatial_scale - roi_offset;
      T roi_ymax = rois_data[3] * spatial_scale - roi_offset;

      T roi_width = roi_xmax - roi_xmin;
      T roi_height = roi_ymax - roi_ymin;
      roi_width = std::max(roi_width, static_cast<T>(1.));
      roi_height = std::max(roi_height, static_cast<T>(1.));
      if (!aligned) {
        roi_width = std::max(roi_width, static_cast<T>(1.));
        roi_height = std::max(roi_height, static_cast<T>(1.));
      }

      T bin_size_h = static_cast<T>(roi_height) / static_cast<T>(pooled_height);
      T bin_size_w = static_cast<T>(roi_width) / static_cast<T>(pooled_width);
      for (int c = 0; c < channels; ++c) {
        T* batch_grad_data =
            in_grad_data + roi_batch_idx * in_stride[0] + c * in_stride[1];
        const T* batch_out_grad_data =
            out_grad_data + n * out_stride[0] + c * out_stride[1];
        for (int ph = 0; ph < pooled_height; ++ph) {
          for (int pw = 0; pw < pooled_width; ++pw) {
            int pool_index = ph * pooled_width + pw;
            T out_grad_this_bin = batch_out_grad_data[pool_index];
            int roi_bin_grid_h = (sampling_ratio > 0)
                                     ? sampling_ratio
                                     : ceil(roi_height / pooled_height);
            int roi_bin_grid_w = (sampling_ratio > 0)
                                     ? sampling_ratio
                                     : ceil(roi_width / pooled_width);
            T count = roi_bin_grid_h * roi_bin_grid_w;
            for (int iy = 0; iy < roi_bin_grid_h; iy++) {
              const T y = roi_ymin + ph * bin_size_h +
                          static_cast<T>(iy + .5f) * bin_size_h /
                              static_cast<T>(roi_bin_grid_h);
              for (int ix = 0; ix < roi_bin_grid_w; ix++) {
                const T x = roi_xmin + pw * bin_size_w +
                            static_cast<T>(ix + .5f) * bin_size_w /
                                static_cast<T>(roi_bin_grid_w);
                bilinear_interpolate_gradient(height, width, y, x,
                                              out_grad_this_bin, count,
                                              batch_grad_data);
              }
            }
          }
        }
      }
      rois_data += roi_stride[0];
    }
  }
};
}  // namespace operators
}  // namespace paddle
