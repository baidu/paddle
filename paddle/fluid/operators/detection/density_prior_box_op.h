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

#pragma once
#include <algorithm>
#include <vector>
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/math/math_function.h"
#include "paddle/fluid/platform/transform.h"

namespace paddle {
namespace operators {

inline void ExpandAspectRatios(const std::vector<float>& input_aspect_ratior,
                               bool flip,
                               std::vector<float>* output_aspect_ratior) {
  constexpr float epsilon = 1e-6;
  output_aspect_ratior->clear();
  output_aspect_ratior->push_back(1.0f);
  for (size_t i = 0; i < input_aspect_ratior.size(); ++i) {
    float ar = input_aspect_ratior[i];
    bool already_exist = false;
    for (size_t j = 0; j < output_aspect_ratior->size(); ++j) {
      if (fabs(ar - output_aspect_ratior->at(j)) < epsilon) {
        already_exist = true;
        break;
      }
    }
    if (!already_exist) {
      output_aspect_ratior->push_back(ar);
      if (flip) {
        output_aspect_ratior->push_back(1.0f / ar);
      }
    }
  }
}

template <typename T>
struct ClipFunctor {
  HOSTDEVICE inline T operator()(T in) const {
    return std::min<T>(std::max<T>(in, 0.), 1.);
  }
};

template <typename T>
class DensityPriorBoxOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* input = ctx.Input<paddle::framework::Tensor>("Input");
    auto* image = ctx.Input<paddle::framework::Tensor>("Image");
    auto* boxes = ctx.Output<paddle::framework::Tensor>("Boxes");
    auto* vars = ctx.Output<paddle::framework::Tensor>("Variances");

    auto min_sizes = ctx.Attr<std::vector<float>>("min_sizes");
    auto max_sizes = ctx.Attr<std::vector<float>>("max_sizes");
    auto input_aspect_ratio = ctx.Attr<std::vector<float>>("aspect_ratios");
    auto variances = ctx.Attr<std::vector<float>>("variances");
    auto flip = ctx.Attr<bool>("flip");
    auto clip = ctx.Attr<bool>("clip");
    auto min_max_aspect_ratios_order =
        ctx.Attr<bool>("min_max_aspect_ratios_order");
    auto fixed_sizes = ctx.Attr<std::vector<float>>("fixed_sizes");
    auto fixed_ratios = ctx.Attr<std::vector<float>>("fixed_ratios");
    auto densities = ctx.Attr<std::vector<int>>("densities");
    std::vector<float> aspect_ratios;

    ExpandAspectRatios(input_aspect_ratio, flip, &aspect_ratios);

    T step_w = static_cast<T>(ctx.Attr<float>("step_w"));
    T step_h = static_cast<T>(ctx.Attr<float>("step_h"));
    T offset = static_cast<T>(ctx.Attr<float>("offset"));

    auto img_width = image->dims()[3];
    auto img_height = image->dims()[2];

    auto feature_width = input->dims()[3];
    auto feature_height = input->dims()[2];

    T step_width, step_height;
    if (step_w == 0 || step_h == 0) {
      step_width = static_cast<T>(img_width) / feature_width;
      step_height = static_cast<T>(img_height) / feature_height;
    } else {
      step_width = step_w;
      step_height = step_h;
    }

    int num_priors = aspect_ratios.size() * min_sizes.size();

    if (fixed_sizes.size() > 0) {
      if (densities.size() > 0) {
        for (size_t i = 0; i < densities.size(); ++i) {
          if (fixed_ratios.size() > 0) {
            num_priors += (fixed_ratios.size()) * (pow(densities[i], 2));
          } else {
            num_priors += (aspect_ratios.size()) * (pow(densities[i], 2));
          }
        }
      }
    }
    if (max_sizes.size() > 0) {
      num_priors += max_sizes.size();
    }
    boxes->mutable_data<T>(ctx.GetPlace());
    vars->mutable_data<T>(ctx.GetPlace());
    auto e_boxes = framework::EigenTensor<T, 4>::From(*boxes).setConstant(0.0);

    int step_average = static_cast<int>((step_width + step_height) * 0.5);

    for (int h = 0; h < feature_height; ++h) {
      for (int w = 0; w < feature_width; ++w) {
        T center_x = (w + offset) * step_width;
        T center_y = (h + offset) * step_height;
        T box_width, box_height;
        int idx = 0
            // Generate density prior boxes with fixed sizes.
            for (size_t s = 0; s < fixed_sizes.size(); ++s) {
          auto fixed_size = fixed_sizes[s];
          box_width = box_height = fixed_size;
          int density = densities[s];
          // Generate density prior boxes with fixed ratios.
          if (fixed_ratios.size() > 0) {
            for (size_t r = 0; r < fixed_ratios.size(); ++r) {
              float ar = fixed_ratios[r];
              int shift = step_average / density;
              float box_width_ratio = fixed_size * sqrt(ar);
              float box_height_ratio = fixed_size / sqrt(ar);
              for (int di = 0; di < density; ++di) {
                for (int dj = 0; dj < density; ++dj) {
                  float center_x_temp =
                      center_x - step_average / 2. + shift / 2. + dj * shift;
                  float center_y_temp =
                      center_y - step_average / 2. + shift / 2. + di * shift;
                  e_boxes(h, w, idx, 0) =
                      (center_x_temp - box_width_ratio / 2.) / img_width >= 0
                          ? (center_x_temp - box_width_ratio / 2.) / img_width
                          : 0;
                  e_boxes(h, w, idx, 1) =
                      (center_y_temp - box_height_ratio / 2.) / img_height >= 0
                          ? (center_y_temp - box_height_ratio / 2.) / img_height
                          : 0;
                  e_boxes(h, w, idx, 2) =
                      (center_x_temp + box_width_ratio / 2.) / img_width <= 1
                          ? (center_x_temp + box_width_ratio / 2.) / img_width
                          : 1;
                  e_boxes(h, w, idx, 3) =
                      (center_y_temp + box_height_ratio / 2.) / img_height <= 1
                          ? (center_y_temp + box_height_ratio / 2.) / img_height
                          : 1;
                  idx++;
                }
              }
            }
          } else {
            // if does not have fixed ratios, apply the aspect ratios to
            // generate density prior boxes instead.
            int shift = fixed_size / density;
            for (int di = 0; di < density; ++di) {
              for (int dj = 0; dj < density; ++dj) {
                float center_x_temp =
                    center_x - fixed_size / 2. + shift / 2. + dj * shift;
                float center_y_temp =
                    center_y - fixed_size / 2. + shift / 2. + di * shift;
                e_boxes(h, w, idx, 0) =
                    (center_x_temp - box_width / 2.) / img_width >= 0
                        ? (center_x_temp - box_width / 2.) / img_width
                        : 0;
                e_boxes(h, w, idx, 1) =
                    (center_y_temp - box_height / 2.) / img_height >= 0
                        ? (center_y_temp - box_height / 2.) / img_height
                        : 0;
                e_boxes(h, w, idx, 2) =
                    (center_x_temp + box_width / 2.) / img_width <= 1
                        ? (center_x_temp + box_width / 2.) / img_width
                        : 1;
                e_boxes(h, w, idx, 3) =
                    (center_y_temp + box_height / 2.) / img_height <= 1
                        ? (center_y_temp + box_height / 2.) / img_height
                        : 1;
                idx++;
              }
            }
            for (size_t r = 0; r < aspect_ratios.size(); ++r) {
              float ar = aspect_ratios[r];
              if (fabs(ar - 1.) < 1e-6) {
                continue;
              }
              int shift = fixed_size / density;
              float box_width_ratio = fixed_size * sqrt(ar);
              float box_height_ratio = fixed_size / sqrt(ar);
              for (int di = 0; di < density; ++di) {
                for (int dj = 0; dj < density; ++dj) {
                  float center_x_temp =
                      center_x - fixed_size / 2. + shift / 2. + dj * shift;
                  float center_y_temp =
                      center_y - fixed_size / 2. + shift / 2. + di * shift;
                  e_boxes(h, w, idx, 0) =
                      (center_x_temp - box_width_ratio / 2.) / img_width >= 0
                          ? (center_x_temp - box_width_ratio / 2.) / img_width
                          : 0;
                  e_boxes(h, w, idx, 1) =
                      (center_y_temp - box_height_ratio / 2.) / img_height >= 0
                          ? (center_y_temp - box_height_ratio / 2.) / img_height
                          : 0;
                  e_boxes(h, w, idx, 2) =
                      (center_x_temp + box_width_ratio / 2.) / img_width <= 1
                          ? (center_x_temp + box_width_ratio / 2.) / img_width
                          : 1;
                  e_boxes(h, w, idx, 3) =
                      (center_y_temp + box_height_ratio / 2.) / img_height <= 1
                          ? (center_y_temp + box_height_ratio / 2.) / img_height
                          : 1;
                  idx++;
                }
              }
            }
          }
        }
        // Same as prior box op, Generate prior boxes in the order of min size,
        // square boxes, aspect ratios.
        // if the min_max_aspect_ratios_order is false, the order changes to
        // aspect ratios, square boxes.
        for (size_t s = 0; s < min_sizes.size(); ++s) {
          auto min_size = min_sizes[s];
          if (min_max_aspect_ratios_order) {
            box_width = box_height = min_size / 2.;
            e_boxes(h, w, idx, 0) = (center_x - box_width) / img_width;
            e_boxes(h, w, idx, 1) = (center_y - box_height) / img_height;
            e_boxes(h, w, idx, 2) = (center_x + box_width) / img_width;
            e_boxes(h, w, idx, 3) = (center_y + box_height) / img_height;
            idx++;
            if (max_sizes.size() > 0) {
              auto max_size = max_sizes[s];
              box_width = box_height = sqrt(min_size * max_size) / 2.;
              e_boxes(h, w, idx, 0) = (center_x - box_width) / img_width;
              e_boxes(h, w, idx, 1) = (center_y - box_height) / img_height;
              e_boxes(h, w, idx, 2) = (center_x + box_width) / img_width;
              e_boxes(h, w, idx, 3) = (center_y + box_height) / img_height;
              idx++;
            }
            for (size_t r = 0; r < aspect_ratios.size(); ++r) {
              float ar = aspect_ratios[r];
              if (fabs(ar - 1.) < 1e-6) {
                continue;
              }
              box_width = min_size * sqrt(ar) / 2.;
              box_height = min_size / sqrt(ar) / 2.;
              e_boxes(h, w, idx, 0) = (center_x - box_width) / img_width;
              e_boxes(h, w, idx, 1) = (center_y - box_height) / img_height;
              e_boxes(h, w, idx, 2) = (center_x + box_width) / img_width;
              e_boxes(h, w, idx, 3) = (center_y + box_height) / img_height;
              idx++;
            }
          } else {
            for (size_t r = 0; r < aspect_ratios.size(); ++r) {
              float ar = aspect_ratios[r];
              box_width = min_size * sqrt(ar) / 2.;
              box_height = min_size / sqrt(ar) / 2.;
              e_boxes(h, w, idx, 0) = (center_x - box_width) / img_width;
              e_boxes(h, w, idx, 1) = (center_y - box_height) / img_height;
              e_boxes(h, w, idx, 2) = (center_x + box_width) / img_width;
              e_boxes(h, w, idx, 3) = (center_y + box_height) / img_height;
              idx++;
            }
            if (max_sizes.size() > 0) {
              auto max_size = max_sizes[s];
              box_width = box_height = sqrt(min_size * max_size) / 2.;
              e_boxes(h, w, idx, 0) = (center_x - box_width) / img_width;
              e_boxes(h, w, idx, 1) = (center_y - box_height) / img_height;
              e_boxes(h, w, idx, 2) = (center_x + box_width) / img_width;
              e_boxes(h, w, idx, 3) = (center_y + box_height) / img_height;
              idx++;
            }
          }
        }
      }
    }
    if (clip) {
      platform::Transform<platform::CPUDeviceContext> trans;
      ClipFunctor<T> clip_func;
      trans(ctx.template device_context<platform::CPUDeviceContext>(),
            boxes->data<T>(), boxes->data<T>() + boxes->numel(),
            boxes->data<T>(), clip_func);
    }
    framework::Tensor var_t;
    var_t.mutable_data<T>(
        framework::make_ddim({1, static_cast<int>(variances.size())}),
        ctx.GetPlace());

    auto var_et = framework::EigenTensor<T, 2>::From(var_t);

    for (size_t i = 0; i < variances.size(); ++i) {
      var_et(0, i) = variances[i];
    }

    int box_num = feature_height * feature_width * num_priors;
    auto var_dim = vars->dims();
    vars->Resize({box_num, static_cast<int>(variances.size())});

    auto e_vars = framework::EigenMatrix<T, Eigen::RowMajor>::From(*vars);

    e_vars = var_et.broadcast(Eigen::DSizes<int, 2>(box_num, 1));

    vars->Resize(var_dim);
  }
};  // namespace operators

}  // namespace operators
}  // namespace paddle
