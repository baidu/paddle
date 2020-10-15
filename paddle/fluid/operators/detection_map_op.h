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
#include <map>
#include <string>
#include <utility>
#include <vector>
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/op_registry.h"

namespace paddle {
namespace operators {

enum APType { kNone = 0, kIntegral, k11point };

APType GetAPType(std::string str) {
  if (str == "integral") {
    return APType::kIntegral;
  } else if (str == "11point") {
    return APType::k11point;
  } else {
    return APType::kNone;
  }
}

template <typename T>
inline bool SortScorePairDescend(const std::pair<float, T>& pair1,
                                 const std::pair<float, T>& pair2) {
  return pair1.first > pair2.first;
}

template <typename T>
inline void GetAccumulation(std::vector<std::pair<T, int>> in_pairs,
                            std::vector<int>* accu_vec) {
  std::stable_sort(in_pairs.begin(), in_pairs.end(), SortScorePairDescend<int>);
  accu_vec->clear();
  size_t sum = 0;
  for (size_t i = 0; i < in_pairs.size(); ++i) {
    auto count = in_pairs[i].second;
    sum += count;
    accu_vec->push_back(sum);
  }
}

template <typename Place, typename T>
class DetectionMAPOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* in_detect = ctx.Input<framework::LoDTensor>("DetectRes");
    auto* in_label = ctx.Input<framework::LoDTensor>("Label");
    auto* out_map = ctx.Output<framework::Tensor>("MAP");

    auto* in_pos_count = ctx.Input<framework::Tensor>("PosCount");
    auto* in_true_pos = ctx.Input<framework::LoDTensor>("TruePos");
    auto* in_false_pos = ctx.Input<framework::LoDTensor>("FalsePos");

    auto* out_pos_count = ctx.Output<framework::Tensor>("AccumPosCount");
    auto* out_true_pos = ctx.Output<framework::LoDTensor>("AccumTruePos");
    auto* out_false_pos = ctx.Output<framework::LoDTensor>("AccumFalsePos");

    float overlap_threshold = ctx.Attr<float>("overlap_threshold");
    bool evaluate_difficult = ctx.Attr<bool>("evaluate_difficult");
    auto ap_type = GetAPType(ctx.Attr<std::string>("ap_type"));
    int class_num = ctx.Attr<int>("class_num");

    auto& label_lod = in_label->lod();
    auto& detect_lod = in_detect->lod();
    PADDLE_ENFORCE_EQ(
        label_lod.size(), 1UL,
        platform::errors::InvalidArgument("Only support LodTensor of lod_level "
                                          "with 1 in label, but received %d.",
                                          label_lod.size()));
    PADDLE_ENFORCE_EQ(label_lod[0].size(), detect_lod[0].size(),
                      platform::errors::InvalidArgument(
                          "The batch_size of input(Label) and input(Detection) "
                          "must be the same, but received %d:%d",
                          label_lod[0].size(), detect_lod[0].size()));

    std::vector<std::map<int, std::vector<Box>>> gt_boxes;
    std::vector<std::map<int, std::vector<std::pair<T, Box>>>> detect_boxes;

    GetBoxes(*in_label, *in_detect, &gt_boxes, detect_boxes);

    std::map<int, int> label_pos_count;
    std::map<int, std::vector<std::pair<T, int>>> true_pos;
    std::map<int, std::vector<std::pair<T, int>>> false_pos;

    auto* has_state = ctx.Input<framework::LoDTensor>("HasState");
    int state = 0;
    if (has_state) {
      state = has_state->data<int>()[0];
    }

    if (in_pos_count != nullptr && state) {
      GetInputPos(*in_pos_count, *in_true_pos, *in_false_pos, &label_pos_count,
                  &true_pos, &false_pos, class_num);
    }

    CalcTrueAndFalsePositive(gt_boxes, detect_boxes, evaluate_difficult,
                             overlap_threshold, &label_pos_count, &true_pos,
                             &false_pos);

    int background_label = ctx.Attr<int>("background_label");
    T map = CalcMAP(ap_type, label_pos_count, true_pos, false_pos,
                    background_label);

    GetOutputPos(ctx, label_pos_count, true_pos, false_pos, out_pos_count,
                 out_true_pos, out_false_pos, class_num);

    T* map_data = out_map->mutable_data<T>(ctx.GetPlace());
    map_data[0] = map;
  }

 protected:
  struct Box {
    Box(T xmin, T ymin, T xmax, T ymax)
        : xmin(xmin), ymin(ymin), xmax(xmax), ymax(ymax), is_difficult(false) {}

    T xmin, ymin, xmax, ymax;
    bool is_difficult;
  };

  inline T JaccardOverlap(const Box& box1, const Box& box2) const {
    if (box2.xmin > box1.xmax || box2.xmax < box1.xmin ||
        box2.ymin > box1.ymax || box2.ymax < box1.ymin) {
      return 0.0;
    } else {
      T inter_xmin = std::max(box1.xmin, box2.xmin);
      T inter_ymin = std::max(box1.ymin, box2.ymin);
      T inter_xmax = std::min(box1.xmax, box2.xmax);
      T inter_ymax = std::min(box1.ymax, box2.ymax);

      T inter_width = inter_xmax - inter_xmin;
      T inter_height = inter_ymax - inter_ymin;
      T inter_area = inter_width * inter_height;

      T bbox_area1 = (box1.xmax - box1.xmin) * (box1.ymax - box1.ymin);
      T bbox_area2 = (box2.xmax - box2.xmin) * (box2.ymax - box2.ymin);

      return inter_area / (bbox_area1 + bbox_area2 - inter_area);
    }
  }

  inline void ClipBBox(const Box& bbox, Box* clipped_bbox) const {
    T one = static_cast<T>(1.0);
    T zero = static_cast<T>(0.0);
    clipped_bbox->xmin = std::max(std::min(bbox.xmin, one), zero);
    clipped_bbox->ymin = std::max(std::min(bbox.ymin, one), zero);
    clipped_bbox->xmax = std::max(std::min(bbox.xmax, one), zero);
    clipped_bbox->ymax = std::max(std::min(bbox.ymax, one), zero);
  }

  void GetBoxes(const framework::LoDTensor& input_label,
                const framework::LoDTensor& input_detect,
                std::vector<std::map<int, std::vector<Box>>>* gt_boxes,
                std::vector<std::map<int, std::vector<std::pair<T, Box>>>>&
                    detect_boxes) const {
    auto labels = framework::EigenTensor<T, 2>::From(input_label);
    auto detect = framework::EigenTensor<T, 2>::From(input_detect);

    auto& label_lod = input_label.lod();
    auto& detect_lod = input_detect.lod();

    int batch_size = label_lod[0].size() - 1;
    auto& label_index = label_lod[0];

    for (int n = 0; n < batch_size; ++n) {
      std::map<int, std::vector<Box>> boxes;
      for (size_t i = label_index[n]; i < label_index[n + 1]; ++i) {
        int label = labels(i, 0);
        if (input_label.dims()[1] == 6) {
          Box box(labels(i, 2), labels(i, 3), labels(i, 4), labels(i, 5));
          auto is_difficult = labels(i, 1);
          if (std::abs(is_difficult - 0.0) < 1e-6)
            box.is_difficult = false;
          else
            box.is_difficult = true;
          boxes[label].push_back(box);
        } else {
          PADDLE_ENFORCE_EQ(
              input_label.dims()[1], 5,
              platform::errors::InvalidArgument(
                  "The input label width"
                  " must be 5, but received %d, please check your input data",
                  input_label.dims()[1]));
          Box box(labels(i, 1), labels(i, 2), labels(i, 3), labels(i, 4));
          boxes[label].push_back(box);
        }
      }
      gt_boxes->push_back(boxes);
    }

    auto detect_index = detect_lod[0];
    for (int n = 0; n < batch_size; ++n) {
      std::map<int, std::vector<std::pair<T, Box>>> boxes;
      for (size_t i = detect_index[n]; i < detect_index[n + 1]; ++i) {
        Box box(detect(i, 2), detect(i, 3), detect(i, 4), detect(i, 5));
        int label = detect(i, 0);
        auto score = detect(i, 1);
        boxes[label].push_back(std::make_pair(score, box));
      }
      detect_boxes.push_back(boxes);
    }
  }

  void GetOutputPos(
      const framework::ExecutionContext& ctx,
      const std::map<int, int>& label_pos_count,
      const std::map<int, std::vector<std::pair<T, int>>>& true_pos,
      const std::map<int, std::vector<std::pair<T, int>>>& false_pos,
      framework::Tensor* output_pos_count,
      framework::LoDTensor* output_true_pos,
      framework::LoDTensor* output_false_pos, const int class_num) const {
    int true_pos_count = 0;
    int false_pos_count = 0;
    for (auto it = true_pos.begin(); it != true_pos.end(); ++it) {
      auto tp = it->second;
      true_pos_count += tp.size();
    }
    for (auto it = false_pos.begin(); it != false_pos.end(); ++it) {
      auto fp = it->second;
      false_pos_count += fp.size();
    }

    int* pos_count_data = output_pos_count->mutable_data<int>(
        framework::make_ddim({class_num, 1}), ctx.GetPlace());

    T* true_pos_data = output_true_pos->mutable_data<T>(
        framework::make_ddim({true_pos_count, 2}), ctx.GetPlace());
    T* false_pos_data = output_false_pos->mutable_data<T>(
        framework::make_ddim({false_pos_count, 2}), ctx.GetPlace());
    true_pos_count = 0;
    false_pos_count = 0;
    std::vector<size_t> true_pos_starts = {0};
    std::vector<size_t> false_pos_starts = {0};
    for (int i = 0; i < class_num; ++i) {
      auto it_count = label_pos_count.find(i);
      pos_count_data[i] = 0;
      if (it_count != label_pos_count.end()) {
        pos_count_data[i] = it_count->second;
      }
      auto it_true_pos = true_pos.find(i);
      if (it_true_pos != true_pos.end()) {
        const std::vector<std::pair<T, int>>& true_pos_vec =
            it_true_pos->second;
        for (const std::pair<T, int>& tp : true_pos_vec) {
          true_pos_data[true_pos_count * 2] = tp.first;
          true_pos_data[true_pos_count * 2 + 1] = static_cast<T>(tp.second);
          true_pos_count++;
        }
      }
      true_pos_starts.push_back(true_pos_count);

      auto it_false_pos = false_pos.find(i);
      if (it_false_pos != false_pos.end()) {
        const std::vector<std::pair<T, int>>& false_pos_vec =
            it_false_pos->second;
        for (const std::pair<T, int>& fp : false_pos_vec) {
          false_pos_data[false_pos_count * 2] = fp.first;
          false_pos_data[false_pos_count * 2 + 1] = static_cast<T>(fp.second);
          false_pos_count++;
        }
      }
      false_pos_starts.push_back(false_pos_count);
    }

    framework::LoD true_pos_lod;
    true_pos_lod.emplace_back(true_pos_starts);
    framework::LoD false_pos_lod;
    false_pos_lod.emplace_back(false_pos_starts);

    output_true_pos->set_lod(true_pos_lod);
    output_false_pos->set_lod(false_pos_lod);
  }

  void GetInputPos(const framework::Tensor& input_pos_count,
                   const framework::LoDTensor& input_true_pos,
                   const framework::LoDTensor& input_false_pos,
                   std::map<int, int>* label_pos_count,
                   std::map<int, std::vector<std::pair<T, int>>>* true_pos,
                   std::map<int, std::vector<std::pair<T, int>>>* false_pos,
                   const int class_num) const {
    const int* pos_count_data = input_pos_count.data<int>();
    for (int i = 0; i < class_num; ++i) {
      (*label_pos_count)[i] = pos_count_data[i];
    }

    auto SetData = [](const framework::LoDTensor& pos_tensor,
                      std::map<int, std::vector<std::pair<T, int>>>& pos) {
      const T* pos_data = pos_tensor.data<T>();
      auto& pos_data_lod = pos_tensor.lod()[0];
      for (size_t i = 0; i < pos_data_lod.size() - 1; ++i) {
        for (size_t j = pos_data_lod[i]; j < pos_data_lod[i + 1]; ++j) {
          T score = pos_data[j * 2];
          int flag = pos_data[j * 2 + 1];
          pos[i].push_back(std::make_pair(score, flag));
        }
      }
    };

    SetData(input_true_pos, *true_pos);
    SetData(input_false_pos, *false_pos);
    return;
  }

  void CalcTrueAndFalsePositive(
      const std::vector<std::map<int, std::vector<Box>>>& gt_boxes,
      const std::vector<std::map<int, std::vector<std::pair<T, Box>>>>&
          detect_boxes,
      bool evaluate_difficult, float overlap_threshold,
      std::map<int, int>* label_pos_count,
      std::map<int, std::vector<std::pair<T, int>>>* true_pos,
      std::map<int, std::vector<std::pair<T, int>>>* false_pos) const {
    int batch_size = gt_boxes.size();
    for (int n = 0; n < batch_size; ++n) {
      auto& image_gt_boxes = gt_boxes[n];
      for (auto& image_gt_box : image_gt_boxes) {
        size_t count = 0;
        auto& labeled_bboxes = image_gt_box.second;
        if (evaluate_difficult) {
          count = labeled_bboxes.size();
        } else {
          for (auto& box : labeled_bboxes) {
            if (!box.is_difficult) {
              ++count;
            }
          }
        }
        if (count == 0) {
          continue;
        }
        int label = image_gt_box.first;
        if (label_pos_count->find(label) == label_pos_count->end()) {
          (*label_pos_count)[label] = count;
        } else {
          (*label_pos_count)[label] += count;
        }
      }
    }

    for (size_t n = 0; n < detect_boxes.size(); ++n) {
      auto image_gt_boxes = gt_boxes[n];
      auto detections = detect_boxes[n];

      if (image_gt_boxes.size() == 0) {
        for (auto it = detections.begin(); it != detections.end(); ++it) {
          auto pred_boxes = it->second;
          int label = it->first;
          for (size_t i = 0; i < pred_boxes.size(); ++i) {
            auto score = pred_boxes[i].first;
            (*true_pos)[label].push_back(std::make_pair(score, 0));
            (*false_pos)[label].push_back(std::make_pair(score, 1));
          }
        }
        continue;
      }

      for (auto it = detections.begin(); it != detections.end(); ++it) {
        int label = it->first;
        auto pred_boxes = it->second;
        if (image_gt_boxes.find(label) == image_gt_boxes.end()) {
          for (size_t i = 0; i < pred_boxes.size(); ++i) {
            auto score = pred_boxes[i].first;
            (*true_pos)[label].push_back(std::make_pair(score, 0));
            (*false_pos)[label].push_back(std::make_pair(score, 1));
          }
          continue;
        }

        auto matched_bboxes = image_gt_boxes.find(label)->second;
        std::vector<bool> visited(matched_bboxes.size(), false);
        // Sort detections in descend order based on scores
        std::sort(pred_boxes.begin(), pred_boxes.end(),
                  SortScorePairDescend<Box>);
        for (size_t i = 0; i < pred_boxes.size(); ++i) {
          T max_overlap = -1.0;
          size_t max_idx = 0;
          auto score = pred_boxes[i].first;
          for (size_t j = 0; j < matched_bboxes.size(); ++j) {
            Box& pred_box = pred_boxes[i].second;
            ClipBBox(pred_box, &pred_box);
            T overlap = JaccardOverlap(pred_box, matched_bboxes[j]);
            if (overlap > max_overlap) {
              max_overlap = overlap;
              max_idx = j;
            }
          }
          if (max_overlap > overlap_threshold) {
            bool match_evaluate_difficult =
                evaluate_difficult ||
                (!evaluate_difficult && !matched_bboxes[max_idx].is_difficult);
            if (match_evaluate_difficult) {
              if (!visited[max_idx]) {
                (*true_pos)[label].push_back(std::make_pair(score, 1));
                (*false_pos)[label].push_back(std::make_pair(score, 0));
                visited[max_idx] = true;
              } else {
                (*true_pos)[label].push_back(std::make_pair(score, 0));
                (*false_pos)[label].push_back(std::make_pair(score, 1));
              }
            }
          } else {
            (*true_pos)[label].push_back(std::make_pair(score, 0));
            (*false_pos)[label].push_back(std::make_pair(score, 1));
          }
        }
      }
    }
  }

  T CalcMAP(APType ap_type, const std::map<int, int>& label_pos_count,
            const std::map<int, std::vector<std::pair<T, int>>>& true_pos,
            const std::map<int, std::vector<std::pair<T, int>>>& false_pos,
            const int background_label) const {
    T mAP = 0.0;
    int count = 0;
    for (auto it = label_pos_count.begin(); it != label_pos_count.end(); ++it) {
      int label = it->first;
      int label_num_pos = it->second;
      if (label_num_pos == background_label) {
        continue;
      }
      if (true_pos.find(label) == true_pos.end()) {
        count++;
        continue;
      }
      auto label_true_pos = true_pos.find(label)->second;
      auto label_false_pos = false_pos.find(label)->second;
      // Compute average precision.
      std::vector<int> tp_sum;
      GetAccumulation<T>(label_true_pos, &tp_sum);
      std::vector<int> fp_sum;
      GetAccumulation<T>(label_false_pos, &fp_sum);
      std::vector<T> precision, recall;
      size_t num = tp_sum.size();
      // Compute Precision.
      for (size_t i = 0; i < num; ++i) {
        precision.push_back(static_cast<T>(tp_sum[i]) /
                            static_cast<T>(tp_sum[i] + fp_sum[i]));
        recall.push_back(static_cast<T>(tp_sum[i]) / label_num_pos);
      }
      // VOC2007 style
      if (ap_type == APType::k11point) {
        std::vector<T> max_precisions(11, 0.0);
        int start_idx = num - 1;
        for (int j = 10; j >= 0; --j)
          for (int i = start_idx; i >= 0; --i) {
            if (recall[i] < j / 10.) {
              start_idx = i;
              if (j > 0) max_precisions[j - 1] = max_precisions[j];
              break;
            } else {
              if (max_precisions[j] < precision[i])
                max_precisions[j] = precision[i];
            }
          }
        for (int j = 10; j >= 0; --j) mAP += max_precisions[j] / 11;
        ++count;
      } else if (ap_type == APType::kIntegral) {
        // Nature integral
        float average_precisions = 0.;
        float prev_recall = 0.;
        for (size_t i = 0; i < num; ++i) {
          if (fabs(recall[i] - prev_recall) > 1e-6)
            average_precisions += precision[i] * fabs(recall[i] - prev_recall);
          prev_recall = recall[i];
        }
        mAP += average_precisions;
        ++count;
      } else {
        PADDLE_THROW(platform::errors::Unimplemented(
            "Unkown ap version %s. Now only supports integral and l1point.",
            ap_type));
      }
    }
    if (count != 0) mAP /= count;
    return mAP;
  }
};  // namespace operators

}  // namespace operators
}  // namespace paddle
