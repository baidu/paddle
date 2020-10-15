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

#include "paddle/fluid/framework/op_compatible_info.h"
#include <iostream>
#include <utility>
#include <vector>
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/string/string_helper.h"

namespace paddle {
namespace framework {

inline std::vector<int> ConvertStr2Int(const std::string& str_text) {
  auto vec_text = string::split_string<std::string>(str_text, ".");
  PADDLE_ENFORCE(
      (vec_text.size() == 2 || vec_text.size() == 3),
      platform::errors::InvalidArgument(
          "Input[%s] is not a right version format [1.6 or 1.6.0].", str_text));

  std::vector<int> vec_res;
  vec_res.reserve(3);
  for (auto& val : vec_text) {
    vec_res.emplace_back(atoi(val.c_str()));
  }

  if (vec_res.size() == 2) {
    vec_res.emplace_back(0);
  }

  return vec_res;
}

/* first version >= second version return true */

inline bool CompareVersion(const std::string& str_first,
                           const std::string& str_second) {
  auto vec_first_version = ConvertStr2Int(str_first);
  auto vec_second_version = ConvertStr2Int(str_second);

  // first version id
  PADDLE_ENFORCE_EQ(vec_first_version.size(), vec_second_version.size(),
                    platform::errors::InvalidArgument(
                        "Version information size is not equal, the first is "
                        "[%d], the second is [%d].",
                        vec_first_version.size(), vec_second_version.size()));

  for (size_t i = 0; i < vec_first_version.size() - 1; ++i) {
    if (vec_first_version[i] != vec_second_version[i]) {
      return vec_first_version[i] > vec_second_version[i];
    }
  }
  return vec_first_version[2] >= vec_second_version[2];
}

void OpCompatibleMap::InitOpCompatibleMap() {
  op_compatible_map_["sequence_pad"] = {"1.6.0", OpCompatibleType::DEFIN_NOT};
  op_compatible_map_["sequence_unpad"] = {"1.6.0", OpCompatibleType::DEFIN_NOT};

  op_compatible_map_["center_loss"] = {"1.6.0", OpCompatibleType::DEFIN_NOT};
  op_compatible_map_["coalesce_tensor"] = {"1.6.0",
                                           OpCompatibleType::DEFIN_NOT};
  op_compatible_map_["crop_tensor"] = {"1.6.0", OpCompatibleType::DEFIN_NOT};
  op_compatible_map_["deformable_conv"] = {"1.6.0",
                                           OpCompatibleType::DEFIN_NOT};
  op_compatible_map_["deformable_conv_v1"] = {"1.6.0",
                                              OpCompatibleType::DEFIN_NOT};
  op_compatible_map_["dpsgd"] = {"1.6.0", OpCompatibleType::DEFIN_NOT};
  op_compatible_map_["eye"] = {"1.6.0", OpCompatibleType::DEFIN_NOT};
  op_compatible_map_["fill_any_like"] = {"1.6.0", OpCompatibleType::DEFIN_NOT};
  op_compatible_map_["filter_by_instag"] = {"1.6.0",
                                            OpCompatibleType::DEFIN_NOT};
  op_compatible_map_["hard_swish"] = {"1.6.0", OpCompatibleType::DEFIN_NOT};
  op_compatible_map_["gather_nd"] = {"1.6.0", OpCompatibleType::DEFIN_NOT};
  op_compatible_map_["instance_norm"] = {"1.6.0", OpCompatibleType::DEFIN_NOT};
  op_compatible_map_["lookup_table_v2"] = {"1.6.0",
                                           OpCompatibleType::DEFIN_NOT};
  op_compatible_map_["match_matrix_tensor"] = {"1.6.0",
                                               OpCompatibleType::DEFIN_NOT};
  op_compatible_map_["multiclass_nms2"] = {"1.6.0",
                                           OpCompatibleType::DEFIN_NOT};
  op_compatible_map_["one_hot_v2"] = {"1.6.0", OpCompatibleType::DEFIN_NOT};
  op_compatible_map_["prroi_pool"] = {"1.6.0", OpCompatibleType::DEFIN_NOT};
  op_compatible_map_["pull_box_sparse"] = {"1.6.0",
                                           OpCompatibleType::DEFIN_NOT};
  op_compatible_map_["scatter_nd_add"] = {"1.6.0", OpCompatibleType::DEFIN_NOT};
  op_compatible_map_["sequence_topk_avg_pooling"] = {
      "1.6.0", OpCompatibleType::DEFIN_NOT};
  op_compatible_map_["shard_index"] = {"1.6.0", OpCompatibleType::DEFIN_NOT};
  op_compatible_map_["size"] = {"1.6.0", OpCompatibleType::DEFIN_NOT};
  op_compatible_map_["strided_slice"] = {"1.6.0", OpCompatibleType::DEFIN_NOT};
  op_compatible_map_["trilinear_interp"] = {"1.6.0",
                                            OpCompatibleType::DEFIN_NOT};
  op_compatible_map_["unfold"] = {"1.6.0", OpCompatibleType::DEFIN_NOT};
  op_compatible_map_["unique"] = {"1.6.0", OpCompatibleType::DEFIN_NOT};
  op_compatible_map_["unique_with_counts"] = {"1.6.0",
                                              OpCompatibleType::DEFIN_NOT};
  op_compatible_map_["var_conv_2d"] = {"1.6.0", OpCompatibleType::DEFIN_NOT};

  op_compatible_map_["reshape2"] = {"1.6.0", OpCompatibleType::possible};
  op_compatible_map_["slice"] = {"1.6.0", OpCompatibleType::possible};
  op_compatible_map_["expand"] = {"1.6.0", OpCompatibleType::possible};
  op_compatible_map_["bilinear_interp"] = {"1.6.0", OpCompatibleType::possible};
  op_compatible_map_["chunk_eval"] = {"1.6.0", OpCompatibleType::possible};
  op_compatible_map_["conditional_block"] = {"1.6.0",
                                             OpCompatibleType::possible};
  op_compatible_map_["conditional_block_infer"] = {"1.6.0",
                                                   OpCompatibleType::possible};
  op_compatible_map_["conv2d"] = {"1.6.0", OpCompatibleType::possible};
  op_compatible_map_["conv2d_transpose"] = {"1.6.0",
                                            OpCompatibleType::possible};
  op_compatible_map_["conv3d"] = {"1.6.0", OpCompatibleType::possible};
  op_compatible_map_["conv3d_transpose"] = {"1.6.0",
                                            OpCompatibleType::possible};
  op_compatible_map_["crf_decoding"] = {"1.6.0", OpCompatibleType::possible};
  op_compatible_map_["ctc_align"] = {"1.6.0", OpCompatibleType::possible};
  op_compatible_map_["data_norm"] = {"1.6.0", OpCompatibleType::possible};
  op_compatible_map_["depthwise_conv2d"] = {"1.6.0",
                                            OpCompatibleType::possible};
  op_compatible_map_["depthwise_conv2d_transpose"] = {
      "1.6.0", OpCompatibleType::possible};
  op_compatible_map_["edit_distance"] = {"1.6.0", OpCompatibleType::possible};
  op_compatible_map_["fc"] = {"1.6.0", OpCompatibleType::possible};
  op_compatible_map_["fused_embedding_seq_pool"] = {"1.6.0",
                                                    OpCompatibleType::possible};
  op_compatible_map_["group_norm"] = {"1.6.0", OpCompatibleType::possible};
  op_compatible_map_["hash"] = {"1.6.0", OpCompatibleType::possible};
  op_compatible_map_["leaky_relu"] = {"1.6.0", OpCompatibleType::possible};
  op_compatible_map_["linear_chain_crf"] = {"1.6.0",
                                            OpCompatibleType::possible};
  op_compatible_map_["lod_reset"] = {"1.6.0", OpCompatibleType::possible};
  op_compatible_map_["matmul"] = {"1.6.0", OpCompatibleType::possible};
  op_compatible_map_["mul"] = {"1.6.0", OpCompatibleType::possible};
  op_compatible_map_["nearest_interp"] = {"1.6.0", OpCompatibleType::possible};
  op_compatible_map_["one_hot"] = {"1.6.0", OpCompatibleType::possible};
  op_compatible_map_["pow"] = {"1.6.0", OpCompatibleType::possible};
  op_compatible_map_["prior_box"] = {"1.6.0", OpCompatibleType::possible};
  op_compatible_map_["uniform_random"] = {"1.6.0", OpCompatibleType::possible};
  op_compatible_map_["uniform_random_batch_size_like"] = {
      "1.6.0", OpCompatibleType::possible};
  op_compatible_map_["warpctc"] = {"1.6.0", OpCompatibleType::possible};

  op_compatible_map_["layer_norm"] = {"1.6.0", OpCompatibleType::bug_fix};
}

CompatibleInfo OpCompatibleMap::GetOpCompatibleInfo(std::string op_name) const {
  auto it = op_compatible_map_.find(op_name);
  if (it != op_compatible_map_.end()) {
    return it->second;
  } else {
    return {default_required_version_, OpCompatibleType::DEFIN_NOT};
  }
}

OpCompatibleType OpCompatibleMap::IsRequireMiniVersion(
    std::string op_name, std::string str_current_version) const {
  auto it = op_compatible_map_.find(op_name);
  if (it != op_compatible_map_.end()) {
    if (CompareVersion(str_current_version, it->second.required_version_)) {
      return OpCompatibleType::compatible;
    } else {
      return it->second.compatible_type_;
    }

  } else {
    if (CompareVersion(str_current_version, default_required_version_)) {
      return OpCompatibleType::compatible;
    } else {
      return OpCompatibleType::DEFIN_NOT;
    }
  }
}

}  // namespace framework
}  // namespace paddle
