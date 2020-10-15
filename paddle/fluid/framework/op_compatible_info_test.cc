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

#include "gtest/gtest.h"
#include "paddle/fluid/framework/program_desc.h"

namespace paddle {
namespace framework {

namespace proto {
class OpCompatibleMap;
}  // namespace proto

TEST(test_op_compatible_info, test_op_compatible) {
  auto comp_map = OpCompatibleMap();
  comp_map.InitOpCompatibleMap();

  ASSERT_NE(comp_map.GetDefaultRequiredVersion(), std::string());
  ASSERT_NE(comp_map.GetOpCompatibleInfo("sequence_pad").required_version_,
            std::string());
  ASSERT_NE(comp_map.GetOpCompatibleInfo("reshape").required_version_,
            std::string());
  ASSERT_NE(comp_map.GetOpCompatibleInfo("layer_norm").required_version_,
            std::string());
  ASSERT_NE(comp_map.GetOpCompatibleInfo("layer_xx").required_version_,
            std::string());

  auto comp_1 = comp_map.IsRequireMiniVersion("sequence_pad", "1.5.0");
  ASSERT_EQ(comp_1, OpCompatibleType::DEFIN_NOT);
  auto comp_2 = comp_map.IsRequireMiniVersion("sequence_pad", "1.6.0");
  ASSERT_EQ(comp_2, OpCompatibleType::compatible);
  auto comp_3 = comp_map.IsRequireMiniVersion("sequence_pad", "1.6.1");
  ASSERT_EQ(comp_3, OpCompatibleType::compatible);
  auto comp_6 = comp_map.IsRequireMiniVersion("sequence_pad", "1.7.0");
  ASSERT_EQ(comp_6, OpCompatibleType::compatible);
  auto comp_7 = comp_map.IsRequireMiniVersion("sequence_pad", "0.7.0");
  ASSERT_EQ(comp_7, OpCompatibleType::DEFIN_NOT);
  auto comp_8 = comp_map.IsRequireMiniVersion("sequence_pad", "2.0.0");
  ASSERT_EQ(comp_8, OpCompatibleType::compatible);

  ASSERT_EQ(comp_map.IsRequireMiniVersion("unkop", "2.0.0"),
            OpCompatibleType::compatible);
  ASSERT_EQ(comp_map.IsRequireMiniVersion("unkop", "0.7.0"),
            OpCompatibleType::DEFIN_NOT);

  ASSERT_EQ(comp_map.IsRequireMiniVersion("slice", "0.7.0"),
            OpCompatibleType::possible);
  ASSERT_EQ(comp_map.IsRequireMiniVersion("slice", "1.6.0"),
            OpCompatibleType::compatible);
}

}  // namespace framework
}  // namespace paddle
