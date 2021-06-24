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

#include <iostream>
#include <string>
#include <vector>
#include "gtest/gtest.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/platform/bfloat16.h"
#include "paddle/fluid/platform/float16.h"

USE_CPU_ONLY_OP(save_combine);
USE_CPU_ONLY_OP(load_combine);

template <typename T, typename U>
T* CreateForSaveCombineOp(int x, int y, const std::vector<int>& lod_info,
                          std::string var_name,
                          const paddle::platform::CPUPlace& place,
                          paddle::framework::Scope* scope,
                          paddle::framework::LoD* expect_lod) {
  auto var = scope->Var(var_name);
  auto tensor = var->GetMutable<paddle::framework::LoDTensor>();
  tensor->Resize({x, y});
  expect_lod->resize(1);
  for (size_t i = 0; i < lod_info.size(); i++) {
    (*expect_lod)[0].push_back(lod_info[i]);
  }
  tensor->set_lod(*expect_lod);
  T* expect = tensor->mutable_data<T>(place);
  for (int64_t i = 0; i < tensor->numel(); ++i) {
    expect[i] = static_cast<T>(
        static_cast<U>(i));  // For FP16, we intend to do float(float16(i))
  }
  return expect;
}

paddle::framework::LoDTensor* GeneratePlaceholderBeforeLoad(
    const std::string out_var_name, paddle::framework::Scope* scope) {
  auto load_var = scope->Var(out_var_name);
  auto target = load_var->GetMutable<paddle::framework::LoDTensor>();
  return target;
}

template <typename T>
T* GetValuesAfterLoadCombineOp(paddle::framework::LoDTensor* target,
                               const paddle::framework::Scope& scope,
                               paddle::framework::LoD* actual_lod) {
  T* actual = target->data<T>();
  *actual_lod = target->lod();
  return actual;
}

template <typename T, typename U>
void CheckValues(T* expect, U* actual, const paddle::framework::LoD& expect_lod,
                 const paddle::framework::LoD& actual_lod, const int& numel) {
  for (int i = 0; i < numel; ++i) {
    EXPECT_EQ(expect[i], static_cast<T>(actual[i]));
  }
  EXPECT_EQ(expect_lod.size(), actual_lod.size());
  for (size_t i = 0; i < expect_lod.size(); ++i) {
    for (size_t j = 0; j < expect_lod[i].size(); ++j) {
      EXPECT_EQ(expect_lod[i][j], actual_lod[i][j]);
    }
  }
}

// Here, we create 4 LoDTensors and use save_combine_op to first save these
// in a single file. Then, we use load_combine_op to load these sequentially
template <typename T, typename U>
void SaveLoadCombineOp() {
  paddle::framework::Scope scope;
  paddle::platform::CPUPlace place;

  std::vector<int> lod1 = {0, 1, 2, 3, 10};
  int numel1 = 100;
  paddle::framework::LoD expect_lod1;
  T* expect1 = CreateForSaveCombineOp<T, U>(10, 10, lod1, "test_var1", place,
                                            &scope, &expect_lod1);

  std::vector<int> lod2 = {0, 2, 5, 10};
  int numel2 = 200;
  paddle::framework::LoD expect_lod2;
  T* expect2 = CreateForSaveCombineOp<T, U>(10, 20, lod2, "test_var2", place,
                                            &scope, &expect_lod2);

  std::vector<int> lod3 = {0, 2, 3, 20};
  int numel3 = 4000;
  paddle::framework::LoD expect_lod3;
  T* expect3 = CreateForSaveCombineOp<T, U>(20, 200, lod3, "test_var3", place,
                                            &scope, &expect_lod3);

  std::vector<int> lod4 = {0, 1, 20};
  int numel4 = 1000;
  paddle::framework::LoD expect_lod4;
  T* expect4 = CreateForSaveCombineOp<T, U>(20, 50, lod4, "test_var4", place,
                                            &scope, &expect_lod4);

  // Set attributes
  std::string filename = "check_tensor.ls";
  paddle::framework::AttributeMap attrs;
  attrs.insert({"file_path", std::string(filename)});

  // Run the save_combine_op
  auto save_combine_op = paddle::framework::OpRegistry::CreateOp(
      "save_combine",
      {{"X", {"test_var1", "test_var2", "test_var3", "test_var4"}}}, {}, attrs);
  save_combine_op->Run(scope, place);

  // Set up output vars
  auto target1 = GeneratePlaceholderBeforeLoad("out_var1", &scope);
  auto target2 = GeneratePlaceholderBeforeLoad("out_var2", &scope);
  auto target3 = GeneratePlaceholderBeforeLoad("out_var3", &scope);
  auto target4 = GeneratePlaceholderBeforeLoad("out_var4", &scope);

  // Run the load_combine_op
  auto load_combine_op = paddle::framework::OpRegistry::CreateOp(
      "load_combine", {},
      {{"Out", {"out_var1", "out_var2", "out_var3", "out_var4"}}}, attrs);
  load_combine_op->Run(scope, place);

  paddle::framework::LoD actual_lod1, actual_lod2, actual_lod3, actual_lod4;
  U* actual1 = GetValuesAfterLoadCombineOp<U>(target1, scope, &actual_lod1);
  U* actual2 = GetValuesAfterLoadCombineOp<U>(target2, scope, &actual_lod2);
  U* actual3 = GetValuesAfterLoadCombineOp<U>(target3, scope, &actual_lod3);
  U* actual4 = GetValuesAfterLoadCombineOp<U>(target4, scope, &actual_lod4);

  CheckValues<T, U>(expect1, actual1, expect_lod1, actual_lod1, numel1);
  CheckValues<T, U>(expect2, actual2, expect_lod2, actual_lod2, numel2);
  CheckValues<T, U>(expect3, actual3, expect_lod3, actual_lod3, numel3);
  CheckValues<T, U>(expect4, actual4, expect_lod4, actual_lod4, numel4);
}

TEST(SaveLoadCombineOp, CPU) { SaveLoadCombineOp<int, int>(); }

TEST(SaveLoadCombineBF16Op, CPU) {
  SaveLoadCombineOp<paddle::platform::bfloat16, paddle::platform::bfloat16>();
}

// FP16 version of SaveLoadCombineOp Test, only altering the saving aspect
// to save as FP16.
TEST(SaveCombineFP16Op, CPU) {
  paddle::framework::Scope scope;
  paddle::platform::CPUPlace place;

  std::vector<int> lod1 = {0, 1, 2, 3, 10};
  int numel1 = 100;
  paddle::framework::LoD expect_lod1;
  float* expect1 = CreateForSaveCombineOp<float, paddle::platform::float16>(
      10, 10, lod1, "test_var1", place, &scope, &expect_lod1);

  std::vector<int> lod2 = {0, 2, 5, 10};
  int numel2 = 200;
  paddle::framework::LoD expect_lod2;
  float* expect2 = CreateForSaveCombineOp<float, paddle::platform::float16>(
      10, 20, lod2, "test_var2", place, &scope, &expect_lod2);

  std::vector<int> lod3 = {0, 20};
  int numel3 = 4000;
  paddle::framework::LoD expect_lod3;
  float* expect3 = CreateForSaveCombineOp<float, paddle::platform::float16>(
      20, 200, lod3, "test_var3", place, &scope, &expect_lod3);

  std::vector<int> lod4 = {0, 1, 20};
  int numel4 = 1000;
  paddle::framework::LoD expect_lod4;
  float* expect4 = CreateForSaveCombineOp<float, paddle::platform::float16>(
      20, 50, lod4, "test_var4", place, &scope, &expect_lod4);

  // Set attributes
  std::string filename = "check_tensor_fp16_save.ls";
  paddle::framework::AttributeMap attrs;
  attrs.insert({"file_path", std::string(filename)});
  attrs.insert({"save_as_fp16", true});

  // Run the save_combine_op
  auto save_combine_op = paddle::framework::OpRegistry::CreateOp(
      "save_combine",
      {{"X", {"test_var1", "test_var2", "test_var3", "test_var4"}}}, {}, attrs);
  save_combine_op->Run(scope, place);

  // Set up output vars
  auto target1 = GeneratePlaceholderBeforeLoad("out_var1", &scope);
  auto target2 = GeneratePlaceholderBeforeLoad("out_var2", &scope);
  auto target3 = GeneratePlaceholderBeforeLoad("out_var3", &scope);
  auto target4 = GeneratePlaceholderBeforeLoad("out_var4", &scope);

  // Run the load_combine_op
  auto load_combine_op = paddle::framework::OpRegistry::CreateOp(
      "load_combine", {},
      {{"Out", {"out_var1", "out_var2", "out_var3", "out_var4"}}}, attrs);
  load_combine_op->Run(scope, place);

  paddle::framework::LoD actual_lod1, actual_lod2, actual_lod3, actual_lod4;
  paddle::platform::float16* actual1 =
      GetValuesAfterLoadCombineOp<paddle::platform::float16>(target1, scope,
                                                             &actual_lod1);
  paddle::platform::float16* actual2 =
      GetValuesAfterLoadCombineOp<paddle::platform::float16>(target2, scope,
                                                             &actual_lod2);
  paddle::platform::float16* actual3 =
      GetValuesAfterLoadCombineOp<paddle::platform::float16>(target3, scope,
                                                             &actual_lod3);
  paddle::platform::float16* actual4 =
      GetValuesAfterLoadCombineOp<paddle::platform::float16>(target4, scope,
                                                             &actual_lod4);

  CheckValues<float, paddle::platform::float16>(expect1, actual1, expect_lod1,
                                                actual_lod1, numel1);
  CheckValues<float, paddle::platform::float16>(expect2, actual2, expect_lod2,
                                                actual_lod2, numel2);
  CheckValues<float, paddle::platform::float16>(expect3, actual3, expect_lod3,
                                                actual_lod3, numel3);
  CheckValues<float, paddle::platform::float16>(expect4, actual4, expect_lod4,
                                                actual_lod4, numel4);
}

// FP16 version of SaveLoadCombineOp Test, only altering the loading aspect
// to load tensors with FP16 precision.
TEST(LoadCombineFP16Op, CPU) {
  paddle::framework::Scope scope;
  paddle::platform::CPUPlace place;

  std::vector<int> lod1 = {0, 1, 2, 3, 10};
  int numel1 = 100;
  paddle::framework::LoD expect_lod1;
  float* expect1 = CreateForSaveCombineOp<float, paddle::platform::float16>(
      10, 10, lod1, "test_var1", place, &scope, &expect_lod1);

  std::vector<int> lod2 = {0, 2, 5, 10};
  int numel2 = 200;
  paddle::framework::LoD expect_lod2;
  float* expect2 = CreateForSaveCombineOp<float, paddle::platform::float16>(
      10, 20, lod2, "test_var2", place, &scope, &expect_lod2);

  std::vector<int> lod3 = {0, 20};
  int numel3 = 4000;
  paddle::framework::LoD expect_lod3;
  float* expect3 = CreateForSaveCombineOp<float, paddle::platform::float16>(
      20, 200, lod3, "test_var3", place, &scope, &expect_lod3);

  std::vector<int> lod4 = {0, 1, 20};
  int numel4 = 1000;
  paddle::framework::LoD expect_lod4;
  float* expect4 = CreateForSaveCombineOp<float, paddle::platform::float16>(
      20, 50, lod4, "test_var4", place, &scope, &expect_lod4);

  // Set attributes
  std::string filename = "check_tensor_fp16_load.ls";
  paddle::framework::AttributeMap attrs;
  attrs.insert({"file_path", std::string(filename)});

  // Run the save_combine_op
  auto save_combine_op = paddle::framework::OpRegistry::CreateOp(
      "save_combine",
      {{"X", {"test_var1", "test_var2", "test_var3", "test_var4"}}}, {}, attrs);
  save_combine_op->Run(scope, place);

  // Set up output vars
  auto load_var1 = scope.Var("out_var1");
  auto load_var2 = scope.Var("out_var2");
  auto load_var3 = scope.Var("out_var3");
  auto load_var4 = scope.Var("out_var4");

  attrs.insert({"load_as_fp16", true});
  // Run the load_combine_op
  auto load_combine_op = paddle::framework::OpRegistry::CreateOp(
      "load_combine", {},
      {{"Out", {"out_var1", "out_var2", "out_var3", "out_var4"}}}, attrs);
  load_combine_op->Run(scope, place);

  auto* target1 = load_var1->GetMutable<paddle::framework::LoDTensor>();
  auto* target2 = load_var2->GetMutable<paddle::framework::LoDTensor>();
  auto* target3 = load_var3->GetMutable<paddle::framework::LoDTensor>();
  auto* target4 = load_var4->GetMutable<paddle::framework::LoDTensor>();

  paddle::framework::LoD actual_lod1, actual_lod2, actual_lod3, actual_lod4;
  paddle::platform::float16* actual1 =
      GetValuesAfterLoadCombineOp<paddle::platform::float16>(target1, scope,
                                                             &actual_lod1);
  paddle::platform::float16* actual2 =
      GetValuesAfterLoadCombineOp<paddle::platform::float16>(target2, scope,
                                                             &actual_lod2);
  paddle::platform::float16* actual3 =
      GetValuesAfterLoadCombineOp<paddle::platform::float16>(target3, scope,
                                                             &actual_lod3);
  paddle::platform::float16* actual4 =
      GetValuesAfterLoadCombineOp<paddle::platform::float16>(target4, scope,
                                                             &actual_lod4);

  CheckValues<float, paddle::platform::float16>(expect1, actual1, expect_lod1,
                                                actual_lod1, numel1);
  CheckValues<float, paddle::platform::float16>(expect2, actual2, expect_lod2,
                                                actual_lod2, numel2);
  CheckValues<float, paddle::platform::float16>(expect3, actual3, expect_lod3,
                                                actual_lod3, numel3);
  CheckValues<float, paddle::platform::float16>(expect4, actual4, expect_lod4,
                                                actual_lod4, numel4);
}

// Test with original SaveLoadTest
TEST(SaveLoadTestWithCombineOp, CPU) {
  paddle::framework::Scope scope;
  paddle::platform::CPUPlace place;

  auto var = scope.Var("test_var");
  auto tensor = var->GetMutable<paddle::framework::LoDTensor>();
  tensor->Resize({3, 4000});
  paddle::framework::LoD expect_lod;
  expect_lod.resize(1);
  expect_lod[0].push_back(0);
  expect_lod[0].push_back(1);
  expect_lod[0].push_back(2);
  expect_lod[0].push_back(3);

  tensor->set_lod(expect_lod);
  int* expect = tensor->mutable_data<int>(place);
  for (int64_t i = 0; i < tensor->numel(); ++i) {
    expect[i] = static_cast<int>(i);
  }
  paddle::framework::AttributeMap attrs;
  attrs.insert({"file_path", std::string("check_t.save")});

  auto save_op = paddle::framework::OpRegistry::CreateOp(
      "save_combine", {{"X", {"test_var"}}}, {}, attrs);
  save_op->Run(scope, place);

  auto load_var = scope.Var("out_var");
  auto target = load_var->GetMutable<paddle::framework::LoDTensor>();
  auto load_op = paddle::framework::OpRegistry::CreateOp(
      "load_combine", {}, {{"Out", {"out_var"}}}, attrs);
  load_op->Run(scope, place);
  int* actual = target->data<int>();
  for (int64_t i = 0; i < tensor->numel(); ++i) {
    EXPECT_EQ(expect[i], actual[i]);
  }
  auto& actual_lod = target->lod();
  EXPECT_EQ(expect_lod.size(), actual_lod.size());
  for (size_t i = 0; i < expect_lod.size(); ++i) {
    for (size_t j = 0; j < expect_lod[i].size(); ++j) {
      EXPECT_EQ(expect_lod[i][j], actual_lod[i][j]);
    }
  }
}
