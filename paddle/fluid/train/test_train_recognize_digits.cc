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

#include <time.h>
#include <fstream>

#include "gflags/gflags.h"
#include "gtest/gtest.h"

#include "paddle/fluid/framework/executor.h"
#include "paddle/fluid/framework/io/fs.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/fluid/inference/io.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/init.h"
#include "paddle/fluid/platform/place.h"

DEFINE_string(dirname, "", "Directory of the train model.");

namespace paddle {

void Train(std::string model_dir) {
  framework::InitDevices(false);
  const auto cpu_place = platform::CPUPlace();
  framework::Executor executor(cpu_place);
  framework::Scope scope;

  auto train_program = inference::Load(
      &executor, &scope, model_dir + "__model_combined__.main_program",
      model_dir + "__params_combined__");

  std::string loss_name = "";
  for (auto op_desc : train_program->Block(0).AllOps()) {
    if (op_desc->Type() == "mean") {
      loss_name = op_desc->Output("Out")[0];
      break;
    }
  }

  PADDLE_ENFORCE_NE(loss_name, "",
                    platform::errors::NotFound("Loss name is not found."));

  // prepare data
  auto x_var = scope.Var("img");
  auto x_tensor = x_var->GetMutable<framework::LoDTensor>();
  x_tensor->Resize({64, 1, 28, 28});

  auto x_data = x_tensor->mutable_data<float>(cpu_place);
  for (int i = 0; i < 64 * 28 * 28; ++i) {
    x_data[i] = 1.0;
  }

  auto y_var = scope.Var("label");
  auto y_tensor = y_var->GetMutable<framework::LoDTensor>();
  y_tensor->Resize({64, 1});
  auto y_data = y_tensor->mutable_data<int64_t>(cpu_place);
  for (int i = 0; i < 64 * 1; ++i) {
    y_data[i] = static_cast<int64_t>(1);
  }

  auto loss_var = scope.Var(loss_name);
  float first_loss = 0.0;
  float last_loss = 0.0;
  for (int i = 0; i < 100; ++i) {
    executor.Run(*train_program, &scope, 0, false, true,
                 {loss_name, "img", "label"});
    if (i == 0) {
      first_loss = loss_var->Get<framework::LoDTensor>().data<float>()[0];
    } else if (i == 99) {
      last_loss = loss_var->Get<framework::LoDTensor>().data<float>()[0];
    }
  }
  EXPECT_LT(last_loss, first_loss);
}

TEST(train, recognize_digits) {
  CHECK(!FLAGS_dirname.empty());
  Train(FLAGS_dirname + "recognize_digits_mlp.train.model/");
  Train(FLAGS_dirname + "recognize_digits_conv.train.model/");
}

}  // namespace paddle
