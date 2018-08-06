// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include <gperftools/profiler.h>
#include <chrono>              // NOLINT
#include <condition_variable>  // NOLINT
#include <fstream>
#include <thread>  // NOLINT
#include "gtest/gtest.h"
#include "paddle/fluid/framework/executor.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/platform/init.h"

USE_OP(uniform_random);
USE_OP(mul);
USE_OP_ITSELF(fill_constant);
USE_OP(gaussian_random);
USE_OP(conv2d);
USE_OP(batch_norm);
USE_OP(relu);
USE_OP(pool2d);
USE_OP(elementwise_add);
USE_OP(softmax);

TEST(MultiThread, main) {
  paddle::framework::Scope param_scope;
  std::unique_ptr<paddle::framework::ProgramDesc> startup_program;
  {
    std::ifstream in("startup.protobin");
    std::string str((std::istreambuf_iterator<char>(in)),
                    std::istreambuf_iterator<char>());
    paddle::framework::proto::ProgramDesc proto_startup_program;
    proto_startup_program.ParseFromString(str);
    startup_program.reset(
        new paddle::framework::ProgramDesc(proto_startup_program));
    paddle::platform::CPUPlace cpu;
    paddle::framework::Executor exe(cpu);
    auto ctx = paddle::framework::Executor::Prepare(*startup_program, 0);
    exe.RunPreparedContext(ctx.get(), &param_scope, false, true, false);
  }
  std::unique_ptr<paddle::framework::ProgramDesc> main_program;
  {
    std::ifstream in("main.protobin");
    std::string str((std::istreambuf_iterator<char>(in)),
                    std::istreambuf_iterator<char>());
    paddle::framework::proto::ProgramDesc proto;
    proto.ParseFromString(str);
    main_program.reset(new paddle::framework::ProgramDesc(proto));
  }
  int cpu_count = std::thread::hardware_concurrency();
  std::unique_ptr<paddle::framework::Scope[]> scopes(
      new paddle::framework::Scope[cpu_count]);
  for (int i = 0; i < cpu_count; ++i) {
    for (auto* var : startup_program->Block(0).AllVars()) {
      auto var_name = var->Name();
      scopes[i]
          .Var(var_name)
          ->GetMutable<paddle::framework::LoDTensor>()
          ->ShareDataWith(
              param_scope.Var(var_name)->Get<paddle::framework::LoDTensor>());
    }
  }

  std::mutex start_flag_mtx;
  bool start_flag = false;
  std::condition_variable start_cv;

  auto thread_main = [&](paddle::framework::Scope* scope) {
    {
      std::unique_lock<std::mutex> lock(start_flag_mtx);
      start_cv.wait(lock, [&] { return start_flag; });
    }
    paddle::platform::CPUPlace cpu;
    paddle::framework::Executor exec(cpu);
    auto& working_scope = *scope;
    working_scope.Var("img")->GetMutable<paddle::framework::LoDTensor>();

    {
      // initialize x
      paddle::framework::OpRegistry::CreateOp(
          "uniform_random", {}, {{"Out", {"img"}}},
          {{"shape", std::vector<int>{1, 3, 224, 224}},
           {"dtype",
            static_cast<int>(paddle::framework::proto::VarType::FP32)}})
          ->Run(working_scope, cpu);
    }
    auto ctx = exec.Prepare(*main_program, 0);
    exec.RunPreparedContext(ctx.get(), &working_scope, false, true, false);
    for (size_t i = 0; i < (1U << 4); ++i) {
      exec.RunPreparedContext(ctx.get(), &working_scope, false, false, false);
    }
  };

  std::unique_ptr<std::chrono::nanoseconds[]> times(
      new std::chrono::nanoseconds[cpu_count]);

  for (int i = 0; i < cpu_count; ++i) {
    std::vector<std::thread> threads;
    auto begin = std::chrono::high_resolution_clock::now();
    for (int j = 0; j < i + 1; ++j) {
      threads.emplace_back(
          [thread_main, &scopes, j] { thread_main(&scopes[j]); });
    }

    ProfilerStart(paddle::string::Sprintf("thread_%d.prof", i + 1).c_str());

    {
      std::unique_lock<std::mutex> lock(start_flag_mtx);
      start_flag = true;
    }

    start_cv.notify_all();

    for (auto& th : threads) {
      th.join();
    }

    ProfilerStop();

    threads.clear();
    auto end = std::chrono::high_resolution_clock::now();
    times[i] =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
    start_flag = false;
  }

  std::unique_ptr<float[]> speed_up_ratio(new float[cpu_count]);
  speed_up_ratio[0] = 1.0f;  // speed_up_ratio[0] is meaning less.
  for (int i = 1; i < cpu_count; ++i) {
    speed_up_ratio[i] = static_cast<float>(
        (static_cast<double>(i + 1) / static_cast<double>(times[i].count())) /
        (1 / static_cast<double>(times[0].count())));
  }
  std::cerr << "speed up ratio:\n";
  for (int i = 0; i < cpu_count; ++i) {
    std::cerr << speed_up_ratio[i] << " ";
  }
  std::cerr << "\n";
}
