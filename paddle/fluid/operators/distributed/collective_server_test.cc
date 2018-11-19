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

#include <unistd.h>
#include <string>
#include <thread>  // NOLINT

#include "gtest/gtest.h"
#include "paddle/fluid/framework/block_desc.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"

#include "paddle/fluid/operators/detail/macros.h"
#include "paddle/fluid/operators/distributed/collective_client.h"
#include "paddle/fluid/operators/distributed/collective_server.h"
#include "paddle/fluid/operators/distributed/request_handler_impl.h"
#include "paddle/fluid/operators/math/math_function.h"

namespace framework = paddle::framework;
namespace platform = paddle::platform;
namespace distributed = paddle::operators::distributed;

std::shared_ptr<distributed::CollectiveServer> StartServer(
    const std::string& ep, int fan_in, framework::Scope* scope,
    platform::DeviceContext* dev_ctx) {
  distributed::CollectiveServer* server =
      new distributed::CollectiveServer(ep, fan_in);

  server->WaitNotInService();
  server->ResetContext(scope, dev_ctx);
  server->SetInService();

  return std::shared_ptr<distributed::CollectiveServer>(server);
}

std::shared_ptr<framework::Scope> GenerateVars(platform::Place place) {
  platform::DeviceContextPool& pool = platform::DeviceContextPool::Instance();
  auto& ctx = *pool.Get(place);

  framework::Scope* scope = new framework::Scope();
  framework::Variable* var = scope->Var("var1");
  auto* slr = var->GetMutable<framework::SelectedRows>();
  slr->set_height(1000);

  auto* tensor = slr->mutable_value();
  auto* rows = slr->mutable_rows();

  tensor->Resize(framework::make_ddim({564, 128}));
  tensor->mutable_data<float>(place);

  // int tensor_numel = 564 * 128;
  paddle::operators::math::set_constant(ctx, tensor, 32.7);
  for (int i = 0; i < 564; ++i) rows->push_back(i);

  std::cout << "src:" << slr->Info();

  return std::shared_ptr<framework::Scope>(scope);
}

TEST(PREFETCH, CPU) {
  platform::CPUPlace place;
  platform::DeviceContextPool& pool = platform::DeviceContextPool::Instance();
  auto& ctx = *pool.Get(place);

  std::string ep = "127.0.0.1:7164";
  auto scope = GenerateVars(place);
  auto server = StartServer(ep, 2, scope.get(), &ctx);

  distributed::CollectiveClient* client =
      distributed::CollectiveClient::GetInstance();

  std::vector<std::string> eps{ep};
  // gather 1
  std::vector<const framework::SelectedRows*> dst1;
  client->Gather(eps, ctx, *scope.get(), "var2", &dst1);
  std::cout << "dst1:" << dst1[0]->Info();

  // gather 2
  std::vector<const framework::SelectedRows*> dst2;
  client->Gather(eps, ctx, *scope.get(), "var3", &dst2);
  std::cout << "dst2:" << dst2[0]->Info();
}
