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

#include <map>
#include <memory>
#include <set>
#include <string>
#include <thread>  // NOLINT
#include <utility>
#include <vector>
#include "gflags/gflags.h"
#include "paddle/fluid/operators/distributed/distributed.h"
#include "paddle/fluid/operators/distributed/request_handler.h"
#include "paddle/fluid/operators/distributed/request_handler_impl.h"
#include "paddle/fluid/operators/distributed/rpc_server.h"

namespace paddle {
namespace framework {
class Variable;
}  // namespace framework
}  // namespace paddle

namespace paddle {
namespace operators {
namespace distributed {

class CollectiveServer;

class GetMonomerHandler final : public RequestHandler {
 public:
  GetMonomerHandler() : RequestHandler(true) {}
  virtual ~GetMonomerHandler() {}
  bool Handle(const std::string& var_name, framework::Scope* scope,
              framework::Variable* var, framework::Variable** outvar,
              const int trainer_id, const std::string& out_var_name = "",
              const std::string& table_name = "") override {
    VLOG(50) << "GetMonomerHandler recv " << var_name;

    *outvar = scope->FindVar(var_name);
    PADDLE_ENFORCE_NOT_NULL(
        outvar, platform::errors::NotFound("var: %s is not found.", var_name));

    return true;
  }
};

class GetMonomerBarrierHandler final : public RequestHandler {
 public:
  GetMonomerBarrierHandler() : RequestHandler(true) {}
  virtual ~GetMonomerBarrierHandler() {}
  bool Handle(const std::string& var_name, framework::Scope* scope,
              framework::Variable* var, framework::Variable** outvar,
              const int trainer_id, const std::string& out_var_name = "",
              const std::string& table_name = "") override {
    VLOG(50) << "GetMonomerHandler recv " << var_name;

    rpc_server_->IncreaseVarBarrier(var_name);

    return true;
  }
};

class CollectiveServer final {
 public:
  explicit CollectiveServer(const std::string& end_point, int fan_in);

  virtual ~CollectiveServer() {}

  void StartServer();

  static CollectiveServer* GetInstance(const std::string& end_point,
                                       int fan_in) {
    std::call_once(init_flag_, [&]() {
      if (collective_server_.get() == nullptr) {
        collective_server_.reset(new CollectiveServer(end_point, fan_in));
        collective_server_->StartServer();
      }
    });

    return collective_server_.get();
  }

  std::shared_ptr<RPCServer> GetRPCServer() { return rpc_server_; }

  void Stop();

 private:
  std::unique_ptr<GetMonomerHandler> get_monomer_handler_;
  std::unique_ptr<GetMonomerBarrierHandler> get_barrier_handler_;

  std::shared_ptr<distributed::RPCServer> rpc_server_;
  std::shared_ptr<std::thread> server_thread_;
  std::shared_ptr<std::thread> loop_thread_;

  bool ready_{false};

  static std::once_flag init_flag_;
  static std::shared_ptr<CollectiveServer> collective_server_;
};

};  // namespace distributed
};  // namespace operators
};  // namespace paddle
