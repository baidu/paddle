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

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/distributed/distributed.h"

namespace paddle {
namespace framework {
class InferShapeContext;
class OpDesc;
class Scope;
template <typename T>
class EmptyGradOpMaker;
}  // namespace framework
namespace imperative {
class OpBase;
}  // namespace imperative
}  // namespace paddle

namespace paddle {
namespace operators {

class CheckpointNotifyOp : public framework::OperatorBase {
 public:
  CheckpointNotifyOp(const std::string& type,
                     const framework::VariableNameMap& inputs,
                     const framework::VariableNameMap& outputs,
                     const framework::AttributeMap& attrs)
      : OperatorBase(type, inputs, outputs, attrs) {}

  void RunImpl(const framework::Scope& scope,
               const platform::Place& place) const override {
    std::vector<std::string> epmap =
        Attr<std::vector<std::string>>("endpoints");
    std::string dirname = Attr<std::string>("dirname");
    std::string varname = Attr<std::string>("varname");
    auto mode = Attr<int>("mode");

    if (mode != 0 && mode != 1 && mode != 2) {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "mode expected in [0/1/2], but got %d", mode));
    }

    std::vector<std::string> slice_varnames =
        Attr<std::vector<std::string>>("slice_varnames");

    std::vector<std::string> remote_varnames =
        Attr<std::vector<std::string>>("remote_varnames");

    distributed::RPCClient* rpc_client =
        distributed::RPCClient::GetInstance<RPCCLIENT_T>(0);

    for (size_t i = 0; i < epmap.size(); i++) {
      auto save_path =
          string::Sprintf("%s/%s/%s", dirname, varname, slice_varnames[i]);

      rpc_client->AsyncCheckpointNotify(epmap[i], save_path, remote_varnames[i],
                                        mode);

      VLOG(3) << "checkpoint notify sending with path: " << save_path
              << " and var:" << slice_varnames[i] << " to " << epmap[i]
              << " with mode " << mode;
    }
    PADDLE_ENFORCE_EQ(
        rpc_client->Wait(), true,
        platform::errors::Fatal("Fail to notify checkpoint."
                                " Internal error occurs in RPCClient."));
  }
};

class CheckpointNotifyOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() {
    AddAttr<std::vector<std::string>>(
        "endpoints",
        "(string vector)"
        "Parameter Server endpoints in the order");
    AddAttr<std::string>("dirname",
                         "(string) indicate the folder checkpoint will use");
    AddAttr<std::string>("varname", "(string)  the var need to be saved");
    AddAttr<std::vector<std::string>>(
        "slice_varnames", "(string vector) the slice vars need to be saved");
    AddAttr<std::vector<std::string>>(
        "remote_varnames", "(string vector) the slice vars need to be saved");
    AddAttr<int>("mode", "mode=0/1/2 means nothing/save base/save delta")
        .SetDefault(0);
    AddComment(R"DOC(
CheckpointNotify operator
This operator will send lookup table and it's checkpoint direcoty to listen_and_serve op at
the parameter server.
)DOC");
  }
};

class CheckpointNotifyOpShapeInference : public framework::InferShapeBase {
 public:
  void operator()(framework::InferShapeContext* ctx) const override {}
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(
    checkpoint_notify, ops::CheckpointNotifyOp,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>,
    ops::CheckpointNotifyOpMaker, ops::CheckpointNotifyOpShapeInference);
