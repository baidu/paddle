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

#include <stdint.h>
#include <fstream>
#include <numeric>
#include <sstream>
#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/framework/data_type_transform.h"
#include "paddle/fluid/framework/framework.pb.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/port.h"

namespace paddle {
namespace operators {

class SaveCombineOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {}
};

class SaveCombineOpProtoMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput(
        "X",
        "(vector) Input LoDTensors that need to be saved together in a file.")
        .AsDuplicable();
    AddComment(R"DOC(
SaveCombine operator

This operator will serialize and write a list of input LoDTensor variables
to a file on disk.
)DOC");
    AddAttr<bool>("overwrite",
                  "(boolean, default true)"
                  "Overwrite the output file if it exists.")
        .SetDefault(true);
    AddAttr<bool>("save_as_fp16",
                  "(boolean, default false)"
                  "If true, the tensor will be converted to float16 data "
                  "type and then saved. Otherwise, the tensor will be "
                  "directly saved without data type conversion.")
        .SetDefault(false);
    AddAttr<std::string>(
        "file_path",
        "(string)"
        "The \"file_path\" where the LoDTensor variables will be saved.")
        .AddCustomChecker(
            [](const std::string &path) { return !path.empty(); });
  }
};

class SaveCombineOpVarTypeInference : public framework::VarTypeInference {
 public:
  void operator()(const framework::OpDesc &op_desc,
                  framework::BlockDesc *block) const override {}
};

template <typename DeviceContext, typename T>
class SaveCombineOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    auto place = ctx.GetPlace();
    auto filename = ctx.Attr<std::string>("file_path");
    auto overwrite = ctx.Attr<bool>("overwrite");
    auto save_as_fp16 = ctx.Attr<bool>("save_as_fp16");

    bool is_present = FileExists(filename);
    if (is_present && !overwrite) {
      PADDLE_THROW("%s exists!, cannot save_combine to it when overwrite=false",
                   filename, overwrite);
    }

    MkDirRecursively(DirName(filename).c_str());
    std::ofstream fout(filename, std::ios::binary);
    PADDLE_ENFORCE(static_cast<bool>(fout), "Cannot open %s to write",
                   filename);

    auto &inp_var_names = ctx.Inputs("X");
    auto &inp_vars = ctx.MultiInputVar("X");
    PADDLE_ENFORCE_GT(static_cast<int>(inp_var_names.size()), 0,
                      "The number of input variables should be greater than 0");

    // get device context from pool
    platform::DeviceContextPool &pool = platform::DeviceContextPool::Instance();
    auto &dev_ctx = *pool.Get(place);

    for (size_t i = 0; i < inp_var_names.size(); i++) {
      PADDLE_ENFORCE(inp_vars[i] != nullptr,
                     "Cannot find variable %s for save_combine_op",
                     inp_var_names[i]);
      PADDLE_ENFORCE(inp_vars[i]->IsType<framework::LoDTensor>(),
                     "SaveCombineOp only supports LoDTensor, %s has wrong type",
                     inp_var_names[i]);

      auto &tensor = inp_vars[i]->Get<framework::LoDTensor>();
      // Serialize tensors one by one

      // Check types to see if a fp16 transformation is required
      auto in_dtype = tensor.type();
      auto out_dtype =
          save_as_fp16 ? framework::proto::VarType::FP16 : in_dtype;

      if (in_dtype != out_dtype) {
        auto in_kernel_type = framework::OpKernelType(in_dtype, place);
        auto out_kernel_type = framework::OpKernelType(out_dtype, place);
        framework::LoDTensor out;
        // copy LoD info to the new tensor
        out.set_lod(tensor.lod());
        framework::TransDataType(in_kernel_type, out_kernel_type, tensor, &out);
        framework::SerializeToStream(fout, out, dev_ctx);
      } else {
        framework::SerializeToStream(fout, tensor, dev_ctx);
      }
    }
    fout.close();
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(save_combine, ops::SaveCombineOp,
                  ops::SaveCombineOpVarTypeInference,
                  paddle::framework::EmptyGradOpMaker,
                  ops::SaveCombineOpProtoMaker);

REGISTER_OP_CPU_KERNEL(
    save_combine,
    ops::SaveCombineOpKernel<paddle::platform::CPUDeviceContext, float>,
    ops::SaveCombineOpKernel<paddle::platform::CPUDeviceContext, double>);
