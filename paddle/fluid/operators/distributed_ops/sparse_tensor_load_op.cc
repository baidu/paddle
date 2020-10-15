/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <future>  // NOLINT
#include <ostream>

#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/version.h"
#include "paddle/fluid/platform/profiler.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using LoDTensor = framework::LoDTensor;
using SelectedRows = framework::SelectedRows;

struct DeserializedDataFunctor {
  DeserializedDataFunctor(void **buf, Tensor *tensor,
                          const platform::Place &place)
      : buf_(buf), tensor_(tensor), place_(place) {}

  template <typename T>
  void apply() {
    *buf_ = tensor_->mutable_data<T>(place_);
  }

  void **buf_;
  Tensor *tensor_;
  platform::Place place_;
};

template <typename DeviceContext, typename T>
class SparseTensorLoadKernel : public paddle::framework::OpKernel<T> {
 public:
  void Compute(const paddle::framework::ExecutionContext &ctx) const override {
    auto place = ctx.GetPlace();
    auto filename = ctx.Attr<std::string>("file_path");
    std::ifstream fin(filename, std::ios::binary);
    PADDLE_ENFORCE_EQ(static_cast<bool>(fin), true,
                      platform::errors::Unavailable(
                          "Load operator fail to open file %s, please check "
                          "whether the model file is complete or damaged.",
                          filename));
    auto name = ctx.OutputNames("Out")[0];
    VLOG(4) << "Sparse Load Var name: " << name;
    auto *out_var = ctx.OutputVar("Out");
    PADDLE_ENFORCE_NOT_NULL(
        out_var, platform::errors::InvalidArgument(
                     "The variable %s to be loaded cannot be found.", name));
    PADDLE_ENFORCE_EQ(out_var->IsType<paddle::framework::LoDTensor>(), true,
                      platform::errors::InvalidArgument(
                          "SparseLoad OP only support LoDTensor"));
    LoadLodTensor(fin, place, out_var, ctx);
  }

  void LoadLodTensor(std::istream &is, const platform::Place &place,
                     paddle::framework::Variable *var,
                     const paddle::framework::ExecutionContext &ctx) const {
    auto *tensor = var->GetMutable<paddle::framework::LoDTensor>();

    auto node_index = ctx.Attr<int64_t>("node_index");
    auto node_num = ctx.Attr<int64_t>("node_num");
    auto shape = ctx.Attr<std::vector<int64_t>>("shape");
    VLOG(4) << "Sparse LoadLodTensor node_num" << node_num;
    VLOG(4) << "Sparse LoadLodTensor node_index" << node_index;
    VLOG(4) << "Sparse LoadLodTensor shape[0]" << shape[0];
    PADDLE_ENFORCE_GE(node_index, 0, platform::errors::InvalidArgument(
                                         "node_num great than or equal to 0"));
    PADDLE_ENFORCE_GE(node_num, 1, platform::errors::InvalidArgument(
                                       "node_num great than or equal to 1"));

    {
      // the 1st field, unit32_t version for LoDTensor
      uint32_t version;
      is.read(reinterpret_cast<char *>(&version), sizeof(version));
      PADDLE_ENFORCE_EQ(paddle::framework::IsTensorVersionSupported(version),
                        true,
                        platform::errors::InvalidArgument(
                            "Tensor version %u is not supported.", version));
      PADDLE_ENFORCE_EQ(version, 0U, platform::errors::InvalidArgument(
                                         "Tensor version %u is not supported, "
                                         "only version 0 is supported.",
                                         version));
    }

    {
      // the 2st field, LoD information
      // Todo sparse load need change LoDTensor's lod level
      uint64_t lod_level;
      is.read(reinterpret_cast<char *>(&lod_level), sizeof(lod_level));
      auto &lod = *tensor->mutable_lod();
      lod.resize(lod_level);
    }

    // the 3st filed, Tensor

    uint32_t version;
    is.read(reinterpret_cast<char *>(&version), sizeof(version));

    PADDLE_ENFORCE_EQ(
        version, 0U,
        platform::errors::InvalidArgument(
            "tensor version %u is not supported, Only version 0 is supported",
            version));

    paddle::framework::proto::VarType::TensorDesc desc;

    {  // int32_t size
      // proto buffer
      int32_t size;
      is.read(reinterpret_cast<char *>(&size), sizeof(size));
      std::unique_ptr<char[]> buf(new char[size]);
      is.read(reinterpret_cast<char *>(buf.get()), size);
      PADDLE_ENFORCE_EQ(
          desc.ParseFromArray(buf.get(), size), true,
          platform::errors::InvalidArgument("Cannot parse tensor desc"));
    }

    {  // read tensor
      std::vector<int64_t> dims;
      dims.reserve(static_cast<size_t>(desc.dims().size()));
      std::copy(desc.dims().begin(), desc.dims().end(),
                std::back_inserter(dims));

      int64_t line_numel = 1;
      for (size_t dim = 1; dim < dims.size(); dim++) {
        line_numel *= dims[dim];
      }
      auto total_line = dims[0];

      tensor->Resize(paddle::framework::make_ddim(shape));

      void *buf;
      auto ctx = platform::CPUDeviceContext();

      paddle::framework::VisitDataType(
          desc.data_type(),
          DeserializedDataFunctor(&buf, tensor, ctx.GetPlace()));

      auto line_size =
          line_numel * paddle::framework::SizeOfType(desc.data_type());
      char *cur_buf = static_cast<char *>(buf);
      char *temp_row = new char[line_size];
      VLOG(4) << "TensorFromStream: line_size " << line_size;
      VLOG(4) << "TensorFromStream: total_line " << total_line;
      for (size_t line_index = 0; line_index < static_cast<size_t>(total_line);
           ++line_index) {
        is.read(temp_row, line_size);
        if (static_cast<int64_t>(line_index) % node_num == node_index) {
          memcpy(cur_buf, temp_row, line_size);
          cur_buf += line_size;
        }
      }
    }
  }
};

class SparseTensorLoadOp : public paddle::framework::OperatorWithKernel {
 public:
  using paddle::framework::OperatorWithKernel::OperatorWithKernel;
  void InferShape(paddle::framework::InferShapeContext *ctx) const override {}

 protected:
  paddle::framework::OpKernelType GetExpectedKernelType(
      const paddle::framework::ExecutionContext &ctx) const override {
    paddle::framework::OpKernelType kt = paddle::framework::OpKernelType(
        paddle::framework::proto::VarType::FP32, ctx.GetPlace());
    return kt;
  }
};

class SparseTensorLoadOpMaker
    : public paddle::framework::OpProtoAndCheckerMaker {
 public:
  void Make() {
    AddOutput("Out", "The LoDTensor / SelectedRows need to be loaded");
    AddAttr<std::string>("file_path",
                         R"(Variable will be loaded from "file_path")")
        .AddCustomChecker(
            [](const std::string &path) { return !path.empty(); });
    AddAttr<int64_t>("node_index", "role id from 0 ~ node_num.").SetDefault(0);
    AddAttr<int64_t>("node_num", "role nums which need load current varibale.")
        .SetDefault(0);
    AddAttr<std::vector<int64_t>>("shape",
                                  "(vector<int64_t>) The shape of the output")
        .SetDefault({});
    AddComment(R"DOC(
    SparseTensorLoad OP, Load sprase tensor on parameter server
    )DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(sparse_tensor_load, ops::SparseTensorLoadOp,
                  ops::SparseTensorLoadOpMaker);

REGISTER_OP_CPU_KERNEL(
    sparse_tensor_load,
    ops::SparseTensorLoadKernel<paddle::platform::CPUDeviceContext, float>)
