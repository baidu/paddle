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
#include "paddle/framework/lod_rank_table.h"
#include "paddle/framework/lod_tensor_array.h"
#include "paddle/framework/op_registry.h"

namespace paddle {
namespace operators {

struct CopyRange {
  size_t begin;
  size_t end;
};

class LoDTensorToArrayOp : public framework::OperatorBase {
 public:
  LoDTensorToArrayOp(const std::string &type,
                     const framework::VariableNameMap &inputs,
                     const framework::VariableNameMap &outputs,
                     const framework::AttributeMap &attrs)
      : OperatorBase(type, inputs, outputs, attrs) {}
  void Run(const framework::Scope &scope,
           const platform::DeviceContext &dev_ctx) const override {
    auto &x = scope.FindVar(Input("X"))->Get<framework::LoDTensor>();
    auto &rank_table =
        scope.FindVar(Input("RankTable"))->Get<framework::LoDRankTable>();
    auto &out =
        *scope.FindVar(Output("Out"))->GetMutable<framework::LoDTensorArray>();

    auto &items = rank_table.items();
    auto max_seq_len = items[0].length;
    auto table_height = items.size();
    auto rank_level = rank_table.level();
    out.resize(max_seq_len);
    std::vector<std::vector<CopyRange>> copy_ranges(max_seq_len);

    // set out[i] lod
    for (size_t i = 0; i < max_seq_len; i++) {
      auto &lod = *out[i].mutable_lod();
      lod.clear();
      for (size_t j = 0; j < table_height; j++) {
        if (i >= items[j].length) {
          break;
        }
        size_t start_idx = x.lod()[rank_level][items[j].index] + i;
        copy_ranges[i].emplace_back();
        auto &range = copy_ranges[i].back();
        std::vector<std::vector<size_t>> lod_length;
        framework::GetFineGrainedLoDLength2(x.lod(), start_idx, start_idx + 1,
                                            rank_level + 1, &lod_length,
                                            &range.begin, &range.end);
        VLOG(10) << "Append Range " << i << " [" << range.begin << ", "
                 << range.end << "]";
        framework::AppendLoD(&lod, lod_length);
      }
    }

    for (size_t i = 0; i < max_seq_len; ++i) {
      auto &ranges = copy_ranges[i];
      size_t height = std::accumulate(
          ranges.begin(), ranges.end(), 0UL,
          [](size_t a, const CopyRange &b) { return a + b.end - b.begin; });
      auto x_dim = x.dims();
      x_dim[0] = static_cast<int64_t>(height);
      out[i].Resize(x_dim);
      out[i].mutable_data(x.place(), x.type());
      size_t offset = 0;
      for (auto &each_range : ranges) {
        size_t len = each_range.end - each_range.begin;
        // out[i][offset: offset+len] = x[each_range.begin: each_range.end]
        out[i]
            .Slice(static_cast<int>(offset), static_cast<int>(offset + len))
            .CopyFrom(x.Slice(static_cast<int>(each_range.begin),
                              static_cast<int>(each_range.end)),
                      x.place(), dev_ctx);
        offset += len;
      }
    }
  }
};

class LoDTensorToArrayOpProtoMaker : public framework::OpProtoAndCheckerMaker {
 public:
  LoDTensorToArrayOpProtoMaker(framework::OpProto *proto,
                               framework::OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X", "");
    AddInput("RankTable", "");
    AddOutput("Out", "");
    AddComment("");
  }
};

class LoDTensorToArrayInferShape : public framework::InferShapeBase {
 public:
  void operator()(framework::InferShapeContext *context) const override {
    PADDLE_ENFORCE(context->HasInput("X"),
                   "Input(X) of LoDTensorToArrayOp should not be null.");
    PADDLE_ENFORCE(
        context->HasInput("RankTable"),
        "Input(RankTable) of LoDTensorToArrayOp should not be null.");

    PADDLE_ENFORCE(context->HasOutput("Out"),
                   "Output(Out) of LoDTensorToArrayOp should not be null.");

    auto x_dim = context->GetInputDim("X");
    // The first dim of each LoDTensor in Output can only be set at run-time.;
    // We still have to Resize each LoDTensor in Output.
    context->SetOutputDim("Out", x_dim);
  }
};

class LoDTensorToArrayInferVarType : public framework::VarTypeInference {
 public:
  void operator()(const framework::OpDescBind &op_desc,
                  framework::BlockDescBind *block) const override {
    for (auto &out_var : op_desc.Output("Out")) {
      block->Var(out_var)->SetType(framework::VarDesc::LOD_TENSOR_ARRAY);
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(lod_tensor_to_array, ops::LoDTensorToArrayOp,
                  ops::LoDTensorToArrayOpProtoMaker,
                  ops::LoDTensorToArrayInferShape,
                  ops::LoDTensorToArrayInferVarType);
