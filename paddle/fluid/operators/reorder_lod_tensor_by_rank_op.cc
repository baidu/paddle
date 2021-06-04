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

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/platform/device_context.h"

namespace paddle {
namespace framework {
class LoDRankTable;
class LoDTensor;
class OpDesc;
class Scope;
}  // namespace framework
namespace imperative {
class OpBase;
}  // namespace imperative
}  // namespace paddle

namespace paddle {
namespace operators {

class ReorderLoDTensorByRankTableOpProtoMaker
    : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X",
             "(LoDTensor), the input lod tensor to be reordered according to "
             "Input(RankTable).");
    AddInput("RankTable",
             "(LoDRankTable), the rank table according to which Input(X) is "
             "reordered.");
    AddOutput("Out", "LoDTensor, the reordered lod tensor.");
    AddComment(R"DOC(ReorderLoDTensorByRankTable operator.

Input(X) is a batch of sequences. Input(RankTable) stores new orders of the
input sequence batch. The reorder_lod_tensor_by_rank operator reorders the
Input(X) according to the information provided by Input(RankTable).

For example:

If the indices stored in the Input(RankTable) are [3, 0, 2, 1], the
Input(X) will be reordered that the fourth sequence in Input(X) will become the
first one, and then followed by the original first, third, and the second one.

This is:
X = [Seq0, Seq1, Seq2, Seq3]. The indices in RankTable are [3, 0, 2, 1].
Out =  [Seq3, Seq0, Seq2, Seq1] with a new LoD information.

If the LoD information of Input(X) is empty, this means Input(X) is not sequence
data. This is also identical to a batch of sequences where each sequence has a
fixed length 1. In this case, the reorder_lod_tensor_by_rank operator reorders
each slice of Input(X) along the first axis according to Input(RankTable).

This is:
X = [Slice0, Slice1, Slice2, Slice3] and its LoD information is empty. The
indices in RankTable are [3, 0, 2, 1].
Out = [Slice3, Slice0, Slice2, Slice1] with no LoD information is appended.

**NOTE**: 
This operator sorts Input(X) according to a given LoDRankTable which does
not need to be calculated according to Input(X). It can be calculated according
to another different sequence, and then this operator sorts Input(X) according
to the given LoDRankTable.

)DOC");
  }
};

class ReorderLoDTensorByRankTableBase : public framework::OperatorBase {
 public:
  ReorderLoDTensorByRankTableBase(const std::string &type,
                                  const framework::VariableNameMap &inputs,
                                  const framework::VariableNameMap &outputs,
                                  const framework::AttributeMap &attrs)
      : OperatorBase(type, inputs, outputs, attrs) {}

 private:
  void RunImpl(const framework::Scope &scope,
               const platform::Place &place) const override {
    auto &x = GET_DATA_SAFELY(scope.FindVar(Input("X")), "Input", "X",
                              "ReorderLoDTensorByRankTable")
                  .Get<framework::LoDTensor>();
    auto &rank_table =
        GET_DATA_SAFELY(scope.FindVar(Input("RankTable")), "Input", "RankTable",
                        "ReorderLoDTensorByRankTable")
            .Get<framework::LoDRankTable>();
    auto &out = *(GET_DATA_SAFELY(scope.FindVar(Output("Out")), "Output", "Out",
                                  "ReorderLoDTensorByRankTable")
                      .GetMutable<framework::LoDTensor>());

    out.Resize(x.dims());
    out.mutable_data(x.place(), x.type());
    this->process(place, x, rank_table, &out);
  }

 protected:
  virtual void process(const platform::Place &place,
                       const framework::LoDTensor &x,
                       const framework::LoDRankTable &rank_table,
                       framework::LoDTensor *out) const = 0;

  struct AbsoluteRankTableItem {
    size_t offset;  // the absolute/accumulated offset.
    size_t length;  // the length
    framework::LoD lod;
  };

  std::vector<AbsoluteRankTableItem> GetAbsoluteOffsetAndLengthByLoDRankTable(
      const framework::LoDTensor &x) const {
    std::vector<AbsoluteRankTableItem> absolute_table;

    if (x.lod().empty()) {
      // For Tensor without lod, such as the output of sequence_pool_op
      size_t size = x.dims()[0];
      absolute_table.reserve(size);
      for (size_t i = 0; i < size; ++i) {
        absolute_table.emplace_back();
        absolute_table.back().length = 1;
        absolute_table.back().offset = i;
      }
    } else {
      size_t level = 0;
      size_t size = x.lod()[level].size();

      for (size_t i = 0; i < size - 1; ++i) {
        auto lod_offset =
            framework::GetSubLoDAndAbsoluteOffset(x.lod(), i, i + 1, level);

        auto &offset = lod_offset.second;

        absolute_table.emplace_back();
        absolute_table.back().length = offset.second - offset.first;
        absolute_table.back().offset = offset.first;
        absolute_table.back().lod = lod_offset.first;
      }
    }

    return absolute_table;
  }

  size_t CopyTensorAndLod(const platform::Place &place,
                          const AbsoluteRankTableItem &item,
                          const framework::LoDTensor &x,
                          framework::LoDTensor *out, size_t out_offset) const {
    auto &out_lod = *out->mutable_lod();
    auto len = item.length;
    auto x_offset = item.offset;

    if (out_lod.empty()) {
      for (size_t i = 0; i < item.lod.size(); ++i) {
        out_lod.emplace_back(std::vector<size_t>({0}));
      }
    }

    for (size_t i = 0; i < out_lod.size(); ++i) {
      auto &out_v = out_lod[i];
      auto &new_lod_v = item.lod[i];

      for (auto &detail : new_lod_v) {
        out_v.push_back(out_v.back() + detail);
      }
    }

    auto x_sliced = x.Slice(x_offset, x_offset + len);
    auto out_sliced = out->Slice(out_offset, out_offset + len);

    platform::DeviceContextPool &pool = platform::DeviceContextPool::Instance();
    auto &dev_ctx = *pool.Get(place);
    framework::TensorCopy(x_sliced, out_sliced.place(), dev_ctx, &out_sliced);
    out_offset += len;
    return out_offset;
  }
};

class ReorderLoDTensorByRankTableOp : public ReorderLoDTensorByRankTableBase {
 public:
  ReorderLoDTensorByRankTableOp(const std::string &type,
                                const framework::VariableNameMap &inputs,
                                const framework::VariableNameMap &outputs,
                                const framework::AttributeMap &attrs)
      : ReorderLoDTensorByRankTableBase(type, inputs, outputs, attrs) {}

 protected:
  void process(const platform::Place &place, const framework::LoDTensor &x,
               const framework::LoDRankTable &rank_table,
               framework::LoDTensor *out) const override {
    auto absolute_table = GetAbsoluteOffsetAndLengthByLoDRankTable(x);
    size_t out_offset = 0;
    out->mutable_lod()->clear();
    for (auto &item : rank_table.items()) {
      PADDLE_ENFORCE_LT(item.index, absolute_table.size(),
                        platform::errors::OutOfRange(
                            "The value of rank_table is out of range."));
      out_offset = CopyTensorAndLod(place, absolute_table[item.index], x, out,
                                    out_offset);
    }
  }
};

class IdentityInferShape : public framework::InferShapeBase {
 public:
  void operator()(framework::InferShapeContext *context) const override {
    context->SetOutputDim("Out", context->GetInputDim("X"));
    // X'lod and Out'lod is different on runtime, so there is no need to call
    // ShareLoD for runtime. While the setting of Out's lod is done in detail
    // kernel implementation.
    if (!context->IsRuntime()) {
      context->ShareLoD("X", /*->*/ "Out");
    }
  }
};

template <typename T>
class ReorderLodTensorByRankGradOpMaker
    : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> grad_op) const override {
    grad_op->SetType("reorder_lod_tensor_by_rank_grad");
    grad_op->SetInput("X", this->OutputGrad("Out"));
    grad_op->SetOutput("Out", this->InputGrad("X"));
    grad_op->SetInput("RankTable", this->Input("RankTable"));
  }
};

class ReorderLoDTensorByRankGradOp : public ReorderLoDTensorByRankTableBase {
 public:
  ReorderLoDTensorByRankGradOp(const std::string &type,
                               const framework::VariableNameMap &inputs,
                               const framework::VariableNameMap &outputs,
                               const framework::AttributeMap &attrs)
      : ReorderLoDTensorByRankTableBase(type, inputs, outputs, attrs) {}

 protected:
  void process(const platform::Place &place, const framework::LoDTensor &x,
               const framework::LoDRankTable &rank_table,
               framework::LoDTensor *out) const override {
    auto absolute_table = GetAbsoluteOffsetAndLengthByLoDRankTable(x);

    // offsets = enumerate([item.index for item in rank_table.items()])
    std::vector<std::pair<size_t, size_t>> offsets;
    offsets.reserve(rank_table.items().size());
    for (size_t i = 0; i < rank_table.items().size(); ++i) {
      offsets.emplace_back(i, rank_table.items()[i].index);
    }

    // offsets.sort(key=lambda x: x[1])
    std::sort(
        offsets.begin(), offsets.end(),
        [](const std::pair<size_t, size_t> &a,
           const std::pair<size_t, size_t> &b) { return a.second < b.second; });

    // Copy TensorAndLod
    size_t out_offset = 0;
    for (auto &offset : offsets) {
      out_offset = this->CopyTensorAndLod(place, absolute_table[offset.first],
                                          x, out, out_offset);
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(
    reorder_lod_tensor_by_rank, ops::ReorderLoDTensorByRankTableOp,
    ops::ReorderLodTensorByRankGradOpMaker<paddle::framework::OpDesc>,
    ops::ReorderLodTensorByRankGradOpMaker<paddle::imperative::OpBase>,
    ops::ReorderLoDTensorByRankTableOpProtoMaker, ops::IdentityInferShape);
REGISTER_OPERATOR(reorder_lod_tensor_by_rank_grad,
                  ops::ReorderLoDTensorByRankGradOp, ops::IdentityInferShape);
