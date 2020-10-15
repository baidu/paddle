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
#include "paddle/fluid/operators/math/concat_and_split.h"
#include "paddle/fluid/platform/device_context.h"

namespace paddle {
namespace framework {
class OpDesc;
class Scope;
}  // namespace framework
namespace imperative {
class OpBase;
}  // namespace imperative
}  // namespace paddle

namespace paddle {
namespace operators {

struct CopyRange {
  size_t begin;
  size_t end;
};

struct LoDTensorToArrayFunctor;

template <typename DeviceContext>
struct LoDTensorToArrayFunctorImpl {
  const LoDTensorToArrayFunctor *prev_functor_;
  DeviceContext *dev_ctx_;
  template <typename T>
  void apply();
};

struct LoDTensorToArrayFunctor : public boost::static_visitor<void> {
  std::vector<const framework::Tensor *> ref_inputs_;
  mutable std::vector<framework::Tensor *> outputs_;
  const framework::Tensor &input_;

  explicit LoDTensorToArrayFunctor(const framework::Tensor &input)
      : input_(input) {}

  void AddOutput(framework::Tensor *t) {
    outputs_.emplace_back(t);
    ref_inputs_.emplace_back(t);
  }

  template <typename Place>
  void operator()(Place place) const {
    auto &pool = platform::DeviceContextPool::Instance();
    auto *dev_ctx = pool.Get(place);
    if (std::is_same<Place, platform::CPUPlace>::value) {
      Apply(static_cast<platform::CPUDeviceContext *>(dev_ctx));
    } else {
#ifdef PADDLE_WITH_CUDA
      Apply(static_cast<platform::CUDADeviceContext *>(dev_ctx));
#else
      PADDLE_THROW(
          platform::errors::Unavailable("Paddle is not compiled with CUDA."));
#endif
    }
  }

  template <typename DeviceContext>
  void Apply(DeviceContext *dev_ctx) const {
    LoDTensorToArrayFunctorImpl<DeviceContext> func;
    func.prev_functor_ = this;
    func.dev_ctx_ = dev_ctx;
    framework::VisitDataType(input_.type(), func);
  }
};

template <typename DeviceContext>
template <typename T>
void LoDTensorToArrayFunctorImpl<DeviceContext>::apply() {
  math::SplitFunctor<DeviceContext, T> func;
  func(*dev_ctx_, prev_functor_->input_, prev_functor_->ref_inputs_, 0,
       &prev_functor_->outputs_);
}

class LoDTensorToArrayOp : public framework::OperatorBase {
 public:
  LoDTensorToArrayOp(const std::string &type,
                     const framework::VariableNameMap &inputs,
                     const framework::VariableNameMap &outputs,
                     const framework::AttributeMap &attrs)
      : OperatorBase(type, inputs, outputs, attrs) {}

 private:
  void RunImpl(const framework::Scope &scope,
               const platform::Place &place) const override {
    auto &x = GET_DATA_SAFELY(scope.FindVar(Input("X")), "Input", "X",
                              "LoDTensorToArray")
                  .Get<framework::LoDTensor>();
    auto &rank_table = GET_DATA_SAFELY(scope.FindVar(Input("RankTable")),
                                       "Input", "RankTable", "LoDTensorToArray")
                           .Get<framework::LoDRankTable>();
    auto &out = *(GET_DATA_SAFELY(scope.FindVar(Output("Out")), "Output", "Out",
                                  "LoDTensorToArray")
                      .GetMutable<framework::LoDTensorArray>());
    auto &items = rank_table.items();
    auto max_seq_len = items[0].length;
    auto rank_level = rank_table.level();

    PADDLE_ENFORCE_LT(
        rank_level, x.lod().size(),
        platform::errors::InvalidArgument(
            "Input should be a LoDTensor, and its lod_level should be at "
            "least %d, but given is %d.",
            rank_level + 1, x.lod().size()));
    out.resize(max_seq_len);
    std::vector<std::vector<CopyRange>> copy_ranges(max_seq_len);

    // set out[i] lod
    for (size_t t = 0; t < max_seq_len; t++) {
      auto &lod = *out[t].mutable_lod();
      lod.clear();
      for (auto &item : items) {
        if (t >= item.length) {
          break;
        }
        size_t start_idx = x.lod()[rank_level][item.index] + t;
        auto lod_and_offset = framework::GetSubLoDAndAbsoluteOffset(
            x.lod(), start_idx, start_idx + 1, rank_level + 1);
        auto &lod_length = lod_and_offset.first;
        framework::AppendLoD(&lod, lod_length);
        size_t start_offset = lod_and_offset.second.first;
        size_t end_offset = lod_and_offset.second.second;
        copy_ranges[t].emplace_back(CopyRange{start_offset, end_offset});
      }
    }

    std::map<size_t, framework::Tensor> outputs;

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
        if (len == 0) {
          continue;
        }
        // out[i][offset: offset+len] = x[each_range.begin: each_range.end]
        auto slice = out[i].Slice(static_cast<int>(offset),
                                  static_cast<int>(offset + len));
        outputs.insert({each_range.begin, slice});
        offset += len;
      }
    }

    LoDTensorToArrayFunctor functor(x);
    for (auto &out_pair : outputs) {
      functor.AddOutput(&out_pair.second);
    }
    platform::VisitPlace(place, functor);
  }
};

class LoDTensorToArrayOpProtoMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X",
             "(LoDTensor), the input lod tensor is a minibatch of sequences, "
             "and will be split to a tensor_array according to "
             "Input(RankTable).");
    AddInput("RankTable", "(LoDRankTable), the rank table.");
    AddOutput("Out",
              "(LoDTensorArray), the result tensor_array, which is actually a "
              "std::vector<LoDTensor>.");
    AddComment(R"DOC(LoDTensorToArray operator.
Input(X) is a minibatch of sequences. Input(RankTable) stores the order of the input sequences.
The lod_tensor_to_array operator will spilt the input sequences to a tensor_array, with each
element stores one sequence, according to the input rank_table.

NOTE: this operator is an internal component of DynamicRNN, and cannot be called by users.
)DOC");
  }
};

class LoDTensorToArrayInferShape : public framework::InferShapeBase {
 public:
  void operator()(framework::InferShapeContext *context) const override {
    PADDLE_ENFORCE_EQ(
        context->HasInput("X"), true,
        platform::errors::NotFound(
            "Input(X) of LoDTensorToArrayOp should not be null."));
    PADDLE_ENFORCE_EQ(
        context->HasInput("RankTable"), true,
        platform::errors::NotFound(
            "Input(RankTable) of LoDTensorToArrayOp should not be null."));

    PADDLE_ENFORCE_EQ(
        context->HasOutput("Out"), true,
        platform::errors::NotFound(
            "Output(Out) of LoDTensorToArrayOp should not be null."));

    auto x_dim = context->GetInputDim("X");
    // For compile-time, the first dim of input X and output Out should be -1.
    // For runtime, the first dim of input X should be the sum of all elements's
    // first dim in output Out. The output's dims will be re-computed in detail
    // kernel implementation.
    context->SetOutputDim("Out", x_dim);

    // The output LoDTensor's lod_level should be input X's lod_level - 1.
    // For compile time, we call SetLoDLevel to set output's lod_level.
    // For runtime, output LoDTensor's lod is determined by input X's lod and
    // the level specified by input RandTable.
    // We cannot get X's detail lod and RankTable's level in this function, so
    // leave this work to the detail kernel implementation.
    if (!context->IsRuntime()) {
      context->SetLoDLevel("Out", context->GetLoDLevel("X") - 1);
    }
  }
};

class LoDTensorToArrayInferVarType : public framework::VarTypeInference {
 public:
  void operator()(framework::InferVarTypeContext *ctx) const override {
    ctx->SetOutputType("Out", framework::proto::VarType::LOD_TENSOR_ARRAY,
                       framework::ALL_ELEMENTS);
  }
};

template <typename T>
class LoDTensorToArrayGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> grad_op) const override {
    grad_op->SetType("array_to_lod_tensor");
    grad_op->SetInput("X", this->OutputGrad("Out"));
    grad_op->SetInput("RankTable", this->Input("RankTable"));
    grad_op->SetOutput("Out", this->InputGrad("X"));
    grad_op->SetAttrMap(this->Attrs());
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(lod_tensor_to_array, ops::LoDTensorToArrayOp,
                  ops::LoDTensorToArrayOpProtoMaker,
                  ops::LoDTensorToArrayInferShape,
                  ops::LoDTensorToArrayInferVarType,
                  ops::LoDTensorToArrayGradMaker<paddle::framework::OpDesc>,
                  ops::LoDTensorToArrayGradMaker<paddle::imperative::OpBase>);
