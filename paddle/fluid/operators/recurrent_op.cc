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

#include "paddle/fluid/operators/recurrent_op.h"

namespace paddle {
namespace framework {
class InferShapeContext;
class LoDTensor;
class OpDesc;
}  // namespace framework
}  // namespace paddle

namespace paddle {
namespace operators {

using StepScopeVar = std::vector<framework::Scope *>;

const char RecurrentBase::kInputs[] = "inputs";
const char RecurrentBase::kInitialStates[] = "initial_states";
const char RecurrentBase::kParameters[] = "parameters";
const char RecurrentBase::kOutputs[] = "outputs";
const char RecurrentBase::kStepScopes[] = "step_scopes";
const char RecurrentBase::kHasStates[] = "has_states";
const char RecurrentBase::kExStates[] = "ex_states";
const char RecurrentBase::kStates[] = "states";
const char RecurrentBase::kStepBlock[] = "sub_block";
const char RecurrentBase::kReverse[] = "reverse";
const char RecurrentBase::kIsTrain[] = "is_train";
const char RecurrentBase::kSkipEagerDeletionVars[] = "skip_eager_deletion_vars";
#define GRAD_SUFFIX "@GRAD"
const char RecurrentBase::kInputGrads[] = "inputs" GRAD_SUFFIX;
const char RecurrentBase::kOutputGrads[] = "outputs" GRAD_SUFFIX;
const char RecurrentBase::kParamGrads[] = "parameters" GRAD_SUFFIX;
const char RecurrentBase::kInitStateGrads[] = "initial_states" GRAD_SUFFIX;

static void ClearStepScopes(const platform::DeviceContext &dev_ctx,
                            framework::Scope *parent_scope,
                            StepScopeVar *step_scopes) {
  if (step_scopes->empty()) return;

  dev_ctx.Wait();

  for (auto *sub_scope : *step_scopes) {
    if (parent_scope->HasKid(sub_scope)) {
      parent_scope->DeleteScope(sub_scope);
    }
  }

  step_scopes->clear();
}

StepScopes::StepScopes(const platform::DeviceContext &dev_ctx,
                       const framework::Scope &parent, StepScopeVar *scopes,
                       bool is_train, size_t seq_len, bool is_backward)
    : counter_(is_backward ? seq_len - 1 : 0UL),
      scopes_(scopes),
      is_train_(is_train),
      is_backward_(is_backward) {
  size_t num_step_scopes = is_train ? seq_len : 2;
  PADDLE_ENFORCE_EQ(is_train || !is_backward, true,
                    platform::errors::PreconditionNotMet(
                        "Cannot backward when is not training"));
  if (!is_backward_) {
    ClearStepScopes(dev_ctx, const_cast<framework::Scope *>(&parent), scopes);
    scopes->reserve(static_cast<size_t>(num_step_scopes));
    for (size_t i = 0; i < num_step_scopes; ++i) {
      scopes->emplace_back(&parent.NewScope());
    }
  }
}

framework::Scope &StepScopes::CurScope() { return GetScope(counter_); }

framework::Scope &StepScopes::ExScope() {
  auto &scope = GetScope(is_backward_ ? counter_ + 1 : counter_ - 1);
  return scope;
}

void StepScopes::BackwardNext(const platform::DeviceContext &dev_ctx,
                              framework::Scope *parent_scope) {
  PADDLE_ENFORCE_EQ(is_backward_, true,
                    platform::errors::PreconditionNotMet(
                        "Cannot get backward next scope when is forward"));
  if (counter_ + 2 == scopes_->size()) {
    parent_scope->DeleteScope((*scopes_)[counter_ + 1]);
    scopes_->pop_back();
    VLOG(3) << "Deleted scope at " << counter_ + 1;
  }
  --counter_;
}

void StepScopes::ForwardNext() {
  PADDLE_ENFORCE_EQ(is_backward_, false,
                    platform::errors::PreconditionNotMet(
                        "Cannot get forward next scope when is backward"));
  ++counter_;
}

framework::Scope &StepScopes::GetScope(size_t scope_id) const {
  if (!is_train_) {
    scope_id %= 2;
  }
  PADDLE_ENFORCE_LT(
      scope_id, scopes_->size(),
      platform::errors::InvalidArgument(
          "Input scope_id is greater than scopes size in RecurrentOp"));
  return *(*scopes_)[scope_id];
}

RecurrentBase::RecurrentBase(const std::string &type,
                             const framework::VariableNameMap &inputs,
                             const framework::VariableNameMap &outputs,
                             const framework::AttributeMap &attrs)
    : OperatorBase(type, inputs, outputs, attrs) {}

// Get SequenceLength from Scope
//   The sequence length is got from input tensor. The input tensor's
//   dimension should be [SEQ_LEN, ..., ...]. The first of the tensor's shape
//   is SEQ_LEN. The second of the tensor's shape could be the batch size or
//   nested sequence length.
int64_t RecurrentBase::GetSequenceLength(const framework::Scope &scope) const {
  // Dim format SEQ_LEN, BATCH_SIZE, ...
  int64_t seq_len = -1;
  auto &all_inputs = Inputs(kInputs);
  PADDLE_ENFORCE_EQ(
      all_inputs.empty(), false,
      platform::errors::InvalidArgument("RecurrentOp gets empty input"));
  for (auto &iname : all_inputs) {
    auto *var = scope.FindVar(iname);
    PADDLE_ENFORCE_NOT_NULL(var,
                            platform::errors::InvalidArgument(
                                "RecurrentOp finds var %s is NULL", iname));
    PADDLE_ENFORCE_EQ(var->IsType<framework::LoDTensor>(), true,
                      platform::errors::InvalidArgument(
                          "RecurrentOp only accepts LoDTensor as input but "
                          "input var %s is not LoDTensor",
                          iname));
    auto &dim = var->Get<framework::LoDTensor>().dims();
    if (seq_len == -1) {
      seq_len = dim[0];
    } else {
      PADDLE_ENFORCE_EQ(seq_len, dim[0],
                        platform::errors::InvalidArgument(
                            "Sequence length of input %s in RecurrentOp is NOT "
                            "equal to sequence length of previous input",
                            iname));
    }
  }
  PADDLE_ENFORCE_GE(seq_len, 0,
                    platform::errors::InvalidArgument(
                        "RecurrentOp gets invalid sequence length. Expected "
                        "seq_len >= 0. Received seq_len = %d",
                        seq_len));
  return seq_len;
}

// for src_tensor, dst_tensor in zip(map(src_scope.FindVar, src_vars),
//                                   map(dst_scope.Var, dst_vars)):
//   dst_tensor.ShareDataWith(src_tensor)
void RecurrentBase::LinkTensor(const framework::Scope &src_scope,
                               const std::vector<std::string> &src_vars,
                               framework::Scope *dst_scope,
                               const std::vector<std::string> &dst_vars) {
  LinkTensorWithCallback(
      src_scope, src_vars, dst_scope, dst_vars,
      [&](const framework::Tensor &src, framework::Tensor *dst) {
        dst->ShareDataWith(src);
      });
}

// (seq_len, shape) -> return [seq_len] + list(shape)
framework::DDim RecurrentBase::PrependDims(size_t seq_len,
                                           const framework::DDim &src) {
  auto dims = framework::vectorize(src);
  dims.insert(dims.begin(), static_cast<int64_t>(seq_len));
  return framework::make_ddim(dims);
}

RecurrentOp::RecurrentOp(const std::string &type,
                         const framework::VariableNameMap &inputs,
                         const framework::VariableNameMap &outputs,
                         const framework::AttributeMap &attrs)
    : RecurrentBase(type, inputs, outputs, attrs) {}

void RecurrentOp::RunImpl(const framework::Scope &scope,
                          const platform::Place &place) const {
  bool has_state = Attr<bool>(kHasStates);
  auto seq_len = static_cast<size_t>(this->GetSequenceLength(scope));

  // get device context from pool
  platform::DeviceContextPool &pool = platform::DeviceContextPool::Instance();
  auto &dev_ctx = *pool.Get(place);

  VLOG(3) << "Static RNN input sequence length = " << seq_len;
  auto reverse = Attr<bool>(kReverse);

  framework::Executor executor(place);
  auto *block = Attr<framework::BlockDesc *>(kStepBlock);

  auto *program = block->Program();
  auto ctx = executor.Prepare(*program, block->ID(),
                              Attr<std::vector<std::string>>(
                                  kSkipEagerDeletionVars), /*skip_ref_cnt_vars*/
                              true);

  static std::mutex mutex;
  std::lock_guard<std::mutex> lock(mutex);
  StepScopes scopes = CreateStepScopes(dev_ctx, scope, seq_len);
  // TODO(gfwm2013) Function CreateStepScopes would make segmentation fault in
  // multithreading in eval process, so we use a mutex before function
  // CreateStepScopes to make sure that the computing process is correct. This
  // problem will fix in next pull request.
  for (size_t i = 0; i < seq_len; ++i) {
    size_t seq_offset = reverse ? seq_len - i - 1 : i;
    VLOG(3) << "Recurrent operate at the time step " << seq_offset;

    auto &cur_scope = scopes.CurScope();

    // Link outside::input --> inside::input
    //   inside::input = outside::input[seq_offset: seq_offset+1]
    LinkTensorWithCallback(
        scope, Inputs(kInputs), &cur_scope, Inputs(kInputs),
        [&seq_offset](const framework::Tensor &outside,
                      framework::Tensor *inside) {
          inside->ShareDataWith(outside.Slice(seq_offset, seq_offset + 1));
          auto dims = framework::vectorize(inside->dims());
          dims.erase(dims.begin());
          inside->Resize(framework::make_ddim(dims));
        });

    if (has_state) {
      if (i == 0) {
        // Link initial states  --> ex_states
        LinkTensor(scope, Inputs(kInitialStates), &cur_scope,
                   Attr<std::vector<std::string>>(kExStates));
      } else {
        auto &ex_scope = scopes.ExScope();
        // Link ex_scope::state --> cur_scope::ex_state
        LinkTensor(ex_scope, Attr<std::vector<std::string>>(kStates),
                   &cur_scope, Attr<std::vector<std::string>>(kExStates));
      }
    }

    // Link inside::output -> outside::output
    //   outside::output[seq_offset: seq_offset + 1] = inside::output
    executor.CreateVariables(ctx->prog_, &cur_scope, ctx->block_id_);

    // Linked now, execute!
    executor.RunPreparedContext(ctx.get(), &cur_scope,
                                false /*create_local_scope*/,
                                false /*create_vars*/, true /* keep_kids */);
    if (i == 0) {
      LinkTensorWithCallback(
          cur_scope, Outputs(kOutputs), scope, Outputs(kOutputs),
          [&](const framework::LoDTensor &src_tensor,
              framework::LoDTensor *dst_tensor) {
            // create output tensor at begin
            dst_tensor->Resize(PrependDims(seq_len, src_tensor.dims()));
            dst_tensor->mutable_data(place, src_tensor.type());

            auto dst_out = dst_tensor->Slice(seq_offset, seq_offset + 1);
            // Explicit copy output since the local RNN scope can be destroyed
            // early.
            framework::TensorCopy(src_tensor, place, dev_ctx, &dst_out);
          });
    } else {
      LinkTensorWithCallback(
          cur_scope, Outputs(kOutputs), scope, Outputs(kOutputs),
          [&](const framework::LoDTensor &src_tensor,
              framework::LoDTensor *dst_tensor) {
            auto dst_out = dst_tensor->Slice(seq_offset, seq_offset + 1);
            framework::TensorCopy(src_tensor, place, dev_ctx, &dst_out);
          });
    }

    scopes.ForwardNext();
  }
}

StepScopes RecurrentOp::CreateStepScopes(const platform::DeviceContext &dev_ctx,
                                         const framework::Scope &scope,
                                         size_t seq_len) const {
  auto *var = scope.FindVar(Output(kStepScopes));
  PADDLE_ENFORCE_NOT_NULL(var, platform::errors::InvalidArgument(
                                   "RecurrentOp gets empty StepScopes var"));
  return StepScopes(dev_ctx, scope, var->GetMutable<StepScopeVar>(),
                    Attr<bool>(kIsTrain), seq_len);
}

RecurrentGradOp::RecurrentGradOp(const std::string &type,
                                 const framework::VariableNameMap &inputs,
                                 const framework::VariableNameMap &outputs,
                                 const framework::AttributeMap &attrs)
    : RecurrentBase(type, inputs, outputs, attrs) {}

void RecurrentGradOp::RunImpl(const framework::Scope &scope,
                              const platform::Place &place) const {
  bool has_state = Attr<bool>(kHasStates);
  const auto seq_len = static_cast<size_t>(GetSequenceLength(scope));

  // get device context from pool
  platform::DeviceContextPool &pool = platform::DeviceContextPool::Instance();
  auto &dev_ctx = *pool.Get(place);

  StepScopes scopes = CreateStepScopes(dev_ctx, scope, seq_len);
  auto reverse = Attr<bool>(kReverse);

  framework::Executor executor(place);
  auto *block = Attr<framework::BlockDesc *>(kStepBlock);
  auto *program = block->Program();
  auto ctx =
      executor.Prepare(*program, block->ID(),
                       Attr<std::vector<std::string>>(
                           kSkipEagerDeletionVars) /*skip_ref_cnt_vars*/);

  for (size_t step_id = 0; step_id < seq_len; ++step_id) {
    size_t seq_offset = reverse ? step_id : seq_len - step_id - 1;
    VLOG(3) << "Recurrent backward operate at the time step " << seq_offset;
    auto &cur_scope = scopes.CurScope();

    // Link outside::output_grads --> inside::output_grads
    //   inside::output_grad = outside::output_grad[seq_offset:seq_offset+1]
    LinkTensorWithCallback(
        scope, Inputs(kOutputGrads), &cur_scope, Inputs(kOutputGrads),
        [&](const framework::Tensor &outside, framework::Tensor *inside) {
          inside->ShareDataWith(outside.Slice(seq_offset, seq_offset + 1));
          auto dims = framework::vectorize(inside->dims());
          dims.erase(dims.begin());
          inside->Resize(framework::make_ddim(dims));
        },
        true /*is_backward*/);
    auto og_set = List2Set(Inputs(kOutputGrads));

    if (VLOG_IS_ON(10)) {
      std::ostringstream sout;
      std::copy(og_set.begin(), og_set.end(),
                std::ostream_iterator<std::string>(sout, ","));
      VLOG(10) << " RNN output gradients = [" << sout.str() << "]";
    }

    if (has_state) {
      // Link states
      //   if cur_scope::cur_state_grad in out_grads:
      //     cur_scope::cur_state_grad += ex_scope::ex_state_grad
      //   else:
      //     ex_scope::ex_state_grad --> cur_scope::cur_state_grad
      if (step_id != 0) {  // not at beginning
        auto &ex_scope = scopes.ExScope();
        auto ex_state_grads =
            GradVarLists(Attr<std::vector<std::string>>(kExStates));
        auto cur_state_grads =
            GradVarLists(Attr<std::vector<std::string>>(kStates));

        PADDLE_ENFORCE_EQ(ex_state_grads.size(), cur_state_grads.size(),
                          platform::errors::InvalidArgument(
                              "lengths of ex_states and cur_states are not "
                              "equal in RecurrentGradOp"));
        for (size_t i = 0; i < ex_state_grads.size(); ++i) {
          auto &cur_grad = cur_state_grads[i];
          auto &ex_grad = ex_state_grads[i];
          auto &ex_grad_tensor =
              ex_scope.FindVar(ex_grad)->Get<framework::LoDTensor>();

          VLOG(10) << " RNN link " << cur_grad << " from " << ex_grad;
          auto *cur_grad_var = cur_scope.Var(cur_grad);
          auto *cur_grad_tensor =
              cur_grad_var->GetMutable<framework::LoDTensor>();
          cur_grad_tensor->ShareDataWith(ex_grad_tensor);
        }
      }
    }

    // Link inside::output -> outside::output
    //   outside::output[seq_offset: seq_offset + 1] = inside::output
    executor.CreateVariables(ctx->prog_, &cur_scope, ctx->block_id_);
    if (step_id > 0) {
      LinkTensorWithCallback(
          scope, Outputs(kInputGrads), cur_scope, GradVarLists(Inputs(kInputs)),
          [&](const framework::LoDTensor &src_tensor,
              framework::LoDTensor *dst_tensor) {
            if (src_tensor.memory_size() ==
                0) {  // Inside Gradient is not created.
              return;
            }
            framework::Tensor src_slice =
                src_tensor.Slice(seq_offset, seq_offset + 1);
            dst_tensor->ShareDataWith(src_slice);
          },
          true /*is_backward*/);
    }

    VLOG(5) << "Recurrent memory linking finished ";
    // Run step block with cur_scope
    executor.RunPreparedContext(ctx.get(), &cur_scope,
                                false /*create_local_scope*/,
                                false /*create_vars*/, true /* keep_kids */);

    VLOG(5) << "executor.Run finished ";

    auto local_var_names = LocalVarNames(cur_scope);

    // Accumulate params
    //   if (step == 0):
    //      outside::param_grad = 0.0
    //   outside::param_grad += inside::param_grad
    {
      auto &pg_names = Outputs(kParamGrads);
      auto &p_names = Inputs(kParameters);
      PADDLE_ENFORCE_EQ(pg_names.size(), p_names.size(),
                        platform::errors::InvalidArgument(
                            "Sizes of Parameters and ParamGrads are not equal "
                            "in RecurrentGradOp"));

      for (size_t param_id = 0; param_id < pg_names.size(); ++param_id) {
        auto inside_grad_name = framework::GradVarName(p_names[param_id]);

        // If does not compute gradient of that variable inside rnn, just
        // continue
        if (local_var_names.find(inside_grad_name) == local_var_names.end()) {
          continue;
        }

        // zero gradient variable in step 0
        if (step_id == 0) {
          auto &inside_tensor =
              cur_scope.FindVar(inside_grad_name)->Get<framework::LoDTensor>();
          framework::AttributeMap attrs;
          attrs["dtype"] = inside_tensor.type();
          attrs["shape"] = framework::vectorize<int>(inside_tensor.dims());
          attrs["value"] = 0.0f;

          auto zero_op = framework::OpRegistry::CreateOp(
              "fill_constant", framework::VariableNameMap{},
              {{"Out", {pg_names[param_id]}}}, attrs);
          zero_op->Run(scope, place);
        }

        auto new_inside_name = cur_scope.Rename(inside_grad_name);

        // sum gradient
        auto sum_op = framework::OpRegistry::CreateOp(
            "sum", {{"X", {pg_names[param_id], new_inside_name}}},
            {{"Out", {pg_names[param_id]}}},
            framework::AttributeMap{{"use_mkldnn", {false}}});
        sum_op->Run(cur_scope, place);

        cur_scope.Rename(new_inside_name, inside_grad_name);
      }
    }
    VLOG(5) << "Accumulate Parameter finished ";

    // Copy input gradient from inside to outside
    //   outside::input_grad[seq_offset: seq_offset + 1] = inside::input_grad
    if (step_id == 0) {
      LinkTensorWithCallback(
          cur_scope, GradVarLists(Inputs(kInputs)), scope, Outputs(kInputGrads),
          [&](const framework::LoDTensor &inside,
              framework::LoDTensor *outside) {
            if (inside.memory_size() == 0) {  // IG is not created.
              return;
            }
            // Alloc outside memory
            outside->Resize(PrependDims(seq_len, inside.dims()));
            outside->mutable_data(place, inside.type());

            auto dst = outside->Slice(seq_offset, seq_offset + 1);
            framework::TensorCopy(inside, place, dev_ctx, &dst);
          },
          true /*is_backward*/);
    }
    VLOG(5) << "Link outside gradient finished ";

    if (has_state) {
      if (step_id + 1 == seq_len) {  // at_end
        // copy initialize states gradient from inside to outside
        LinkTensorWithCallback(
            cur_scope, GradVarLists(Attr<std::vector<std::string>>(kExStates)),
            scope, Outputs(kInitStateGrads),
            [&](const framework::LoDTensor &inside,
                framework::LoDTensor *outside) {
              outside->Resize(inside.dims());
              outside->mutable_data(place, inside.type());
              framework::TensorCopy(inside, place, dev_ctx, outside);
            },
            true /*is_backward*/);
        VLOG(5) << "Link initialize state gradient finished ";
      }
    }
    scopes.BackwardNext(dev_ctx, const_cast<framework::Scope *>(&scope));
  }
  // Delete the scope of StepScopes
  auto *var = scope.FindVar(Input(kStepScopes));
  PADDLE_ENFORCE_NOT_NULL(var,
                          platform::errors::InvalidArgument(
                              "StepScopes var is empty in RecurrentGradOp"));
  auto *step_scopes = var->GetMutable<StepScopeVar>();
  ClearStepScopes(dev_ctx, const_cast<framework::Scope *>(&scope), step_scopes);
}

StepScopes RecurrentGradOp::CreateStepScopes(
    const platform::DeviceContext &dev_ctx, const framework::Scope &scope,
    size_t seq_len) const {
  auto *var = scope.FindVar(Input(kStepScopes));
  PADDLE_ENFORCE_NOT_NULL(var,
                          platform::errors::InvalidArgument(
                              "StepScopes var is empty in RecurrentGradOp"));
  return StepScopes(dev_ctx, scope, var->GetMutable<StepScopeVar>(),
                    Attr<bool>(kIsTrain), seq_len, true /*is_backward*/);
}

std::unordered_set<std::string> RecurrentGradOp::List2Set(
    const std::vector<std::string> &list) const {
  std::unordered_set<std::string> local_var_name_set;
  local_var_name_set.reserve(list.size());
  for (auto &each : list) {
    local_var_name_set.insert(each);
  }
  return local_var_name_set;
}

std::unordered_set<std::string> RecurrentGradOp::LocalVarNames(
    const framework::Scope &scope) const {
  return this->List2Set(scope.LocalVarNames());
}

std::vector<std::string> RecurrentGradOp::GradVarLists(
    const std::vector<std::string> &var_names) {
  std::vector<std::string> retv;
  retv.reserve(var_names.size());
  std::transform(var_names.begin(), var_names.end(), std::back_inserter(retv),
                 framework::GradVarName);
  return retv;
}

class RecurrentOpProtoMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput(RecurrentBase::kInputs, "rnn inputs").AsDuplicable();
    AddInput(RecurrentBase::kInitialStates, "rnn initial states")
        .AsDuplicable();
    AddInput(RecurrentBase::kParameters,
             "Parameters are used by step block as its input. However, the "
             "input is not a sequence tensor. Every time step, each operator "
             "in step block just use the parameter directly.")
        .AsDuplicable();
    AddOutput(RecurrentBase::kOutputs,
              "The output sequence of RNN. The sequence length must be same.")
        .AsDuplicable();
    AddOutput(RecurrentBase::kStepScopes,
              "StepScopes contain all local variables in each time step.");
    AddAttr<bool>(RecurrentBase::kHasStates, "Whether has states.")
        .SetDefault(false);
    AddAttr<std::vector<std::string>>(
        RecurrentBase::kExStates,
        string::Sprintf(
            R"DOC(The ex-state variable names.
The ex-state means the state value in the ex-timestep or the previous time step
[%s, %s, %s] must be the same order)DOC",
            RecurrentBase::kExStates, RecurrentBase::kStates,
            RecurrentBase::kInitStateGrads));
    AddAttr<std::vector<std::string>>(
        RecurrentBase::kStates,
        string::Sprintf(
            "The state variable names. [%s, %s, %s] must be the same order",
            RecurrentBase::kExStates, RecurrentBase::kStates,
            RecurrentBase::kInitStateGrads));
    AddAttr<framework::BlockDesc *>(RecurrentBase::kStepBlock,
                                    "The step block inside RNN");
    AddAttr<bool>(RecurrentBase::kReverse, R"DOC(Calculate RNN reversely or not.
By default reverse=False

Assume the input data is [A, B, C, D]

if reverse is False:
  the computation of RNN is like
      A          B          C         D
      |          |          |         |
      v          v          v         v
     rnn -----> rnn -----> rnn ----> rnn
      |          |          |         |
      v          v          v         v
      o          o          o         o

if reverse is True
  the computation of RNN is like
      A          B          C         D
      |          |          |         |
      v          v          v         v
     rnn <----- rnn <----- rnn <---- rnn
      |          |          |         |
      v          v          v         v
      o          o          o         o
)DOC")
        .SetDefault(false);
    AddAttr<bool>(RecurrentBase::kIsTrain, "").SetDefault(true);
    AddAttr<std::vector<std::string>>(RecurrentBase::kSkipEagerDeletionVars,
                                      "Vars that would skip eager deletion."
                                      "Users should not set this manually.")
        .SetDefault(std::vector<std::string>());

    AddComment(R"DOC(
Static Length Recurrent Operator.

The static length recurrent operator can only operate on fixed size sequence
data, i.e. in each mini-batch, the sequence length of all inputs are the same.

)DOC");
  }
};

template <typename T>
class RecurrentGradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> grad) const override {
    grad->SetType("recurrent_grad");
    for (auto &input_param : this->InputNames()) {
      grad->SetInput(input_param, this->Input(input_param));
      grad->SetOutput(framework::GradVarName(input_param),
                      this->InputGrad(input_param, false));
    }

    for (auto &output_param : this->OutputNames()) {
      if (output_param == RecurrentBase::kStepScopes) {
        grad->SetInput(output_param, this->Output(output_param));
        grad->SetInput(framework::GradVarName(output_param),
                       this->Output(output_param));
      } else {
        grad->SetInput(output_param, this->Output(output_param));
        grad->SetInput(framework::GradVarName(output_param),
                       this->OutputGrad(output_param));
      }
    }
    grad->SetAttrMap(this->Attrs());
    grad->SetBlockAttr(RecurrentBase::kStepBlock, this->grad_block_[0]);
  }
};

class RecurrentGradOpShapeInference : public framework::InferShapeBase {
 public:
  void operator()(framework::InferShapeContext *ctx) const override {
    std::vector<std::string> output{RecurrentBase::kOutputs};

    // In some case the kInitialStates is empty.
    // If the kInitialStates is empty, all the states should be empty.
    if (!ctx->HasInputs(RecurrentBase::kInitialStates)) {
      PADDLE_ENFORCE_EQ(
          ctx->Attrs()
              .Get<std::vector<std::string>>(RecurrentBase::kExStates)
              .size(),
          0,
          platform::errors::InvalidArgument("The Attr(%s) should be empty.",
                                            RecurrentBase::kExStates));
      PADDLE_ENFORCE_EQ(
          ctx->Attrs()
              .Get<std::vector<std::string>>(RecurrentBase::kStates)
              .size(),
          0,
          platform::errors::InvalidArgument("The Attr(%s) should be empty.",
                                            RecurrentBase::kStates));
    }

    PADDLE_ENFORCE_EQ(
        ctx->HasInputs(RecurrentBase::kInputs), true,
        platform::errors::InvalidArgument("The input(%s) should not be empty.",
                                          RecurrentBase::kInputs));
    PADDLE_ENFORCE_EQ(
        ctx->HasInputs(RecurrentBase::kOutputs), true,
        platform::errors::InvalidArgument("The input(%s) should not be empty.",
                                          RecurrentBase::kOutputs));

    // In some case the kInitialStates is empty.
    if (ctx->HasInputs(RecurrentBase::kInitialStates) &&
        ctx->HasOutputs(
            framework::GradVarName(RecurrentBase::kInitialStates))) {
      ctx->SetOutputsDim(framework::GradVarName(RecurrentBase::kInitialStates),
                         ctx->GetInputsDim(RecurrentBase::kInitialStates));
    }

    PADDLE_ENFORCE_EQ(
        ctx->HasOutputs(framework::GradVarName(RecurrentBase::kInputs)), true,
        platform::errors::InvalidArgument(
            "The output of(%s) should not be empty.",
            framework::GradVarName(RecurrentBase::kInputs)));
    ctx->SetOutputsDim(framework::GradVarName(RecurrentBase::kInputs),
                       ctx->GetInputsDim(RecurrentBase::kInputs));

    // In some case the kParameters is empty.
    if (ctx->HasInputs(RecurrentBase::kParameters)) {
      PADDLE_ENFORCE_EQ(
          ctx->HasOutputs(framework::GradVarName(RecurrentBase::kParameters)),
          true,
          platform::errors::InvalidArgument(
              "The output of(%s) should not be empty.",
              framework::GradVarName(RecurrentBase::kParameters)));
      ctx->SetOutputsDim(framework::GradVarName(RecurrentBase::kParameters),
                         ctx->GetInputsDim(RecurrentBase::kParameters));
    }
  }
};

}  // namespace operators
}  // namespace paddle

REGISTER_OPERATOR(
    recurrent, paddle::operators::RecurrentOp,
    paddle::operators::RecurrentOpProtoMaker,
    paddle::operators::RecurrentGradOpMaker<paddle::framework::OpDesc>);
REGISTER_OPERATOR(recurrent_grad, paddle::operators::RecurrentGradOp,
                  paddle::operators::RecurrentGradOpShapeInference);
