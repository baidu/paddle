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

#include "paddle/fluid/framework/executor.h"
#include <memory>
#include "paddle/fluid/framework/feed_fetch_method.h"
#include "paddle/fluid/framework/trainer_desc.pb.h"
#include "paddle/fluid/framework/trainer_factory.h"
#include "paddle/fluid/operators/controlflow/conditional_block_op_helper.h"
#include "paddle/fluid/operators/controlflow/recurrent_op_helper.h"
#include "paddle/fluid/operators/controlflow/while_op_helper.h"
#include "paddle/fluid/platform/place.h"
#include "paddle/fluid/platform/profiler.h"
#ifdef PADDLE_WITH_MKLDNN
#include "paddle/fluid/platform/mkldnn_helper.h"
#endif
#include "paddle/fluid/framework/executor_gc_helper.h"

DECLARE_bool(benchmark);
DECLARE_bool(use_mkldnn);

namespace paddle {
namespace framework {
namespace {
// block id starts from 0. This id is used to represent the codeblock
// wrapping the first block 0.
int kProgramId = -1;
}  // namespace

ExecutorPrepareContext::ExecutorPrepareContext(
    const framework::ProgramDesc& prog, size_t block_id)
    : prog_(prog), block_id_(block_id) {}

void ExecutorPrepareContext::PrepareUnusedVars(
    const std::vector<std::string>& keep_vars, bool force_disable_gc) {
  // If gc is enabled and block size > 1
  if (prog_.Size() > 1) {
    operators::PrepareSafeEagerDeletionOnConditionalOpAndConditionalGradOp(
        prog_, block_id_, ops_);
    operators::PrepareSafeEagerDeletionOnWhileOpAndWhileGradOp(prog_, block_id_,
                                                               ops_);
    operators::PrepareSafeEagerDeletionOnRecurrentOpAndRecurrentGradOp(
        prog_, block_id_, ops_);
  }

  force_disable_gc_ = force_disable_gc;
  if (GetEagerDeletionThreshold() < 0 || force_disable_gc_) {
    return;
  }

  unused_vars_ = GetUnusedVars(prog_.Block(block_id_), ops_, keep_vars);
}

ExecutorPrepareContext::~ExecutorPrepareContext() {
  VLOG(5) << "destroy ExecutorPrepareContext";
}

Executor::Executor(const platform::Place& place) : place_(place) {
  if (platform::is_npu_place(place)) {
#ifndef PADDLE_WITH_ASCEND_CL
    PADDLE_THROW(platform::errors::Unavailable(
        "Cannot run operator on place %s, please recompile paddle or "
        "reinstall Paddle with NPU support.",
        place));
#else
    auto dev_id = BOOST_GET_CONST(platform::NPUPlace, place).device;
    platform::SetNPUDeviceId(dev_id);
#endif
  }
}

Executor::~Executor() {
#ifdef PADDLE_WITH_MKLDNN
  // Clear mkl-dnn cache,
  // this is needed to have mkl-dnn unit tests working
  ClearMKLDNNCache(place_, this);
#endif
}

void Executor::Close() {
  // #ifdef PADDLE_WITH_DISTRIBUTE
  //   // TODO(typhoonzero): complete message will need to use real trainer_id,
  //   // except 0.
  //   auto client =
  //       paddle::operators::distributed::RPCClient::GetInstance<RPCCLIENT_T>(0);
  //   client->SendComplete();
  // #endif
}

void Executor::CreateVariables(const ProgramDesc& pdesc, Scope* scope,
                               int block_id) {
  VLOG(3) << "Creating Variables for block " << block_id;
  auto& global_block = pdesc.Block(block_id);
  const Scope* ancestor_scope = scope;
  while (ancestor_scope->parent()) {
    ancestor_scope = ancestor_scope->parent();
  }
  if (ancestor_scope != scope) {
    for (auto& var : global_block.AllVars()) {
      if (var->Name() == framework::kEmptyVarName) {
        continue;
      }

      if (var->Persistable()) {
        auto* ptr = const_cast<Scope*>(ancestor_scope)->Var(var->Name());
        InitializeVariable(ptr, var->GetType());
        VLOG(3) << "Create Variable " << var->Name()
                << " global, which pointer is " << ptr;
      } else {
        auto* ptr = scope->Var(var->Name());
        InitializeVariable(ptr, var->GetType());
        VLOG(3) << "Create Variable " << var->Name()
                << " locally, which pointer is " << ptr;
      }
    }
  } else {
    for (auto& var : global_block.AllVars()) {
      auto* ptr = scope->Var(var->Name());
      InitializeVariable(ptr, var->GetType());
      VLOG(3) << "Create variable " << var->Name() << ", which pointer is "
              << ptr;
    }
  }
}

std::shared_ptr<TrainerBase> Executor::InitForDataset(
    const ProgramDesc& main_program, const std::string& trainer_desc_str,
    Scope* scope, Dataset* dataset) {
  VLOG(3) << "Start to RunFromDataset in executor";
  TrainerDesc trainer_desc;
  bool success = trainer_desc.ParseFromString(trainer_desc_str);
  PADDLE_ENFORCE_EQ(success, true,
                    platform::errors::PreconditionNotMet(
                        "Fail to parse TrainerDesc from string:\n%s",
                        trainer_desc_str.c_str()));
  VLOG(3) << "Going to create trainer, trainer class is "
          << trainer_desc.class_name();
  std::shared_ptr<TrainerBase> trainer;
  trainer = TrainerFactory::CreateTrainer(trainer_desc.class_name());
  // initialize trainer
  VLOG(3) << "Going to initialize trainer";
  trainer->Initialize(trainer_desc, dataset);
  VLOG(3) << "Set root scope here";
  trainer->SetScope(scope);
  // prepare training environment and helper environment
  VLOG(3) << "Try to init train environment";
  trainer->InitTrainerEnv(main_program, place_);
  VLOG(3) << "Try to init other environment";
  trainer->InitOtherEnv(main_program);
  return trainer;
}

void Executor::RunFromDataset(std::shared_ptr<TrainerBase> trainer) {
  PADDLE_ENFORCE_NOT_NULL(
      trainer, platform::errors::InvalidArgument(
                   "Trainer is nullptr, invoke InitForDataset first"));
  // training and finalize training
  VLOG(3) << "Trainer starts to run";
  trainer->Run();
}

void Executor::ReleaseTrainer(std::shared_ptr<TrainerBase> trainer) {
  VLOG(3) << "Trainer going to finalize";
  trainer->Finalize();
}

void Executor::Run(const ProgramDesc& pdesc, Scope* scope, int block_id,
                   bool create_local_scope, bool create_vars,
                   const std::vector<std::string>& skip_ref_cnt_vars,
                   bool force_disable_gc, bool keep_kid_scopes) {
  platform::RecordBlock b(block_id);
  if (FLAGS_use_mkldnn) EnableMKLDNN(pdesc);
#ifdef PADDLE_WITH_MKLDNN
  platform::AttachPointerHashToMKLDNNKey(this, place_);
#endif
  auto ctx = Prepare(pdesc, block_id, skip_ref_cnt_vars, force_disable_gc);
  RunPreparedContext(ctx.get(), scope, create_local_scope, create_vars,
                     keep_kid_scopes);
}

// Check whether the block already has feed operators and feed_holder.
// Return false if the block does not have any feed operators.
// If some feed operators have been prepended to the block, check that
// the info contained in these feed operators matches the feed_targets
// and feed_holder_name. Raise exception when any mismatch is found.
// Return true if the block has feed operators and holder of matching info.
static bool has_feed_operators(
    const BlockDesc& block,
    const std::map<std::string, const LoDTensor*>& feed_targets,
    const std::string& feed_holder_name) {
  size_t feed_count = 0;
  for (auto* op : block.AllOps()) {
    if (op->Type() == kFeedOpType) {
      feed_count++;
      // The input variable's name of feed_op should be feed_holder_name.
      PADDLE_ENFORCE_EQ(
          op->Input("X")[0], feed_holder_name,
          platform::errors::PreconditionNotMet(
              "Input to feed op should be '%s', but received '%s'.",
              feed_holder_name, op->Input("X")[0]));
      std::string feed_target_name = op->Output("Out")[0];
      PADDLE_ENFORCE_NE(feed_targets.find(feed_target_name), feed_targets.end(),
                        platform::errors::PreconditionNotMet(
                            "Feed operator output name '%s' cannot be found in "
                            "'feed_targets'",
                            feed_target_name));
    }
  }

  if (feed_count > 0) {
    PADDLE_ENFORCE_EQ(
        feed_count, feed_targets.size(),
        platform::errors::PreconditionNotMet(
            "The number of feed operators should match 'feed_targets', but "
            "received feed_count: %zu, required feed_targets.size(): %zu.",
            feed_count, feed_targets.size()));

    if (!feed_holder_name.empty()) {
      // When feed operator are present, so should be feed_holder.
      auto var = block.FindVar(feed_holder_name);
      PADDLE_ENFORCE_NOT_NULL(
          var,
          platform::errors::PreconditionNotMet(
              "Block should already have a '%s' variable", feed_holder_name));
      PADDLE_ENFORCE_EQ(
          var->GetType(), proto::VarType::FEED_MINIBATCH,
          platform::errors::PreconditionNotMet(
              "'%s' variable should be 'FEED_MINIBATCH' type, but received "
              "'%s'.",
              feed_holder_name, DataTypeToString(var->GetType())));
    }
  }

  return feed_count > 0;
}

// Check whether the block already has fetch operators and fetch_holder.
// Return false if the block does not have any fetch operators.
// If some fetch operators have been appended to the block, check that
// the info contained in these fetch operators matches the fetch_targets
// and fetch_holder_name. Raise exception when any mismatch is found.
// Return true if the block has fetch operators and holder of matching info.
static bool has_fetch_operators(
    const BlockDesc& block,
    const std::map<std::string, FetchType*>& fetch_targets,
    const std::string& fetch_holder_name) {
  size_t fetch_count = 0;
  for (auto* op : block.AllOps()) {
    if (op->Type() == kFetchOpType) {
      fetch_count++;
      // The output variable's name of fetch_op should be fetch_holder_name.
      PADDLE_ENFORCE_EQ(
          op->Output("Out")[0], fetch_holder_name,
          platform::errors::PreconditionNotMet(
              "Output of fetch op should be '%s', but received '%s'.",
              fetch_holder_name, op->Output("Out")[0]));
      std::string fetch_target_name = op->Input("X")[0];
      PADDLE_ENFORCE_NE(fetch_targets.find(fetch_target_name),
                        fetch_targets.end(),
                        platform::errors::NotFound(
                            "Fetch operator input name '%s' cannot be found in "
                            "'fetch_targets'.",
                            fetch_target_name));
    }
  }

  if (fetch_count > 0) {
    PADDLE_ENFORCE_EQ(
        fetch_count, fetch_targets.size(),
        platform::errors::PreconditionNotMet(
            "The number of fetch operators should match 'fetch_targets', but "
            "received fetch_count: %zu, required fetch_targets.size(): %zu.",
            fetch_count, fetch_targets.size()));

    if (!fetch_holder_name.empty()) {
      // When fetch operator are present, so should be fetch_holder.
      auto var = block.FindVar(fetch_holder_name);
      PADDLE_ENFORCE_NOT_NULL(
          var,
          platform::errors::PreconditionNotMet(
              "Block should already have a '%s' variable.", fetch_holder_name));
      PADDLE_ENFORCE_EQ(
          var->GetType(), proto::VarType::FETCH_LIST,
          platform::errors::PreconditionNotMet(
              "'%s' variable should be 'FETCH_LIST' type, but received '%s'.",
              fetch_holder_name, DataTypeToString(var->GetType())));
    }
  }

  return fetch_count > 0;
}

void Executor::Run(const ProgramDesc& program, Scope* scope,
                   std::map<std::string, const LoDTensor*>* feed_targets,
                   std::map<std::string, FetchType*>* fetch_targets,
                   bool create_local_scope, bool create_vars,
                   const std::string& feed_holder_name,
                   const std::string& fetch_holder_name) {
  platform::RecordBlock b(kProgramId);
  if (FLAGS_use_mkldnn) EnableMKLDNN(program);
#ifdef PADDLE_WITH_MKLDNN
  platform::AttachPointerHashToMKLDNNKey(this, place_);
#endif
  bool has_feed_ops =
      has_feed_operators(program.Block(0), *feed_targets, feed_holder_name);
  bool has_fetch_ops =
      has_fetch_operators(program.Block(0), *fetch_targets, fetch_holder_name);

  ProgramDesc* copy_program = const_cast<ProgramDesc*>(&program);
  std::unique_ptr<ProgramDesc> unique_ptr_of_copy_program;
  if (!has_feed_ops || !has_fetch_ops) {
    unique_ptr_of_copy_program.reset(new ProgramDesc(program));
    copy_program = unique_ptr_of_copy_program.get();
  }
  auto* global_block = copy_program->MutableBlock(0);

  if (!has_feed_ops) {
    // create feed_holder variable
    auto* feed_holder = global_block->Var(feed_holder_name);
    feed_holder->SetType(proto::VarType::FEED_MINIBATCH);
    feed_holder->SetPersistable(true);

    int i = 0;
    for (auto& feed_target : (*feed_targets)) {
      std::string var_name = feed_target.first;
      VLOG(3) << "feed target's name: " << var_name;

      // prepend feed op
      auto* op = global_block->PrependOp();
      op->SetType(kFeedOpType);
      op->SetInput("X", {feed_holder_name});
      op->SetOutput("Out", {var_name});
      op->SetAttr("col", {static_cast<int>(i)});
      op->CheckAttrs();

      i++;
    }
  }

  if (!has_fetch_ops) {
    // create fetch_holder variable
    auto* fetch_holder = global_block->Var(fetch_holder_name);
    fetch_holder->SetType(proto::VarType::FETCH_LIST);
    fetch_holder->SetPersistable(true);

    int i = 0;
    for (auto& fetch_target : (*fetch_targets)) {
      std::string var_name = fetch_target.first;
      VLOG(3) << "fetch target's name: " << var_name;

      // append fetch op
      auto* op = global_block->AppendOp();
      op->SetType(kFetchOpType);
      op->SetInput("X", {var_name});
      op->SetOutput("Out", {fetch_holder_name});
      op->SetAttr("col", {static_cast<int>(i)});
      op->CheckAttrs();

      i++;
    }
  }

  auto ctx = Prepare(*copy_program, 0);
  RunPreparedContext(ctx.get(), scope, feed_targets, fetch_targets,
                     create_local_scope, create_vars, feed_holder_name,
                     fetch_holder_name);
}

std::unique_ptr<ExecutorPrepareContext> Executor::Prepare(
    const ProgramDesc& program, int block_id,
    const std::vector<std::string>& skip_ref_cnt_vars, bool force_disable_gc) {
  std::unique_ptr<ExecutorPrepareContext> ctx(
      new ExecutorPrepareContext(program, block_id));
  PADDLE_ENFORCE_LT(static_cast<size_t>(block_id), program.Size(),
                    platform::errors::InvalidArgument(
                        "Input block id = %d, but it should be less than "
                        "program.size() which is %d",
                        static_cast<size_t>(block_id), program.Size()));
  auto& block = program.Block(block_id);
  for (auto& op_desc : block.AllOps()) {
    ctx->ops_.push_back(OpRegistry::CreateOp(*op_desc));
  }
  ctx->PrepareUnusedVars(skip_ref_cnt_vars, force_disable_gc);
  return ctx;
}

std::vector<std::shared_ptr<ExecutorPrepareContext>> Executor::Prepare(
    const ProgramDesc& program, const std::vector<int>& block_ids,
    const std::vector<std::vector<std::string>>& skip_ref_cnt_vars,
    bool force_disable_gc) {
  PADDLE_ENFORCE_EQ(
      skip_ref_cnt_vars.empty() || skip_ref_cnt_vars.size() == block_ids.size(),
      true,
      platform::errors::InvalidArgument("skip_ref_cnt_vars should be either "
                                        "empty or equals to block number %d",
                                        block_ids.size()));
  std::vector<std::shared_ptr<ExecutorPrepareContext>> result;
  size_t idx = 0;
  for (auto& bid : block_ids) {
    PADDLE_ENFORCE_LT(static_cast<size_t>(bid), program.Size(),
                      platform::errors::InvalidArgument(
                          "Input block id = %zu, but it should be less than "
                          "program.size() which is %zu",
                          static_cast<size_t>(bid), program.Size()));
    auto* ctx = new ExecutorPrepareContext(program, bid);
    auto& block = program.Block(bid);
    for (auto& op_desc : block.AllOps()) {
      ctx->ops_.push_back(OpRegistry::CreateOp(*op_desc));
    }
    if (skip_ref_cnt_vars.empty()) {
      ctx->PrepareUnusedVars(std::vector<std::string>(), force_disable_gc);
    } else {
      ctx->PrepareUnusedVars(skip_ref_cnt_vars[idx], force_disable_gc);
    }
    result.push_back(std::shared_ptr<ExecutorPrepareContext>(ctx));
    ++idx;
  }
  return result;
}

void Executor::RunPartialPreparedContext(ExecutorPrepareContext* ctx,
                                         Scope* scope, int64_t start_op_index,
                                         int64_t end_op_index,
                                         bool create_local_scope,
                                         bool create_vars, bool keep_kids) {
  platform::RecordBlock b(kProgramId);
  PADDLE_ENFORCE_NOT_NULL(
      scope, platform::errors::InvalidArgument("Scope shouldn't be null"));
  Scope* local_scope = scope;
  if (create_vars) {
    if (create_local_scope) {
      local_scope = &scope->NewScope();
    }
    CreateVariables(ctx->prog_, local_scope, ctx->block_id_);
  }

  int64_t max_memory_size = GetEagerDeletionThreshold();
  std::unique_ptr<GarbageCollector> gc;
  if (!ctx->force_disable_gc_ && max_memory_size >= 0) {
    if (platform::is_gpu_place(place_)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      if (IsFastEagerDeletionModeEnabled()) {
        gc.reset(new UnsafeFastGPUGarbageCollector(
            BOOST_GET_CONST(platform::CUDAPlace, place_), max_memory_size));
      } else {
        gc.reset(new DefaultStreamGarbageCollector(
            BOOST_GET_CONST(platform::CUDAPlace, place_), max_memory_size));
      }
#else
      PADDLE_THROW(
          platform::errors::Unimplemented("No GPU gc found in CPU/XPU paddle"));
#endif
    } else if (platform::is_cpu_place(place_)) {
      gc.reset(new CPUGarbageCollector(
          BOOST_GET_CONST(platform::CPUPlace, place_), max_memory_size));
    } else if (platform::is_xpu_place(place_)) {
#ifdef PADDLE_WITH_XPU
      gc.reset(new XPUGarbageCollector(
          BOOST_GET_CONST(platform::XPUPlace, place_), max_memory_size));
#else
      PADDLE_THROW(
          platform::errors::Unimplemented("No XPU gc found in CPU/GPU paddle"));
#endif
    } else if (platform::is_npu_place(place_)) {
#ifdef PADDLE_WITH_ASCEND_CL
      if (IsFastEagerDeletionModeEnabled()) {
        VLOG(4) << "Use unsafe fast gc for NPU.";
        gc.reset(new NPUUnsafeFastGarbageCollector(
            BOOST_GET_CONST(platform::NPUPlace, place_), max_memory_size));
      } else {
        PADDLE_THROW(platform::errors::Unimplemented(
            "Please set FLAGS_fast_eager_deletion_mode=true to use "
            "GarbageCollector on NPU."));
        // TODO(zhiqiu): fix bugs and enable NPUDefaultStreamGarbageCollector.
        VLOG(4) << "Use default stream gc for NPU.";
        gc.reset(new NPUDefaultStreamGarbageCollector(
            BOOST_GET_CONST(platform::NPUPlace, place_), max_memory_size));
      }
#else
      PADDLE_THROW(
          platform::errors::Unimplemented("No NPU gc found in CPU/NPU paddle"));
#endif
    }
  }

  for (int64_t i = start_op_index; i < end_op_index; ++i) {
    auto& op = ctx->ops_[i];
    op->Run(*local_scope, place_);
    if (gc) {
      DeleteUnusedTensors(*local_scope, op.get(), ctx->unused_vars_, gc.get());
    }
  }

  auto callback = [scope, local_scope, keep_kids]() {
    if (local_scope != scope) {
      VLOG(4) << "Delete scope: " << local_scope;
      scope->DeleteScope(local_scope);
    } else {
      if (!keep_kids) {
        VLOG(4) << "Drop kids: " << scope;
        // By default, we should delete all kid scopes after run executor
        // because
        // some operators may create local scope when running, such as while_op.
        // But when while_op also create a local executor to run it's sub block,
        // the sub scopes it created should not be dropped immediately, because
        // while_grad_op will use some variables created during while_op run, so
        // we need to keep the kids and wait for the outer executor to drop
        // them.

        scope->DropKids();
      }
      VLOG(4) << "Keep kids: " << scope;
    }
  };

  if (gc) {
    VLOG(4) << "Async deleting scope";
    gc->DirectClearCallback(callback);
  } else {
    VLOG(4) << "Sync deleting scope";
    platform::DeviceContextPool::Instance().Get(place_)->Wait();
    callback();
  }
}

void Executor::RunPreparedContext(ExecutorPrepareContext* ctx, Scope* scope,
                                  bool create_local_scope, bool create_vars,
                                  bool keep_kids) {
  int64_t start_op_index = 0;
  int64_t end_op_index = ctx->ops_.size();
  RunPartialPreparedContext(ctx, scope, start_op_index, end_op_index,
                            create_local_scope, create_vars, keep_kids);
}

void Executor::RunPreparedContext(
    ExecutorPrepareContext* ctx, Scope* scope,
    std::map<std::string, const LoDTensor*>* feed_targets,
    std::map<std::string, FetchType*>* fetch_targets, bool create_local_scope,
    bool create_vars, const std::string& feed_holder_name,
    const std::string& fetch_holder_name) {
  auto& global_block = ctx->prog_.Block(ctx->block_id_);

  PADDLE_ENFORCE_EQ(
      has_feed_operators(global_block, *feed_targets, feed_holder_name), true,
      platform::errors::PreconditionNotMet(
          "Program in ExecutorPrepareContext should has feed_ops."));
  PADDLE_ENFORCE_EQ(
      has_fetch_operators(global_block, *fetch_targets, fetch_holder_name),
      true, platform::errors::PreconditionNotMet(
                "Program in the prepared context should has fetch_ops."));

  // map the data of feed_targets to feed_holder
  for (auto* op : global_block.AllOps()) {
    if (op->Type() == kFeedOpType) {
      std::string feed_target_name = op->Output("Out")[0];
      int idx = BOOST_GET_CONST(int, op->GetAttr("col"));
      SetFeedVariable(scope, *(*feed_targets)[feed_target_name],
                      feed_holder_name, idx);
    }
  }

  RunPreparedContext(ctx, scope, create_local_scope, create_vars);

  // obtain the data of fetch_targets from fetch_holder
  for (auto* op : global_block.AllOps()) {
    if (op->Type() == kFetchOpType) {
      std::string fetch_target_name = op->Input("X")[0];
      int idx = BOOST_GET_CONST(int, op->GetAttr("col"));
      *(*fetch_targets)[fetch_target_name] =
          GetFetchVariable(*scope, fetch_holder_name, idx);
    }
  }
}

void Executor::EnableMKLDNN(const ProgramDesc& program) {
#ifdef PADDLE_WITH_MKLDNN
  VLOG(3) << "use_mkldnn=True";
  for (size_t bid = 0; bid < program.Size(); ++bid) {
    auto* block = const_cast<ProgramDesc&>(program).MutableBlock(bid);
    for (auto* op : block->AllOps()) {
      if (op->HasAttr("use_mkldnn")) {
        op->SetAttr("use_mkldnn", true);
      }
    }
  }
#else
  LOG(WARNING)
      << "'MKLDNN' is not supported, Please re-compile with WITH_MKLDNN option";
#endif
}
}  // namespace framework
}  // namespace paddle
