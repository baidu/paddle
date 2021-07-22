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

#include "paddle/fluid/framework/details/build_strategy.h"

#include <glog/logging.h>
#include "paddle/fluid/framework/details/reduce_op_handle.h"
#include "paddle/fluid/framework/ir/graph_printer.h"
#include "paddle/fluid/framework/ir/multi_devices_graph_pass/multi_devices_graph_pass.h"

DECLARE_bool(use_mkldnn);
DECLARE_bool(convert_all_blocks);

namespace paddle {
namespace framework {
namespace details {

static inline bool SeqOnlyAllReduceOps(const BuildStrategy &strategy) {
  // Should fix the allreduce op order if scheduling
  // them in multiple threads or processes to avoid hang.
  // NOTE: ParallelGraph would execute this pass on each graph, so
  // don't need to append it here.
  return (!strategy.enable_sequential_execution_ &&
          strategy.num_trainers_ > 1) &&
         !strategy.enable_parallel_graph_;
}

static inline void ConvertDefaultValue(boost::optional<bool> *default_value) {
  if (*default_value == boost::none) {
    *default_value = true;
  }
}

class ParallelExecutorPassBuilder : public ir::PassBuilder {
 public:
  explicit ParallelExecutorPassBuilder(const BuildStrategy &strategy)
      : ir::PassBuilder(), strategy_(strategy) {
    ResolveOptionConfliction();

    AppendPrintGraphPass("graph_viz_pass", "_original_graph");
    AppendPassWithCheck(strategy_.enable_sequential_execution_,
                        "sequential_execution_pass");
    AppendPassWithCheck(strategy_.sync_batch_norm_, "sync_batch_norm_pass");

    AppendOpFusePasses();
    AppendPrintGraphPass("graph_viz_pass", "_fused_graph");

    AppendAddReaderDependencyPass();
    AppendMultiDevPass();
    AppendMultiGraphOptPasses();

    AppendPassToSetMkldnnAttr("mkldnn_placement_pass");
    // runtime_context_cache pass should be the last pass to enable the attr of
    // all original and fused operators. But no operators can be enabled this
    // attr if putting it after MultiDevPass.
    AppendPassWithCheck(strategy_.cache_runtime_context_,
                        "runtime_context_cache_pass");
    AppendPassWithCheck(strategy_.remove_unnecessary_lock_,
                        "modify_op_lock_and_record_event_pass");
    // Note: This pass is used to check whether the multi_device_graph is right.
    AppendPass("multi_devices_check_pass");

    SetCollectiveContext();
  }

  void ResolveOptionConfliction() {
    // Specifies the restrictions between different pass.
    if (strategy_.enable_parallel_graph_) {
      LOG_IF(WARNING, strategy_.fuse_all_optimizer_ops_ == true)
          << "Currently, fuse_all_optimizer_ops doesn't work under "
             "parallel_graph.";
      strategy_.fuse_all_optimizer_ops_ = false;
      LOG_IF(WARNING, strategy_.fuse_all_reduce_ops_ == true)
          << "fuse_all_reduce_ops doesn't work under "
             "parallel_graph.";
      strategy_.fuse_all_reduce_ops_ = false;
    }
    if (strategy_.is_distribution_) {
      LOG_IF(WARNING, strategy_.fuse_all_optimizer_ops_ == true)
          << "Currently, fuse_all_optimizer_ops only works under "
             "Non-distributed mode.";
      strategy_.fuse_all_optimizer_ops_ = false;
      LOG_IF(WARNING, strategy_.fuse_all_reduce_ops_ == true)
          << "Currently, fuse_all_reduce_ops_ only works under "
             "Non-distributed mode.";
      strategy_.fuse_all_reduce_ops_ = false;
    }
    if (strategy_.reduce_ == BuildStrategy::ReduceStrategy::kReduce) {
      LOG_IF(WARNING, strategy_.fuse_all_optimizer_ops_ == true)
          << "Currently, fuse_all_optimizer_ops only works under AllReduce "
             "mode.";
      strategy_.fuse_all_optimizer_ops_ = false;
      LOG_IF(WARNING, strategy_.fuse_all_reduce_ops_ == true)
          << "fuse_all_optimizer_ops only works under AllReduce "
             "mode.";
      strategy_.fuse_all_reduce_ops_ = false;
    }
    if (strategy_.reduce_ == BuildStrategy::ReduceStrategy::kAllReduce) {
      LOG_IF(WARNING, strategy_.fuse_broadcast_ops_ == true)
          << "Currently, fuse_broadcast_ops only works under Reduce "
             "mode.";
      strategy_.fuse_broadcast_ops_ = false;
    }

    ConvertDefaultValue(&strategy_.fuse_all_optimizer_ops_);
    ConvertDefaultValue(&strategy_.fuse_all_reduce_ops_);
    ConvertDefaultValue(&strategy_.fuse_broadcast_ops_);

    if (strategy_.fuse_all_optimizer_ops_ == true) {
      LOG_IF(WARNING, strategy_.async_mode_)
          << "Currently, fuse_all_optimizer_ops doesn't work under "
             "async mode.";
      strategy_.fuse_all_optimizer_ops_ = !strategy_.async_mode_;
    }
    if (strategy_.fuse_all_reduce_ops_ == true) {
      LOG_IF(WARNING, strategy_.async_mode_)
          << "Currently, fuse_all_reduce_ops doesn't work under "
             "async mode.";
      strategy_.fuse_all_reduce_ops_ = !strategy_.async_mode_;
    }
  }

  void AppendMultiGraphOptPasses() {
    // NOTE: fuse_all_reduce_ops will count the number of all_reduce operator
    // first, if the number is zero, fuse_all_reduce_ops will do nothing.
    AppendPassWithCheck(strategy_.fuse_all_reduce_ops_,
                        "fuse_all_reduce_op_pass");
    AppendPrintGraphPass("multi_devices_print_pass", "_multi_devices_graph");

    // experimental shows that the program will be faster if append
    // all_reduce_deps_pass here.
    bool append_all_reduce_deps_pass =
        !strategy_.enable_parallel_graph_ &&
        (SeqOnlyAllReduceOps(strategy_) ||
         strategy_.reduce_ == BuildStrategy::ReduceStrategy::kAllReduce);
    AppendPassWithCheck(append_all_reduce_deps_pass, "all_reduce_deps_pass");

    bool append_backward_optimizer_op_deps_pass =
        strategy_.num_trainers_ > 1 && !strategy_.async_mode_ &&
        !strategy_.is_distribution_ &&
        strategy_.enable_backward_optimizer_op_deps_;
    AppendPassWithCheck(append_backward_optimizer_op_deps_pass,
                        "backward_optimizer_op_deps_pass");
  }

  void AppendOpFusePasses() {
    AppendPassWithCheck(strategy_.fuse_relu_depthwise_conv_,
                        "fuse_relu_depthwise_conv_pass");
    AppendPassWithCheck(strategy_.fuse_bn_act_ops_, "fuse_bn_act_pass");
    AppendPassWithCheck(strategy_.fuse_bn_add_act_ops_, "fuse_bn_add_act_pass");
#if (defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)) && \
    !defined(_WIN32) && !defined(__APPLE__)
    AppendPassWithCheck(strategy_.enable_auto_fusion_, "fusion_group_pass");
#endif
    AppendPassWithCheck(strategy_.fuse_elewise_add_act_ops_,
                        "fuse_elewise_add_act_pass");
    // for single card training, fuse_all_reduce_ops is unnecessary.
    // coalesce_grad_tensor_pass should be before of MultiDevPass.
    AppendPassWithCheck(strategy_.fuse_all_reduce_ops_,
                        "coalesce_grad_tensor_pass");
    // Fuse all the optimization operators.
    // NOTE: fuse_all_xx_ops will count the number of xx operator first,
    // if the number is zero, fuse_all_reduce_ops will do nothing.
    // Currently, only one type of optimization algorithm can be fused.
    if (strategy_.fuse_all_optimizer_ops_ == true) {
      AppendPass("fuse_adam_op_pass");
      AppendPass("fuse_sgd_op_pass");
      AppendPass("fuse_momentum_op_pass");
    }
  }

  void SetCollectiveContext() const {
    CollectiveContext *context = CollectiveContext::GetInstance();
    context->endpoints_ = strategy_.trainers_endpoints_;
    context->trainer_id_ = strategy_.trainer_id_;
    PADDLE_ENFORCE_GE(
        strategy_.trainer_id_, 0,
        platform::errors::InvalidArgument(
            "The trainer_id_ of strategy_ must be greater than or equal to 0, "
            "but received strategy_.trainer_id_ = %d.",
            strategy_.trainer_id_));

    if (strategy_.trainer_id_ > 0 && strategy_.trainers_endpoints_.size() > 0) {
      PADDLE_ENFORCE_LT(
          static_cast<size_t>(strategy_.trainer_id_),
          strategy_.trainers_endpoints_.size(),
          platform::errors::InvalidArgument(
              "The trainer_id_ of strategy_ must be less than the "
              "size of vector strategy_.trainers_endpoints_, "
              "but received strategy_.trainer_id_ = %d, "
              "the size of strategy_.trainers_endpoints_ is %d.",
              static_cast<size_t>(strategy_.trainer_id_),
              strategy_.trainers_endpoints_.size()));
    }
    VLOG(1) << "CollectiveContext:" << context->String();
  }

  void AppendAddReaderDependencyPass() {
    AppendPass("add_reader_dependency_pass");
  }

  // Convert graph to run on multi-devices.
  void AppendMultiDevPass() {
    ir::Pass *multi_devices_pass = nullptr;
    if (strategy_.async_mode_) {
      multi_devices_pass = AppendPass("async_multi_devices_pass").get();
    } else if (strategy_.is_distribution_) {
      multi_devices_pass = AppendPass("dist_multi_devices_pass").get();
    } else {
      switch (strategy_.reduce_) {
        case BuildStrategy::ReduceStrategy::kAllReduce:
          multi_devices_pass =
              AppendPass("all_reduce_mode_multi_devices_pass").get();
          break;
        case BuildStrategy::ReduceStrategy::kReduce:
          multi_devices_pass =
              AppendPass("reduce_mode_multi_devices_pass").get();
          break;
        default:
          PADDLE_THROW(
              platform::errors::Unimplemented("Unknown reduce strategy."));
      }
    }
    multi_devices_pass->SetNotOwned<const BuildStrategy>("strategy",
                                                         &strategy_);
  }

  void AppendPrintGraphPass(const std::string &pass_name,
                            const std::string &debug_file_suffix) {
    if (!strategy_.debug_graphviz_path_.empty()) {
      auto viz_pass = AppendPass(pass_name);
      const std::string graph_path = string::Sprintf(
          "%s%s", strategy_.debug_graphviz_path_.c_str(), debug_file_suffix);
      viz_pass->Set<std::string>(ir::kGraphvizPath,
                                 new std::string(graph_path));
    }
  }

  void AppendPassWithCheck(const boost::optional<bool> &append_pass,
                           const std::string &pass_name) {
    AppendPassWithCheck(append_pass == true, pass_name);
  }

  void AppendPassWithCheck(bool append_pass, const std::string &pass_name) {
    if (append_pass) {
      AppendPass(pass_name);
    }
  }

  void AppendPassToSetMkldnnAttr(const std::string &pass_name) {
#ifdef PADDLE_WITH_MKLDNN
    if (FLAGS_use_mkldnn) {
      AppendPass(pass_name);
    } else if (!strategy_.mkldnn_enabled_op_types_.empty()) {
      VLOG(1) << "mkldnn_enabled_op_types specify the operator type list to "
                 "use MKLDNN acceleration. It is null in default, means "
                 "that all the operators supported by MKLDNN will be "
                 "accelerated. And it should not be set when "
                 "FLAGS_use_mkldnn=false.";
    }
#else
    PADDLE_ENFORCE_NE(FLAGS_use_mkldnn, true,
                      platform::errors::PreconditionNotMet(
                          "FLAGS_use_mkldnn has been set to True, but "
                          "PaddlePaddle is compiled without MKLDNN. "
                          "Please compile PaddlePaddle with MKLDNN first."));
#endif
  }

 private:
  BuildStrategy strategy_;
};

std::shared_ptr<ir::PassBuilder> BuildStrategy::CreatePassesFromStrategy(
    bool finalize_strategy) const {
  if (is_finalized_) {
    return pass_builder_;
  }
  pass_builder_.reset(new ParallelExecutorPassBuilder(*this));
  if (finalize_strategy) {
    is_finalized_ = true;
  }
  return pass_builder_;
}

bool BuildStrategy::IsMultiDevPass(const std::string &pass_name) const {
  return framework::ir::MultiDevSSAGraphBuilder().count(pass_name) > 0;
}

ir::Graph *BuildStrategy::Apply(ir::Graph *graph,
                                const std::vector<platform::Place> &places,
                                const std::string &loss_var_name,
                                const std::vector<Scope *> &local_scopes,
                                const size_t &nranks,
#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
                                DeviceType use_device,
                                platform::NCCLCommunicator *nccl_ctxs) const {
#elif defined(PADDLE_WITH_XPU) && defined(PADDLE_WITH_XPU_BKCL)
                                DeviceType use_device,
                                platform::BKCLCommunicator *bkcl_ctxs) const {
#else
                                DeviceType use_device) const {
#endif
  VLOG(1) << "apply all passes";
  if (FLAGS_convert_all_blocks) {
    PADDLE_ENFORCE_EQ(
        graph->IsMainGraph(), true,
        platform::errors::InvalidArgument("This graph is not main_graph"));
  }
  // Create a default one if not finalized by user.
  CreatePassesFromStrategy(false);

  for (std::shared_ptr<ir::Pass> &pass : pass_builder_->AllPasses()) {
    VLOG(1) << "BuildStrategy::Apply pass:" << pass->Type();
    if (IsMultiDevPass(pass->Type())) {
      pass->Erase(kPlaces);
      pass->SetNotOwned<const std::vector<platform::Place>>(kPlaces, &places);
      pass->Erase(ir::kLossVarName);
      pass->SetNotOwned<const std::string>(ir::kLossVarName, &loss_var_name);
      pass->Erase(kLocalScopes);
      pass->SetNotOwned<const std::vector<Scope *>>(kLocalScopes,
                                                    &local_scopes);
      pass->Erase(kNRanks);
      pass->Set<size_t>(kNRanks, new size_t(nranks));

#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
      platform::NCCLCommunicator *nctx =
          (use_device == p::kCUDA) ? nccl_ctxs : nullptr;
      pass->Erase(kNCCLCtxs);
      pass->SetNotOwned<platform::NCCLCommunicator>(kNCCLCtxs, nctx);
#elif defined(PADDLE_WITH_XPU) && defined(PADDLE_WITH_XPU_BKCL)
      // ToDo: more check
      platform::BKCLCommunicator *bkcl_ctx =
          (use_device == p::kXPU) ? bkcl_ctxs : nullptr;
      pass->Erase(kBKCLCtxs);
      pass->SetNotOwned<platform::BKCLCommunicator>(kBKCLCtxs, bkcl_ctx);
#endif
    } else if (pass->Type() == "fuse_all_reduce_op_pass") {
      pass->Erase(kNRanks);
      pass->Set<size_t>(kNRanks, new size_t(nranks));
      pass->Erase(kPlaces);
      pass->SetNotOwned<const std::vector<platform::Place>>(kPlaces, &places);
      pass->Erase(kLocalScopes);
      pass->SetNotOwned<const std::vector<Scope *>>(kLocalScopes,
                                                    &local_scopes);
#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
      platform::NCCLCommunicator *nctx =
          (use_device == p::kCUDA) ? nccl_ctxs : nullptr;
      pass->Erase(kNCCLCtxs);
      pass->SetNotOwned<platform::NCCLCommunicator>(kNCCLCtxs, nctx);
      pass->Erase(kUseHierarchicalAllReduce);
      pass->Set<bool>(kUseHierarchicalAllReduce,
                      new bool(use_hierarchical_allreduce_));
#elif defined(PADDLE_WITH_XPU) && defined(PADDLE_WITH_XPU_BKCL)
      platform::BKCLCommunicator *nctx =
          (use_device == p::kXPU) ? bkcl_ctxs : nullptr;
      pass->Erase(kBKCLCtxs);
      pass->SetNotOwned<platform::BKCLCommunicator>(kBKCLCtxs, nctx);
      pass->Erase(kUseHierarchicalAllReduce);
      PADDLE_ENFORCE_EQ(use_hierarchical_allreduce_, false,
                        platform::errors::Unimplemented(
                            "xpu doesn't support hierarchical_allreduce"));
      pass->Set<bool>(kUseHierarchicalAllReduce,
                      new bool(use_hierarchical_allreduce_));
#endif
    } else if (pass->Type() == "coalesce_grad_tensor_pass") {
      pass->Erase(kNRanks);
      pass->Set<size_t>(kNRanks, new size_t(nranks));
    } else if (pass->Type() == "sequential_execution_pass") {
      LOG(INFO) << "set enable_sequential_execution:"
                << enable_sequential_execution_;
    } else if (pass->Type() == "all_reduce_deps_pass") {
#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
      platform::NCCLCommunicator *nctx =
          (use_device == p::kCUDA) ? nccl_ctxs : nullptr;
      pass->Erase(kNCCLCtxs);
      pass->SetNotOwned<platform::NCCLCommunicator>(kNCCLCtxs, nctx);
      pass->Erase(kUseHierarchicalAllReduce);
      pass->Set<bool>(kUseHierarchicalAllReduce,
                      new bool(use_hierarchical_allreduce_));
#elif defined(PADDLE_WITH_XPU) && defined(PADDLE_WITH_XPU_BKCL)
      platform::BKCLCommunicator *nctx =
          (use_device == p::kXPU) ? bkcl_ctxs : nullptr;
      pass->Erase(kBKCLCtxs);
      pass->SetNotOwned<platform::BKCLCommunicator>(kBKCLCtxs, nctx);
      pass->Erase(kUseHierarchicalAllReduce);
      PADDLE_ENFORCE_EQ(use_hierarchical_allreduce_, false,
                        platform::errors::Unimplemented(
                            "xpu doesn't support hierarchical_allreduce"));
      pass->Set<bool>(kUseHierarchicalAllReduce,
                      new bool(use_hierarchical_allreduce_));
#endif
      VLOG(1) << "SeqOnlyAllReduceOps:" << SeqOnlyAllReduceOps(*this)
              << ", num_trainers:" << num_trainers_;
    } else if (pass->Type() == "fuse_relu_depthwise_conv_pass") {
      if (use_device != p::kCUDA) {
        VLOG(1) << "fuse_relu_depthwise_conv_pass is only supported on "
                   "GPU, skipped.";
        continue;
      }
    } else if (pass->Type() == "fusion_group_pass") {
      pass->Set<bool>("use_gpu", new bool((use_device == p::kCUDA)));
      if (use_device != p::kCUDA) {
        VLOG(1) << "fusion_group_pass is only supported on GPU, skipped.";
        continue;
      }
    } else if (pass->Type() == "fuse_bn_act_pass") {
      if (use_device != p::kCUDA) {
        VLOG(1) << "fuse_bn_act_pass is only supported on "
                   "GPU, skipped.";
        continue;
      }
    } else if (pass->Type() == "fuse_bn_add_act_pass") {
      if (use_device != p::kCUDA) {
        VLOG(1) << "fuse_bn_add_act_pass is only supported on "
                   "GPU, skipped.";
        continue;
      }
    } else if (pass->Type() == "mkldnn_placement_pass") {
      pass->Set("mkldnn_enabled_op_types",
                new std::unordered_set<std::string>(mkldnn_enabled_op_types_));
    } else if (pass->Type() == "backward_optimizer_op_deps_pass") {
      if (use_device != p::kCUDA) {
        VLOG(1) << "backward_optimizer_op_deps_pass is only supported on "
                   "GPU, skipped.";
        continue;
      }
    }
    VLOG(1) << "Start Apply Pass " << pass->Type();
    if (FLAGS_convert_all_blocks) {
      for (size_t i = 0; i < graph->SubGraphsSize(); ++i) {
        VLOG(3) << "Apply Pass " << pass->Type() << "to SubGraph " << i;
        pass->Apply(graph->GetSubGraph(i));
      }
    } else {
      graph = pass->Apply(graph);
    }
    VLOG(1) << "Finish Apply Pass " << pass->Type();
  }
  VLOG(1) << "All Passes Applied";
  return graph;
}

}  // namespace details
}  // namespace framework
}  // namespace paddle

USE_PASS(sync_batch_norm_pass);
USE_PASS(fuse_relu_depthwise_conv_pass);
USE_PASS(fuse_elewise_add_act_pass);
USE_PASS(fuse_bn_act_pass);
USE_PASS(fuse_bn_add_act_pass);
USE_PASS(graph_viz_pass);
USE_PASS(multi_batch_merge_pass);
USE_PASS(reduce_mode_multi_devices_pass);
USE_PASS(all_reduce_mode_multi_devices_pass);
USE_PASS(dist_multi_devices_pass);
USE_PASS(multi_devices_check_pass);
USE_PASS(multi_devices_print_pass);
USE_PASS(sequential_execution_pass);
USE_PASS(all_reduce_deps_pass);
USE_PASS(backward_optimizer_op_deps_pass);
USE_PASS(modify_op_lock_and_record_event_pass);
USE_PASS(lock_free_optimize_pass);
USE_PASS(coalesce_grad_tensor_pass);
USE_PASS(graph_to_program_pass);
USE_PASS(fuse_adam_op_pass);
USE_PASS(fuse_sgd_op_pass);
USE_PASS(fuse_momentum_op_pass);
USE_PASS(fuse_all_reduce_op_pass);
USE_PASS(runtime_context_cache_pass);
USE_PASS(add_reader_dependency_pass);
#ifdef PADDLE_WITH_MKLDNN
USE_PASS(mkldnn_placement_pass);
#endif
#if (defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)) && \
    !defined(_WIN32) && !defined(__APPLE__)
USE_PASS(fusion_group_pass);
#endif
