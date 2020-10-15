/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/distributed/communicator.h"

#include <gflags/gflags.h>
#include <paddle/fluid/framework/program_desc.h>

#include <algorithm>
#include <chrono>  // NOLINT
#include <map>
#include <thread>  // NOLINT
#include <unordered_set>

#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/selected_rows.h"
#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/fluid/framework/threadpool.h"
#include "paddle/fluid/framework/variable_helper.h"
#include "paddle/fluid/operators/distributed/distributed.h"
#include "paddle/fluid/operators/distributed/parameter_recv.h"
#include "paddle/fluid/operators/distributed/parameter_send.h"
#include "paddle/fluid/string/printf.h"
#include "paddle/fluid/string/split.h"

namespace paddle {
namespace operators {
namespace distributed {

using Tree =
    std::map<std::string, std::map<std::string, std::vector<std::string>>>;
using RpcCtxMap = operators::distributed::RpcCtxMap;

inline double GetCurrentUS() {
  struct timeval time;
  gettimeofday(&time, NULL);
  return 1e+6 * time.tv_sec + time.tv_usec;
}

Communicator::Communicator() {}

std::once_flag Communicator::init_flag_;
std::shared_ptr<Communicator> Communicator::communicator_(nullptr);

void AsyncCommunicator::InitImpl(const RpcCtxMap &send_varname_to_ctx,
                                 const RpcCtxMap &recv_varname_to_ctx,
                                 Scope *recv_scope) {
  send_varname_to_ctx_ = std::move(send_varname_to_ctx);
  recv_varname_to_ctx_ = std::move(recv_varname_to_ctx);
  recv_scope_ = std::move(recv_scope);

  if (send_varname_to_ctx.size() == 0) {
    VLOG(0) << "nothing need to be send, will not start send_thread";
  } else {
    send_scope_.reset(new Scope());
    for (auto &iter : send_varname_to_ctx_) {
      send_varname_to_queue_[iter.first] =
          std::make_shared<BlockingQueue<std::shared_ptr<Variable>>>(
              send_queue_size_);
    }
    send_threadpool_.reset(new ::ThreadPool(thread_pool_size_));
  }

  if (recv_varname_to_ctx.size() == 0) {
    VLOG(0) << "nothing need to be received, will not start recv_thread";
  } else {
    recv_threadpool_.reset(new ::ThreadPool(thread_pool_size_));
  }

  InitParams();
}

void AsyncCommunicator::InitParams() { RecvNoBarrier(); }

AsyncCommunicator::~AsyncCommunicator() {
  running_ = false;
  if (main_thread_) main_thread_->join();
}

void AsyncCommunicator::SendGlobalStep(int batches) {
  if (!need_global_step_) {
    return;
  }

  if (batches == 0) {
    return;
  }

  auto &var_name = STEP_COUNTER;
  auto *out_var = send_scope_->Var(var_name);
  auto *out_t = out_var->GetMutable<framework::LoDTensor>();
  auto *data = out_t->mutable_data<int64_t>({1}, platform::CPUPlace());
  data[0] = static_cast<int64_t>(batches);

  auto &ctx = send_varname_to_ctx_.at(var_name);
  auto send_functor = distributed::ParameterSend<float>();
  send_functor(ctx, *send_scope_, true, 1);
}

void AsyncCommunicator::SendByCommunicator(int batches) {
  std::vector<std::future<void>> task_futures;
  task_futures.reserve(send_varname_to_ctx_.size());
  VLOG(3) << "run send graph";
  auto before_run_send_graph = GetCurrentUS();
  for (auto &iter : send_varname_to_queue_) {
    auto &var_name = iter.first;
    auto &var_queue = iter.second;

    auto send_task = [this, batches, &var_name, &var_queue] {
      if (var_name == STEP_COUNTER) {
        return;
      }

      VLOG(3) << var_name << " merge and send";
      std::vector<std::shared_ptr<Variable>> vars;
      vars.reserve(batches);

      for (int i = 0; i < batches; ++i) {
        vars.push_back(var_queue->Pop());
      }

      auto &ctx = send_varname_to_ctx_.at(var_name);

      auto before_merge = GetCurrentUS();
      MergeVars<float>(var_name, vars, send_scope_.get(), ctx.merge_add);
      auto after_merge = GetCurrentUS();
      VLOG(3) << "merge " << batches << " " << var_name << " use time "
              << after_merge - before_merge;

      auto send_functor = distributed::ParameterSend<float>();
      send_functor(ctx, *send_scope_, true, 1);
      auto after_send = GetCurrentUS();
      VLOG(3) << "send " << var_name << " use time "
              << after_send - after_merge;
    };
    task_futures.emplace_back(send_threadpool_->enqueue(std::move(send_task)));
  }
  for (auto &task_f : task_futures) {
    task_f.wait();
  }
  auto after_run_send_graph = GetCurrentUS();

  VLOG(3) << "run send graph use time "
          << after_run_send_graph - before_run_send_graph;
}

void AsyncCommunicator::MainThread() {
  VLOG(3) << "MainThread start and wait";

  while (waiting_ && running_) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    VLOG(3) << "wait for running";
  }

  while (running_) {
    int batches = BatchesCounter();

    if (batches > 0) {
      SendGlobalStep(batches);
      SendByCommunicator(batches);
      BarrierSend();
      RecvByCommunicator();
      BarrierRecv();
      BarrierWeakUp();
    } else {
      VLOG(1) << "get nothing from sending queue, will skip send/recv";
    }
  }
  VLOG(1) << "communicator stopped, send thread exit";
}

void AsyncCommunicator::RecvByCommunicator() {
  VLOG(3) << "parallel run recv graph";
  if (!running_) return;
  RecvNoBarrier();
  VLOG(3) << "run recv graph use time";
}

void AsyncCommunicator::RecvNoBarrier() {
  std::vector<std::future<void>> task_futures;
  task_futures.reserve(recv_varname_to_ctx_.size());

  for (auto &iter : recv_varname_to_ctx_) {
    auto recv_task = [this, &iter] {
      auto &var_name = iter.first;
      VLOG(4) << "recv var " << var_name;
      auto recv_functor = distributed::ParameterRecv<float>();
      recv_functor(iter.second, *recv_scope_);
    };
    task_futures.emplace_back(recv_threadpool_->enqueue(std::move(recv_task)));
  }

  for (auto &task : task_futures) {
    task.wait();
  }
}

int AsyncCommunicator::BatchesCounter() {
  auto &step_queue = send_varname_to_queue_.at(STEP_COUNTER);

  size_t merged_var_num = 0;
  size_t wait_times = 0;

  while (merged_var_num < static_cast<size_t>(max_merge_var_num_)) {
    if (step_queue->Size() == 0) {
      VLOG(3) << "wait_times -> " << wait_times;
      if (wait_times >= static_cast<size_t>(send_wait_times_)) {
        break;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
      wait_times++;
      continue;
    } else {
      step_queue->Pop();
      wait_times = 0;
      merged_var_num++;
    }
  }

  return merged_var_num;
}

void AsyncCommunicator::Start() {
  VLOG(1) << "Communicator start";
  if (!communicator_) {
    VLOG(0) << "Communicator is not inited, do nothing";
  } else {
    VLOG(1) << "start send thread and recv thread";
    waiting_ = true;
    running_ = true;
    BarrierTriggerReset(max_merge_var_num_);
    // start send and recv thread
    main_thread_.reset(
        new std::thread(std::bind(&AsyncCommunicator::MainThread, this)));
  }
}

void AsyncCommunicator::Stop() {
  VLOG(1) << "Communicator stop";
  running_ = false;
  if (!communicator_) {
    VLOG(0) << "Communicator is not inited, do nothing";
  } else {
    if (main_thread_) {
      VLOG(1) << "stop send thread";
      main_thread_->join();
      main_thread_.reset(nullptr);
    }
  }
  VLOG(1) << "Communicator stop done";
}

void AsyncCommunicator::Send(const std::vector<std::string> &var_names,
                             const std::vector<std::string> &var_tables,
                             const framework::Scope &scope) {
  waiting_ = false;

  PADDLE_ENFORCE_EQ(
      var_tables.size(), 1,
      platform::errors::InvalidArgument("var_tables.size() == 1 is permitted"));

  auto table_name = var_tables[0];
  auto &queue = send_varname_to_queue_.at(table_name);

  if (table_name == STEP_COUNTER) {
    auto tmp_var = std::make_shared<Variable>();
    auto *tensor = tmp_var->GetMutable<framework::LoDTensor>();
    tensor->Resize(framework::make_ddim({1}));
    auto *out_d = tensor->mutable_data<int64_t>(platform::CPUPlace());
    out_d[0] = 1;
    VLOG(3) << "send to " << table_name << " with queue size " << queue->Size();
    queue->Push(tmp_var);
  } else {
    PADDLE_ENFORCE_GE(var_names.size(), 1,
                      platform::errors::InvalidArgument(
                          "var_names.size() >= 1 is permitted"));

    auto *var = scope.FindVar(var_names[0]);

    PADDLE_ENFORCE_EQ(
        var->IsInitialized(), true,
        platform::errors::InvalidArgument("grad var should be inited"));

    auto tmp_var = std::make_shared<Variable>();
    if (var->IsType<framework::SelectedRows>()) {
      framework::CopyVariable(*var, tmp_var.get());
      VLOG(3) << "send to " << table_name << " with queue size "
              << queue->Size();
      queue->Push(tmp_var);
    } else if (var->IsType<framework::LoDTensor>()) {
      // push var into send queue by var_name
      auto var_name = var_names[0];
      framework::CopyVariable(*var, tmp_var.get());
      VLOG(3) << "send to " << table_name << " with queue size "
              << queue->Size();
      queue->Push(tmp_var);
    } else {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "unknown var type to copy, only support LoDTensor/SelectedRows"));
    }
  }
}

void HalfAsyncCommunicator::Clean() {
  for (auto &iter : send_varname_to_queue_) {
    auto &var_name = iter.first;
    auto &var_queue = iter.second;

    while (var_queue->Size() > 0) {
      var_queue->Pop();
    }

    VLOG(3) << "clean var: " << var_name << " done";
  }
}

int HalfAsyncCommunicator::BatchesCounter() {
  while (running_) {
    if (barrier_counter_.load() >= barrier_trigger_.load() &&
        barrier_trigger_.load() != 0) {
      break;
    } else {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
  }

  return barrier_counter_.load();
}

void HalfAsyncCommunicator::Barrier() {
  barrier_counter_++;

  if (!running_) {
    VLOG(3) << "Communicator is not running, release barrier";
    return;
  }

  {
    std::unique_lock<std::mutex> lk(barrier_mutex_);
    barrier_cond_.wait(lk, [this] { return (barrier_counter_ == 0); });
  }
}

void HalfAsyncCommunicator::BarrierTriggerDecrement() {
  barrier_trigger_--;
  VLOG(3) << "BarrierTriggerDecrement decrement barrier trigger to "
          << barrier_trigger_.load();
}

void HalfAsyncCommunicator::BarrierTriggerReset(int initial_val) {
  barrier_trigger_.store(initial_val);

  VLOG(3) << "BarrierTriggerReset reset barrier trigger to "
          << barrier_trigger_.load();
}

void HalfAsyncCommunicator::BarrierWeakUp() {
  barrier_counter_.store(0);
  barrier_cond_.notify_all();
}

void SyncCommunicator::BarrierSend() {
  if (!running_) return;

  distributed::RPCClient *rpc_client =
      distributed::RPCClient::GetInstance<RPCCLIENT_T>(trainer_id_);

  std::vector<distributed::VarHandlePtr> rets;

  for (auto &ep : pserver_endpoints_) {
    rets.push_back(rpc_client->AsyncSendBatchBarrier(ep));
  }

  for (size_t i = 0; i < rets.size(); i++) {
    PADDLE_ENFORCE_NE(rets[i]->Wait(), 0U, platform::errors::External(
                                               "internal error in RPCClient"));
  }

  VLOG(4) << "BarrierSend with SyncCommunicator";
}

void SyncCommunicator::BarrierRecv() {
  if (!running_) return;

  distributed::RPCClient *rpc_client =
      distributed::RPCClient::GetInstance<RPCCLIENT_T>(trainer_id_);

  std::vector<distributed::VarHandlePtr> rets;
  for (auto &ep : pserver_endpoints_) {
    rets.push_back(rpc_client->AsyncSendFetchBarrier(ep));
  }

  for (size_t i = 0; i < rets.size(); i++) {
    PADDLE_ENFORCE_NE(rets[i]->Wait(), 0U, platform::errors::External(
                                               "internal error in RPCClient"));
  }

  VLOG(4) << "BarrierRecv with SyncCommunicator";
}

void GeoCommunicator::InitImpl(const RpcCtxMap &send_varname_to_ctx,
                               const RpcCtxMap &recv_varname_to_ctx,
                               Scope *recv_scope) {
  send_varname_to_ctx_ = std::move(send_varname_to_ctx);
  recv_varname_to_ctx_ = std::move(recv_varname_to_ctx);
  recv_scope_ = std::move(recv_scope);

  PADDLE_ENFORCE_GT(
      send_varname_to_ctx.size(), 0,
      platform::errors::InvalidArgument("send var contexts can not be zero"));

  send_scope_.reset(new Scope());
  for (auto &iter : send_varname_to_ctx_) {
    auto &varname = iter.first;

    if (varname == STEP_COUNTER) {
      send_varname_to_queue_[varname] =
          std::make_shared<BlockingQueue<std::shared_ptr<Variable>>>(
              send_queue_size_);
    } else {
      auto &send_ctx = iter.second;

      send_var_nums_ += send_ctx.splited_varnames.size();
      if (!send_ctx.is_sparse) {
        continue;
      }
      int pserver_num = static_cast<int>(send_ctx.epmap.size());
      for (int ep_idx = 0; ep_idx < pserver_num; ep_idx++) {
        sparse_id_queues_.insert(
            std::pair<std::string, std::shared_ptr<BlockingQueue<
                                       std::shared_ptr<std::vector<int64_t>>>>>(
                send_ctx.splited_varnames[ep_idx],
                std::make_shared<
                    BlockingQueue<std::shared_ptr<std::vector<int64_t>>>>(
                    send_queue_size_)));
      }
    }
  }
  send_threadpool_.reset(new ::ThreadPool(thread_pool_size_));

  if (recv_varname_to_ctx.size() == 0) {
    VLOG(0) << "nothing need to be received, will not start recv_thread";
  } else {
    recv_threadpool_.reset(new ::ThreadPool(thread_pool_size_));
  }

  delta_scope_.reset(new Scope());
  old_scope_.reset(new Scope());
  pserver_scope_.reset(new Scope());

  InitParams();
}

void GeoCommunicator::Send(const std::vector<std::string> &var_names,
                           const std::vector<std::string> &var_tables,
                           const framework::Scope &scope) {
  waiting_ = false;
  PADDLE_ENFORCE_EQ(
      var_tables.size(), 1,
      platform::errors::InvalidArgument("var_tables.size() == 1 is permitted"));

  auto table_name = var_tables[0];
  if (table_name == STEP_COUNTER) return;

  auto before_send = GetCurrentUS();
  size_t splited_var_nums =
      send_varname_to_ctx_[table_name].splited_varnames.size();

  std::unordered_map<std::string, std::unordered_set<int64_t>> ids_table;

  for (size_t j = 0; j < splited_var_nums; j++) {
    ids_table.insert(std::pair<std::string, std::unordered_set<int64_t>>(
        send_varname_to_ctx_[table_name].splited_varnames[j],
        std::unordered_set<int64_t>()));
  }
  auto *var = scope.FindVar(var_names[0]);
  auto &rows = var->Get<framework::SelectedRows>().rows();

  // insert ids which has not been record
  for (size_t j = 0; j < rows.size(); j++) {
    auto ep_idx = rows[j] % splited_var_nums;
    ids_table.at(send_varname_to_ctx_[table_name].splited_varnames[ep_idx])
        .insert(rows[j]);
  }

  auto before_push = GetCurrentUS();
  for (auto &iter : ids_table) {
    auto &key = iter.first;
    auto &sparse_ids_set = iter.second;
    auto sparse_ids_vec = std::make_shared<std::vector<int64_t>>();
    sparse_ids_vec->assign(sparse_ids_set.begin(), sparse_ids_set.end());
    sparse_id_queues_.at(key)->Push(sparse_ids_vec);
    VLOG(3) << "push " << sparse_ids_vec->size() << " ids to " << key
            << "'s queue";
  }
  auto after_send = GetCurrentUS();
  VLOG(3) << "run send " << table_name << " op finish. using "
          << (before_push - before_send) << "; " << (after_send - before_push);
}

void GeoCommunicator::MainThread() {
  VLOG(3) << "MainThread start and wait";

  while (waiting_ && running_) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    VLOG(3) << "wait for running";
  }

  while (running_) {
    std::vector<std::future<void>> tasks;
    tasks.reserve(send_var_nums_);

    for (auto &iter : send_varname_to_ctx_) {
      auto &var_name = iter.first;
      auto &send_ctx = iter.second;
      int pserver_num = static_cast<int>(send_ctx.epmap.size());
      if (send_ctx.is_sparse) {
        for (int ep_idx = 0; ep_idx < pserver_num; ep_idx++) {
          auto send_recv_task = [this, ep_idx, &var_name] {
            auto before_send_sparse = GetCurrentUS();
            if (var_name == STEP_COUNTER) {
              return;
            }
            auto send_varname =
                send_varname_to_ctx_.at(var_name).splited_varnames[ep_idx];
            auto sparse_ids = MergeSparseIds(send_varname);
            if (sparse_ids.size() == 0) {
              return;
            }
            SendSparse(var_name, ep_idx, sparse_ids);
            auto after_send_sparse = GetCurrentUS();
            RecvSparse(var_name, ep_idx);
            auto after_recv_sparse = GetCurrentUS();
            VLOG(3)
                << "send recv "
                << send_varname_to_ctx_.at(var_name).splited_varnames[ep_idx]
                << " finish, using " << (after_send_sparse - before_send_sparse)
                << " and " << (after_recv_sparse - after_send_sparse)
                << "; total = " << (after_recv_sparse - before_send_sparse);
          };
          tasks.emplace_back(
              send_threadpool_->enqueue(std::move(send_recv_task)));
        }
      } else {
        auto send_recv_task = [this, &var_name, &send_ctx] {
          if (var_name == STEP_COUNTER) {
            return;
          }
          SendDense(var_name);
          RecvDense(var_name);
        };
        tasks.emplace_back(
            send_threadpool_->enqueue(std::move(send_recv_task)));
      }
    }
    for (auto &task : tasks) {
      task.wait();
    }
  }
}

std::vector<int64_t> GeoCommunicator::MergeSparseIds(
    const std::string &send_varname) {
  size_t merge_num = 0, wait_times = 0;
  std::unordered_set<int64_t> sparse_ids;
  while (merge_num < static_cast<size_t>(max_merge_var_num_)) {
    VLOG(3) << "Merge Number of " << send_varname << " = " << merge_num;
    if (sparse_id_queues_.at(send_varname)->Size() > 0) {
      wait_times = 0;
      std::shared_ptr<std::vector<int64_t>> pop_ids =
          sparse_id_queues_.at(send_varname)->Pop();
      for (size_t j = 0; j < pop_ids->size(); j++) {
        sparse_ids.insert(pop_ids->at(j));
      }
      merge_num += 1;
      VLOG(3) << "sparse_id_queues_(" << send_varname << ") pushed";
    } else if (sparse_id_queues_.at(send_varname)->Size() == 0) {
      VLOG(3) << "wait_times -> " << wait_times;
      if (wait_times >= static_cast<size_t>(send_wait_times_)) {
        break;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
      wait_times++;
      continue;
    }
  }
  std::vector<int64_t> res;
  res.assign(sparse_ids.begin(), sparse_ids.end());
  return res;
}
void GeoCommunicator::SendSparse(const std::string &varname, int ep_idx,
                                 const std::vector<int64_t> &sparse_ids) {
  auto &rpc_ctx = send_varname_to_ctx_.at(varname);
  auto send_varname = rpc_ctx.splited_varnames[ep_idx];
  auto trainer_id = rpc_ctx.trainer_id;
  auto endpoint = rpc_ctx.epmap[ep_idx];
  auto pserver_num = rpc_ctx.epmap.size();

  auto *var_latest = recv_scope_->FindVar(varname);

  PADDLE_ENFORCE_EQ(var_latest->IsInitialized(), true,
                    platform::errors::Unavailable(
                        "%s is not initialized, please check", varname));
  auto &t_latest = var_latest->Get<framework::LoDTensor>();

  auto dims1 = t_latest.dims()[1];

  auto cpu_ctx = paddle::platform::CPUDeviceContext();
  auto *var_delta = delta_scope_->Var(send_varname);
  auto *t_delta = var_delta->GetMutable<framework::SelectedRows>();

  auto *t_value = t_delta->mutable_value();
  t_value->mutable_data<float>(
      framework::make_ddim({static_cast<int64_t>(sparse_ids.size()), dims1}),
      cpu_ctx.GetPlace());

  std::vector<std::vector<std::vector<float> *>> values;
  auto *ins = distributed::LargeScaleKV::GetInstance();
  ins->Get(varname)->Get(sparse_ids, {"Param"}, &values);

  auto blas = math::GetBlas<platform::CPUDeviceContext, float>(cpu_ctx);
  float coefficient = 1.0 / static_cast<float>(trainers_);

  for (auto j = 0; j < static_cast<int>(sparse_ids.size()); ++j) {
    blas.VSUB(dims1, t_latest.data<float>() + sparse_ids[j] * dims1,
              values[j][0]->data(), t_value->data<float>() + j * dims1);
    blas.SCAL(dims1, coefficient, t_value->data<float>() + j * dims1);
    blas.VADD(dims1, values[j][0]->data(), t_value->data<float>() + j * dims1,
              values[j][0]->data());
  }

  std::vector<int64_t> send_rows;
  send_rows.reserve(sparse_ids.size());
  for (auto idx : sparse_ids) {
    send_rows.push_back(idx / pserver_num);
  }
  t_delta->set_height(rpc_ctx.height_sections[ep_idx]);
  t_delta->set_rows(send_rows);

  platform::DeviceContextPool &pool = platform::DeviceContextPool::Instance();
  auto &cpu_ctx_send = *pool.Get(platform::CPUPlace());
  distributed::RPCClient *rpc_client =
      distributed::RPCClient::GetInstance<RPCCLIENT_T>(trainer_id);

  auto ret = rpc_client->AsyncSendVar(endpoint, cpu_ctx_send,
                                      *delta_scope_.get(), send_varname);
  ret->Wait();
}

void GeoCommunicator::SendDense(const std::string &varname) {
  auto *var_latest = recv_scope_->FindVar(varname);
  auto *var_timestamp = old_scope_->FindVar(varname);

  PADDLE_ENFORCE_EQ(var_latest->IsInitialized(), true,
                    platform::errors::Unavailable(
                        "%s is not initialized, please check", varname));
  PADDLE_ENFORCE_EQ(var_timestamp->IsInitialized(), true,
                    platform::errors::Unavailable(
                        "%s is not initialized, please check", varname));

  auto &t_latest = var_latest->Get<framework::LoDTensor>();
  auto t_timestamp = var_timestamp->GetMutable<framework::LoDTensor>();

  auto cpu_ctx = paddle::platform::CPUDeviceContext();
  auto *var_delta = delta_scope_->Var(varname);
  auto *t_delta = var_delta->GetMutable<framework::LoDTensor>();
  t_delta->mutable_data<float>(t_latest.dims(), cpu_ctx.GetPlace());

  auto blas = math::GetBlas<platform::CPUDeviceContext, float>(cpu_ctx);
  blas.VSUB(t_latest.numel(), t_latest.data<float>(),
            t_timestamp->data<float>(), t_delta->data<float>());

  float coefficient = 1.0 / static_cast<float>(trainers_);
  blas.SCAL(t_latest.numel(), coefficient, t_delta->data<float>());

  blas.VADD(t_latest.numel(), t_timestamp->data<float>(),
            t_delta->data<float>(), t_timestamp->data<float>());

  auto &ctx = send_varname_to_ctx_.at(varname);
  auto send = distributed::ParameterSend<float>();
  send(ctx, *delta_scope_, true, 1);
}

void GeoCommunicator::RecvByCommunicator() { return; }

void GeoCommunicator::RecvSparse(const std::string &varname, int ep_idx) {
  auto train_id = recv_varname_to_ctx_.at(varname).trainer_id;
  auto endpoint = recv_varname_to_ctx_.at(varname).epmap[ep_idx];
  auto splited_var_name =
      recv_varname_to_ctx_.at(varname).splited_varnames[ep_idx];
  auto pserver_num = recv_varname_to_ctx_.at(varname).epmap.size();

  platform::DeviceContextPool &pool = platform::DeviceContextPool::Instance();
  auto &cpu_ctx_recv = *pool.Get(platform::CPUPlace());
  distributed::RPCClient *rpc_client =
      distributed::RPCClient::GetInstance<RPCCLIENT_T>(train_id);

  auto *var_psrever = pserver_scope_->Var(splited_var_name);
  auto handle = rpc_client->AsyncGetVar(endpoint, cpu_ctx_recv,
                                        *pserver_scope_.get(), splited_var_name,
                                        splited_var_name, splited_var_name);
  handle->Wait();

  auto *var_latest = recv_scope_->FindVar(varname);

  PADDLE_ENFORCE_EQ(
      var_psrever->IsInitialized(), true,
      platform::errors::Unavailable(
          "%s in pserver scope is not initialized, please check", varname));

  std::vector<int64_t> ids;
  ids.assign(var_psrever->Get<framework::SelectedRows>().rows().begin(),
             var_psrever->Get<framework::SelectedRows>().rows().end());

  for (size_t j = 0; j < ids.size(); j++) {
    ids[j] = ids[j] * pserver_num + ep_idx;
  }

  VLOG(3) << "RecvSparse receive var: " << splited_var_name
          << " ids Size: " << ids.size();

  auto t_psrever = var_psrever->Get<framework::SelectedRows>().value();

  std::vector<std::vector<std::vector<float> *>> old_values;

  auto *ins = distributed::LargeScaleKV::GetInstance();
  ins->Get(varname)->Get(ids, {"Param"}, &old_values);

  auto *t_latest = var_latest->GetMutable<framework::LoDTensor>();

  auto dims1 = t_latest->dims()[1];
  auto numel = ids.size() * dims1;

  std::vector<float> v_delta;
  v_delta.resize(numel);

  auto cpu_ctx = paddle::platform::CPUDeviceContext();
  auto blas = math::GetBlas<platform::CPUDeviceContext, float>(cpu_ctx);

  for (auto j = 0; j < static_cast<int>(ids.size()); ++j) {
    blas.VSUB(dims1, t_psrever.data<float>() + j * dims1,
              old_values[j][0]->data(), v_delta.data() + j * dims1);
    blas.VADD(dims1, t_latest->data<float>() + ids[j] * dims1,
              v_delta.data() + j * dims1,
              t_latest->data<float>() + ids[j] * dims1);
    blas.VCOPY(dims1, t_psrever.data<float>() + j * dims1,
               old_values[j][0]->data());
  }
}

void GeoCommunicator::RecvDense(const std::string &varname) {
  auto *var_latest = recv_scope_->FindVar(varname);
  auto *var_timestamp = old_scope_->FindVar(varname);
  auto *var_psrever = pserver_scope_->Var(varname);

  auto &ctx = recv_varname_to_ctx_.at(varname);
  auto recv = distributed::ParameterRecv<float>();
  recv(ctx, *pserver_scope_);

  PADDLE_ENFORCE_EQ(
      var_psrever->IsInitialized(), true,
      platform::errors::Unavailable(
          "%s in pserver scope is not initialized, please check", varname));

  auto t_psrever = var_psrever->Get<framework::LoDTensor>();
  auto t_latest = var_latest->GetMutable<framework::LoDTensor>();
  auto t_timestamp = var_timestamp->GetMutable<framework::LoDTensor>();

  auto cpu_ctx = paddle::platform::CPUDeviceContext();
  auto *var_delta = delta_scope_->Var(varname);
  auto *t_delta = var_delta->GetMutable<framework::LoDTensor>();
  t_delta->mutable_data<float>(t_latest->dims(), cpu_ctx.GetPlace());

  auto blas = math::GetBlas<platform::CPUDeviceContext, float>(cpu_ctx);
  blas.VSUB(t_latest->numel(), t_psrever.data<float>(),
            t_timestamp->data<float>(), t_delta->data<float>());
  blas.VADD(t_latest->numel(), t_latest->data<float>(), t_delta->data<float>(),
            t_latest->data<float>());
  blas.VCOPY(t_latest->numel(), t_psrever.data<float>(),
             t_timestamp->data<float>());
}

void GeoCommunicator::InitParams() {
  std::vector<std::future<void>> tasks;
  tasks.reserve(recv_varname_to_ctx_.size());

  for (auto &iter : recv_varname_to_ctx_) {
    auto &var_name = iter.first;
    auto &recv_ctx = iter.second;

    auto recv_task = [this, &var_name, &recv_ctx] {
      if (!recv_ctx.is_sparse) {
        InitDense(var_name);
      }
    };
    tasks.emplace_back(send_threadpool_->enqueue(std::move(recv_task)));
  }

  for (auto &task : tasks) {
    task.wait();
  }
  InitSparse();
}

void GeoCommunicator::InitDense(const std::string varname) {
  auto &ctx = recv_varname_to_ctx_.at(varname);
  auto recv = distributed::ParameterRecv<float>();
  recv(ctx, *recv_scope_);

  auto *global_var = recv_scope_->FindVar(varname);
  global_var->GetMutable<framework::LoDTensor>();

  auto *old_var = old_scope_->Var(varname);
  old_var->GetMutable<framework::LoDTensor>();

  framework::CopyVariable(*global_var, old_var);
  VLOG(1) << "init dense variable " << varname << " done";
}

void GeoCommunicator::InitSparse() {
  auto sparse_metas = string::split_string<std::string>(sparse_attrs_, "#");

  std::vector<distributed::SparseMeta> metas;
  std::vector<int64_t> dicts;

  for (auto &sparse_meta : sparse_metas) {
    auto attrs = string::split_string<std::string>(sparse_meta, ":");

    auto meta = distributed::SparseMeta();
    meta.name = attrs[0];
    meta.value_names = {"Param"};

    auto dic = string::split_string<std::string>(attrs[1], ",");
    dicts.push_back(std::stoi(dic[0]));
    meta.value_dims = {std::stoi(dic[1])};
    meta.mode = distributed::Mode::training;
    meta.grad_name = "none";
    meta.cached_varnames = {};
    meta.initializer_attrs = string::split_string<std::string>(attrs[2]);
    meta.entry = "none";

    VLOG(3) << "add sparse meta: " << meta.ToString();
    metas.push_back(meta);
  }

  LargeScaleKV::Init(metas);

  for (auto &meta : metas) {
    auto &ctx = recv_varname_to_ctx_.at(meta.name);
    auto recv = distributed::ParameterRecv<float>();

    auto *global_var = recv_scope_->FindVar(meta.name);
    auto global_value = global_var->Get<framework::LoDTensor>();
    auto rows = global_value.dims()[0];
    auto dim1 = global_value.dims()[1];

    recv(ctx, *recv_scope_);
    VLOG(1) << "recv " << meta.name << " with global scope for init";

    auto n_rows = global_var->Get<framework::LoDTensor>().dims()[0];

    PADDLE_ENFORCE_EQ(
        rows, n_rows,
        platform::errors::InvalidArgument(
            "global var: %s origin dim must equal recved rows", meta.name));

    std::vector<int64_t> ids(rows);
    std::iota(ids.begin(), ids.end(), 0);

    auto *ins = distributed::LargeScaleKV::GetInstance();
    std::vector<std::vector<std::vector<float> *>> values;

    ins->Get(meta.name)->Init(ids);
    ins->Get(meta.name)->Get(ids, {"Param"}, &values);

    auto blas = math::GetBlas<platform::CPUDeviceContext, float>(
        paddle::platform::CPUDeviceContext());

    for (auto &id : ids) {
      blas.VCOPY(dim1, global_value.data<float>() + id * dim1,
                 values[id][0]->data());
    }
  }

  VLOG(3) << "init sparse variable done";
}

}  // namespace distributed
}  // namespace operators
}  // namespace paddle
