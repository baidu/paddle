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
#include <fcntl.h>

#ifdef _POSIX_C_SOURCE
#undef _POSIX_C_SOURCE
#endif

#ifdef _XOPEN_SOURCE
#undef _XOPEN_SOURCE
#endif

#include "paddle/fluid/pybind/fleet_py.h"

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "paddle/fluid/distributed/common/sparse_sharding_merge.h"
#include "paddle/fluid/distributed/communicator_common.h"
#include "paddle/fluid/distributed/fleet.h"
#include "paddle/fluid/distributed/index_dataset/index_dataset.pb.h"
#include "paddle/fluid/distributed/index_dataset/index_sampler.h"
#include "paddle/fluid/distributed/index_dataset/index_wrapper.h"
#include "paddle/fluid/distributed/service/communicator.h"
#include "paddle/fluid/distributed/service/env.h"
#include "paddle/fluid/distributed/service/graph_brpc_client.h"
#include "paddle/fluid/distributed/service/graph_py_service.h"
#include "paddle/fluid/distributed/service/heter_client.h"

namespace py = pybind11;
using paddle::distributed::CommContext;
using paddle::distributed::Communicator;
using paddle::distributed::FleetWrapper;
using paddle::distributed::HeterClient;
using paddle::distributed::GraphPyService;
using paddle::distributed::GraphNode;
using paddle::distributed::GraphPyServer;
using paddle::distributed::GraphPyClient;
using paddle::distributed::FeatureNode;
using paddle::distributed::GraphIndex;
using paddle::distributed::GraphItem;
using paddle::distributed::ShardingMerge;

namespace paddle {
namespace pybind {
void BindDistFleetWrapper(py::module* m) {
  py::class_<FleetWrapper, std::shared_ptr<FleetWrapper>>(*m,
                                                          "DistFleetWrapper")
      .def(py::init([]() { return FleetWrapper::GetInstance(); }))
      .def("load_sparse", &FleetWrapper::LoadSparseOnServer)
      .def("load_model", &FleetWrapper::LoadModel)
      .def("load_one_table", &FleetWrapper::LoadModelOneTable)
      .def("init_server", &FleetWrapper::InitServer)
      .def("run_server",
           (uint64_t (FleetWrapper::*)(void)) & FleetWrapper::RunServer)
      .def("run_server", (uint64_t (FleetWrapper::*)(          // NOLINT
                             const std::string&, uint32_t)) &  // NOLINT
                             FleetWrapper::RunServer)
      .def("init_worker", &FleetWrapper::InitWorker)
      .def("push_dense_params", &FleetWrapper::PushDenseParamSync)
      .def("pull_dense_params", &FleetWrapper::PullDenseVarsSync)
      .def("save_all_model", &FleetWrapper::SaveModel)
      .def("save_one_model", &FleetWrapper::SaveModelOneTable)
      .def("recv_and_save_model", &FleetWrapper::RecvAndSaveTable)
      .def("sparse_table_stat", &FleetWrapper::PrintTableStat)
      .def("stop_server", &FleetWrapper::StopServer)
      .def("stop_worker", &FleetWrapper::FinalizeWorker)
      .def("barrier", &FleetWrapper::BarrierWithTable)
      .def("shrink_sparse_table", &FleetWrapper::ShrinkSparseTable);
}

void BindPSHost(py::module* m) {
  py::class_<distributed::PSHost>(*m, "PSHost")
      .def(py::init<const std::string&, uint32_t, uint32_t>())
      .def("serialize_to_string", &distributed::PSHost::serialize_to_string)
      .def("parse_from_string", &distributed::PSHost::parse_from_string)
      .def("to_uint64", &distributed::PSHost::serialize_to_uint64)
      .def("from_uint64", &distributed::PSHost::parse_from_uint64)
      .def("to_string", &distributed::PSHost::to_string);
}

void BindSparseShardingTools(py::module* m) {
  py::class_<ShardingMerge>(*m, "ShardingMerge")
      .def(py::init<>())
      .def("merge", &ShardingMerge::Merge);
}

void BindCommunicatorContext(py::module* m) {
  py::class_<CommContext>(*m, "CommContext")
      .def(
          py::init<const std::string&, const std::vector<std::string>&,
                   const std::vector<std::string>&, const std::vector<int64_t>&,
                   const std::vector<std::string>&, int, bool, bool, bool, int,
                   bool>())
      .def("var_name", [](const CommContext& self) { return self.var_name; })
      .def("trainer_id",
           [](const CommContext& self) { return self.trainer_id; })
      .def("table_id", [](const CommContext& self) { return self.table_id; })
      .def("split_varnames",
           [](const CommContext& self) { return self.splited_varnames; })
      .def("split_endpoints",
           [](const CommContext& self) { return self.epmap; })
      .def("sections",
           [](const CommContext& self) { return self.height_sections; })
      .def("aggregate", [](const CommContext& self) { return self.merge_add; })
      .def("is_sparse", [](const CommContext& self) { return self.is_sparse; })
      .def("is_distributed",
           [](const CommContext& self) { return self.is_distributed; })
      .def("origin_varnames",
           [](const CommContext& self) { return self.origin_varnames; })
      .def("is_tensor_table",
           [](const CommContext& self) { return self.is_tensor_table; })
      .def("__str__", [](const CommContext& self) { return self.print(); });
}

using paddle::distributed::AsyncCommunicator;
using paddle::distributed::GeoCommunicator;
using paddle::distributed::RecvCtxMap;
using paddle::distributed::RpcCtxMap;
using paddle::distributed::SyncCommunicator;
using paddle::framework::Scope;

void BindDistCommunicator(py::module* m) {
  // Communicator is already used by nccl, change to DistCommunicator
  py::class_<Communicator, std::shared_ptr<Communicator>>(*m,
                                                          "DistCommunicator")
      .def(py::init([](const std::string& mode, const std::string& dist_desc,
                       const std::vector<std::string>& host_sign_list,
                       const RpcCtxMap& send_ctx, const RecvCtxMap& recv_ctx,
                       Scope* param_scope,
                       std::map<std::string, std::string>& envs) {
        if (mode == "ASYNC") {
          Communicator::InitInstance<AsyncCommunicator>(
              send_ctx, recv_ctx, dist_desc, host_sign_list, param_scope, envs);
        } else if (mode == "SYNC") {
          Communicator::InitInstance<SyncCommunicator>(
              send_ctx, recv_ctx, dist_desc, host_sign_list, param_scope, envs);
        } else if (mode == "GEO") {
          Communicator::InitInstance<GeoCommunicator>(
              send_ctx, recv_ctx, dist_desc, host_sign_list, param_scope, envs);
        } else {
          PADDLE_THROW(platform::errors::InvalidArgument(
              "unsuported communicator MODE"));
        }
        return Communicator::GetInstantcePtr();
      }))
      .def("stop", &Communicator::Stop)
      .def("start", &Communicator::Start)
      .def("push_sparse_param", &Communicator::RpcSendSparseParam)
      .def("is_running", &Communicator::IsRunning)
      .def("init_params", &Communicator::InitParams);
  //  .def("recv", &Communicator::RecvNoBarrier);
}

void BindHeterClient(py::module* m) {
  py::class_<HeterClient, std::shared_ptr<HeterClient>>(*m, "HeterClient")
      .def(py::init(
          [](const std::vector<std::string>& endpoint, const int& trainer_id) {
            return HeterClient::GetInstance(endpoint, trainer_id);
          }))
      .def("stop", &HeterClient::Stop);
}

void BindGraphNode(py::module* m) {
  py::class_<GraphNode>(*m, "GraphNode")
      .def(py::init<>())
      .def("get_id", &GraphNode::get_id)
      .def("get_feature", &GraphNode::get_feature);
}
void BindGraphPyFeatureNode(py::module* m) {
  py::class_<FeatureNode>(*m, "FeatureNode")
      .def(py::init<>())
      .def("get_id", &GraphNode::get_id)
      .def("get_feature", &GraphNode::get_feature);
}

void BindGraphPyService(py::module* m) {
  py::class_<GraphPyService>(*m, "GraphPyService").def(py::init<>());
}

void BindGraphPyServer(py::module* m) {
  py::class_<GraphPyServer>(*m, "GraphPyServer")
      .def(py::init<>())
      .def("start_server", &GraphPyServer::start_server)
      .def("set_up", &GraphPyServer::set_up)
      .def("add_table_feat_conf", &GraphPyServer::add_table_feat_conf);
}
void BindGraphPyClient(py::module* m) {
  py::class_<GraphPyClient>(*m, "GraphPyClient")
      .def(py::init<>())
      .def("load_edge_file", &GraphPyClient::load_edge_file)
      .def("load_node_file", &GraphPyClient::load_node_file)
      .def("set_up", &GraphPyClient::set_up)
      .def("add_table_feat_conf", &GraphPyClient::add_table_feat_conf)
      .def("pull_graph_list", &GraphPyClient::pull_graph_list)
      .def("start_client", &GraphPyClient::start_client)
      .def("batch_sample_neighboors", &GraphPyClient::batch_sample_neighboors)
      .def("random_sample_nodes", &GraphPyClient::random_sample_nodes)
      .def("stop_server", &GraphPyClient::stop_server)
      .def("get_node_feat",
           [](GraphPyClient& self, std::string node_type,
              std::vector<uint64_t> node_ids,
              std::vector<std::string> feature_names) {
             auto feats =
                 self.get_node_feat(node_type, node_ids, feature_names);
             std::vector<std::vector<py::bytes>> bytes_feats(feats.size());
             for (int i = 0; i < feats.size(); ++i) {
               for (int j = 0; j < feats[i].size(); ++j) {
                 bytes_feats[i].push_back(py::bytes(feats[i][j]));
               }
             }
             return bytes_feats;
           })
      .def("bind_local_server", &GraphPyClient::bind_local_server);
}

using paddle::distributed::TreeIndex;
using paddle::distributed::IndexWrapper;
using paddle::distributed::IndexNode;

void BindIndexNode(py::module* m) {
  py::class_<IndexNode>(*m, "IndexNode")
      .def(py::init<>())
      .def("id", [](IndexNode& self) { return self.id(); })
      .def("is_leaf", [](IndexNode& self) { return self.is_leaf(); })
      .def("probability", [](IndexNode& self) { return self.probability(); });
}

void BindGraphItem(py::module* m) {
  py::class_<GraphItem>(*m, "GraphItem")
      .def(py::init<>())
      .def("item_id", [](GraphItem& self) { return self.item_id(); })
      .def("path_id", [](GraphItem& self) { return self.path_id(); });
  //  .def("item_path_nums",
  //       [](GraphItem& self) { return self.item_path_nums(); })
}

void BindGraphIndex(py::module* m) {
  py::class_<GraphIndex, std::shared_ptr<GraphIndex>>(*m, "GraphIndex")
      // .def(py::init([](const std::string name, const std::string path) {
      //   auto index_wrapper = IndexWrapper::GetInstancePtr();
      //   index_wrapper->insert_graph_index(name, path);
      //   return index_wrapper->GetGraphIndex(name);
      // }))
      .def(py::init<>())
      .def("initialize",
           [](GraphIndex& self, int height, int width, int item_path_nums) {
             self.set_height(height);
             self.set_width(width);
             self.set_item_path_nums(item_path_nums);
           })
      .def("reset_mapping", &GraphIndex::reset_mapping)
      .def("add_item", &GraphIndex::add_item)
      .def("save", &GraphIndex::save)
      .def("load", &GraphIndex::load)
      .def("get_item_path_dict", &GraphIndex::get_item_path_dict)
      .def("height", [](GraphIndex& self) { return self.height(); })
      .def("width", [](GraphIndex& self) { return self.width(); })
      .def("item_path_nums", &GraphIndex::item_path_nums)
      .def("gather_unique_items_of_paths",
           &GraphIndex::gather_unique_items_of_paths)
      .def("get_path_of_item",
           [](GraphIndex& self, std::vector<uint64_t>& items) {
             return self.get_path_of_item(items);
           })
      .def("get_item_of_path",
           [](GraphIndex& self, std::vector<uint32_t>& paths) {
             return self.get_item_of_path(paths);
           })
      .def("update_Jpath_of_item", &GraphIndex::update_Jpath_of_item);
}

void BindTreeIndex(py::module* m) {
  py::class_<TreeIndex, std::shared_ptr<TreeIndex>>(*m, "TreeIndex")
      .def(py::init([](const std::string name, const std::string path) {
        auto index_wrapper = IndexWrapper::GetInstancePtr();
        index_wrapper->insert_tree_index(name, path);
        return index_wrapper->get_tree_index(name);
      }))
      .def("height", [](TreeIndex& self) { return self.Height(); })
      .def("branch", [](TreeIndex& self) { return self.Branch(); })
      .def("total_node_nums",
           [](TreeIndex& self) { return self.TotalNodeNums(); })
      .def("emb_size", [](TreeIndex& self) { return self.EmbSize(); })
      .def("get_all_leafs", [](TreeIndex& self) { return self.GetAllLeafs(); })
      .def("get_nodes",
           [](TreeIndex& self, const std::vector<uint64_t>& codes) {
             return self.GetNodes(codes);
           })
      .def("get_layer_codes",
           [](TreeIndex& self, int level) { return self.GetLayerCodes(level); })
      .def("get_ancestor_codes",
           [](TreeIndex& self, const std::vector<uint64_t>& ids, int level) {
             return self.GetAncestorCodes(ids, level);
           })
      .def("get_children_codes",
           [](TreeIndex& self, uint64_t ancestor, int level) {
             return self.GetChildrenCodes(ancestor, level);
           })
      .def("get_travel_codes",
           [](TreeIndex& self, uint64_t id, int start_level) {
             return self.GetTravelCodes(id, start_level);
           });
}

void BindIndexWrapper(py::module* m) {
  py::class_<IndexWrapper, std::shared_ptr<IndexWrapper>>(*m, "IndexWrapper")
      .def(py::init([]() { return IndexWrapper::GetInstancePtr(); }))
      .def("insert_tree_index", &IndexWrapper::insert_tree_index)
      .def("get_tree_index", &IndexWrapper::get_tree_index)
      .def("clear_tree", &IndexWrapper::clear_tree)
      .def("insert_graph_index", &IndexWrapper::insert_graph_index)
      .def("get_graph_index", &IndexWrapper::GetGraphIndex)
      .def("insert_graph_index_by_meta_info",
           &IndexWrapper::insert_graph_index_by_meta_info)
      .def("save_graph_index_to_file", &IndexWrapper::save_graph_index_to_file);
}

using paddle::distributed::IndexSampler;
using paddle::distributed::LayerWiseSampler;

void BindIndexSampler(py::module* m) {
  py::class_<IndexSampler, std::shared_ptr<IndexSampler>>(*m, "IndexSampler")
      .def(py::init([](const std::string& mode, const std::string& name) {
        if (mode == "by_layerwise") {
          return IndexSampler::Init<LayerWiseSampler>(name);
        } else {
          PADDLE_THROW(platform::errors::InvalidArgument(
              "Unsupported IndexSampler Type!"));
        }
      }))
      .def("init_layerwise_conf", &IndexSampler::init_layerwise_conf)
      .def("init_beamsearch_conf", &IndexSampler::init_beamsearch_conf)
      .def("sample", &IndexSampler::sample);
}

}  // end namespace pybind
}  // namespace paddle
