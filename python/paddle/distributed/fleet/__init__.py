#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# TODO: define distributed api under this directory,
from .base.role_maker import Role, UserDefinedRoleMaker, PaddleCloudRoleMaker
from .base.distributed_strategy import DistributedStrategy
from .base.fleet_base import Fleet
from .base.util_factory import UtilBase
from .dataset import *
from .data_generator import MultiSlotDataGenerator, MultiSlotStringDataGenerator
from . import metrics

__all__ = [
    "DistributedStrategy",
    "UtilBase",
    "UserDefinedRoleMaker",
    "PaddleCloudRoleMaker",
    "Fleet",
    "MultiSlotDataGenerator",
    "MultiSlotStringDataGenerator",
    "Role",
]

fleet = Fleet()
_final_strategy = fleet._final_strategy
_get_applied_meta_list = fleet._get_applied_meta_list
_get_applied_graph_list = fleet._get_applied_graph_list
init = fleet.init
is_first_worker = fleet.is_first_worker
worker_index = fleet.worker_index
worker_num = fleet.worker_num
is_worker = fleet.is_worker
worker_endpoints = fleet.worker_endpoints
server_num = fleet.server_num
server_index = fleet.server_index
server_endpoints = fleet.server_endpoints
is_server = fleet.is_server
util = UtilBase()
barrier_worker = fleet.barrier_worker
init_worker = fleet.init_worker
init_server = fleet.init_server
run_server = fleet.run_server
stop_worker = fleet.stop_worker
distributed_optimizer = fleet.distributed_optimizer
save_inference_model = fleet.save_inference_model
save_persistables = fleet.save_persistables
minimize = fleet.minimize
distributed_model = fleet.distributed_model
step = fleet.step
clear_grad = fleet.clear_grad
set_lr = fleet.set_lr
get_lr = fleet.get_lr
state_dict = fleet.state_dict
set_state_dict = fleet.set_state_dict
