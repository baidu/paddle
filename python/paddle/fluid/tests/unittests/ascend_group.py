# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

import os
import sys
import time
import paddle.fluid as fluid
from paddle.fluid import unique_name
import paddle.fluid.core as core
import paddle
from paddle.fluid.layer_helper import LayerHelper
from paddle.distributed import fleet
from paddle.distributed.fleet.meta_optimizers.ascend import ascend_parser, ascend_optimizer
from collections import namedtuple

Block = namedtuple('Block', ['program'])
Loss = namedtuple('Loss', ['block'])

paddle.enable_static()

OpRole = core.op_proto_and_checker_maker.OpRole
OP_ROLE_KEY = core.op_proto_and_checker_maker.kOpRoleAttrName()
OP_ROLE_VAR_KEY = core.op_proto_and_checker_maker.kOpRoleVarAttrName()

role = fleet.PaddleCloudRoleMaker(is_collective=True)
fleet.init(role)


def init_communicator(startup_program, main_program, current_endpoint,
                      endpoints, ring_id):
    nranks = len(endpoints)
    other_endpoints = endpoints[:]
    other_endpoints.remove(current_endpoint)
    group_rank = endpoints.index(current_endpoint)
    assert group_rank >= 0

    block = startup_program.global_block()
    nccl_id_var = block.create_var(
        name=unique_name.generate('nccl_id'),
        persistable=True,
        type=core.VarDesc.VarType.RAW)
    block.append_op(
        type='c_gen_nccl_id',
        inputs={},
        outputs={'Out': nccl_id_var},
        attrs={
            'rank': group_rank,
            'endpoint': current_endpoint,
            'other_endpoints': other_endpoints,
            OP_ROLE_KEY: OpRole.Forward,
        })
    block.append_op(
        type='c_comm_init',
        inputs={'X': nccl_id_var},
        outputs={},
        attrs={
            'nranks': nranks,
            'rank': group_rank,
            'ring_id': ring_id,
            OP_ROLE_KEY: OpRole.Forward,
        })

    # add input op for test
    fill_var_name = "tensor@Filled"
    fill_var = block.create_var(
        name=fill_var_name,
        shape=[10, 10],
        dtype='float32',
        persistable=False,
        stop_gradient=True)
    block.append_op(
        type="fill_constant",
        outputs={"Out": fill_var_name},
        attrs={
            "shape": [10, 10],
            "dtype": fill_var.dtype,
            "value": 1.0,
            "place_type": 1
        })

    with fluid.program_guard(main_program):
        op_type = "c_allreduce_sum"
        data = fluid.layers.fill_constant(shape=[1], dtype='float32', value=2.5)
        helper = LayerHelper(op_type, **locals())
        helper.append_op(
            type=op_type,
            inputs={'X': [data]},
            outputs={'Out': [data]},
            attrs={'ring_id': ring_id,
                   'use_calc_stream': True})

    print("startup program:", startup_program)
    print("main program:", main_program)


def train(world_endpoints, world_device_ids, local_device_ids, local_rank):
    startup_programs = []
    main_programs = []

    #trainer_endpoints=["127.0.0.1:6071","127.0.0.1:6072","127.0.0.1:6073","127.0.0.1:6074"]
    trainer_endpoints = world_endpoints
    groups = [[], [], []]
    groups[0] = [trainer_endpoints[0], trainer_endpoints[1]]
    groups[1] = [trainer_endpoints[2], trainer_endpoints[3]]
    groups[2] = [trainer_endpoints[0], trainer_endpoints[2]]
    print("groups:", groups)

    for i in range(len(trainer_endpoints)):
        startup_programs.append(fluid.Program())
        main_programs.append(fluid.Program())

    for idx, group in enumerate(groups):
        for te in group:
            te_idx = trainer_endpoints.index(te)
            startup_program = startup_programs[te_idx]
            main_program = main_programs[te_idx]
            init_communicator(startup_program, main_program, te, group, idx)

    print(len(startup_programs))
    print(startup_programs[local_rank])
    print(main_programs[local_rank])

    print("local rank: ", local_rank)
    print("local startup program: ", startup_programs[local_rank])

    startup_program = startup_programs[local_rank]
    main_program = main_programs[local_rank]
    loss = Loss(Block(main_program))
    optimizer = ascend_optimizer.AscendOptimizer(None, fetch_list=[])
    optimizer.minimize(
        loss,
        startup_program,
        auto_dp=True,
        rank_table_file=os.getenv("RANK_TABLE_FILE", None))

    exe = paddle.static.Executor(paddle.CPUPlace())
    exe.run(startup_program)
    exe.run(main_program)


worker_endpoints = fleet.worker_endpoints()
world_device_ids = fleet.world_device_ids()
local_device_ids = fleet.local_device_ids()
local_rank = int(fleet.local_rank())

print("worker_endpoints:", worker_endpoints)
print("world_device_ids:", world_device_ids)
print("local_device_ids:", local_device_ids)
print("local_rank:", local_rank)

train(worker_endpoints, world_device_ids, local_device_ids, local_rank)
