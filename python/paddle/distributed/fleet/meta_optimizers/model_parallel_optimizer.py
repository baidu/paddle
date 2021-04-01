#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import print_function
from __future__ import division

import paddle.fluid as fluid
from paddle.fluid import core, unique_name
from ..base.private_helper_function import wait_server_ready
from .meta_optimizer_base import MetaOptimizerBase
from .common import OpRole, OP_ROLE_KEY, OP_ROLE_VAR_KEY, CollectiveHelper, is_update_op, is_loss_grad_op, is_backward_op, is_optimizer_op


class ModelParallelHelper(object):
    def __init__(self, role_maker, wait_port=True, megatron_dp=False):
        self.wait_port = wait_port
        self.role_maker = role_maker
        self.megatron_dp = megatron_dp

    def update_startup_program(self,
                               startup_program=None,
                               inner_parallelism=None):
        self.startup_program = startup_program

        nranks = self.role_maker._worker_num()
        rank = self.role_maker._worker_index()
        endpoints = self.role_maker._get_trainer_endpoints()
        current_endpoint = endpoints[rank]

        # Create ring 0 for all model parallel parts within a single model
        mp_endpoints = []
        mp_rank = rank % inner_parallelism
        mp_id = rank // inner_parallelism
        for idx, ep in enumerate(endpoints):
            if idx // inner_parallelism == mp_id:
                mp_endpoints.append(ep)
        print("model parallel eps:{}, rank{}".format(mp_endpoints, mp_rank))
        self._init_communicator(self.startup_program, current_endpoint,
                                mp_endpoints, mp_rank, 0, self.wait_port)
        self._broadcast_params(0, broadcast_distributed_weight=False)

        print("megatron group size: {}".format(inner_parallelism))
        print("megatron rank: {}".format(mp_rank))
        print("megatron endpoints: {}".format(mp_endpoints))

        if self.megatron_dp:
            mp_num = len(endpoints) // inner_parallelism
            if mp_num == 1: return
            # Create rings for gpus as the same model parallel part
            eps = []
            dp_rank = rank // inner_parallelism
            dp_id = rank % inner_parallelism
            #if dp_rank == 1: dp_rank =0
            #if dp_rank == 0: dp_rank =1
            ring_id = 1
            for idx, ep in enumerate(endpoints):
                if idx % inner_parallelism == dp_id:
                    eps.append(ep)
            #ep = eps.pop(0)
            #eps.insert(1, ep)
            print("data parallel eps:{}, rank{}".format(eps, dp_rank))
            self._init_communicator(self.startup_program, current_endpoint, eps,
                                    dp_rank, ring_id, self.wait_port)
            self._broadcast_params(ring_id, broadcast_distributed_weight=True)

    def _init_communicator(self, program, current_endpoint, endpoints, rank,
                           ring_id, wait_port):
        nranks = len(endpoints)
        other_endpoints = endpoints[:]
        other_endpoints.remove(current_endpoint)
        if rank == 0 and wait_port:
            wait_server_ready(other_endpoints)

        block = program.global_block()
        nccl_id_var = block.create_var(
            name=unique_name.generate('nccl_id'),
            persistable=True,
            type=core.VarDesc.VarType.RAW)
        block.append_op(
            type='c_gen_nccl_id',
            inputs={},
            outputs={'Out': nccl_id_var},
            attrs={
                'rank': rank,
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
                'rank': rank,
                'ring_id': ring_id,
                OP_ROLE_KEY: OpRole.Forward,
            })

    def _broadcast_params(self, ring_id, broadcast_distributed_weight):
        block = self.startup_program.global_block()
        for param in block.iter_parameters():
            if not broadcast_distributed_weight and param.is_distributed:
                continue

            block.append_op(
                type='c_broadcast',
                inputs={'X': param},
                outputs={'Out': param},
                attrs={
                    'ring_id': ring_id,
                    'root': 0,
                    OP_ROLE_KEY: OpRole.Forward
                })

        block.append_op(
            type='c_sync_comm_stream',
            inputs={'X': param},
            outputs={'Out': param},
            attrs={'ring_id': ring_id,
                   OP_ROLE_KEY: OpRole.Forward})


class ModelParallelOptimizer(MetaOptimizerBase):
    def __init__(self, optimizer):
        super(ModelParallelOptimizer, self).__init__(optimizer)
        self.inner_opt = optimizer
        self.meta_optimizers_white_list = [
            "RecomputeOptimizer",
            "AMPOptimizer",
            "LarsOptimizer",
            "LambOptimizer",
        ]
        self.meta_optimizers_black_list = ["GraphExecutionOptimizer", ]
        self.megatron_dp = False

    def _set_basic_info(self, loss, role_maker, user_defined_optimizer,
                        user_defined_strategy):
        super(ModelParallelOptimizer, self)._set_basic_info(
            loss, role_maker, user_defined_optimizer, user_defined_strategy)
        self.inner_parallelism = user_defined_strategy.model_parallel_configs[
            'parallelism']

    def _can_apply(self):
        if not self.role_maker._is_collective:
            return False

        if self.user_defined_strategy.model_parallel == True:
            return True
        return False

    def _disable_strategy(self, dist_strategy):
        dist_strategy.model_parallel = False
        dist_strategy.model_parallel_configs = {}

    def _enable_strategy(self, dist_strategy, context):
        dist_strategy.model_parallel = True
        dist_strategy.model_parallel_configs = {"parallelism": 1, }

    # the following function will be used by AMP if both Megatron and AMP are turn on together.
    def apply_gradients(self, params_grads):
        return self.minimize_impl(params_grads=params_grads)

    def minimize_impl(self,
                      loss,
                      startup_program=None,
                      parameter_list=None,
                      no_grad_set=None):
        endpoints = self.role_maker._get_trainer_endpoints()
        current_endpoint = endpoints[self.role_maker._worker_index()]
        self.startup_program = startup_program
        if startup_program is None:
            self.startup_program = fluid.default_startup_program()

        # (TODO) check the order of metaoptimizer
        # (TODO) check the params_grads
        optimize_ops, params_grads = self.inner_opt.minimize(
            loss, self.startup_program, parameter_list, no_grad_set)

        self.main_program = loss.block.program
        self.inner_parallelism = self.inner_parallelism
        self.nranks = len(endpoints)

        pipeline_helper = ModelParallelHelper(self.role_maker)
        pipeline_helper.update_startup_program(self.startup_program,
                                               self.inner_parallelism)

        assert self.nranks % self.inner_parallelism == 0

        if self.megatron_dp:
            # data parallelism
            dp_parallelism = self.nranks // self.inner_parallelism

            self._transpile_main_program(loss, dp_parallelism)
        return optimize_ops, params_grads

    def _transpile_main_program(self, loss, dp_parallelism):
        self._insert_loss_grad_ops(loss, dp_parallelism)
        ring_id = 1
        print("ring_id: ", ring_id)
        # for ring_id in range(1, dp_parallelism + 1):
        self._insert_allreduce_ops(loss, ring_id)

    def _insert_loss_grad_ops(self, loss, dp_parallelism):
        """
        In order to keep the learning rate consistent in different numbers of
        training workers, we scale the loss grad by the number of workers
        """
        block = loss.block
        for idx, op in reversed(list(enumerate(block.ops))):
            if is_loss_grad_op(op):
                loss_grad_var = block.vars[op.output_arg_names[0]]
                block._insert_op(
                    idx + 1,
                    type='scale',
                    inputs={'X': loss_grad_var},
                    outputs={'Out': loss_grad_var},
                    attrs={
                        'scale': 1.0 / dp_parallelism,
                        OP_ROLE_KEY: OpRole.Backward
                    })

    def _insert_allreduce_ops(self, loss, ring_id):
        block = loss.block
        grad = None
        for idx, op in reversed(list(enumerate(block.ops))):
            if is_backward_op(op) and \
                    OP_ROLE_VAR_KEY in op.attr_names:
                op_role_var = op.all_attrs()[OP_ROLE_VAR_KEY]
                if len(op_role_var) == 0:
                    continue
                assert len(op_role_var) % 2 == 0
                offset = idx
                for i in range(0, len(op_role_var), 2):
                    param = block.vars[op_role_var[i]]
                    grad = block.vars[op_role_var[i + 1]]
                    #if param.is_distributed:
                    #    continue
                    if offset == idx:
                        offset += 1
                        block._insert_op(
                            offset,
                            type='c_sync_calc_stream',
                            inputs={'X': grad},
                            outputs={'Out': grad},
                            attrs={OP_ROLE_KEY: OpRole.Backward})
                        offset += 1

                    block._insert_op(
                        offset,
                        type='c_allreduce_sum',
                        inputs={'X': grad},
                        outputs={'Out': grad},
                        attrs={
                            'ring_id': ring_id,
                            OP_ROLE_KEY: OpRole.Backward
                        })

        if grad is None:
            return

        for idx, op in list(enumerate(block.ops)):
            if is_optimizer_op(op):
                block._insert_op(
                    idx,
                    type='c_sync_comm_stream',
                    inputs={'X': grad},
                    outputs={'Out': grad},
                    attrs={'ring_id': ring_id,
                           OP_ROLE_KEY: OpRole.Backward})
            break
