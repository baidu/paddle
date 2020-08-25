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

import paddle
from paddle.fluid.framework import core
from paddle.fluid import compiler
from .meta_optimizer_base import MetaOptimizerBase
from ..base.private_helper_function import wait_server_ready
import logging


class GraphExecutionOptimizer(MetaOptimizerBase):
    def __init__(self, optimizer):
        super(GraphExecutionOptimizer, self).__init__(optimizer)
        self.inner_opt = optimizer
        # we do not allow meta optimizer to be inner optimizer currently
        self.meta_optimizers_white_list = []
        self.meta_optimizers_black_list = []

    def _is_graph_out(self):
        return True

    def _can_apply(self):
        """
        Basically, this is PE, and almost all programs can be executed here
        """
        if not self.role_maker._is_collective:
            # update me. currently, if parameter server is used
            # graph execution optimizer can not be applied
            return False
        return True

    def backward(self,
                 loss,
                 startup_program=None,
                 parameter_list=None,
                 no_grad_set=None,
                 callbacks=None):
        pass

    # should fix the variable 
    def _setup_nccl_op(self, startup_program, main_program, build_strategy):
        trainer_endpoints = self.role_maker.get_trainer_endpoints()
        trainers = trainer_endpoints
        trainer_id = self.role_maker.worker_index()
        current_endpoint = self.role_maker.get_trainer_endpoints()[trainer_id]
        trainer_endpoints_env = ",".join(trainer_endpoints)
        trainers_num = self.role_maker.worker_num()
        nccl_id_var = startup_program.global_block().create_var(
            name="NCCLID", persistable=True, type=core.VarDesc.VarType.RAW)
        for i in range(1, build_strategy.nccl_comm_num):
            startup_program.global_block().create_var(
                name="NCCLID_{}".format(i),
                persistable=True,
                type=core.VarDesc.VarType.RAW)

        if build_strategy.use_hierarchical_allreduce:
            for i in range(0, build_strategy.nccl_comm_num):
                startup_program.global_block().create_var(
                    name="Hierarchical_inter_NCCLID_{}".format(i),
                    persistable=True,
                    type=core.VarDesc.VarType.RAW)
                startup_program.global_block().create_var(
                    name="Hierarchical_exter_NCCLID_{}".format(i),
                    persistable=True,
                    type=core.VarDesc.VarType.RAW)

        startup_program.global_block().append_op(
            type="gen_nccl_id",
            inputs={},
            outputs={"NCCLID": nccl_id_var},
            attrs={
                "trainers": trainer_endpoints,
                "trainer_id": trainer_id,
                "nccl_comm_num": build_strategy.nccl_comm_num,
                "use_hierarchical_allreduce":
                build_strategy.use_hierarchical_allreduce,
                "hierarchical_allreduce_inter_ranks":
                build_strategy.hierarchical_allreduce_inter_nranks
            })

    def _try_to_compile(self, startup_program, main_program, loss):
        import copy
        dist_strategy = self.user_defined_strategy
        local_build_strategy = paddle.fluid.BuildStrategy()
        local_build_strategy.enable_sequential_execution = \
                    dist_strategy.build_strategy.enable_sequential_execution
        local_build_strategy.fuse_elewise_add_act_ops = \
                    dist_strategy.build_strategy.fuse_elewise_add_act_ops
        local_build_strategy.fuse_bn_act_ops = \
                    dist_strategy.build_strategy.fuse_bn_act_ops
        local_build_strategy.enable_auto_fusion = \
                    dist_strategy.build_strategy.enable_auto_fusion
        local_build_strategy.fuse_relu_depthwise_conv = \
                    dist_strategy.build_strategy.fuse_relu_depthwise_conv
        local_build_strategy.fuse_broadcast_ops = \
                    dist_strategy.build_strategy.fuse_broadcast_ops
        local_build_strategy.fuse_all_optimizer_ops = \
                    dist_strategy.build_strategy.fuse_all_optimizer_ops
        local_build_strategy.enable_inplace = \
                    dist_strategy.build_strategy.enable_inplace
        local_build_strategy.use_hierarchical_allreduce = \
                    dist_strategy.use_hierarchical_allreduce
        local_build_strategy.hierarchical_allreduce_inter_nranks = \
                    dist_strategy.hierarchical_allreduce_inter_nranks
        local_build_strategy.sync_batch_norm = \
                    dist_strategy.sync_batch_norm
        local_build_strategy.fuse_all_reduce_ops = \
                    dist_strategy.fuse_all_reduce_ops
        local_build_strategy.nccl_comm_num = \
                    dist_strategy.nccl_comm_num
        if dist_strategy.localsgd:
            local_build_strategy.reduce_strategy = \
                    paddle.fluid.BuildStrategy.ReduceStrategy.UserDefined

        if self.user_defined_strategy.recompute == True:
            logging.warn(
                "set enable_sequential_execution=True since you have enable the recompute strategy"
            )
            local_build_strategy.enable_sequential_execution = True

        exe_strategy = self.user_defined_strategy.execution_strategy
        worker_num = self.role_maker.worker_num()
        node_num = self.role_maker.node_num()

        if self.role_maker._is_collective:
            assert worker_num >= 1, "nccl2 worker_num must >= 1, now:{}" % worker_num

        if worker_num <= 1:
            # local mode
            if local_build_strategy.nccl_comm_num > 1:
                logging.warn("set nccl_comm_num=1 since you only have 1 node.")
            local_build_strategy.nccl_comm_num = 1

        if node_num <= 1:
            if local_build_strategy.use_hierarchical_allreduce:
                logging.warn(
                    "set hierachical_allreduce=False since you only have 1 node."
                )
            local_build_strategy.use_hierarchical_allreduce = False

        sync_allreduce = dist_strategy.sync_nccl_allreduce
        if sync_allreduce:
            paddle.fluid.framework.set_flags({
                "FLAGS_sync_nccl_allreduce": True
            })
            exe_strategy.num_threads = local_build_strategy.nccl_comm_num + 1
            if local_build_strategy.use_hierarchical_allreduce:
                exe_strategy.num_threads = 2 * local_build_strategy.nccl_comm_num + 1
            if exe_strategy.num_threads > 4:
                logging.warn(
                    "if you use hierachical_allreduce or "
                    "with multi nccl comm, please set distributed_strategy.sync_nccl_allreduce=False"
                )

        sync_batch_norm = local_build_strategy.sync_batch_norm
        if sync_batch_norm:
            local_build_strategy.nccl_comm_num = 1
            local_build_strategy.use_hierarchical_allreduce = False
            exe_strategy.num_threads = 1
            logging.warn(
                "use sync_batch_norm will hang when set num_threads > 1, so "
                "set num_threads=1, nccl_comm_num=1, hierachical_allreduce=False."
            )

        # TODO(guru4elephant): should be an independent optimizer
        self._setup_nccl_op(startup_program, main_program, local_build_strategy)

        local_build_strategy.num_trainers = self.role_maker.worker_num()
        local_build_strategy.trainer_id = self.role_maker.worker_index()
        local_build_strategy.trainers_endpoints = self.role_maker.get_trainer_endpoints(
        )
        local_build_strategy.enable_backward_optimizer_op_deps = True

        self._compiled_program = compiler.CompiledProgram(main_program)

        self._compiled_program.with_data_parallel(
            loss_name=loss.name,
            build_strategy=local_build_strategy,
            exec_strategy=exe_strategy,
            share_vars_from=None)

        return self._compiled_program

    def _disable_strategy(self, dist_strategy):
        # TODO(guru4elephant): should close all PE related flags here
        pass

    def minimize(self,
                 loss,
                 startup_program=None,
                 parameter_list=None,
                 no_grad_set=None):
        if startup_program == None:
            startup_program = paddle.static.default_startup_program()
        compiled_program = self._try_to_compile(startup_program,
                                                loss.block.program, loss)
        loss.block.program._graph = compiled_program

        # just return self.optimizer_ops and self.param_grads
        return None, None
