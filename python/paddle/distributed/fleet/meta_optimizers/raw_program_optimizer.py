#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
import os
import collections
import numpy as np

import paddle.fluid as fluid
from paddle.fluid import core, unique_name
from paddle.fluid.dygraph import Layer, LayerList
from ..base.private_helper_function import wait_server_ready
from .meta_optimizer_base import MetaOptimizerBase
from .common import OpRole, OP_ROLE_KEY, OP_ROLE_VAR_KEY, CollectiveHelper, is_loss_grad_op, is_backward_op, is_optimizer_op


class RawProgramOptimizer(MetaOptimizerBase):
    def __init__(self, optimizer):
        super(RawProgramOptimizer, self).__init__(optimizer)
        self.inner_opt = optimizer
        self.meta_optimizers_white_list = [
            "RecomputeOptimizer",
            "AMPOptimizer",
        ]
        self.meta_optimizers_black_list = ["GraphExecutionOptimizer", ]
        self.global_ring_id = 0

    def _set_basic_info(self, loss, role_maker, user_defined_optimizer,
                        user_defined_strategy):
        super(RawProgramOptimizer, self)._set_basic_info(
            loss, role_maker, user_defined_optimizer, user_defined_strategy)
        self.without_graph_optimization = user_defined_strategy.without_graph_optimization
        self.fuse_all_reduce_ops = user_defined_strategy.fuse_all_reduce_ops
        if self.fuse_all_reduce_ops:
            self.fuse_grad_size_in_num = user_defined_strategy.fuse_grad_size_in_num
            self.calc_comm_same_stream = user_defined_strategy._calc_comm_same_stream

    def _can_apply(self):
        if not self.role_maker._is_collective:
            return False

        if self.without_graph_optimization == True:
            return True
        return False

    def _disable_strategy(self, dist_strategy):
        dist_strategy.without_graph_optimization = False

    def _enable_strategy(self, dist_strategy, context):
        dist_strategy.without_graph_optimization = True

    def _broadcast_params(self, ring_id):
        block = self.startup_program.global_block()
        param = None
        for param in block.iter_parameters():
            if param.is_distributed:
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

        if not param: return  # no parameter on this device
        block.append_op(
            type='c_sync_comm_stream',
            inputs={'X': param},
            outputs={'Out': param},
            attrs={'ring_id': ring_id,
                   OP_ROLE_KEY: OpRole.Forward})

    def _get_process_group_info(self):
        # global ring info
        self.global_endpoints = self.endpoints
        self.global_rank = self.rank
        self.global_nranks = self.nranks

    def _init_process_group(self):
        self._get_process_group_info()
        collective_helper = CollectiveHelper(self.role_maker, wait_port=False)
        # Create global ring for all gpus (ring_id = 0)
        collective_helper._init_communicator(
            self.startup_program, self.current_endpoint, self.global_endpoints,
            self.global_rank, self.global_ring_id, True, self.global_ring_id,
            True)
        self._broadcast_params(self.global_ring_id)

    def minimize_impl(self,
                      loss,
                      startup_program=None,
                      parameter_list=None,
                      no_grad_set=None):
        self.endpoints = self.role_maker._get_trainer_endpoints()
        self.current_endpoint = self.endpoints[self.role_maker._worker_index()]
        self.rank = self.role_maker._worker_index()
        self.nranks = self.role_maker._worker_num()
        if startup_program is None:
            startup_program = fluid.default_startup_program()
        self.startup_program = startup_program

        block = loss.block
        program = block.program
        self.main_program = program

        optimize_ops, params_grads = self.inner_opt.minimize(
            loss, startup_program, parameter_list, no_grad_set)
        if self.nranks == 1:
            return optimize_ops, params_grads
        self._init_process_group()

        self.main_program = program
        if self.nranks > 1:
            self._transpile_main_program(loss)
        return optimize_ops, params_grads

    def _transpile_main_program(self, loss):
        self._insert_loss_grad_ops(loss)
        if self.fuse_all_reduce_ops:
            self._allreduce_fusion_program()
        else:
            self._insert_allreduce_ops()

    def _insert_loss_grad_ops(self, loss):
        """
        In order to keep the learning rate consistent in different numbers of
        training workers, we scale the loss grad by the number of workers
        """
        block = self.main_program.global_block()
        for idx, op in reversed(list(enumerate(block.ops))):
            if is_loss_grad_op(op):
                loss_grad_var = block.vars[op.output_arg_names[0]]
                block._insert_op(
                    idx + 1,
                    type='scale',
                    inputs={'X': loss_grad_var},
                    outputs={'Out': loss_grad_var},
                    attrs={
                        'scale': 1.0 / self.nranks,
                        OP_ROLE_KEY: OpRole.Backward
                    })

    def _insert_allreduce_ops(self):
        block = self.main_program.global_block()
        ring_id = self.global_ring_id
        grad = None
        for idx, op in reversed(list(enumerate(block.ops))):
            if is_backward_op(op) and \
                    OP_ROLE_VAR_KEY in op.attr_names:
                op_role_var = op.attr(OP_ROLE_VAR_KEY)
                if len(op_role_var) == 0:
                    continue
                assert len(op_role_var) % 2 == 0
                offset = 1
                for i in range(0, len(op_role_var), 2):
                    param_name = op_role_var[i]
                    param = block.var(param_name)
                    grad_name = op_role_var[i + 1]
                    grad = block.var(grad_name)
                    if param.is_distributed:
                        continue

                    block._insert_op(
                        idx + offset,
                        type='c_sync_calc_stream',
                        inputs={'X': grad},
                        outputs={'Out': grad},
                        attrs={OP_ROLE_KEY: OpRole.Backward, })
                    offset += 1
                    block._insert_op(
                        idx + offset,
                        type='c_allreduce_sum',
                        inputs={'X': grad},
                        outputs={'Out': grad},
                        attrs={
                            'ring_id': ring_id,
                            OP_ROLE_KEY: OpRole.Backward
                        })

        if grad is None:
            return

        for idx, op in enumerate(block.ops):
            if is_optimizer_op(op):
                block._insert_op(
                    idx,
                    type='c_sync_comm_stream',
                    inputs={'X': grad},
                    outputs={'Out': grad},
                    attrs={'ring_id': ring_id,
                           OP_ROLE_KEY: OpRole.Backward})
                break

    # This function helps reduce the number of allreduce by integrating op, which can save communication time.
    # to use allreduce fuse, follow these codes:
    # strategy = paddle.distributed.fleet.DistributedStrategy()
    # strategy.without_graph_optimization = True
    # strategy.fuse_all_reduce_ops = True
    # strategy.calc_comm_same_stream = False
    # strategy.fuse_grad_size_in_num = 8
    def _allreduce_fusion_program(self):
        block = self.main_program.global_block()
        ring_id = self.global_ring_id
        record_idx, allreduce_input_vars, allreduce_output_vars = [], [], []
        ops = list(enumerate(block.ops))

        for idx, op in reversed(ops):
            # we travers the ops reversely
            if is_backward_op(op) and \
                    OP_ROLE_VAR_KEY in op.attr_names:
                op_role_var = op.attr(OP_ROLE_VAR_KEY)
                if len(op_role_var) == 0:
                    continue
                assert len(op_role_var) % 2 == 0, "vars need to be one param var followed by one grad var, " \
                                                  "but got odd number of vars"
                for i in range(0, len(op_role_var), 2):
                    # handle vars in each op, each time handle a param and a grad
                    param_name = op_role_var[i]
                    param = block.var(param_name)
                    grad_name = op_role_var[i + 1]
                    grad = block.var(grad_name)
                    if param.is_distributed:
                        continue
                    if ".cast_fp16@GRAD" in grad_name:
                        # when amp=True get the fp16 param
                        param_name = param_name + ".cast_fp16"
                        if not block.has_var(param_name):
                            raise ValueError("op cast name error {}".format(
                                op.type))
                        else:
                            param = block.var(param_name)

                    if len(allreduce_output_vars) == 0 or \
                            len(allreduce_output_vars[-1]) == \
                            self.fuse_grad_size_in_num:
                        # start of the fusion or last group meets the config size
                        allreduce_output_vars.append([grad])
                        allreduce_input_vars.append([param])
                        # add the start and end idx to the record idx
                        record_idx.append([idx, idx])
                    else:
                        # Current group's size is below the config size
                        # append grad and param to the last group (current group)
                        # update the start idx to current op's idx
                        # Since we travers the ops reversely, the idx is descending
                        # we update the first entry of each entry for record_idx
                        allreduce_output_vars[-1].append(grad)
                        allreduce_input_vars[-1].append(param)
                        record_idx[-1][0] = idx

        assert len(allreduce_output_vars) == len(
            record_idx
        ), "It has different lens between the allreduce_output_vars and record_idx."

        if not allreduce_output_vars or not allreduce_input_vars:
            # nothing needs to be allreduced
            return

        self.vars = collections.OrderedDict()
        index, pos, offset = 0, 0, 0
        start, end = record_idx[index]
        for idx, op in reversed(ops):
            if idx == start:
                pos = 0
                done_output_vars, done_input_vars = self._split_fuction(
                    allreduce_output_vars[index],  # grad
                    allreduce_input_vars[index]  # param
                )
                for id_, done_output_var in enumerate(done_output_vars):
                    tmp_var = block.create_var(
                        name=unique_name.generate('FusedOutput_{}'.format(
                            done_output_var[0].name)),
                        dtype=done_output_var[0].dtype,
                        persistable=False,
                        stop_gradient=True)
                    self.vars['FusedOutput_{}'.format(done_output_var[0]
                                                      .name)] = tmp_var

                    block._insert_op(
                        idx + id_,
                        type="coalesce_tensor",
                        inputs={"Input": done_input_vars[id_]},
                        outputs={
                            "Output": done_output_var,
                            "FusedOutput": tmp_var
                        },
                        attrs={
                            "copy_data": False,
                            "use_align": True,
                            "dtype": done_output_var[0].dtype,
                            OP_ROLE_KEY: OpRole.Backward
                        })
                    pos += 1

                for id_ in range(len(done_output_vars)):
                    x = self.vars['FusedOutput_{}'.format(done_output_vars[id_][
                        0].name)]
                    out = x

                    # NOTE: there still some optimize space if use EVENT instead of sync
                    if not self.calc_comm_same_stream:
                        # need sync if the calc and comm stream are not the same
                        block._insert_op(
                            end + id_ + pos + 1,
                            type='c_sync_calc_stream',
                            inputs={'X': x},
                            outputs={'Out': out},
                            attrs={OP_ROLE_KEY: OpRole.Backward})

                    block._insert_op(
                        end + id_ + pos + 1
                        if self.calc_comm_same_stream else end + id_ + pos + 2,
                        type='c_allreduce_sum',
                        inputs={'X': x},
                        outputs={'Out': out},
                        attrs={
                            'ring_id': ring_id,
                            'use_calc_stream': self.calc_comm_same_stream,
                            OP_ROLE_KEY: OpRole.Backward
                        })

                index += 1
                if len(record_idx) == index:
                    break
                start, end = record_idx[index]

        if not self.calc_comm_same_stream:
            # need sync if the calc and comm stream are not the same
            for idx, op in enumerate(block.ops):
                if is_optimizer_op(op):
                    block._insert_op(
                        idx,
                        type='c_sync_comm_stream',
                        inputs={'X': block.create_var()},
                        outputs={'Out': block.create_var()},
                        attrs={
                            'ring_id': ring_id,
                            OP_ROLE_KEY: OpRole.Backward
                        })
                    break

    # Integrate grads of the same type to form a combination.
    # If combination is selected, will return grads of the same type in a groups.
    # For example:[(fp16, fp16), (fp32), (fp16)] -> [(fp16, fp16, fp16), (fp32)]
    def _split_fuction(self,
                       allreduce_output_vars,
                       allreduce_input_vars,
                       combination=True):
        input_vars, final_input_vars, output_vars, final_output_vars = [], [], [], []
        if len(allreduce_output_vars) == 1:
            # only have one var to handle
            final_output_vars.append(allreduce_output_vars)
            final_input_vars.append(allreduce_input_vars)
            return final_output_vars, final_input_vars

        for idx in range(len(allreduce_input_vars) - 1):
            # the last var needs to be handled differently
            if allreduce_input_vars[idx].dtype == allreduce_input_vars[idx +
                                                                       1].dtype:
                # if current var and next var are in same type
                # append current var to input_vars
                input_vars.append(allreduce_input_vars[idx])
                if idx == len(allreduce_input_vars) - 2:
                    # if current var is the second last var
                    # append the last var to input_vars
                    # and update the final_input_vars
                    input_vars.append(allreduce_input_vars[idx + 1])
                    final_input_vars.append(input_vars)
            else:
                # the current var and next var are in different types
                # append current var to input_vars
                # update the final_input_vars
                # reset input_vars to receive a new type
                input_vars.append(allreduce_input_vars[idx])
                final_input_vars.append(input_vars)
                input_vars = []
                if idx == len(allreduce_input_vars) - 2:
                    # if current var is the second last var
                    # append the last var to a reset input_vars since they are in different types
                    # and update the final_input_vars
                    input_vars.append(allreduce_input_vars[idx + 1])
                    final_input_vars.append(input_vars)

        for idx in range(len(allreduce_output_vars) - 1):
            # the procedure for the output vars is the same with that for the input vars
            if allreduce_output_vars[idx].dtype == allreduce_output_vars[
                    idx + 1].dtype:
                output_vars.append(allreduce_output_vars[idx])
                if idx == len(allreduce_output_vars) - 2:
                    output_vars.append(allreduce_output_vars[idx + 1])
                    final_output_vars.append(output_vars)
            else:
                output_vars.append(allreduce_output_vars[idx])
                final_output_vars.append(output_vars)
                output_vars = []
                if idx == len(allreduce_output_vars) - 2:
                    output_vars.append(allreduce_output_vars[idx + 1])
                    final_output_vars.append(output_vars)

        # at this time, all vars in each group in final_input_vars and final_output_vars are in the same type

        if combination:
            input_fp16_vars, input_fp32_vars, output_fp16_vars, output_fp32_vars = [], [], [], []
            for final_input_var in final_input_vars:
                if final_input_var[0].dtype == core.VarDesc.VarType.FP16:
                    # extend the group
                    input_fp16_vars.extend(final_input_var)
                else:
                    input_fp32_vars.extend(final_input_var)

            for final_output_var in final_output_vars:
                if final_output_var[0].dtype == core.VarDesc.VarType.FP16:
                    output_fp16_vars.extend(final_output_var)
                else:
                    output_fp32_vars.extend(final_output_var)

            final_output_vars, final_input_vars = [], []
            if output_fp16_vars:
                final_output_vars.append(output_fp16_vars)
            if output_fp32_vars:
                final_output_vars.append(output_fp32_vars)
            if input_fp16_vars:
                final_input_vars.append(input_fp16_vars)
            if input_fp32_vars:
                final_input_vars.append(input_fp32_vars)

        return final_output_vars, final_input_vars
