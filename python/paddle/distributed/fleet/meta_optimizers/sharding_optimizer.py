# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
from paddle.fluid import unique_name, core
import paddle.fluid as fluid
import paddle.distributed.fleet as fleet
from paddle.distributed.fleet.meta_optimizers.common import OpRole, OP_ROLE_VAR_KEY, CollectiveHelper
from paddle.distributed.fleet.meta_optimizers.common import is_backward_op
from paddle.distributed.fleet.meta_optimizers.meta_optimizer_base import MetaOptimizerBase
from paddle.distributed.fleet.meta_optimizers.sharding.shard import Shard, ProgramSegment
from paddle.distributed.fleet.meta_optimizers.sharding.fp16_helper import FP16Utils
from paddle.distributed.fleet.meta_optimizers.sharding.weight_decay_helper import WeightDecayHelper
from paddle.distributed.fleet.meta_optimizers.sharding.gradient_clip_helper import GradientClipHelper
from paddle.distributed.fleet.meta_optimizers.sharding.prune import ProgramDeps
from paddle.distributed.fleet.meta_optimizers.sharding.utils import *
from paddle.fluid.framework import Program, Variable, name_scope, default_main_program, default_startup_program, device_guard

from paddle.fluid import layers

import logging
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')
from functools import reduce

__all__ = ["ShardingOptimizer"]


class ShardingOptimizer(MetaOptimizerBase):
    def __init__(self, optimizer):
        super(ShardingOptimizer, self).__init__(optimizer)
        self.inner_opt = optimizer
        self.meta_optimizers_white_list = [
            "RecomputeOptimizer",
            "AMPOptimizer",
            "LarsOptimizer",
            "LambOptimizer",
            "ModelParallelOptimizer",
        ]
        self.meta_optimizers_black_list = ["GraphExecutionOptimizer", ]
        self._main_program = None
        self._startup_program = None
        self._segments = []
        # params and fp16 params is for broadcast
        self._params = set([])
        self._broadcast_vars = set([])
        # reduced grads to param name
        self._reduced_grads_to_param = {}
        self._shard = Shard()

        # use sharding as outer parallelism (e.g. inner:Megatron & outer sharding)
        self._as_outer_parallelism = False
        self._inner_parallelism_size = None

    def _can_apply(self):
        if not self.role_maker._is_collective:
            return False
        if self.role_maker._worker_num() <= 1:
            return False
        return self.user_defined_strategy.sharding

    def _disable_strategy(self, dist_strategy):
        dist_strategy.sharding = False
        dist_strategy.sharding_configs = {}

    def _enable_strategy(self, dist_strategy, context):
        dist_strategy.sharding = True
        dist_strategy.sharding_configs = {"broadcast_MB": 32}

    def minimize_impl(self,
                      loss,
                      startup_program=None,
                      parameter_list=None,
                      no_grad_set=None):
        # TODO: (JZ-LIANG) support multiple comm in future
        # self._nrings = self.user_defined_strategy.nccl_comm_num
        logging.info("########### into sharding minimize: wait at end")
        self._nrings_sharding = 1
        self._nrings_dp = 1
        self.hybrid_dp = self.user_defined_strategy.sharding_configs[
            "hybrid_dp"]
        self._as_outer_parallelism = self.user_defined_strategy.sharding_configs[
            "as_outer_parallelism"]
        self._inner_parallelism_size = int(
            self.user_defined_strategy.sharding_configs[
                "inner_parallelism_size"])
        self._sharding_segment_strategy = str(self.user_defined_strategy.sharding_configs[
            "sharding_segment_strategy"])

        if self._sharding_segment_strategy == "broadcast_size":
            self._broadcast_MB = int(self.user_defined_strategy.sharding_configs[
                "broadcast_MB"])
            assert self._broadcast_MB > 0, "segment size should larger than zero !" 
        elif self._sharding_segment_strategy == "anchors":
            self._sharding_segment_anchors = self.user_defined_strategy.sharding_configs[
                "sharding_segment_anchors"]
            assert len(self._sharding_segment_anchors) > 0, "you should set the sharding segment anchors !" 
            self._backward_remain_anchors = self._sharding_segment_anchors[:]
            self._forward_remain_anchors = []
        else:
            raise NotImplementedError(
                "the sharding segment strategy [{}] is not implemented".format(str(self._sharding_segment_strategy)))   
        self._gradient_merge_acc_step = int(self.user_defined_strategy.sharding_configs[
                "gradient_merge_acc_step"])
        self._grad2merged_grad = dict()
        
        if self.inner_opt is None:
            raise ValueError(
                "self.inner_opt of ShardingOptimizer should not be None.")
        optimize_ops, params_grads = self.inner_opt.minimize(
            loss, startup_program, parameter_list, no_grad_set)
        logging.info("########### after inner opt")

        if startup_program is None:
            startup_program = default_startup_program()
        main_block = loss.block
        startup_block = startup_program.global_block()
        self._main_program = main_block.program
        self._startup_program = startup_program

        # step1: set_up
        self._set_up(params_grads)
        logging.info("########### after inner _set_up")

        # step2: split_program
        self._split_program(main_block)
        logging.info("########### after inner _split_program")

        # step3: add broadcast and reduce ops
        self._add_broadcast_allreduce(main_block)
        main_block._sync_with_cpp()
        startup_block._sync_with_cpp()
        logging.info("########### after inner _add_broadcast_allreduce")

        # step4: insert reduce_sum for grad
        grad_scale_coeff = self.role_maker._worker_num()
        if self._as_outer_parallelism:
            grad_scale_coeff = grad_scale_coeff / self._inner_parallelism_size
        insert_scale_loss_grad_ops(main_block, scale=1.0 / grad_scale_coeff)
        main_block._sync_with_cpp()

        # step5: remove unneeded ops and vars from block
        self._prune_main_program(main_block)
        self._prune_startup_program(startup_block)
        if self.hybrid_dp:
            self._initialization_broadcast(startup_program)
        logging.info("########### after inner prune")

        with open("main_before_gm_%d" % self.role_maker._worker_index(), 'w') as f:
            f.writelines(str(self._main_program))
    
        # step6: optional gradient merge
        if self._gradient_merge_acc_step > 1:
            self._sharding_gradient_merge(main_block)

        with open("main_after_gm_%d" % self.role_maker._worker_index(), 'w') as f:
            f.writelines(str(self._main_program))

        # # check op dependecy
        # check_broadcast(main_block)
        # check_allreduce_sum(main_block, self._shard, self.sharding_ring_id,
        #                     self.dp_ring_id)
        logging.info("########### after inner check")
        self._wait()
        logging.info("########### after inner wait")


        return optimize_ops, params_grads

    def _set_up(self, params_grads):
        # step 1: initialize nccl
        self.global_word_size = self.role_maker._worker_num()
        self.global_rank = self.role_maker._worker_index()
        self.endpoints = self.role_maker._get_trainer_endpoints()
        self.current_endpoint = self.endpoints[self.global_rank]
        self._collective_helper = CollectiveHelper(self.role_maker,
                                                   self._nrings_sharding)
        # config sharding & dp groups
        self._init_comm()
        # sharding
        self._collective_helper._init_communicator(
            self._startup_program, self.current_endpoint,
            self.sharding_group_endpoints, self.sharding_rank,
            self.sharding_ring_id, False)

        # inner & outer model parallelism
        if self._as_outer_parallelism:
            self._collective_helper._init_communicator(
                self._startup_program, self.current_endpoint,
                self.mp_group_endpoints, self.mp_rank,
                self.mp_group_id, False)

        # dp
        if self.hybrid_dp:
            self._collective_helper._init_communicator(
                self._startup_program, self.current_endpoint,
                self.dp_group_endpoints, self.dp_rank, self.dp_ring_id, False)

        startup_block = self._startup_program.global_block()
        startup_block._sync_with_cpp()

        # step 2: split params
        self._params = set([x[0].name for x in params_grads])
        self._shard.setup(params_grads, self.sharding_rank,
                          self.sharding_group_size)

        # step 3: get broadcast vars
        self._broadcast_vars = self._shard.find_broadcast_params(
            self._main_program.global_block())

    def _wait(self, ):
        # only the first parallelsm group that init nccl need to be wait. 
        endpoints = self.sharding_group_endpoints[:]
        current_endpoint = endpoints[self.sharding_rank]
        if self.sharding_rank == 0:
            self._collective_helper._wait(current_endpoint, endpoints)

    def collect_segment(self, segment, op_idx, block):
        segment._start_idx = op_idx + 1
        self._segments.insert(0, segment)
        new_segment = ProgramSegment(block)
        new_segment._end_idx = op_idx + 1

        return new_segment

    def _split_program(self, block):
        for op_idx, op in reversed(list(enumerate(block.ops))):
            if int(op.attr('op_role')) != int(OpRole.Optimize):
                last_backward_op_idx = op_idx + 1
                break

        var2broadcast_time = dict()
        segment = ProgramSegment(block)
        segment._end_idx = last_backward_op_idx
        for op_idx in reversed(range(last_backward_op_idx)):
            op = block.ops[op_idx]
            assert (int(op.attr('op_role')) != int(OpRole.Optimize))
            if self._sharding_segment_strategy == "broadcast_size":
                if segment._param_mem >= self._broadcast_MB:
                    segment = self.collect_segment(segment, op_idx, block)
                    
            elif self._sharding_segment_strategy == "anchors":
                if int(op.attr('op_role')) == int(OpRole.Backward):
                    for input_name in op.desc.input_arg_names():

                        # NOTE (JZ-LIANG) naive rule to support amp, if amp change, should modify here accordingly
                        if 'AMPOptimizer' in fleet._get_applied_meta_list():
                            # if ".cast_fp16@GRAD" not in input_name:
                            #     continue
                            # else:
                            #     input_name = input_name[:input_name.find(".cast_fp16@GRAD")]
                            if (op.type != "cast" and op.type != "layer_norm") or "@Fetch_0" not in input_name:
                                continue
                            else:
                                input_name = input_name[:input_name.find("@Fetch_0")]
                                                        
                        if input_name in self._backward_remain_anchors:
                            logging.info("backward segment:")
                            logging.info("op [{}] input [{}] output [{}]".format(op.desc.type() ,op.desc.input_arg_names(), op.desc.output_arg_names()))
                            segment = self.collect_segment(segment, op_idx, block)
                            assert input_name not in self._forward_remain_anchors, "segment anchor [{}] met twice !".format(input_name)
                            self._backward_remain_anchors.remove(input_name)
                            self._forward_remain_anchors.append(input_name)
                elif int(op.attr('op_role')) == int(OpRole.Forward):
                    for output_name in op.desc.output_arg_names():
                        if output_name in self._forward_remain_anchors:
                            logging.info("forward segment:")
                            logging.info("op [{}] input [{}] output [{}]".format(op.desc.type() ,op.desc.input_arg_names(), op.desc.output_arg_names()))
                            segment = self.collect_segment(segment, op_idx, block)
                            self._forward_remain_anchors.remove(output_name)
    

            # find broadcast vars
            for input_name in op.desc.input_arg_names():
                if input_name not in self._broadcast_vars:
                    continue
                if input_name in segment._param2broadcast:
                    # skip broadcast because it reuse the old broadcast var
                    broadcast_name = segment._param2broadcast[input_name]
                    if input_name != broadcast_name:
                        op._rename_input(input_name, broadcast_name)
                    continue
                if self._shard.has_param(input_name):
                    broadcast_var_name = input_name
                else:
                    broadcast_var_name = unique_name.generate(input_name +
                                                              "@BroadCast")
                    segment._fill_constant_vars.append(broadcast_var_name)

                # (JZ-LIANG) should use Param base name ?
                broadcast_var_base_name = input_name
                if "subprog" in broadcast_var_base_name:
                    # remove suffix
                    broadcast_var_base_name = broadcast_var_base_name[:broadcast_var_base_name.find(".subprog")]

                var2broadcast_time[broadcast_var_base_name] = var2broadcast_time.get(broadcast_var_base_name, 0) + 1 

                segment._param2broadcast[input_name] = broadcast_var_name
                segment._broadcast_vars.append((broadcast_var_name,
                                                self._shard.device(input_name)))
                segment._param_mem += get_var_size(
                    self._main_program.global_block().var(input_name))

            # find reduce vars
            if is_backward_op(op) and \
                    OP_ROLE_VAR_KEY in op.attr_names:
                op_role_var = op.all_attrs()[OP_ROLE_VAR_KEY]
                if len(op_role_var) != 0:
                    assert len(op_role_var) % 2 == 0
                    for i in range(0, len(op_role_var), 2):
                        param, reduced_grad = op_role_var[i], op_role_var[i + 1]
                        segment._allreduce_vars.append(reduced_grad)
                        assert (
                            reduced_grad not in self._reduced_grads_to_param)
                        self._reduced_grads_to_param[reduced_grad] = param

            # find cast op
            if FP16Utils.is_fp16_cast_op(block, op, self._params):
                fp32_param = op.desc.input_arg_names()[0]
                fp16_param = op.desc.output_arg_names()[0]
                if self._shard.has_param(fp32_param):
                    segment._cast_ops[fp16_param] = fp32_param

        if segment._param_mem > 0:
            segment._start_idx = 0
            self._segments.insert(0, segment)

        if self._sharding_segment_strategy == "anchors":
            assert len(self._forward_remain_anchors) == 0, "remain anchors {}".format(self._forward_remain_anchors)
            assert len(self._backward_remain_anchors) == 0,  "remain anchors {}".format(self._backward_remain_anchors)

        for varname in sorted(var2broadcast_time, key=var2broadcast_time.get, reverse=True):
            logging.info("Sharding broadcast: [{}] times [{}]".format(var2broadcast_time[varname], varname))
        for idx_ in range(len(self._segments)):
            logging.info("segment [{}] :".format(idx_))
            logging.info("start op: [{}]  [{}]".format(block.ops[self._segments[idx_]._start_idx].desc.type(),
            block.ops[self._segments[idx_]._start_idx].desc.input_arg_names()))
            logging.info("end   op: [{}]  [{}]".format(block.ops[self._segments[idx_]._end_idx].desc.type(),
            block.ops[self._segments[idx_]._end_idx].desc.input_arg_names()))
        return

    def _prune_main_program(self, block):
        """
        calculate deps from allredce op to optimize op,
        remove ops and vars not needed in this worker

        1. prune regularization (weight decay)
        2. prune cast_fp32_to_fp16; update amp_infine_checking
        3. prune gradient_clip related; update global_norm_sum
        4. prune optimizer op + param + gradient
            
        """
        weightdecay_helper = WeightDecayHelper()
        weightdecay_helper.prune_weight_decay(block, self._shard)
        # NOTE (JZ-LIANG) the sync of FoundInfinite should among one entire Model Parallelism
        # group. and each Data Parallelism group should have its own sync of FoundInfinite
        Model_Paramllelism_ring_id = self.sharding_ring_id
        if self._as_outer_parallelism:
            Model_Paramllelism_ring_id = self.mp_group_id
        FP16Utils.prune_fp16(block, self._shard, self._reduced_grads_to_param,
                             Model_Paramllelism_ring_id)
        gradientclip_helper = GradientClipHelper(Model_Paramllelism_ring_id)
        gradientclip_helper.prune_gradient_clip(block, self._shard)

        # build prog deps
        reduced_grads = []
        for idx, op in enumerate(block.ops):
            input_names = op.desc.input_arg_names()
            output_names = op.desc.output_arg_names()
            if op.type == "c_allreduce_sum":
                assert (len(output_names) == 1)
                output_name = output_names[0]
                reduced_grads.append(output_name)

        # prune optimizer state and param
        pruned_opti_vars = []
        for var_name in list(block.vars.keys()):
            if self._shard.is_opti_var(var_name) and \
              not self._shard.has_opt_var(var_name):
                pruned_opti_vars.append(var_name)
        program_deps = ProgramDeps(block, reduced_grads, pruned_opti_vars)

        # Init
        for var_name in program_deps._end_vars:
            program_deps._should_removed_var.add(var_name)

        # Prune
        for idx, op in reversed(list(enumerate(block.ops))):
            if op.type in [
                    "c_allreduce_sum", "c_sync_comm_stream",
                    "c_calc_comm_stream", "c_gen_nccl_id", "c_comm_init"
            ]:
                pass
            elif op.type == "conditional_block":
                assert (op.desc.has_attr("sub_block"))
                subblock_idx = op.desc.attr("sub_block").id
                subblock_deps = program_deps.get_sub_block_deps(subblock_idx)
                # only prune amp subblock
                if subblock_deps is None or not self._is_amp_subblock(op):
                    continue
                # init
                reversed_output_vars = []
                for output_name in op.desc.output("Out"):
                    if output_name in program_deps._should_removed_var:
                        subblock_deps._should_removed_var.add(output_name)
                        program_deps.crop_output_var_from_op(idx, output_name)
                    else:
                        reversed_output_vars.append(output_name)
                # prune
                for sub_op_idx, _ in reversed(
                        list(enumerate(subblock_deps._block.ops))):
                    if subblock_deps.should_remove_op(sub_op_idx):
                        subblock_deps.remove_op(sub_op_idx)
                reversed_input_vars = []
                for input_name in op.desc.input('Input'):
                    if input_name not in subblock_deps._should_removed_var:
                        reversed_input_vars.append(input_name)
                    else:
                        program_deps.crop_input_var_from_op(idx, input_name)
                op.desc.set_input('Input', reversed_input_vars)
                op.desc.set_output('Out', reversed_output_vars)
            else:
                # if all outputs of this op are in _should_removed_var
                # _should_removed_var: opt state not cur shard
                if program_deps.should_remove_op(idx):
                    program_deps.remove_op(idx)

        block._sync_with_cpp()
        return

    def _add_broadcast_allreduce(self, block):
        """
        add broadcast allreduce op
        if enable gradient_merge, insert related ops
        """
        if len(self._segments) < 1:
            return
        # sharding
        if self._segments[-1]._allreduce_vars:
            shard_allredue_vars = self._shard.filter_grads(self._segments[-1]
                                                           ._allreduce_vars)
            if self._gradient_merge_acc_step <= 1:
                if self.hybrid_dp and len(shard_allredue_vars) >= 1:
                    insert_sync_comm_ops(block, self._segments[-1]._end_idx,
                                        self.dp_ring_id, shard_allredue_vars)
                    insert_allreduce_ops(block, self._segments[-1]._end_idx,
                                        self.dp_ring_id, shard_allredue_vars)
            # gradient merge 
            else:
                self.create_persistable_gradients_and_insert_merge_ops(
                    block, 
                    self._startup_program.global_block(), 
                    self._segments[-1]._end_idx, 
                    shard_allredue_vars, 
                    self._shard)

            insert_sync_comm_ops(block, self._segments[-1]._end_idx,
                                 self.sharding_ring_id,
                                 self._segments[-1]._allreduce_vars)
            # allreduce --> reduce 
            insert_reduce_ops(block, self._segments[-1]._end_idx,
                                 self.sharding_ring_id,
                                 self._segments[-1]._allreduce_vars, self._shard)

        for idx, segment in reversed(list(enumerate(self._segments))):
            allreduce_vars = self._segments[
                idx - 1]._allreduce_vars if idx > 0 else []
            broadcast_vars = self._segments[idx +
                                            1]._broadcast_vars if idx < len(
                                                self._segments) - 1 else []
            fill_constant_vars = self._segments[
                idx + 2]._fill_constant_vars if idx < len(
                    self._segments) - 2 else []
            cast_ops = self._segments[idx + 2]._cast_ops if idx < len(
                self._segments) - 2 else {}

            for op_idx in reversed(range(segment._start_idx, segment._end_idx)):
                op = block.ops[op_idx]
                for input_name in op.desc.input_arg_names():
                    if input_name in segment._param2broadcast and \
                        input_name != segment._param2broadcast[input_name]:
                        op._rename_input(input_name,
                                         segment._param2broadcast[input_name])

            for param_name, broadcast_name in segment._param2broadcast.items():
                if param_name != broadcast_name:
                    block.create_var(
                        name=broadcast_name,
                        shape=self._main_program.global_block().var(
                            param_name).shape,
                        dtype=self._main_program.global_block().var(param_name)
                        .dtype,
                        persistable=False)

            # step1: remove cast ops
            block._sync_with_cpp()
            segment._end_idx += FP16Utils.remove_cast_op(block, self._params,
                                                         segment, 0)

            # step2: add Sync ops
            shard_allredue_vars = self._shard.filter_grads(allreduce_vars)

            if self._gradient_merge_acc_step <= 1:
                if self.hybrid_dp and len(shard_allredue_vars) >= 1:
                    insert_sync_comm_ops(block, segment._end_idx, self.dp_ring_id,
                                        shard_allredue_vars)

                    broad_cast_vars = [x[0] for x in broadcast_vars]
                    if len(broad_cast_vars) > 0:
                        insert_sync_comm_ops(block, segment._end_idx,
                                            self.sharding_ring_id, broad_cast_vars)
                else:
                    comm_dep_vars = allreduce_vars + [x[0] for x in broadcast_vars]
                    if len(comm_dep_vars) > 0:
                        insert_sync_comm_ops(block, segment._end_idx,
                                            self.sharding_ring_id, comm_dep_vars)
            # gradient merge
            else:
                broad_cast_vars = [x[0] for x in broadcast_vars]
                if len(broad_cast_vars) > 0:
                    insert_sync_comm_ops(block, segment._end_idx,
                                        self.sharding_ring_id, broad_cast_vars)                

            calc_dep_vars = fill_constant_vars + [
                k for k, v in cast_ops.items()
            ] + self._segments[idx]._allreduce_vars

            if len(calc_dep_vars) > 0:
                insert_sync_calc_op(block, segment._end_idx,
                                    [calc_dep_vars[-1]])

            # step3: insert `fill_constant` ops 
            insert_fill_constant_ops(block, segment._end_idx,
                                     fill_constant_vars)

            # step4: add `cast` ops     
            insert_cast_ops(block, segment._end_idx, cast_ops)

            # step5: add broadcast ops
            # gradient merge
            if self._gradient_merge_acc_step > 1:
                self.create_persistable_gradients_and_insert_merge_ops(
                    block, 
                    self._startup_program.global_block(), 
                    segment._start_idx, 
                    shard_allredue_vars, 
                    self._shard)

            insert_broadcast_ops(block, segment._start_idx,
                                 self.sharding_ring_id, broadcast_vars)

            # step6: add all_reduce ops
            # dp
            if self._gradient_merge_acc_step <= 1:
                if self.hybrid_dp and len(shard_allredue_vars) >= 1:
                    insert_allreduce_ops(block, segment._start_idx, self.dp_ring_id,
                                        shard_allredue_vars)
                    insert_sync_comm_ops(block, segment._start_idx,
                                        self.sharding_ring_id, allreduce_vars)
            # gradient merge
            else:
                insert_sync_comm_ops(block, segment._start_idx,
                                        self.sharding_ring_id, allreduce_vars)
            # sharding
            # allreduce --> reduce 
            insert_reduce_ops(block, segment._start_idx,
                                 self.sharding_ring_id, allreduce_vars, self._shard)

            block._sync_with_cpp()

        if self._segments[0]._broadcast_vars:
            broadcast_vars = [x[0] for x in self._segments[0]._broadcast_vars]
            insert_sync_comm_ops(block, self._segments[0]._start_idx,
                                 self.sharding_ring_id, broadcast_vars)
            insert_broadcast_ops(block, self._segments[0]._start_idx,
                                 self.sharding_ring_id,
                                 self._segments[0]._broadcast_vars)

        fill_constant_vars = []
        for x in self._segments[:2]:
            fill_constant_vars += x._fill_constant_vars

        # Join
        cast_ops = {}
        for x in self._segments[:2]:
            for k, v in x._cast_ops.items():
                cast_ops[k] = v

        calc_deps_vars = fill_constant_vars + [k for k, v in cast_ops.items()]
        if fill_constant_vars or cast_ops:
            insert_sync_calc_op(block, self._segments[0]._start_idx,
                                [calc_deps_vars[-1]])

        if fill_constant_vars:
            insert_fill_constant_ops(block, self._segments[0]._start_idx,
                                     fill_constant_vars)

        if cast_ops:
            insert_cast_ops(block, self._segments[0]._start_idx, cast_ops)

        return

    def _prune_startup_program(self, block):
        for idx, op in reversed(list(enumerate(block.ops))):
            for output_name in op.desc.output_arg_names():
                if self._shard.has_var(output_name):
                    continue
                #TODO why do we remove op, when only one var is removed
                block._remove_op(idx, sync=False)
                break

        for var_name in list(block.vars.keys()):
            if self._shard.has_var(var_name):
                continue
            block._remove_var(var_name, sync=False)
        block._sync_with_cpp()

    def _init_comm(self):

        if self.hybrid_dp:
            assert self._as_outer_parallelism == False, "hybrid dp is conflict when using sharding as outer parallelism"
            self.sharding_group_size = self.user_defined_strategy.sharding_configs[
                "sharding_group_size"]
            self.sharding_ring_id = 0
            self.sharding_rank = self.global_rank % self.sharding_group_size

            self.dp_group_size = self.global_word_size // self.sharding_group_size
            self.dp_rank = self.global_rank // self.sharding_group_size
            self.dp_ring_id = self.sharding_rank + 1

            self.sharding_group_endpoints = [
                ep for idx, ep in enumerate(self.endpoints)
                if (idx // self.sharding_group_size) == self.dp_rank
            ]
            self.dp_group_endpoints = [
                ep for idx, ep in enumerate(self.endpoints)
                if (idx % self.sharding_group_size) == self.sharding_rank
            ]
            assert self.global_word_size > self.sharding_group_size, \
                "global_word_size: {} should be larger than sharding_group_size: {}".format(self.global_word_size, self.sharding_group_size)
            assert self.global_word_size % self.sharding_group_size == 0, \
                "global_word_size: {} should be divisible to the sharding_group_size: {}".format(self.global_word_size, self.sharding_group_size)
            assert self.dp_group_size *  self.sharding_group_size == self.global_word_size, \
                "global_word_size: {} should be equal to the product of sharding_group_size: {} and dp_group_size: {}".format(
                self.global_word_size,
                self.sharding_group_size,
                self.dp_group_size)

            # sharding parallelism is the only model parallelism in the current setting
            self.mp_group_id = self.sharding_ring_id
            self.mp_rank = self.sharding_rank
            self.mp_group_size = self.sharding_group_size
            self.mp_group_endpoints = self.sharding_group_endpoints[:]


            logging.info("Using Sharing&DP mode !")
        else:
            if self._as_outer_parallelism:
                self.sharding_ring_id = 1
                assert self.global_word_size > self._inner_parallelism_size, \
                    "global_word_size: {} should be larger than inner_parallelism_size: {}".format(self.global_word_size, self._inner_parallelism_size)
                assert self.global_word_size % self._inner_parallelism_size == 0, \
                    "global_word_size: {} should be divisible to the inner_parallelism_size: {}".format(self.global_word_size, self._inner_parallelism_size)
                self.sharding_rank = self.global_rank // self._inner_parallelism_size
                self.sharding_group_size = self.role_maker._worker_num(
                ) // self._inner_parallelism_size
                _offset = self.global_rank % self._inner_parallelism_size
                self.sharding_group_endpoints = [
                    ep for idx, ep in enumerate(self.endpoints)
                    if idx % self._inner_parallelism_size == _offset
                ]

                # the current entire model parallelism group is the combination of innert & sharding parallelism
                self.mp_group_id = 2
                self.mp_rank = self.global_rank
                self.mp_group_size = self.role_maker._worker_num()
                self.mp_group_endpoints = self.endpoints[:]
                logging.info("Using Sharing as Outer parallelism mode !")

                # print(
                #     "init the nccl comm for megatron paramllelism, this should be done in Megatron Metaoptimizer"
                # )
                # partition_idx = self.global_rank // self._inner_parallelism_size
                # magetron_endpoints = self.endpoints[
                #     partition_idx * self._inner_parallelism_size:partition_idx *
                #     self._inner_parallelism_size + self._inner_parallelism_size]
                # magetron_rank = self.global_rank % self._inner_parallelism_size

                # self._collective_helper._init_communicator(
                #     program=self._startup_program,
                #     current_endpoint=self.current_endpoint,
                #     endpoints=magetron_endpoints,
                #     rank=magetron_rank,
                #     ring_id=0,
                #     wait_port=True)
                # logging.info("megatron group size: {}".format(
                #     self._inner_parallelism_size))
                # logging.info("megatron rank: {}".format(magetron_rank))
                # logging.info("megatron endpoints: {}".format(
                #     magetron_endpoints))
            else:
                self.sharding_ring_id = 0
                self.sharding_rank = self.global_rank
                self.sharding_group_size = self.role_maker._worker_num()
                self.sharding_group_endpoints = self.endpoints

                # sharding parallelism is the only model parallelism in the current setting
                self.mp_group_id = self.sharding_ring_id
                self.mp_rank = self.sharding_rank
                self.mp_group_size = self.sharding_group_size
                self.mp_group_endpoints = self.sharding_group_endpoints[:]

                logging.info("Using Sharing alone mode !")

            self.dp_ring_id = -1
            self.dp_rank = -1
            self.dp_group_size = None
            self.dp_group_endpoints = None

        logging.info("global word size: {}".format(self.global_word_size))
        logging.info("global rank: {}".format(self.global_rank))     
        logging.info("sharding group_size: {}".format(self.sharding_group_size))
        logging.info("sharding rank: {}".format(self.sharding_rank))
        logging.info("current model parallelism group_size: {}".format(self.mp_group_size))
        logging.info("current model parallelism rank: {}".format(self.mp_rank))  
        logging.info("dp group size: {}".format(self.dp_group_size))
        logging.info("dp rank: {}".format(self.dp_rank))
        logging.info("current endpoint: {}".format(self.current_endpoint))
        logging.info("global word endpoints: {}".format(self.endpoints))
        logging.info("sharding group endpoints: {}".format(
            self.sharding_group_endpoints))
        logging.info("current model parallelism group endpoints: {}".format(
            self.mp_group_endpoints))
        logging.info("dp group endpoints: {}".format(self.dp_group_endpoints))

        return

    def _initialization_broadcast(self, startup_prog):
        """
        this funtion is to ensure the initialization between dp group to be 
        identical when hybrid-dp is used.
        """
        block = startup_prog.global_block()
        params = []
        for param in block.iter_parameters():
            params.append(param)
            block.append_op(
                type='c_broadcast',
                inputs={'X': param},
                outputs={'Out': param},
                attrs={
                    'ring_id': self.dp_ring_id,
                    'root': 0,
                    OP_ROLE_KEY: OpRole.Forward
                })
        block.append_op(
            type='c_sync_comm_stream',
            inputs={'X': params},
            outputs={'Out': params},
            attrs={'ring_id': self.dp_ring_id,
                   OP_ROLE_KEY: OpRole.Forward})

    # sharding gradient merge
    def create_persistable_gradients_and_insert_merge_ops(self, main_block, startup_block, insert_idx, grad_names, shard):

        for grad_name in grad_names:
            assert get_grad_device(grad_name, shard) == shard.worker_idx, "try to merge gradient not belong to current shard: [{}]".format(grad_name)
            persistable_grad_name = grad_name + '@GradiantMerge'
            assert grad_name not in self._grad2merged_grad, "grad [{}] already in grad2merged_grad, maybe you meet sharing weight case !".format(grad_name)
            self._grad2merged_grad[grad_name] = persistable_grad_name
            grad_var = main_block.var(grad_name)
            # create var
            gradient_merge_var = main_block.create_var(
                name=persistable_grad_name,
                shape=grad_var.shape,
                dtype=grad_var.dtype,
                persistable=True)
            startup_gradient_merge_var = startup_block.create_var(
                name=persistable_grad_name,
                shape=grad_var.shape,
                dtype=grad_var.dtype,
                persistable=True)

            # merge gradient
            main_block._insert_op_without_sync(
                insert_idx,
                type="elementwise_add",
                inputs={'X': grad_name,
                        'Y': gradient_merge_var},
                outputs={'Out': gradient_merge_var},
                attrs={'axis': -1,
                        'use_mkldnn': False,
                        OP_ROLE_KEY: OpRole.Backward})

            # startup initialization
            startup_block.append_op(
                type="fill_constant",
                outputs={"Out": startup_gradient_merge_var},
                attrs={
                    "shape": grad_var.shape,
                    "dtype": grad_var.dtype,
                    "value": float(0),
                })

        main_block._sync_with_cpp()
        startup_block._sync_with_cpp()



    def _create_gm_cond(self, main_block):
        # Add const var
        acc_step_var = layers.create_global_var(
            name="gradient_merge_acc_step",
            shape=[1],
            value=int(self._gradient_merge_acc_step),
            dtype='int32',
            persistable=True,
            force_cpu=True)

        zero_var = layers.create_global_var(
            name="gradient_merge_zero",
            shape=[1],
            value=int(0),
            dtype='int32',
            persistable=True,
            force_cpu=True)

        # Add step var & cond var
        current_step_var = layers.create_global_var(
            name="gradient_merge_current_step",
            shape=[1],
            value=int(0),
            dtype='int32',
            persistable=True,
            force_cpu=True)

        cond_var = layers.create_global_var(
            name="gradient_merge_cond",
            shape=[1],
            value=bool(0),
            dtype='bool',
            persistable=False,
            force_cpu=True)

        with device_guard("cpu"):
            # step_var = (step_var + 1) % k_step
            main_block.append_op(
                type='increment',
                inputs={'X': [current_step_var]},
                outputs={'Out': [current_step_var]},
                attrs={'step': float(1),
                OP_ROLE_KEY: OpRole.Optimize})

            main_block.append_op(
                type='elementwise_mod',
                inputs={'X': current_step_var,
                        'Y': acc_step_var},
                outputs={'Out': current_step_var},
                attrs={'axis': -1,
                        OP_ROLE_KEY: OpRole.Optimize,
                        'use_mkldnn': False})

            # cond_var = (step_var == 0)
            main_block.append_op(
                type='equal',
                inputs={'X': current_step_var,
                        'Y': zero_var},
                outputs={'Out': cond_var},
                attrs={OP_ROLE_KEY: OpRole.Optimize}
                )
        paddle.static.Print(current_step_var, message="in FWBW last conditional")
        return cond_var

    def _true_apply_gradient(self):
        """
        allreduce grad@gradientmerge in dp group
        grad@gradientmerge / acc_step
        re-create all optimize ops of origin main block and rename them
            cast(backward)
            amp 
            clip
            opt
        # fill constant grad@gradientmerge

        """
        # current conditional block
        main_block = self._main_program.global_block()
        cur_block_idx = self._main_program.current_block_idx
        cur_block = self._main_program.current_block()
        self.cond_block = self._main_program.current_block()
    
        # cur_block's forward_block & backward_block is itself
        cur_block._set_forward_block_idx(cur_block_idx)

        # allreduce grad@gradientmerge  
        if self.hybrid_dp:
            assert self.dp_ring_id >= 0, "dp_ring_id should larger than 0 when in sharding&DP mode"
            for grad, merged_grad in self._grad2merged_grad.items():
                merged_grad_var = main_block.var(merged_grad)
                cur_block.append_op(
                    type='c_allreduce_sum',
                    inputs={'X': merged_grad_var},
                    outputs={'Out': merged_grad_var},
                    attrs={'ring_id': self.dp_ring_id,
                            'use_calc_stream': True,
                            OP_ROLE_KEY: OpRole.Optimize})
        print("after allreduce grad@gradientmerge ")
        
        # grad@gradientmerge / acc_step
        for grad, merged_grad in self._grad2merged_grad.items():
            # grad /= k_steps
            merged_grad_var = main_block.var(merged_grad)
            cur_block.append_op(
                type='scale',
                inputs={'X': merged_grad_var},
                outputs={'Out': merged_grad_var},
                attrs={
                    'scale': 1.0 / float(self._gradient_merge_acc_step),
                    'bias': 0.0,
                    'bias_after_scale': False,
                    OP_ROLE_KEY: OpRole.Optimize
                })
        print("after allreduce grad@gradientmerge / acc_step")

        # re-create optimize ops
        already_moved_var_names = []
        for op_desc in self.original_optimize_ops_desc:
            new_op_desc = cur_block.desc.append_op()
            new_op_desc.copy_from(op_desc)

            if op_desc.type in ["check_finite_and_unscale", "update_loss_scaling"]:
                print("check @@@@@: ", op_desc.type)
                print("input @@@@@: ", op_desc.output_arg_names)

            for input_name in new_op_desc.input_arg_names():
                if input_name in self._grad2merged_grad:
                    new_op_desc._rename_input(input_name, self._grad2merged_grad[input_name])

            for output_name in new_op_desc.output_arg_names():
                if output_name in self._grad2merged_grad:
                    new_op_desc._rename_output(output_name, self._grad2merged_grad[output_name])
            
                # move non temp optimize vars from block0 to cond block
                if output_name not in already_moved_var_names and output_name not in self._grad2merged_grad.keys():
                    var_ = self._main_program.global_block().var(output_name)
                    if not var_.persistable:
                        print("remove: ", output_name)
                        # move
                        name_ = var_.name
                        shape_ = var_.shape
                        type_ = var_.dtype
                        self._main_program.global_block()._remove_var(var_.name, sync=False)
                        self.cond_block.create_var(
                            name=name_, 
                            shape=shape_,
                            dtype=type_,
                            persistable=False)
                        already_moved_var_names.append(name_)
                        
        self._main_program.global_block()._sync_with_cpp()
        cur_block._sync_with_cpp()

        # fill zero to grad@gradientmerge
        for grad, merged_grad in self._grad2merged_grad.items():
            merged_grad_var = main_block.var(merged_grad)
            cur_block.append_op(
                type='fill_constant',
                outputs={'Out': merged_grad_var},
                attrs={
                    "shape": merged_grad_var.shape,
                    "dtype": merged_grad_var.dtype,
                    "value": float(0),
                    OP_ROLE_KEY: OpRole.Optimize
                })

        lr_var = main_block.var("@LR_DECAY_COUNTER@")
        paddle.static.Print(lr_var, message="in OPTIMIZE last conditional")


    def _sharding_gradient_merge(self, main_block):

        """
        copy all optimize ops in origin main block
        remove all optimize ops in origin main block
        create cond block

        """
        # copy original optimize ops to temp ops desc list
        # remove them from block 0
        tmp_copy_block = self._main_program._create_block()

        self.original_optimize_ops_desc = []
        for op_idx, op in reversed(list(enumerate(main_block.ops))):
            if int(op.attr('op_role')) != int(OpRole.Optimize):
                continue
            else:
                tmp_op_desc = tmp_copy_block.desc.append_op()
                tmp_op_desc.copy_from(op.desc)
                self.original_optimize_ops_desc.append(tmp_op_desc)
                main_block._remove_op(op_idx, sync=False)
        tmp_copy_block._sync_with_cpp()
        self.original_optimize_ops_desc = list(reversed(self.original_optimize_ops_desc))
        print("original_optimize_ops_desc :")
        for desc in self.original_optimize_ops_desc:
            print(desc.type(), desc.input_arg_names())

        # back to block 0
        self._main_program._rollback()
        print("after first rollback")

        # create cond vars and ops at the end of block 0
        cond = self._create_gm_cond(main_block)
        print("after _create_gm_cond")


        # create cond block
        cond_block = self._main_program._create_block()
        self._true_apply_gradient()
        print("after _true_apply_gradient")
        
        # back to block 0
        self._main_program._rollback()

        # cond op
        step_scope = self._main_program.global_block().create_var(
            type=core.VarDesc.VarType.STEP_SCOPES)
        conditional_block_op = self._main_program.global_block().append_op(
            type='conditional_block',
            inputs={
                'Cond': cond,
                'Input': [],
            },
            outputs={'Out': [],
                     'Scope': [step_scope]},
            attrs={
                'sub_block': cond_block,
                'is_scalar_condition': True,
            })











        