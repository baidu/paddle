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
from paddle.fluid import core
from functools import reduce
from paddle.distributed.fleet.meta_optimizers.common import is_loss_grad_op
from paddle.distributed.fleet.meta_optimizers.common import OpRole, OP_ROLE_KEY, OP_ROLE_VAR_KEY

import re
import os


def check_broadcast(block):
    """
    if a var is broadcasted, it should have a sync_comm before
    this var is used, if not, raise error.
    if the broadcasted var has a fill_constant op, the fill_constant
    op should stay forward before the broadcast op, and before a
    sync_calc op. Otherwise, raise error.
    """
    broadcast_vars = {}
    for idx, op in enumerate(block.ops):
        if op.type == "c_broadcast":
            var_name = op.desc.input_arg_names()[0]
            if "@BroadCast" in var_name:
                if var_name in broadcast_vars:
                    raise ValueError("var_name areadly exist: {}"
                                     "the old pos is {}, the new pos is {}".
                                     format(var_name, broadcast_vars[var_name][
                                         "broadcast_pos"], idx))
                broadcast_vars[var_name] = {
                    "fill_constant_pos": -1,
                    "broadcast_pos": idx,
                }

    for idx, op in enumerate(block.ops):
        if op.type == "fill_constant":
            var_name = op.desc.output_arg_names()[0]
            if var_name in broadcast_vars:
                broadcast_vars[var_name]["fill_constant_pos"] = idx
            continue

    last_sync_comm_op_idx = -1
    last_sync_calc_op_idx = -1
    for idx, op in enumerate(block.ops):
        if op.type == "c_sync_comm_stream":
            last_sync_comm_op_idx = idx
            continue
        if op.type == "c_sync_calc_stream":
            last_sync_calc_op_idx = idx
            continue
        if op.type == "c_broadcast":
            var_name = op.desc.input_arg_names()[0]
            if "@BroadCast" in var_name:
                if broadcast_vars[var_name]["fill_constant_pos"] != -1:
                    assert (last_sync_calc_op_idx != -1)
                    assert (broadcast_vars[var_name]["fill_constant_pos"] <
                            last_sync_calc_op_idx)
                    assert (last_sync_calc_op_idx < idx)
                continue
        for input_name in op.desc.input_arg_names():
            if input_name in broadcast_vars:
                assert (broadcast_vars[input_name]["broadcast_pos"] != -1)
                assert (broadcast_vars[input_name]["broadcast_pos"] <
                        last_sync_comm_op_idx)
                assert (last_sync_comm_op_idx < idx)
    return


def check_allreduce_sum(block, shard, dp_ring_id=-1):
    """
    the op order should be:
        grad:
            - 0: op that generate Var
            - 1: sync_calc
            - 2: allreduce_sum_sharding
            - 3: sync_comm
            - 4: allreuce_sum_dp (dp_grads)
            - 5: sync_comm (dp_grads)
            - 6: op that use Var (dp_grads & sum)
    """
    vars_status = {}
    dp_grads_status = {}
    idx_last_grad_allreduce = -1
    idx_amp_allreduce = -1
    idx_gradient_clip_allreduce = -1
    for idx, op in enumerate(block.ops):
        if op.type == "c_allreduce_sum":
            ring_id = op.desc.attr("ring_id")
            var_name = op.desc.input_arg_names()[0]
            param = var_name.split("@")[0]

            assert 'sum' in var_name or ("@GRAD" in var_name)
            if 'sum' in var_name or (not shard.has_param(param)):
                vars_status[var_name] = -1
            else:
                dp_grads_status[var_name] = -1

            if ring_id != 0:
                assert shard.has_param(param)
                assert ring_id == dp_ring_id

            if "sum" in var_name:
                idx_amp_allreduce = idx
            elif "@GRAD":
                idx_last_grad_allreduce = idx

        if op.type == "c_allreduce_max":
            idx_gradient_clip_allreduce = idx

    for op in block.ops:
        if op.type == "c_sync_calc_stream":
            for var_name in vars_status:
                if var_name in vars_status and vars_status[var_name] == 0:
                    vars_status[var_name] = 1
            for var_name in dp_grads_status:
                if var_name in dp_grads_status and dp_grads_status[
                        var_name] == 0:
                    dp_grads_status[var_name] = 1

        elif op.type == "c_allreduce_sum":
            var_name = op.desc.input_arg_names()[0]
            ring_id = op.desc.attr("ring_id")
            if ring_id == 0:
                if var_name in vars_status:
                    _status = vars_status[var_name]
                else:
                    _status = dp_grads_status[var_name]
                if _status == -1:
                    raise ValueError("{} is not generated, but you are"
                                     "trying to all-reduce it".format(var_name))
                if _status == 0:
                    raise ValueError("There should be a sync_calc op "
                                     "after generate Var: {} and before the"
                                     "c_allreduce_sum op".format(var_name))
                assert (_status == 1)
                if var_name in vars_status:
                    vars_status[var_name] = 2
                else:
                    dp_grads_status[var_name] = 2
            else:
                assert ring_id == dp_ring_id
                param = var_name.split("@")[0]
                assert shard.has_param(param)
                assert dp_grads_status[var_name] == 3
                dp_grads_status[var_name] = 4

        elif op.type == "c_sync_comm_stream":
            var_name = op.desc.input_arg_names()[0]
            ring_id = op.desc.attr("ring_id")
            if ring_id == 0:
                for var_name in op.desc.input_arg_names():
                    if var_name in vars_status:
                        assert vars_status[var_name] == 2
                        vars_status[var_name] = 3
                    elif var_name in dp_grads_status:
                        assert dp_grads_status[var_name] == 2
                        dp_grads_status[var_name] = 3
            else:
                for var_name in op.desc.input_arg_names():
                    param = var_name.split("@")[0]
                    assert ring_id == dp_ring_id
                    assert shard.has_param(param)
                    assert dp_grads_status[var_name] == 4
                    dp_grads_status[var_name] = 5
        else:
            for input_name in op.desc.input_arg_names():
                if input_name in vars_status:
                    if vars_status[input_name] != 3:
                        raise ValueError("There should be a sync_comm op "
                                         "after allreduce the Var: {}".format(
                                             input_name))
                if input_name in dp_grads_status:
                    if dp_ring_id == -1:
                        if dp_grads_status[input_name] != 3:
                            raise ValueError("There should be a sync_comm op "
                                             "after allreduce the Var: {}".
                                             format(input_name))
                    else:
                        if dp_grads_status[input_name] != 5:
                            raise ValueError(
                                "The grad in shard should be allreduce and sync"
                                "twice before usage {}".format(input_name))

            for output_name in op.desc.output_arg_names():
                if output_name in vars_status and \
                    vars_status[output_name] == -1:
                    vars_status[output_name] = 0
                if output_name in dp_grads_status and  \
                    dp_grads_status[output_name] == -1:
                    dp_grads_status[output_name] = 0

    # check sharding with amp
    if idx_amp_allreduce != -1:
        assert idx_amp_allreduce > idx_last_grad_allreduce

    # check sharding with gradient_clip_by_global_norm
    if idx_gradient_clip_allreduce != -1:
        assert idx_gradient_clip_allreduce > idx_last_grad_allreduce

    return


def get_valid_op_role(block, insert_idx):
    """
    return OpRole.Forward or OpRole.Backward
    """
    op_role = block.ops[insert_idx].attr('op_role')
    if (insert_idx >= len(block.ops)) or (
            op_role in [int(OpRole.Backward), int(OpRole.Optimize)]):
        return OpRole.Backward
    if op_role in [int(OpRole.Forward), int(OpRole.Loss)]:
        return OpRole.Forward

    return get_valid_op_role(block, insert_idx + 1)


def insert_sync_calc_op(block, insert_idx, calc_dep_vars):
    """
    _insert_sync_calc_op
    """
    op_role = get_valid_op_role(block, insert_idx)
    block._insert_op_without_sync(
        insert_idx,
        type='c_sync_calc_stream',
        inputs={'X': calc_dep_vars},
        outputs={'Out': calc_dep_vars},
        attrs={OP_ROLE_KEY: op_role})
    return


def insert_sync_comm_op(block, insert_idx, ring_id, comm_dep_vars):
    """
    insert sync_comm_op for single var
    """
    op_role = get_valid_op_role(block, insert_idx)
    block._insert_op_without_sync(
        insert_idx,
        type='c_sync_comm_stream',
        inputs={'X': comm_dep_vars},
        outputs={'Out': comm_dep_vars},
        attrs={'ring_id': ring_id,
               OP_ROLE_KEY: op_role})
    return 1


def insert_sync_comm_ops(block, insert_idx, ring_id, comm_dep_vars):
    """
    insert sync_comm_op for vars
    """
    op_role = get_valid_op_role(block, insert_idx)
    block._insert_op_without_sync(
        insert_idx,
        type='c_sync_comm_stream',
        inputs={'X': comm_dep_vars},
        outputs={'Out': comm_dep_vars},
        attrs={'ring_id': int(ring_id),
               OP_ROLE_KEY: op_role})
    return 1


def insert_fill_constant_ops(block, insert_idx, fill_constant_vars):
    """
    _add_fill_constant_ops
    """
    op_role = get_valid_op_role(block, insert_idx)
    for broadcast_name in fill_constant_vars:
        broadcast_var = block.var(broadcast_name)
        block._insert_op_without_sync(
            insert_idx,
            type="fill_constant",
            outputs={"Out": broadcast_var.name},
            attrs={
                "shape": broadcast_var.shape,
                "dtype": broadcast_var.dtype,
                "value": 0.0,
                OP_ROLE_KEY: op_role
            })
    return


def insert_cast_ops(block, insert_idx, cast_ops):
    """
    _add_cast_ops
    """
    op_role = get_valid_op_role(block, insert_idx)
    for fp16_name, fp32_name in cast_ops.items():
        block._insert_op_without_sync(
            insert_idx,
            type="cast",
            inputs={"X": fp32_name},
            outputs={"Out": fp16_name},
            attrs={
                "in_dtype": core.VarDesc.VarType.FP32,
                "out_dtype": core.VarDesc.VarType.FP16,
                OP_ROLE_KEY: op_role
            })
    return


def insert_allreduce_ops(block, insert_idx, ring_id, allreduce_vars):
    """
    _add_allreduce_ops
    """
    for var in allreduce_vars:
        block._insert_op_without_sync(
            insert_idx,
            type='c_allreduce_sum',
            inputs={'X': var},
            outputs={'Out': var},
            attrs={'ring_id': ring_id,
                   OP_ROLE_KEY: OpRole.Backward})

    return


def insert_broadcast_ops(block, insert_idx, ring_id, broadcast2root):
    """
    _add_broadcast_ops
    """
    op_role = get_valid_op_role(block, insert_idx)
    for broadcast_name, root_device in broadcast2root:
        block._insert_op_without_sync(
            insert_idx,
            type='c_broadcast',
            inputs={'X': broadcast_name},
            outputs={'Out': broadcast_name},
            attrs={
                'ring_id': ring_id,
                'root': root_device,
                OP_ROLE_KEY: op_role
            })

    return


DtypeToSize = {
    core.VarDesc.VarType.FP16: 2,
    core.VarDesc.VarType.FP32: 4,
    core.VarDesc.VarType.FP64: 8,
    core.VarDesc.VarType.INT16: 2,
    core.VarDesc.VarType.INT32: 4,
    core.VarDesc.VarType.INT64: 8,
    core.VarDesc.VarType.BOOL: 1,
    core.VarDesc.VarType.UINT8: 1,
}


def get_var_size(param):
    """
    input:
        - param: var
    return:
        var size in MB
    """
    assert -1 not in param.shape
    return reduce(lambda x, y: x * y,
                  param.shape) * DtypeToSize[param.dtype] / 1024.0 / 1024.0


def insert_scale_loss_grad_ops(block, scale=1.0):
    '''
    In order to keep the learning rate consistent in different numbers of
    training workers, we scale the loss grad by the number of workers
    '''
    for idx, op in reversed(list(enumerate(block.ops))):
        if is_loss_grad_op(op):
            loss_grad_var = block.vars[op.output_arg_names[0]]
            block._insert_op_without_sync(
                idx + 1,
                type='scale',
                inputs={'X': loss_grad_var},
                outputs={'Out': loss_grad_var},
                attrs={'scale': scale,
                       OP_ROLE_KEY: OpRole.Backward})


def comm_analyse(main_program):
    """
    Analyse the parameter size that need to be broadcast/allreduce during sharding training 
    """
    reduce_vars = {}
    broadcast_vars = {}
    block = main_program.global_block()
    for op in block.ops:
        if op.type == "c_broadcast":
            var_name = op.desc.input_arg_names()[0]
            # convert MB to KB
            broadcast_vars[var_name] = get_var_size(block.var(
                var_name)) * 1024.0
        elif op.type == "c_allreduce_sum":
            var_name = op.desc.input_arg_names()[0]
            reduce_vars[var_name] = get_var_size(block.var(var_name)) * 1024.0

    varsize_count = {}
    gap = 1

    for k, v in broadcast_vars.items():
        print("broadcast: {}: {} KB".format(k, v))
        if (int(v / gap) in varsize_count):
            varsize_count[int(v / gap)] += 1
        else:
            varsize_count[int(v / gap)] = 1

    for k, v in reduce_vars.items():
        print("allreduce: {}: {} KB".format(k, v))
        if (int(v / gap) in varsize_count):
            varsize_count[int(v / gap)] += 1
        else:
            varsize_count[int(v / gap)] = 1

    with open("nccl_size.txt", 'w') as f:
        sorted_varsize = sorted(varsize_count.items(), key=lambda x: x[0])
        for varsize, count in sorted_varsize:
            print("NCCL size {}~{} KB: {}".format(varsize, varsize + 1, count))
            f.write("NCCL size {}~{} KB: {}\n".format(varsize, varsize + 1,
                                                      count))


def add_sync_comm(program, dist_strategy):
    """
    When clone a test prog by clone from the sharding main prog, 
    part of the sync_comm op maybe be pruned by mistake, this function
    add the sync_comm op for the test prog.

    """
    #NOTE (liangjianzhong): only support one comm stream by now, use more than one 
    # comm streams will cause error. should be revise in future.

    block = program.global_block()
    not_sync_vars = set([])
    for op in block.ops:
        if op.type in ["c_broadcast", "c_allreduce"]:
            for input_name in op.desc.input_arg_names():
                not_sync_vars.add(input_name)
        if op.type == "c_sync_comm_stream":
            for input_name in op.desc.input_arg_names():
                not_sync_vars.remove(input_name)
    if not_sync_vars:
        for nccl_id in range(dist_strategy.nccl_comm_num):
            block.append_op(
                type='c_sync_comm_stream',
                inputs={'X': list(not_sync_vars)},
                outputs={'Out': list(not_sync_vars)},
                attrs={
                    'ring_id': nccl_id,
                    'op_role': core.op_proto_and_checker_maker.OpRole.Forward
                })
    return


def save_persistables(exe, dirname, main_program, filename=None):
    """
    When use sharding, part of persistable vars are unique and are partitioned in different ranks,
    and part of persistable vars are duplicated and exist in all the ranks with different values.
    This function handles the model saving for sharding training.
    """

    def is_opt_vars(var):
        # NOTE(liangjianzhong): The checks should be updated when add new compatible optimizer
        # now only Momentum and adam are compatible with sharding
        checks = [
            "_moment1_0", "_moment2_0", "_beta1_pow_acc_0", "_beta2_pow_acc_0",
            "_velocity_0"
        ]
        for check in checks:
            if var.name.endswith(check):
                return True
        return False

    def is_trainable(var):
        return isinstance(var,
                          paddle.fluid.framework.Parameter) and var.trainable

    def sharding_predicate(var):
        return is_trainable(var) or is_opt_vars(var)

    if int(os.environ.get('PADDLE_TRAINER_ID', 0)) == 0:
        paddle.fluid.io.save_persistables(
            exe, dirname, main_program=main_program, filename=None)
    else:
        paddle.fluid.io.save_vars(
            exe,
            dirname,
            main_program=main_program,
            predicate=sharding_predicate,
            filename=None)

    return
