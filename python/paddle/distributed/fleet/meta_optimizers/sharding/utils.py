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

from paddle.fluid import core
from functools import reduce
from paddle.distributed.fleet.meta_optimizers.common import is_loss_grad_op
from paddle.distributed.fleet.meta_optimizers.common import OpRole, OP_ROLE_KEY, OP_ROLE_VAR_KEY

import re


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


def check_allreduce_sum(block):
    """
    if a Var is allreduced, the op order should be:
        - 0: op that generate Var
        - 1: sync_calc
        - 2: allreduce_sum op
        - 3: sync_comm
        - 4: op that use Var
    """
    var_status = {}
    for op in block.ops:
        if op.type == "c_allreduce_sum":
            var_name = op.desc.input_arg_names()[0]
            var_status[var_name] = -1

    for op in block.ops:
        if op.type == "c_sync_calc_stream":
            for var_name in var_status:
                if var_name in var_status and var_status[var_name] == 0:
                    var_status[var_name] = 1
        elif op.type == "c_allreduce_sum":
            var_name = op.desc.input_arg_names()[0]
            if var_status[var_name] == -1:
                raise ValueError("{} is not generated, but you are"
                                 "trying to all-reduce it".format(var_name))
            if var_status[var_name] == 0:
                raise ValueError("There should be a sync_calc op "
                                 "after generate Var: {} and before the"
                                 "c_allreduce_sum op".format(var_name))
            assert (var_status[var_name] == 1)
            var_status[var_name] = 2
        elif op.type == "c_sync_comm_stream":
            for var_name in op.desc.input_arg_names():
                if var_name in var_status and var_status[var_name] == 2:
                    var_status[var_name] = 3
        else:
            for input_name in op.desc.input_arg_names():
                if input_name in var_status:
                    if var_status[input_name] != 3:
                        raise ValueError("There should be a sync_comm op "
                                         "after allreduce the Var: {}".format(
                                             var_name))
            for output_name in op.desc.output_arg_names():
                if output_name in var_status and \
                    var_status[output_name] == -1:
                    var_status[output_name] = 0
    return


def get_valid_op_role(block, insert_idx):
    """
    return OpRole.Forward or OpRole.Backward
    """
    if insert_idx >= len(block.ops):
        return OpRole.Backward

    op_role = block.ops[insert_idx].attr('op_role')
    if op_role == int(OpRole.Forward):
        return OpRole.Forward
    if op_role in [int(OpRole.Backward), int(OpRole.Optimize)]:
        return OpRole.Backward
    if op_role == int(OpRole.Loss):
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


def insert_sync_comm_ops(block, insert_idx, nrings, comm_dep_vars):
    """
    _insert_sync_comm_ops
    """
    op_role = get_valid_op_role(block, insert_idx)
    for i in range(nrings):
        block._insert_op_without_sync(
            insert_idx,
            type='c_sync_comm_stream',
            inputs={'X': comm_dep_vars},
            outputs={'Out': comm_dep_vars},
            attrs={'ring_id': i,
                   OP_ROLE_KEY: op_role})
    return nrings


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


def insert_allreduce_ops(block, insert_idx, nrings, allreduce_vars):
    """
    _add_allreduce_ops
    """
    ring_id = -1
    for var in allreduce_vars:
        ring_id = (ring_id + 1) % nrings
        block._insert_op_without_sync(
            insert_idx,
            type='c_allreduce_sum',
            inputs={'X': var},
            outputs={'Out': var},
            attrs={'ring_id': ring_id,
                   OP_ROLE_KEY: OpRole.Backward})
    return


def insert_broadcast_ops(block, insert_idx, nrings, broadcast2root):
    """
    _add_broadcast_ops
    """
    ring_id = -1
    op_role = get_valid_op_role(block, insert_idx)
    for broadcast_name, root_device in broadcast2root:
        ring_id = (ring_id + 1) % nrings
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
        var size in Bytes
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
