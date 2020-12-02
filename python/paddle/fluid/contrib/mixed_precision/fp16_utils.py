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
# limitations under the License.

from __future__ import print_function

from ... import core
from ... import layers
from ... import global_scope
from ...log_helper import get_logger
import logging
import numpy as np
_logger = get_logger(
    __name__, logging.INFO, fmt='%(asctime)s-%(levelname)s: %(message)s')


def _rename_arg(op, old_name, new_name):
    """
    If an op has old_name input and output, rename these input 
    args new_name.

    Args:
        op (Operator): Current operator.
        old_name (str): The old name of input args.
        new_name (str): The new name of input args.
    """
    op_desc = op.desc
    if isinstance(op_desc, tuple):
        op_desc = op_desc[0]
    op_desc._rename_input(old_name, new_name)
    op_desc._rename_output(old_name, new_name)


def _dtype_to_str(dtype):
    """
    Convert specific variable type to its corresponding string.

    Args:
        dtype (VarType): Variable type.
    """
    if dtype == core.VarDesc.VarType.FP16:
        return 'fp16'
    else:
        return 'fp32'


def _insert_cast_op(block, op, idx, src_dtype, dest_dtype):
    """
    Insert cast op and rename args of input and output.

    Args:
        block (Program): The block in which the operator is.
        op (Operator): The operator to insert cast op.
        idx (int): The index of current operator.
        src_dtype (VarType): The input variable dtype of cast op.
        dest_dtype (VarType): The output variable dtype of cast op.

    Returns:
        num_cast_op (int): The number of cast ops that have been inserted.
    """
    num_cast_ops = 0
    valid_types = [
        core.VarDesc.VarType.LOD_TENSOR, core.VarDesc.VarType.SELECTED_ROWS,
        core.VarDesc.VarType.LOD_TENSOR_ARRAY
    ]

    for in_name in op.input_names:
        if src_dtype == core.VarDesc.VarType.FP32 and op.type in [
                'batch_norm', 'fused_bn_add_activation', 'layer_norm'
        ]:
            if in_name not in {'X', 'Z'}:
                continue
        for in_var_name in op.input(in_name):
            in_var = block.var(in_var_name)
            if in_var.type not in valid_types or in_var.dtype == dest_dtype:
                continue
            if in_var.dtype == src_dtype:
                cast_name = in_var.name + '.cast_' + _dtype_to_str(dest_dtype)
                out_var = block.vars.get(cast_name)
                if out_var is None or out_var.dtype != dest_dtype:
                    out_var = block.create_var(
                        name=cast_name,
                        dtype=dest_dtype,
                        persistable=False,
                        stop_gradient=in_var.stop_gradient)

                    block._insert_op(
                        idx,
                        type="cast",
                        inputs={"X": in_var},
                        outputs={"Out": out_var},
                        attrs={
                            "in_dtype": in_var.dtype,
                            "out_dtype": out_var.dtype
                        })
                    num_cast_ops += 1
                _rename_arg(op, in_var.name, out_var.name)
            else:
                if op.has_attr('in_dtype'):
                    op._set_attr('in_dtype', dest_dtype)
    if src_dtype == core.VarDesc.VarType.FP32 and dest_dtype == core.VarDesc.VarType.FP16:
        for out_name in op.output_names:
            if op.type in [
                    'batch_norm', 'fused_bn_add_activation', 'layer_norm'
            ] and out_name != 'Y':
                continue
            for out_var_name in op.output(out_name):
                out_var = block.var(out_var_name)
                if out_var.type not in valid_types:
                    continue
                if out_var.dtype == core.VarDesc.VarType.FP32:
                    out_var.desc.set_dtype(core.VarDesc.VarType.FP16)
                    if op.has_attr('out_dtype'):
                        op._set_attr('out_dtype', core.VarDesc.VarType.FP16)
    return num_cast_ops


def find_true_prev_op(ops, cur_op, var_name):
    """
    Find the true prev op that outputs var_name variable.

    Args:
        ops (list): A list of ops.
        cur_op (Operator): Current operator which has var_name variable.
        var_name (string): Variable name.
    """
    prev_op = []
    for op in ops:
        if op == cur_op:
            break
        for out_name in op.output_names:
            for out_var_name in op.output(out_name):
                if out_var_name == var_name:
                    prev_op.append(op)
    if prev_op:
        if not len(prev_op) == 1:
            raise ValueError("There must be only one previous op "
                             "that outputs {0} variable".format(var_name))
        else:
            return prev_op[0]
    return None


def find_true_post_op(ops, cur_op, var_name):
    """
    if there are post ops, return them, if there is no post op,
    return None instead.
    Args:
        ops (list): A list of ops.
        cur_op (Operator): Current operator which has var_name variable.
        var_name (string): Variable name.
    """
    post_op = []
    for idx, op in enumerate(ops):
        if op == cur_op:
            break

    for i in range(idx + 1, len(ops)):
        op = ops[i]
        for in_name in op.input_names:
            for in_var_name in op.input(in_name):
                if in_var_name == var_name:
                    post_op.append(op)
    if post_op != []:
        return post_op
    return None


def find_op_index(block_desc, cur_op_desc):
    """
    """
    for idx in range(block_desc.op_size()):
        if cur_op_desc == block_desc.op(idx):
            return idx
    return -1


def _is_in_black_varnames(op, amp_lists):
    for in_name in op.input_arg_names:
        if in_name in amp_lists.black_varnames:
            return True

    for out_name in op.output_arg_names:
        if out_name in amp_lists.black_varnames:
            return True

    return False


def cast_model_to_fp16(main_program):
    """
    Traverse all ops in the whole model and set their inputs and outputs
    to the fp16 data type. This function will do some special process for
    the batch normalization, which keeps the computational process of
    batchnorms in FP32.
    Args:
        main_program (Program): The main program for training.
    """
    valid_types = [
        core.VarDesc.VarType.LOD_TENSOR, core.VarDesc.VarType.SELECTED_ROWS,
        core.VarDesc.VarType.LOD_TENSOR_ARRAY
    ]
    global_block = main_program.global_block()

    for block in main_program.blocks:
        ops = block.ops
        for op in ops:
            if op.type == 'create_py_reader' or op.type == 'read':
                continue
            for in_name in op.input_names:
                if op.type in {
                        'batch_norm', 'fused_bn_add_activation', 'layer_norm'
                } and in_name not in {'X', 'Z'}:
                    continue
                for in_var_name in op.input(in_name):
                    in_var = None
                    try:
                        in_var = block.var(in_var_name)
                    except ValueError as e:
                        _logger.debug(
                            "-- {}, try to get it in the global block. --".
                            format(e))
                        in_var = global_block.var(in_var_name)
                        if in_var is not None:
                            _logger.debug(
                                "-- var {} is got in the global block. --".
                                format(in_var_name))

                    if in_var is None or in_var.type not in valid_types:
                        continue

                    if in_var.dtype == core.VarDesc.VarType.FP32:
                        in_var.desc.set_dtype(core.VarDesc.VarType.FP16)

                    _logger.debug(
                        "-- op type: {}, in var name: {}, in var dtype: {} --".
                        format(op.type, in_var_name, in_var.dtype))

            for out_name in op.output_names:
                if op.type in {
                        'batch_norm', 'fused_bn_add_activation', 'layer_norm'
                } and out_name != 'Y':
                    continue
                for out_var_name in op.output(out_name):
                    out_var = None
                    try:
                        out_var = block.var(out_var_name)
                    except ValueError as e:
                        _logger.debug(
                            "-- {}, try to get it in the global block. --".
                            format(e))
                        out_var = global_block.var(out_var_name)
                        if out_var is not None:
                            _logger.debug(
                                "-- var {} is got in the global block. --".
                                format(out_var_name))

                    if out_var is None or out_var.type not in valid_types:
                        continue

                    if out_var.dtype == core.VarDesc.VarType.FP32:
                        out_var.desc.set_dtype(core.VarDesc.VarType.FP16)

                    _logger.debug(
                        "-- op type: {}, out var name: {}, out var dtype: {} --".
                        format(op.type, out_var_name, out_var.dtype))
            if op.has_attr('in_dtype') and op.attr(
                    'in_dtype') == core.VarDesc.VarType.FP32:
                op._set_attr('in_dtype', core.VarDesc.VarType.FP16)
            if op.has_attr('out_dtype') and op.attr(
                    'out_dtype') == core.VarDesc.VarType.FP32:
                op._set_attr('out_dtype', core.VarDesc.VarType.FP16)
            if op.has_attr('dtype') and op.attr(
                    'dtype') == core.VarDesc.VarType.FP32:
                op._set_attr('dtype', core.VarDesc.VarType.FP16)


def cast_parameters_to_fp16(place, main_program, scope=None):
    """
    Traverse all parameters in the whole model and set them to the fp16 data type.
    Whereas, this function will keep parameters of batchnorms in FP32.
    Args:
        place(fluid.CPUPlace|fluid.CUDAPlace): place is used to restore the weight tensors.
        main_program (Program): The main program for training.
        scope(fluid.Scope, optional): scope is used to get the weight tensor values.
        Default is None.
    """
    all_ops = []
    for block in main_program.blocks:
        all_ops.extend(block.ops)
    bn_params = set()
    for op in all_ops:
        if op.type not in {
                'batch_norm', 'fused_bn_add_activation', 'layer_norm'
        }:
            continue
        for in_name in op.input_names:
            if in_name not in {'X', 'Z'}:
                for in_var_name in op.input(in_name):
                    bn_params.add(in_var_name)
    global_block = main_program.global_block()
    all_parameters = global_block.all_parameters()
    var_scope = scope if scope is not None else global_scope()
    for param in all_parameters:
        if param.name not in bn_params:
            param_t = var_scope.find_var(param.name).get_tensor()
            data = np.array(param_t)
            param_t.set(np.float16(data), place)


def rewrite_program(main_prog, amp_lists):
    """
    Traverse all ops in current block and insert cast op according to 
    which set current op belongs to.

    1. When an op belongs to the black list, add it to black set
    2. When an op belongs to the white list, add it to white set
    3. When an op belongs to the gray list. If one 
       of its inputs is the output of black set op or black list op, 
       add it to black set. If all of its previous ops are not black 
       op and one of its inputs is the output of white set op or 
       white list op, add it to white set.
    4. When an op isn't in the lists, add it to black op set.
    5. Add necessary cast ops to make sure that black set op will be 
       computed in fp32 mode, while white set op will be computed in 
       fp16 mode.

    Args:
        main_prog (Program): The main program for training.
    """
    block = main_prog.global_block()
    ops = block.ops
    white_op_set = set()
    black_op_set = set()
    for op in ops:

        # NOTE(zhiqiu): 'create_py_reader' and 'read' is used in non-iterable DataLoder, 
        # we don't need to handle reader op and the input of 'create_py_reader' is not 
        # in block, which may result in errors.
        # See GeneratorLoader._init_non_iterable() for details.
        if op.type == 'create_py_reader' or op.type == 'read':
            continue

        if amp_lists.black_varnames is not None and _is_in_black_varnames(
                op, amp_lists):
            black_op_set.add(op)
            continue

        if op.type in amp_lists.black_list:
            black_op_set.add(op)
        elif op.type in amp_lists.white_list:
            white_op_set.add(op)
        elif op.type in amp_lists.gray_list:
            is_black_op = False
            is_white_op = False
            for in_name in op.input_names:
                # if this op has inputs
                if in_name:
                    for in_var_name in op.input(in_name):
                        in_var = block.var(in_var_name)
                        # this in_var isn't the output of other op
                        if in_var.op is None:
                            continue
                        elif in_var.op is op:
                            prev_op = find_true_prev_op(ops, op, in_var_name)
                            if prev_op is None:
                                continue
                        else:
                            prev_op = in_var.op
                        # if it's one of inputs
                        if prev_op in black_op_set or \
                                prev_op.type in amp_lists.black_list:
                            is_black_op = True
                        elif prev_op in white_op_set or \
                                prev_op.type in amp_lists.white_list:
                            is_white_op = True
            if is_black_op:
                black_op_set.add(op)
            elif is_white_op:
                white_op_set.add(op)
            else:
                pass
        else:
            # For numerical safe, we apply fp32 computation on ops that
            # are not determined which list they should stay.
            black_op_set.add(op)

    idx = 0
    while idx < len(ops):
        op = ops[idx]
        num_cast_ops = 0
        if op in black_op_set:
            num_cast_ops = _insert_cast_op(block, op, idx,
                                           core.VarDesc.VarType.FP16,
                                           core.VarDesc.VarType.FP32)
        elif op in white_op_set:
            num_cast_ops = _insert_cast_op(block, op, idx,
                                           core.VarDesc.VarType.FP32,
                                           core.VarDesc.VarType.FP16)
        else:
            pass

        idx += num_cast_ops + 1


def update_role_var_grad(main_prog, params_grads):
    """
    Update op_role_var attr for some ops to make sure the gradients
    transferred across GPUs is FP16.
    1. Check whether the op that outputs gradient is cast or not.
    2. If op is cast and gradient is FP32, remove the op_role_var
       and find the prev op which outputs FP16 gradient
    3. Update the op_role_var of the prev op.

    Args:
        main_prog (Program): The main program for training.
        params_grads (list): A list of params and grads.
    """
    block = main_prog.global_block()
    BACKWARD = core.op_proto_and_checker_maker.OpRole.Backward
    OPTIMIZE = core.op_proto_and_checker_maker.OpRole.Optimize
    for p, g in params_grads:
        op = g.op
        if g.dtype == core.VarDesc.VarType.FP32 and op.type == 'cast':
            role = op.attr('op_role')
            if role & int(BACKWARD) and op.has_attr('op_role_var'):
                op.desc.remove_attr("op_role_var")
            else:
                raise ValueError("The cast op {0} must be in BACKWARD role "
                                 "and have op_role_var attr.".format(op))

            fp16_grad_name = op.input(op.input_names[0])[0]
            op_for_fp16_grad = find_true_prev_op(block.ops, op, fp16_grad_name)
            op_role_var_attr_name = \
                core.op_proto_and_checker_maker.kOpRoleVarAttrName()
            attr_val = [p.name, fp16_grad_name]
            if op_for_fp16_grad.has_attr(op_role_var_attr_name):
                attr_val.extend(op_for_fp16_grad.attr(op_role_var_attr_name))
            op_for_fp16_grad._set_attr(op_role_var_attr_name, attr_val)

            # Maximize the all_reduce overlap, and perform the cast
            # operation after gradients transfer.
            op._set_attr('op_role', OPTIMIZE)
            # optimize op should stay behind forward and backward ops
            if op == block.ops[-1]:
                continue
            post_ops = find_true_post_op(block.ops, op, g.name)
            if post_ops is not None:
                raise ValueError("The cast op {0}'s output should not be"
                                 "used by a non-optimize op, however, it"
                                 "is used by {1}".format(op, post_ops[0]))
            new_op_desc = block.desc.append_op()
            new_op_desc.copy_from(op.desc)

            op_idx = find_op_index(block.desc, op.desc)
            if op_idx == -1:
                raise ValueError("The op {0} is not in program".format(op))
            block.desc._remove_op(op_idx, op_idx + 1)
        block._sync_with_cpp()
