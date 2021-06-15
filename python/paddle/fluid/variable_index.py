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
# limitations under the License.

import sys
import numpy as np
from . import unique_name
from . import core

MAX_INTEGER = 2**31 - 1


def replace_ellipsis(var, item):
    from .framework import Variable
    # Use slice(None) to replace Ellipsis.
    # For var, var.shape = [3,4,5,6]
    #
    #   var[..., 1:2] -> var[:, :, :, 1:2]
    #   var[0, ...] -> var[0]
    #   var[0, ..., 1:2] -> var[0, :, :, 1:2]

    item = list(item)

    # Remove Variable to skip bug when counting Ellipsis
    item_remove_var = [ele for ele in item if not isinstance(ele, Variable)]
    ell_count = item_remove_var.count(Ellipsis)
    if ell_count == 0:
        return item
    elif ell_count > 1:
        raise IndexError("An index can only have a single ellipsis ('...')")

    ell_idx = item.index(Ellipsis)

    if ell_idx == len(item) - 1:
        return item[:-1]
    else:
        item[ell_idx:ell_idx + 1] = [slice(None)] * (
            len(var.shape) - len(item) + 1)

    return item


def is_integer_or_scalar_tensor(ele):
    from .framework import Variable
    if isinstance(ele, int):
        return True
    elif isinstance(ele, Variable):
        if len(ele.shape) == 1 and ele.shape[0] == 1:
            return True
    return False


def deal_attrs(attrs, attr, attr_name, tensor_attr_name, inputs, infer_flags):
    from .framework import Variable
    from .layers import utils

    if utils._contain_var(attr):
        inputs[tensor_attr_name] = utils._convert_to_tensor_list(
            attr, dtype="int64")
        for i, dim in enumerate(attr):
            if isinstance(dim, Variable):
                attrs[attr_name].append(-1)
                infer_flags[i] = -1
            else:
                attrs[attr_name].append(dim)
    else:
        attrs[attr_name] = attr


def _getitem_impl_(var, item):
    """
    Slice the variable.

    Args:
        item(int/slice/tuple) : the index.

    Returns:
        Sliced variable
    """
    from .framework import default_main_program

    if not isinstance(item, tuple):
        item = (item, )

    decrease_axes = []
    axes = []
    starts = []
    ends = []
    steps = []
    reverse_axis = []

    use_strided_slice = False

    for dim, slice_item in enumerate(item):
        if is_integer_or_scalar_tensor(slice_item):
            decrease_axes.append(dim)
            start = slice_item
            step = 1
            end = slice_item + 1 if slice_item != -1 else MAX_INTEGER

        elif isinstance(slice_item, slice):
            start = slice_item.start
            end = slice_item.stop
            step = slice_item.step

            if start is None and end is None and step is None:
                continue

            step = 1 if step is None else step

            if start is None and end is None:
                assert (step == -1)
                reverse_axis.append(dim)
                continue

            start = 0 if start is None else start
            end = MAX_INTEGER if end is None else end

        else:
            raise IndexError(
                "Valid index accept int or slice or ellipsis, but received {}.".
                format(slice_item))

        axes.append(dim)
        starts.append(start)
        ends.append(end)
        steps.append(step)
        use_strided_slice = True if step != 1 else use_strided_slice

    inputs = {'Input': [var]}
    attrs = {
        'axes': axes,
        'starts': [],
        'ends': [],
        'decrease_axis': decrease_axes
    }
    if use_strided_slice:
        attrs['strides'] = []

    infer_flags = [1] * len(axes)
    deal_attrs(attrs, starts, "starts", "StartsTensorList", inputs, infer_flags)
    deal_attrs(attrs, ends, "ends", "EndsTensorList", inputs, infer_flags)
    deal_attrs(attrs, steps, "strides", "StridesTensorList", inputs,
               infer_flags)
    attrs['infer_flags'] = infer_flags

    out = var
    if len(axes) > 0:
        target_block = default_main_program().current_block()
        op_type = "strided_slice" if use_strided_slice else "slice"

        slice_out_var = target_block.create_var(
            name=unique_name.generate_with_ignorable_key(var.name + "_" +
                                                         op_type),
            dtype=var.dtype)
        target_block.append_op(
            type=op_type,
            inputs=inputs,
            outputs={'Out': [slice_out_var]},
            attrs=attrs)
        out = slice_out_var

    if len(reverse_axis) > 0:
        from .layers.tensor import reverse
        out = reverse(out, axis=reverse_axis)

    return out


def _setitem_impl_(var, item, value):
    from .framework import default_main_program, Variable

    inputs = {'Input': var}

    # 1. Parse item
    if not isinstance(item, tuple):
        item = (item, )

    decrease_axes = []
    axes = []
    starts = []
    ends = []
    steps = []

    item = replace_ellipsis(var, item)

    for dim, slice_item in enumerate(item):
        if is_integer_or_scalar_tensor(slice_item):
            decrease_axes.append(dim)
            start = slice_item
            end = slice_item + 1 if slice_item != -1 else MAX_INTEGER
            step = 1

        elif isinstance(slice_item, slice):
            start = slice_item.start
            end = slice_item.stop
            step = slice_item.step

            if start is None and end is None and step is None:
                continue

            step = 1 if step is None else step

            if not isinstance(step, Variable) and step == 0:
                raise ValueError(
                    "When assign a value to a paddle.Tensor, step can not be 0, "
                    "but received step is {}.".format(step))

            if isinstance(step, Variable) and (start is None or end is None):
                raise ValueError(
                    "When assign a value to a paddle.Tensor, it's not supported that "
                    "the start or end is None when the type of step is paddle.Tensor."
                )

            if start is None:
                start = 0 if step > 0 else MAX_INTEGER

            if end is None:
                end = MAX_INTEGER if step > 0 else (0 - MAX_INTEGER)
        else:
            raise IndexError(
                "Valid index accept int or slice or ellipsis, but received {}.".
                format(slice_item))

        axes.append(dim)
        starts.append(start)
        ends.append(end)
        steps.append(step)

    attrs = {
        'axes': axes,
        'starts': starts,
        'ends': ends,
        'steps': steps,
        'decrease_axes': decrease_axes
    }

    from .layers import utils
    if utils._contain_var(starts):
        inputs['StartsTensorList'] = utils._convert_to_tensor_list(starts)
        del attrs['starts']
    if utils._contain_var(ends):
        inputs['EndsTensorList'] = utils._convert_to_tensor_list(ends)
        del attrs['ends']
    if utils._contain_var(steps):
        inputs['StepsTensorList'] = utils._convert_to_tensor_list(steps)
        del attrs['steps']

    # 2. Parse value
    dtype = var.dtype
    attrs['dtype'] = dtype

    from .data_feeder import convert_dtype
    #  2.1 value is an integer of float
    if isinstance(value, (int, float)):
        value = np.array([value]).astype(convert_dtype(dtype))

    #  2.2 value is a np.ndarray
    if isinstance(value, np.ndarray):
        shape = list(value.shape)
        if dtype == core.VarDesc.VarType.BOOL:
            value_name = "bool_values"
            values = [bool(v) for v in value.flat]
        elif dtype == core.VarDesc.VarType.FP32:
            value_name = "fp32_values"
            values = [float(v) for v in value.flat]
        elif dtype == core.VarDesc.VarType.FP64:
            value_name = "fp64_values"
            values = [float(v) for v in value.flat]
        elif dtype == core.VarDesc.VarType.INT32:
            value_name = "int32_values"
            values = [int(v) for v in value.flat]
        elif dtype == core.VarDesc.VarType.INT64:
            value_name = "int64_values"
            values = [int(v) for v in value.flat]
        else:
            raise TypeError(
                "When assign a numpy.ndarray, integer or float to a paddle.Tensor, "
                "the data type of the paddle.Tensor must be bool, float32, int32 or int64, but "
                "received %s." % convert_dtype(dtype))
        attrs[value_name] = values
        attrs["shape"] = shape

    elif isinstance(value, Variable):
        inputs["ValueTensor"] = value
    else:
        raise TypeError(
            "Only support to assign an integer, float, numpy.ndarray or "
            "paddle.Tensor to a paddle.Tensor, but received {}".format(
                type(value)))

    cur_block = default_main_program().current_block()
    cur_block.append_op(
        type="set_value", inputs=inputs, outputs={'Out': var}, attrs=attrs)

    return var
