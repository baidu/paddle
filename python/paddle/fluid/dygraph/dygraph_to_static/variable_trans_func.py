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

from __future__ import print_function

import six
import gast

from paddle.fluid import core
from paddle.fluid.framework import Variable
from paddle.fluid.layers import assign, fill_constant, slice
from paddle.fluid.layers.control_flow import array_length, create_array
from paddle.fluid.layers.control_flow import array_read, array_write
from paddle.fluid.layers.control_flow import cond, while_loop
from paddle.fluid.layers.control_flow import less_than, increment
from paddle.fluid.layer_helper import LayerHelper

__all__ = [
    'to_static_variable_gast_node', 'create_static_variable_gast_node',
    'data_layer_not_check', 'DynamicList'
]


def data_layer_not_check(name, shape, dtype='float32', lod_level=0):
    """
    This function creates a variable on the global block. Unlike
    `paddle.fluid.data` , the created variable doesn't check the dtype and the
    shape of feed data because dygraph input data can be variable-length.
    This API is used in translating dygraph into static graph.

     Note: 
        The default :code:`stop_gradient` attribute of the Variable created by
        this API is true, which means the gradient won't be passed backward
        through the data Variable. Set :code:`var.stop_gradient = False` If
        user would like to pass backward gradient.

    Args:
       name (str): The name/alias of the variable, see :ref:`api_guide_Name`
           for more details.
       shape (list|tuple): List|Tuple of integers declaring the shape. You can
           set "None" at a dimension to indicate the dimension can be of any
           size. For example, it is useful to set changeable batch size as "None" 
       dtype (np.dtype|VarType|str, optional): The type of the data. Supported
           dtype: bool, float16, float32, float64, int8, int16, int32, int64,
           uint8. Default: float32
       lod_level (int, optional): The LoD level of the LoDTensor. Usually users
           don't have to set this value. For more details about when and how to
           use LoD level, see :ref:`user_guide_lod_tensor` . Default: 0

    Returns:
        Variable: The global variable that gives access to the data.
    """
    helper = LayerHelper('data', **locals())
    shape = list(shape)
    for i in six.moves.range(len(shape)):
        if shape[i] is None:
            shape[i] = -1

    return helper.create_global_variable(
        name=name,
        shape=shape,
        dtype=dtype,
        type=core.VarDesc.VarType.LOD_TENSOR,
        stop_gradient=True,
        lod_level=lod_level,
        is_data=True,
        need_check_feed=False)


def to_static_variable_gast_node(name):
    func_code = "{} = fluid.dygraph.dygraph_to_static.variable_trans_func\
        .to_static_variable({})".format(name, name)
    return gast.parse(func_code).body[0]


def create_static_variable_gast_node(name):
    func_code = "{} = fluid.dygraph.dygraph_to_static.variable_trans_func\
        .data_layer_not_check(name='{}', shape=[-1], dtype='float32')".format(
        name, name)
    return gast.parse(func_code).body[0]


def create_fill_constant_node(name, value):
    func_code = "{} = fluid.layers.fill_constant(shape=[1], ".format(name)
    if isinstance(value, bool):
        func_code += "dtype='bool', value={})".format(value)
        return gast.parse(func_code).body[0]
    if isinstance(value, float):
        func_code += "dtype='float64', value={})".format(value)
        return gast.parse(func_code).body[0]

    if six.PY2:
        if isinstance(value, int):
            func_code += "dtype='int32', value={})".format(value)
            return gast.parse(func_code).body[0]
        if isinstance(value, long):
            func_code += "dtype='int64', value={})".format(value)
            return gast.parse(func_code).body[0]
    else:
        if isinstance(value, int):
            func_code += "dtype='int64', value={})".format(value)
            return gast.parse(func_code).body[0]


def to_static_variable(x):
    '''
    Translate a Python variable to PaddlePaddle static graph variable
    '''
    if isinstance(x, bool):
        return fill_constant(shape=[1], dtype='bool', value=x)
    if isinstance(x, float):
        return fill_constant(shape=[1], dtype='float64', value=x)

    if isinstance(x, six.integer_types):
        return fill_constant(shape=[1], dtype='int64', value=x)

    return x


def slice_tensor_array(array, start, end):
    def true_fn():
        null_array = create_array("float32")
        return null_array

    def false_fn(array, start, end):
        new_array = slice(array, starts=[start], ends=[end], axes=[0])
        return new_array

    new_array = cond(start == end, true_fn, lambda: false_fn(array, start, end))
    return new_array


def dynamic_list_as_tensor_array(x):
    if isinstance(x, DynamicList) and x.private_type == Variable:
        return x.private_array
    return x


class DynamicList(list):
    '''
    A class casts python list to LoDTensorArray dynamically at run time.
    '''

    def __init__(self, iterable=None):
        if iterable == None:
            self.private_type = Variable
            self.private_array = create_array("float32")
            self.private_size = 0
            return

        self.private_type = self._get_same_type(iterable)
        # self.private_array is LoDTensorArray if all types are LoDTensor
        # else python list
        if self.private_type == Variable:
            self.private_array = create_array("float32")
            self.private_size = 0
            for i in iterable:
                self.append(i)
        else:
            self.private_size = 0
            self.private_array = None
            super(DynamicList, self).__init__(iterable)

    def __str__(self):
        if self.private_type == Variable:
            return "DynamicList : " + str(self.private_array)
        return super(DynamicList, self).__len__()

    def __len__(self):
        if self.private_type == Variable:
            return self.private_size
        return super(DynamicList, self).__len__()

    def __getitem__(self, i):
        if self.private_type == Variable:
            if isinstance(i, int):
                idx = fill_constant(shape=[1], dtype='int64', value=i)
                return array_read(self.private_array, idx)
            else:
                # i should be python slice here
                if i.step is not None:
                    raise NotImplementedError(
                        "LoDTensorArray hasn't implemented read by slice with step."
                    )
                return slice(
                    self.private_array,
                    axes=[0],
                    starts=[i.start],
                    ends=[i.stop])
        return super(DynamicList, self).__getitem__(i)

    def __setitem__(self, i, elem):
        if self.private_type == Variable:
            if isinstance(elem, Variable):
                if isinstance(i, int):
                    array_write(elem, i, self.private_array)
                else:
                    # i should be python slice here
                    raise NotImplementedError(
                        "LoDTensorArray hasn't implemented write by slice")
            else:
                # Transfer from LoDTensorArray to python list
                tmp_list = self.private_array._move_to_list()
                tmp_list[i] = elem
                self.private_type = self._get_same_type(tmp_list)
                self.private_array = None
                super(DynamicList, self).__init__(tmp_list)
        else:
            super(DynamicList, self).__setitem__(i, elem)

    def __iter__(self):
        if self.private_type == Variable:
            for i in range(self.private_size):
                idx = fill_constant(shape=[1], dtype='int64', value=i)
                yield array_read(self.private_array, idx)
        else:
            for i in super(DynamicList, self).__iter__():
                yield i

    def append(self, elem):
        if self.private_type == Variable:
            if isinstance(elem, Variable):
                append_index = array_length(self.private_array)
                array_write(x=elem, i=append_index, array=self.private_array)
                self.private_size += 1
            else:
                # Transfer from LoDTensorArray to python list
                tmp_list = self.private_array._move_to_list()
                tmp_list.append(elem)
                self.private_type = self._get_same_type(tmp_list)
                self.private_array = None
                self.private_size = 0
                super(DynamicList, self).__init__(tmp_list)
        else:
            super(DynamicList, self).append(elem)

    def pop(self, i=None):
        if self.private_type == Variable:
            arr_length = array_length(self.private_array)
            if i == None:
                remove_idx = arr_length - 1
                pop_elem = array_read(self.private_array, remove_idx)
                new_array = slice_tensor_array(self.private_array, 0,
                                               arr_length - 1)
                assign(input=new_array, output=self.private_array)
                self.private_size -= 1
            else:
                if i < 0:
                    remove_idx = i + arr_length
                else:
                    remove_idx = fill_constant(
                        shape=[1], dtype='int64', value=i)
                pop_elem = array_read(self.private_array, remove_idx)
                new_array = slice_tensor_array(self.private_array, 0,
                                               remove_idx)

                i = remove_idx + 1

                def cond(i, new_array):
                    return less_than(i, arr_length)

                def body(i, new_array):
                    item = array_read(array=self.private_array, i=i)
                    array_write(item, array_length(new_array), new_array)
                    i = increment(i)
                    return i, new_array

                _, new_array = while_loop(cond, body, [i, new_array])

                assign(input=new_array, output=self.private_array)
                self.private_size -= 1
            return pop_elem
        ret = super(DynamicList, self).pop()
        if super(DynamicList, self).__len__() == 0:
            self.private_type = Variable
            self.private_array = create_array("float32")
            self.private_size = 0
        return ret

    def extend(self, iterable):
        if self.private_type == Variable:
            for i in iterable:
                self.append(i)
            return
        super(DynamicList, self).extend(iterable)

    def insert(self, i, x):
        if self.private_type == Variable:
            raise NotImplementedError(
                "LoDTensorArray hasn't implemented 'insert' method.")
        super(DynamicList, self).insert(i, x)

    def remove(self, x):
        if self.private_type == Variable:
            raise NotImplementedError(
                "LoDTensorArray hasn't implemented 'remove' method.")
        super(DynamicList, self).remove(x)

    def clear(self):
        self.private_type = Variable
        self.private_array = create_array("float32")
        self.private_size = 0
        super(DynamicList, self).clear()

    def index(self, x, start=None, end=None):
        if self.private_type == Variable:
            raise NotImplementedError(
                "LoDTensorArray hasn't implemented 'index' method.")
        return super(DynamicList, self).index(x, start, end)

    def count(self, x):
        if self.private_type == Variable:
            raise NotImplementedError(
                "LoDTensorArray hasn't implemented 'count' method.")
        return super(DynamicList, self).count(x)

    def sort(self, key=None, reverse=None):
        if self.private_type == Variable:
            raise NotImplementedError(
                "LoDTensorArray hasn't implemented 'sort' method.")
        return super(DynamicList, self).sort(key, reverse)

    def reverse(self):
        if self.private_type == Variable:
            raise NotImplementedError(
                "LoDTensorArray hasn't implemented 'reverse' method.")
        return super(DynamicList, self).reverse()

    def copy(self):
        return DynamicList(self)

    def _get_same_type(self, iterable):
        """
        Returns the type if all iterable have same type. Returns LoDTensor if
        iterable is empty. Returns None if iterable contains different types.
        """
        len_args = len(iterable)
        if len_args == 0:
            return Variable
        ret = "Undefined"
        for i in iterable:
            if ret == "Undefined":
                ret = type(i)
            elif not isinstance(i, ret):
                return None
        return ret
