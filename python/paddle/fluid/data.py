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

import executor
import numpy as np

from . import core
from .framework import convert_np_dtype_to_dtype_
from .layer_helper import LayerHelper

__all__ = ['data']


def data(name, shape, dtype='float32', type=core.VarDesc.VarType.LOD_TENSOR):
    """
    **Data Layer**

    This function takes in the input and based on whether the data has
    to be returned back as a minibatch, it creates the global variable by using
    the helper functions. The global variables can be accessed by all the
    following operators in the graph.

    All the input variables of this function are passed in as local variables
    to the LayerHelper constructor.

    Note: Unlike `paddle.fluid.layers.data` which set shape at compile time but
       not check the shape of feeded data, this `paddle.fluid.data` checks the
       shape of data feeded by Executor/ParallelExecutor during run time.

    Args:
       name (None|str): The name/alias of the variable
       shape (list|tuple): List|Tuple of integers declaring the shape.
       dtype (np.dtype|VarType|str): The type of the data: float32, int64, etc.
       type (VarType): The output type. Default: LOD_TENSOR.

    Returns:
        Variable: The global variable that gives access to the data.

    Examples:
        .. code-block:: python

          import paddle.fluid as fluid
          data = fluid.data(name='x', shape=[784], dtype='float32')

    """
    helper = LayerHelper('data', **locals())
    return helper.create_global_variable(
        name=name,
        shape=shape,
        dtype=dtype,
        type=type,
        stop_gradient=True,
        lod_level=0,
        is_data=True,
        need_check_feed=True)


def check_feed_shape_type(var, feed):
    """
    Returns True if the variable doesn't require feed check or it is compatible
    with the shape and have same dtype as the feeded value.

    A dimension is compatible with the other if:
    1. The length of the dimensions are same.
    2. Each non-negative number of the two dimentions are same.
    3. For negative number or 'None' in a dimention, it means unknown so it
       is compatible with any number.
    
    Args:
        var (Variable): the Variable object
        feed (list|np.array): the feeded value
    Returns:
        True if the shape and dtype of variable is compatible with the feed value
    Raises:
        ValueError: if the shape or dtype of the variable is not compatible with
            the feed value
    """
    if var.desc.need_check_feed():
        numpy_feed = executor.as_numpy(feed) if isinstance(
            feed, core.LoDTensorArray) else np.array(
                feed, copy=False)
        if not dimension_is_compatible_with(numpy_feed.shape, var.shape):
            raise ValueError('Cannot feed value of shape %r for Variable %r, '
                             'which has shape %r' %
                             (numpy_feed.shape, var.name, var.shape))
        if not dtype_is_compatible_with(numpy_feed.dtype, var.dtype):
            raise ValueError('Cannot feed value of type %r for Variable %r, '
                             'which has type %r' %
                             (numpy_feed.dtype, var.name, var.dtype))
    return True


def dtype_is_compatible_with(first, second):
    """
    Returns True if the first dtype can be compatible the second one.
    Currently, we require the two dtype's have to be same.
      
    Args:
        dtype (np.dtype|VarType|str): The type of data : float32, int64, etc.
    
    Returns:
        True if the two types are same.
    """
    if not isinstance(first, core.VarDesc.VarType):
        first = convert_np_dtype_to_dtype_(first)
    if not isinstance(second, core.VarDesc.VarType):
        second = convert_np_dtype_to_dtype_(second)
    return first == second


def dimension_is_compatible_with(first, second):
    """
    Returns True if the two dimensions are compatible.

    A dimension is compatible with the other if:
    1. The length of the dimensions are same.
    2. Each non-negative number of the two dimentions are same.
    3. For negative number or 'None' in a dimention, it means unknown so it
       is compatible with any number.

    Args:
        first (list/tuple): integers representing shape. "None" or negative
            number means unknown.
        second (list/tuple): integers representing shape. "None" or negative
            number means unknown.

    Returns:
        True if the two dimensions are compatible.
    """

    dim_len = len(first)
    if dim_len != len(second):
        return False

    for i in range(dim_len):
        if first[i] is None or first[i] < 0:
            continue
        if second[i] is None or second[i] < 0:
            continue
        if first[i] != second[i]:
            return False

    return True
