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
import numpy as np
from ..fluid.layer_helper import LayerHelper
from ..fluid.data_feeder import check_variable_and_dtype, check_type, check_dtype
from ..fluid import core, layers

# TODO: define searching & indexing functions of a tensor  
# from ..fluid.layers import has_inf  #DEFINE_ALIAS
# from ..fluid.layers import has_nan  #DEFINE_ALIAS

__all__ = [
    'argmax',
    'argmin',
    'argsort',
    'masked_select',
    'topk',
    'where',
    'index_select',
    'nonzero',
    'sort',
    'index_sample',
]

from paddle.common_ops_import import *


def argsort(x, axis=-1, descending=False, name=None):
    """
	:alias_main: paddle.argsort
	:alias: paddle.argsort,paddle.tensor.argsort,paddle.tensor.search.argsort

    This OP sorts the input along the given axis, and returns the corresponding index tensor for the sorted output values. The default sort algorithm is ascending, if you want the sort algorithm to be descending, you must set the :attr:`descending` as True.

    Args:
        x(Tensor): An input N-D Tensor with type float32, float64, int16,
            int32, int64, uint8.
        axis(int, optional): Axis to compute indices along. The effective range
            is [-R, R), where R is Rank(x). when axis<0, it works the same way
            as axis+R. Default is 0.
        descending(bool, optional) : Descending is a flag, if set to true,
            algorithm will sort by descending order, else sort by
            ascending order. Default is false.
        name(str, optional): The default value is None. Normally there is no
            need for user to set this property. For more information, please
            refer to :ref:`api_guide_Name`.

    Returns:
        Tensor: sorted indices(with the same shape as ``x``
        and with data type int64).

    Examples:

        .. code-block:: python

            import paddle
            
            paddle.disable_static()
            x = paddle.to_tensor([[[5,8,9,5],
                                   [0,0,1,7],
                                   [6,9,2,4]],
                                  [[5,2,4,2],
                                   [4,7,7,9],
                                   [1,7,0,6]]], 
                                dtype='float32')
            out1 = paddle.argsort(x=x, axis=-1)
            out2 = paddle.argsort(x=x, axis=0)
            out3 = paddle.argsort(x=x, axis=1)
            print(out1.numpy())
            #[[[0 3 1 2]
            #  [0 1 2 3]
            #  [2 3 0 1]]
            # [[1 3 2 0]
            #  [0 1 2 3]
            #  [2 0 3 1]]]
            print(out2.numpy())
            #[[[0 1 1 1]
            #  [0 0 0 0]
            #  [1 1 1 0]]
            # [[1 0 0 0]
            #  [1 1 1 1]
            #  [0 0 0 1]]]
            print(out3.numpy())
            #[[[1 1 1 2]
            #  [0 0 2 0]
            #  [2 2 0 1]]
            # [[2 0 2 0]
            #  [1 1 0 2]
            #  [0 2 1 1]]]
    """
    if in_dygraph_mode():
        _, ids = core.ops.argsort(x, 'axis', axis, 'descending', descending)
        return ids
    check_variable_and_dtype(
        x, 'x', ['float32', 'float64', 'int16', 'int32', 'int64', 'uint8'],
        'argsort')

    helper = LayerHelper("argsort", **locals())
    out = helper.create_variable_for_type_inference(
        dtype=x.dtype, stop_gradient=True)
    ids = helper.create_variable_for_type_inference(
        VarDesc.VarType.INT64, stop_gradient=True)
    helper.append_op(
        type='argsort',
        inputs={'X': x},
        outputs={'Out': out,
                 'Indices': ids},
        attrs={'axis': axis,
               'descending': descending})
    return ids


def argmax(x, axis=None, keepdim=False, dtype="int64", name=None):
    """
    This OP computes the indices of the max elements of the input tensor's
    element along the provided axis.

    Args:
        x(Tensor): An input N-D Tensor with type float32, float64, int16,
            int32, int64, uint8.
        axis(int, optional): Axis to compute indices along. The effective range
            is [-R, R), where R is x.ndim. when axis < 0, it works the same way
            as axis + R. Default is None, the input `x` will be into the flatten tensor, and selecting the min value index.
        keepdim(bool, optional): Keep the axis that selecting max. The defalut value is False.
        dtype(str|np.dtype, optional): Data type of the output tensor which can
                    be int32, int64. The default value is 'int64', and it will
                    return the int64 indices.
        name(str, optional): The default value is None. Normally there is no
            need for user to set this property. For more information, please
            refer to :ref:`api_guide_Name`.

    Returns:
        Tensor, return the tensor of `int32` if set :attr:`dtype` is `int32`, otherwise return the tensor of `int64`

    Examples:
        .. code-block:: python

            import paddle

            paddle.disable_static()
            x =  paddle.to_tensor([[5,8,9,5],
                                     [0,0,1,7],
                                     [6,9,2,4]])
            out1 = paddle.argmax(x)
            print(out1.numpy()) # 2
            out2 = paddle.argmax(x, axis=1)
            print(out2.numpy()) 
            # [2 3 1]
            out3 = paddle.argmax(x, axis=-1)
            print(out3.numpy()) 
            # [2 3 1]
    """
    if axis is not None and not isinstance(axis, int):
        raise TypeError(
            "The type of 'axis'  must be int or None in argmax, but received %s."
            % (type(axis)))

    if dtype is None:
        raise ValueError(
            "the value of 'dtype' in argmax could not be None, but received None"
        )

    var_dtype = convert_np_dtype_to_dtype_(dtype)
    check_dtype(var_dtype, 'dtype', ['int32', 'int64'], 'argmin')
    flatten = False
    if axis is None:
        flatten = True
        axis = 0

    if in_dygraph_mode():
        out = core.ops.arg_max(x, 'axis', axis, 'dtype', var_dtype, 'keepdims',
                               keepdim, 'flatten', flatten)
        return out

    helper = LayerHelper("argmax", **locals())
    check_variable_and_dtype(
        x, 'x', ['float32', 'float64', 'int16', 'int32', 'int64', 'uint8'],
        'paddle.argmax')
    attrs = {}
    out = helper.create_variable_for_type_inference(var_dtype)
    attrs['keepdims'] = keepdim
    attrs['axis'] = axis
    attrs['flatten'] = flatten
    attrs['dtype'] = var_dtype
    helper.append_op(
        type='arg_max', inputs={'X': x}, outputs={'Out': [out]}, attrs=attrs)
    out.stop_gradient = True
    return out


def argmin(x, axis=None, keepdim=False, dtype="int64", name=None):
    """
    This OP computes the indices of the min elements of the input tensor's
    element along the provided axis.

    Args:
        x(Tensor): An input N-D Tensor with type float32, float64, int16,
            int32, int64, uint8.
        axis(int, optional): Axis to compute indices along. The effective range
            is [-R, R), where R is x.ndim. when axis < 0, it works the same way
            as axis + R. Default is None, the input `x` will be into the flatten tensor, and selecting the min value index.
        keepdim(bool, optional): Keep the axis that selecting min. The defalut value is False.
        dtype(str): Data type of the output tensor which can
                    be int32, int64. The default value is 'int64', and it will
                    return the int64 indices.
        name(str, optional): The default value is None. Normally there is no
            need for user to set this property. For more information, please
            refer to :ref:`api_guide_Name`.

    Returns:
        Tensor, return the tensor of `int32` if set :attr:`dtype` is `int32`, otherwise return the tensor of `int64`

    Examples:
        .. code-block:: python

            import paddle

            paddle.disable_static()
            x =  paddle.to_tensor([[5,8,9,5],
                                     [0,0,1,7],
                                     [6,9,2,4]])
            out1 = paddle.argmin(x)
            print(out1.numpy()) # 4
            out2 = paddle.argmin(x, axis=1)
            print(out2.numpy()) 
            # [0 0 2]
            out3 = paddle.argmin(x, axis=-1)
            print(out3.numpy()) 
            # [0 0 2]
    """
    if axis is not None and not isinstance(axis, int):
        raise TypeError(
            "The type of 'axis'  must be int or None in argmin, but received %s."
            % (type(axis)))

    if dtype is None:
        raise ValueError(
            "the value of 'dtype' in argmin could not be None, but received None"
        )

    var_dtype = convert_np_dtype_to_dtype_(dtype)
    check_dtype(var_dtype, 'dtype', ['int32', 'int64'], 'argmin')
    flatten = False
    if axis is None:
        flatten = True
        axis = 0

    if in_dygraph_mode():
        out = core.ops.arg_min(x, 'axis', axis, 'dtype', var_dtype, 'keepdims',
                               keepdim, 'flatten', flatten)
        return out

    helper = LayerHelper("argmin", **locals())
    check_variable_and_dtype(
        x, 'x', ['float32', 'float64', 'int16', 'int32', 'int64', 'uint8'],
        'paddle.argmin')
    out = helper.create_variable_for_type_inference(var_dtype)
    attrs = {}
    attrs['keepdims'] = keepdim
    attrs['axis'] = axis
    attrs['flatten'] = flatten
    attrs['dtype'] = var_dtype
    helper.append_op(
        type='arg_min', inputs={'X': x}, outputs={'Out': [out]}, attrs=attrs)
    out.stop_gradient = True
    return out


def index_select(x, index, axis=0, name=None):
    """

    Returns a new tensor which indexes the ``input`` tensor along dimension ``axis`` using 
    the entries in ``index`` which is a Tensor. The returned tensor has the same number 
    of dimensions as the original ``x`` tensor. The dim-th dimension has the same 
    size as the length of ``index``; other dimensions have the same size as in the ``x`` tensor. 

    Args:
        x (Tensor): The input Tensor to be operated. The data of ``x`` can be one of float32, float64, int32, int64.
        index (Tensor): The 1-D Tensor containing the indices to index. The data type of ``index`` must be int32 or int64.
        axis (int, optional): The dimension in which we index. Default: if None, the ``axis`` is 0.
        name(str, optional): The default value is None. Normally there is no
            need for user to set this property. For more information, please
            refer to :ref:`api_guide_Name`.

    Returns:
        Tensor: A Tensor with same data type as ``x``.
    
    Examples:
        .. code-block:: python
            
            import paddle

            x = paddle.to_tensor([[1.0, 2.0, 3.0, 4.0],
                                  [5.0, 6.0, 7.0, 8.0],
                                  [9.0, 10.0, 11.0, 12.0]])
            index = paddle.to_tensor([0, 1, 1], dtype='int32')
            out_z1 = paddle.index_select(x=x, index=index)
            #[[1. 2. 3. 4.]
            # [5. 6. 7. 8.]
            # [5. 6. 7. 8.]]
            out_z2 = paddle.index_select(x=x, index=index, axis=1)
            #[[ 1.  2.  2.]
            # [ 5.  6.  6.]
            # [ 9. 10. 10.]]
    """

    if in_dygraph_mode():
        return core.ops.index_select(x, index, 'dim', axis)

    helper = LayerHelper("index_select", **locals())
    check_variable_and_dtype(x, 'x', ['float32', 'float64', 'int32', 'int64'],
                             'paddle.tensor.search.index_select')
    check_variable_and_dtype(index, 'index', ['int32', 'int64'],
                             'paddle.tensor.search.index_select')

    out = helper.create_variable_for_type_inference(x.dtype)

    helper.append_op(
        type='index_select',
        inputs={'X': x,
                'Index': index},
        outputs={'Out': out},
        attrs={'dim': axis})
    return out


def nonzero(x, as_tuple=False):
    """
    Return a tensor containing the indices of all non-zero elements of the `input` 
    tensor. If as_tuple is True, return a tuple of 1-D tensors, one for each dimension 
    in `input`, each containing the indices (in that dimension) of all non-zero elements 
    of `input`. Given a n-Dimensional `input` tensor with shape [x_1, x_2, ..., x_n], If 
    as_tuple is False, we can get a output tensor with shape [z, n], where `z` is the 
    number of all non-zero elements in the `input` tensor. If as_tuple is True, we can get 
    a 1-D tensor tuple of length `n`, and the shape of each 1-D tensor is [z, 1].

    Args:
        x (Tensor): The input tensor variable.
        as_tuple (bool): Return type, Tensor or tuple of Tensor.

    Returns:
        Tensor. The data type is int64.

    Examples:
    
        .. code-block:: python


            import paddle

            x1 = paddle.to_tensor([[1.0, 0.0, 0.0],
                          [0.0, 2.0, 0.0],
                          [0.0, 0.0, 3.0]])
            x2 = paddle.to_tensor([0.0, 1.0, 0.0, 3.0])
            x3 = paddle.to_tensor([0.0, 0.0, 0.0])
            out_z1 = paddle.nonzero(x1)
            print(out_z1.numpy())
            #[[0 0]
            # [1 1]
            # [2 2]]
            out_z1_tuple = paddle.nonzero(x1, as_tuple=True)
            for out in out_z1_tuple:
                print(out.numpy())
            #[[0]
            # [1]
            # [2]]
            #[[0]
            # [1]
            # [2]]
            out_z2 = paddle.nonzero(x2)
            print(out_z2.numpy())
            #[[1]
            # [3]]
            out_z2_tuple = paddle.nonzero(x2, as_tuple=True)
            for out in out_z2_tuple:
                print(out.numpy())
            #[[1]
            # [3]]
            out_z3 = paddle.nonzero(x3)
            print(out_z3.numpy())
            #[]
            out_z3_tuple = paddle.nonzero(x3, as_tuple=True)
            for out in out_z3_tuple:
                print(out.numpy())
            #[]                    
    """
    list_out = []
    shape = x.shape
    rank = len(shape)

    if in_dygraph_mode():
        outs = core.ops.where_index(x)
    else:
        outs = layers.where(x)

    if not as_tuple:
        return outs
    elif rank == 1:
        return tuple([outs])
    else:
        for i in range(rank):
            list_out.append(
                layers.slice(
                    outs, axes=[rank - 1], starts=[i], ends=[i + 1]))
        return tuple(list_out)


def sort(x, axis=-1, descending=False, name=None):
    """

    This OP sorts the input along the given axis, and returns the sorted output tensor. The default sort algorithm is ascending, if you want the sort algorithm to be descending, you must set the :attr:`descending` as True.

    Args:
        x(Tensor): An input N-D Tensor with type float32, float64, int16,
            int32, int64, uint8.
        axis(int, optional): Axis to compute indices along. The effective range
            is [-R, R), where R is Rank(x). when axis<0, it works the same way
            as axis+R. Default is 0.
        descending(bool, optional) : Descending is a flag, if set to true,
            algorithm will sort by descending order, else sort by
            ascending order. Default is false.
        name(str, optional): The default value is None. Normally there is no
            need for user to set this property. For more information, please
            refer to :ref:`api_guide_Name`.
    Returns:
        Tensor: sorted tensor(with the same shape and data type as ``x``).
    Examples:
        .. code-block:: python

            import paddle
            
            x = paddle.to_tensor([[[5,8,9,5],
                                   [0,0,1,7],
                                   [6,9,2,4]],
                                  [[5,2,4,2],
                                   [4,7,7,9],
                                   [1,7,0,6]]], 
                                 dtype='float32')
            out1 = paddle.sort(x=x, axis=-1)
            out2 = paddle.sort(x=x, axis=0)
            out3 = paddle.sort(x=x, axis=1)
            print(out1)
            #[[[5. 5. 8. 9.]
            #  [0. 0. 1. 7.]
            #  [2. 4. 6. 9.]]
            # [[2. 2. 4. 5.]
            #  [4. 7. 7. 9.]
            #  [0. 1. 6. 7.]]]
            print(out2)
            #[[[5. 2. 4. 2.]
            #  [0. 0. 1. 7.]
            #  [1. 7. 0. 4.]]
            # [[5. 8. 9. 5.]
            #  [4. 7. 7. 9.]
            #  [6. 9. 2. 6.]]]
            print(out3)
            #[[[0. 0. 1. 4.]
            #  [5. 8. 2. 5.]
            #  [6. 9. 9. 7.]]
            # [[1. 2. 0. 2.]
            #  [4. 7. 4. 6.]
            #  [5. 7. 7. 9.]]]
    """
    if in_dygraph_mode():
        out, _ = core.ops.argsort(x, 'axis', axis, 'descending', descending)
        return out
    helper = LayerHelper("sort", **locals())
    out = helper.create_variable_for_type_inference(
        dtype=x.dtype, stop_gradient=False)
    ids = helper.create_variable_for_type_inference(
        VarDesc.VarType.INT64, stop_gradient=True)
    helper.append_op(
        type='argsort',
        inputs={'X': x},
        outputs={'Out': out,
                 'Indices': ids},
        attrs={'axis': axis,
               'descending': descending})
    return out


def where(condition, x, y, name=None):
    """
	:alias_main: paddle.where
	:alias: paddle.where,paddle.tensor.where,paddle.tensor.search.where

    Return a tensor of elements selected from either $x$ or $y$, depending on $condition$.

    .. math::

      out_i =
      \\begin{cases}
      x_i, \quad  \\text{if}  \\ condition_i \\  is \\ True \\\\
      y_i, \quad  \\text{if}  \\ condition_i \\  is \\ False \\\\
      \\end{cases}


    Args:
        condition(Variable): The condition to choose x or y.
        x(Variable): x is a Tensor Variable with data type float32, float64, int32, int64.
        y(Variable): y is a Tensor Variable with data type float32, float64, int32, int64.

        name(str, optional): The default value is None. Normally there is no
            need for user to set this property. For more information, please
            refer to :ref:`api_guide_Name`.

    Returns:
        Variable: A Tensor with the same data dype as x. 

    Examples:
        .. code-block:: python

          import paddle

          paddle.disable_static()
          x = paddle.to_tensor([0.9383, 0.1983, 3.2, 1.2])
          y = paddle.to_tensor([1.0, 1.0, 1.0, 1.0])
          out = paddle.where(x>1, x, y)

          print(out.numpy())
          #out: [1.0, 1.0, 3.2, 1.2]
    """
    if not in_dygraph_mode():
        check_variable_and_dtype(condition, 'condition', ['bool'], 'where')
        check_variable_and_dtype(
            x, 'x', ['float32', 'float64', 'int32', 'int64'], 'where')
        check_variable_and_dtype(
            y, 'y', ['float32', 'float64', 'int32', 'int64'], 'where')

    x_shape = list(x.shape)
    y_shape = list(y.shape)
    if x_shape == y_shape:
        if in_dygraph_mode():
            return core.ops.where(condition, x, y)
        else:
            helper = LayerHelper("where", **locals())
            out = helper.create_variable_for_type_inference(dtype=x.dtype)

            helper.append_op(
                type='where',
                inputs={'Condition': condition,
                        'X': x,
                        'Y': y},
                outputs={'Out': [out]})
            return out
    else:
        cond_int = layers.cast(condition, x.dtype)
        cond_not_int = layers.cast(layers.logical_not(condition), x.dtype)
        out1 = layers.elementwise_mul(x, cond_int)
        out2 = layers.elementwise_mul(y, cond_not_int)
        out = layers.elementwise_add(out1, out2)
        return out


def index_sample(x, index):
    """
    **IndexSample Layer**

    IndexSample OP returns the element of the specified location of X, 
    and the location is specified by Index. 

    .. code-block:: text


                Given:

                X = [[1, 2, 3, 4, 5],
                     [6, 7, 8, 9, 10]]

                Index = [[0, 1, 3],
                         [0, 2, 4]]

                Then:

                Out = [[1, 2, 4],
                       [6, 8, 10]]

    Args:
        x (Tensor): The source input tensor with 2-D shape. Supported data type is 
            int32, int64, float32, float64.
        index (Tensor): The index input tensor with 2-D shape, first dimension should be same with X. 
            Data type is int32 or int64.

    Returns:
        output (Tensor): The output is a tensor with the same shape as index.

    Examples:

        .. code-block:: python

            import paddle

            x = paddle.to_tensor([[1.0, 2.0, 3.0, 4.0],
                                  [5.0, 6.0, 7.0, 8.0],
                                  [9.0, 10.0, 11.0, 12.0]], dtype='float32')
            index = paddle.to_tensor([[0, 1, 2],
                                      [1, 2, 3],
                                      [0, 0, 0]], dtype='int32')
            target = paddle.to_tensor([[100, 200, 300, 400],
                                       [500, 600, 700, 800],
                                       [900, 1000, 1100, 1200]], dtype='int32')
            out_z1 = paddle.index_sample(x, index)
            print(out_z1.numpy())
            #[[1. 2. 3.]
            # [6. 7. 8.]
            # [9. 9. 9.]]

            # Use the index of the maximum value by topk op
            # get the value of the element of the corresponding index in other tensors
            top_value, top_index = paddle.topk(x, k=2)
            out_z2 = paddle.index_sample(target, top_index)
            print(top_value.numpy())
            #[[ 4.  3.]
            # [ 8.  7.]
            # [12. 11.]]

            print(top_index.numpy())
            #[[3 2]
            # [3 2]
            # [3 2]]

            print(out_z2.numpy())
            #[[ 400  300]
            # [ 800  700]
            # [1200 1100]]

    """
    if in_dygraph_mode():
        return core.ops.index_sample(x, index)

    helper = LayerHelper("index_sample", **locals())
    check_variable_and_dtype(x, 'x', ['float32', 'float64', 'int32', 'int64'],
                             'paddle.tensor.search.index_sample')
    check_variable_and_dtype(index, 'index', ['int32', 'int64'],
                             'paddle.tensor.search.index_sample')
    out = helper.create_variable_for_type_inference(dtype=x.dtype)

    helper.append_op(
        type='index_sample',
        inputs={'X': x,
                'Index': index},
        outputs={'Out': out})
    return out


def masked_select(x, mask, name=None):
    """
    This OP Returns a new 1-D tensor which indexes the input tensor according to the ``mask``
    which is a tensor with data type of bool.

    Args:
        x (Tensor): The input Tensor, the data type can be int32, int64, float32, float64. 
        mask (Tensor): The Tensor containing the binary mask to index with, it's data type is bool.
        name(str, optional): The default value is None. Normally there is no
            need for user to set this property. For more information, please
            refer to :ref:`api_guide_Name`.

    Returns: A 1-D Tensor which is the same data type  as ``x``.
    
    Examples:

        .. code-block:: python

            import paddle


            x = paddle.to_tensor([[1.0, 2.0, 3.0, 4.0],
                                  [5.0, 6.0, 7.0, 8.0],
                                  [9.0, 10.0, 11.0, 12.0]])
            mask = paddle.to_tensor([[True, False, False, False],
                                     [True, True, False, False],
                                     [True, False, False, False]])
            out = paddle.masked_select(x, mask)
            #[1.0 5.0 6.0 9.0]
    """

    if in_dygraph_mode():
        return core.ops.masked_select(x, mask)

    helper = LayerHelper("masked_select", **locals())
    check_variable_and_dtype(x, 'x', ['float32', 'float64', 'int32', 'int64'],
                             'paddle.tensor.search.mask_select')
    check_variable_and_dtype(mask, 'mask', ['bool'],
                             'paddle.tensor.search.masked_select')
    out = helper.create_variable_for_type_inference(dtype=x.dtype)
    helper.append_op(
        type='masked_select', inputs={'X': x,
                                      'Mask': mask}, outputs={'Y': out})
    return out


def topk(x, k, axis=None, largest=True, sorted=True, name=None):
    """
    This OP is used to find values and indices of the k largest or smallest at the optional axis.
    If the input is a 1-D Tensor, finds the k largest or smallest values and indices.
    If the input is a Tensor with higher rank, this operator computes the top k values and indices along the :attr:`axis`.

    Args:
        x(Tensor): Tensor, an input N-D Tensor with type float32, float64, int32, int64.
        k(int, Tensor): The number of top elements to look for along the axis.
        axis(int, optional): Axis to compute indices along. The effective range
            is [-R, R), where R is x.ndim. when axis < 0, it works the same way
            as axis + R. Default is -1.
        largest(bool, optional) : largest is a flag, if set to true,
            algorithm will sort by descending order, otherwise sort by
            ascending order. Default is True.
        sorted(bool, optional): controls whether to return the elements in sorted order, default value is True. In gpu device, it always return the sorted value. 
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        tuple(Tensor), return the values and indices. The value data type is the same as the input `x`. The indices data type is int64.

    Examples:

        .. code-block:: python

           import paddle


           tensor_1 = paddle.to_tensor([1, 4, 5, 7])
           value_1, indices_1 = paddle.topk(tensor_1, k=1)
           print(value_1)
           # [7]
           print(indices_1)
           # [3] 
           tensor_2 = paddle.to_tensor([[1, 4, 5, 7], [2, 6, 2, 5]])
           value_2, indices_2 = paddle.topk(tensor_2, k=1)
           print(value_2)
           # [[7]
           #  [6]]
           print(indices_2)
           # [[3]
           #  [1]]
           value_3, indices_3 = paddle.topk(tensor_2, k=1, axis=-1)
           print(value_3)
           # [[7]
           #  [6]]
           print(indices_3)
           # [[3]
           #  [1]]
           value_4, indices_4 = paddle.topk(tensor_2, k=1, axis=0)
           print(value_4)
           # [[2 6 5 7]]
           print(indices_4)
           # [[1 1 0 0]]

    """
    if in_dygraph_mode():
        k = k.numpy().item(0) if isinstance(k, Variable) else k
        if axis is None:
            out, indices = core.ops.top_k_v2(x, 'k',
                                             int(k), 'largest', largest,
                                             'sorted', sorted)
        else:
            out, indices = core.ops.top_k_v2(x, 'k',
                                             int(k), 'axis', axis, 'largest',
                                             largest, 'sorted', sorted)
        return out, indices

    helper = LayerHelper("top_k_v2", **locals())
    inputs = {"X": [x]}
    attrs = {}
    if isinstance(k, Variable):
        inputs['K'] = [k]
    else:
        attrs = {'k': k}
    attrs['largest'] = largest
    attrs['sorted'] = sorted
    if axis is not None:
        attrs['axis'] = axis

    values = helper.create_variable_for_type_inference(dtype=x.dtype)
    indices = helper.create_variable_for_type_inference(dtype="int64")

    helper.append_op(
        type="top_k_v2",
        inputs=inputs,
        outputs={"Out": [values],
                 "Indices": [indices]},
        attrs=attrs)
    indices.stop_gradient = True
    return values, indices
