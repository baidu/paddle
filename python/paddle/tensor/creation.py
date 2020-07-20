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
from ..fluid.framework import Variable
from ..fluid.initializer import Constant
from ..fluid.layers import core
from ..fluid.layer_helper import LayerHelper
from ..fluid.data_feeder import check_variable_and_dtype, check_type, check_dtype, convert_dtype
from ..fluid.framework import convert_np_dtype_to_dtype_, in_dygraph_mode, _varbase_creator, device_guard, OpProtoHolder
from ..fluid.layers import fill_constant
from paddle.common_ops_import import *
import paddle

# TODO: define functions to get create a tensor  
from ..fluid.layers import crop_tensor  #DEFINE_ALIAS
from ..fluid.layers import diag  #DEFINE_ALIAS
from ..fluid.layers import eye  #DEFINE_ALIAS
from ..fluid.layers import fill_constant  #DEFINE_ALIAS
from ..fluid.layers import create_tensor  #DEFINE_ALIAS
from ..fluid.layers import linspace  #DEFINE_ALIAS

__all__ = [
    'create_tensor',
    #       'create_lod_tensor',
    #       'create_random_int_lodtensor',
    'crop_tensor',
    'diag',
    'eye',
    'fill_constant',
    #       'get_tensor_from_selected_rows',
    'linspace',
    'ones',
    'ones_like',
    'zeros',
    'zeros_like',
    'arange',
    'eye',
    'full',
    'full_like',
    'triu',
    'tril',
    'meshgrid'
]


def full_like(x, fill_value, dtype=None, name=None):
    """
	:alias_main: paddle.full_like
	:alias: paddle.full_like,paddle.tensor.full_like,paddle.tensor.creation.full_like

    **full_like**
    This function creates a tensor filled with `fill_value` which has identical shape and dtype 
    with `input`.

    Args:
        x(Variable): The input tensor which specifies shape and data type. The data type can be bool, float16, float32, float64, int32, int64.
        fill_value(bool|float|int|Variable): The value to fill the tensor with. Note: this value shouldn't exceed the range of the output data type.
        dtype(np.dtype|core.VarDesc.VarType|str, optional): The data type of output. The data type can be one
            of bool, float16, float32, float64, int32, int64. The default value is None, which means the output 
            data type is the same as input.
        name(str, optional): The default value is None. Normally there is no need for user to set this property. For more information, please refer to :ref:`api_guide_Name`
    
    Returns:
        out(Variable): The Tensor variable storing the output.
    
    Examples:
        .. code-block:: python

          import paddle
          import numpy as np
          
          paddle.enable_imperative()  # Now we are in imperative mode 
          input = paddle.full(shape=[2, 3], fill_value=0.0, dtype='float32', name='input')
          output = paddle.full_like(input, 2.0)
          #output result : [array([[2., 2., 2.], [2., 2., 2.]], dtype=float32)]
    """

    if dtype is None:
        dtype = x.dtype
    else:
        if not isinstance(dtype, core.VarDesc.VarType):
            dtype = convert_np_dtype_to_dtype_(dtype)

    if in_dygraph_mode():
        return core.ops.fill_any_like(x, 'value', fill_value, 'dtype', dtype)

    helper = LayerHelper("full_like", **locals())
    check_dtype(dtype, 'dtype',
                ['bool', 'float16', 'float32', 'float64', 'int32', 'int64'],
                'full_like/zeros_like')
    out = helper.create_variable_for_type_inference(dtype=dtype)

    helper.append_op(
        type='fill_any_like',
        inputs={'X': [x]},
        attrs={'value': fill_value,
               "dtype": dtype},
        outputs={'Out': [out]})
    out.stop_gradient = True
    return out


def ones(shape, dtype=None, out=None, device=None):
    """
	:alias_main: paddle.ones
	:alias: paddle.ones,paddle.tensor.ones,paddle.tensor.creation.ones

    The OP creates a tensor of specified :attr:`shape` and :attr:`dtype`, and fills it with 1.

    Args:
        shape(tuple|list): Shape of output tensor.
        dtype(np.dtype|core.VarDesc.VarType|str): Data type of output tensor, it supports
            bool, float16, float32, float64, int32 and int64.
        out(Variable, optional): Optional output which can be any created 
            Variable that meets the requirements to store the result of operation.
            if out is None, a new Varibale will be create to store the result.
        device(str, optional): Which device to run the operator. The :attr:`device` must be
            None,'cpu', 'gpu'. If :attr:`device` is None, it will be choose the device that the user set in 
            the paddle program. Default value is False.

    Returns:
        Variable: A tensor of data type :attr:`dtype` with shape :attr:`shape` and all elements set to 1.

    Examples:
        .. code-block:: python

          import paddle
          data = paddle.ones(shape=[3, 2], dtype='float32') # [[1., 1.], [1., 1.], [1., 1.]]
          data = paddle.ones(shape=[2, 2], dtype='float32', device='cpu') # [[1., 1.], [1., 1.]]
    """
    check_dtype(dtype, 'create data type',
                ['bool', 'float16', 'float32', 'float64', 'int32', 'int64'],
                'zeros')

    if device is not None:
        if device not in ['cpu', 'gpu']:
            raise ValueError(
                "The value of 'device' in zeros_op must be cpu or gpu, but received %s."
                % (device))
        with fluid.device_guard(device):
            return fill_constant(value=1.0, shape=shape, dtype=dtype, out=out)
    return fill_constant(value=1.0, shape=shape, dtype=dtype, out=out)


def ones_like(input, dtype=None, device=None, name=None):
    """
	:alias_main: paddle.ones_like
	:alias: paddle.ones_like,paddle.tensor.ones_like,paddle.tensor.creation.ones_like

    This function creates a ones tensor which has identical shape and dtype 
    with `input`.

    Args:
        input(Variable): The input tensor which specifies shape and dtype.The dtype of input can be 
            float32, float64, int32, int64.
        dtype(np.dtype|core.VarDesc.VarType|str, optional): The data type can be set bool, float32, float64, int32, int64. 
            The default value is None, the dtype is the same as input.
        device(str, optional): Which device to run the operator. The :attr:`device` must be
            None, 'cpu', 'gpu'. If :attr:`device` is None, it will be choose the device that the user set in 
            the paddle program. Default value is None.
        name(str, optional): The name of output variable, normally there is no need for user to set this this property. 
            Default value is None, the framework set the name of output variable.  
    Returns:
        out(Variable): The tensor variable storing the output.

    Examples:
        .. code-block:: python

          import paddle
          import paddle.fluid as fluid

          x = fluid.data(name='x', dtype='float32', shape=[3])
          data = paddle.ones_like(x) # data=[1.0, 1.0, 1.0]
          data1 = paddle.ones_like(input=x, device="gpu") data1=[1.0, 1.0. 1.0]

    """

    helper = LayerHelper("zeros_like", **locals())

    attrs = {"value": 1.0}
    var_dtype = None
    if dtype is not None:
        check_dtype(
            dtype, 'create data type',
            ['bool', 'float16', 'float32', 'float64', 'int32', 'int64'],
            'zeros_like')
        var_dtype = convert_np_dtype_to_dtype_(dtype)
        attrs["dtype"] = var_dtype
    else:
        var_dtype = input.dtype

    out = helper.create_variable_for_type_inference(dtype=var_dtype)

    if device is not None:
        if device not in ['cpu', 'gpu']:
            raise ValueError(
                "The value of 'device' in zeros_op must be cpu or gpu, but received %s."
                % (device))
        with fluid.device_guard(device):
            helper.append_op(
                type='fill_any_like',
                inputs={'X': [input]},
                attrs=attrs,
                outputs={'Out': [out]})
            return out
    helper.append_op(
        type='fill_any_like',
        inputs={'X': [input]},
        attrs=attrs,
        outputs={'Out': [out]})
    out.stop_gradient = True
    return out


def zeros(shape, dtype=None, name=None):
    """
	:alias_main: paddle.zeros
	:alias: paddle.zeros,paddle.tensor.zeros,paddle.tensor.creation.zeros

    The OP creates a tensor of specified :attr:`shape` and :attr:`dtype`, and fills it with 0.

    Args:
        shape(tuple|list): Shape of output tensor.
        dtype(np.dtype|core.VarDesc.VarType|str, optional): Data type of output tensor, it supports
            bool, float16, float32, float64, int32 and int64. Default: if None, the date type is float32.
        name(str, optional): The default value is None.  Normally there is no need for user to set this
            property.  For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Variable: A tensor of data type :attr:`dtype` with shape :attr:`shape` and all elements set to 0.

    Examples:
        .. code-block:: python

          import paddle
          
          paddle.enable_imperative()  # Now we are in imperative mode
          data = paddle.zeros(shape=[3, 2], dtype='float32') # [[0., 0.], [0., 0.], [0., 0.]]
          data = paddle.zeros(shape=[2, 2], dtype='int32', name='zeros') # [[0, 0], [0, 0]]
    """
    if dtype is None:
        dtype = 'float32'
    return fill_constant(value=0.0, shape=shape, dtype=dtype, name=name)


def zeros_like(x, dtype=None, name=None):
    """
	:alias_main: paddle.zeros_like
	:alias: paddle.zeros_like, paddle.tensor.zeros_like, paddle.tensor.creation.zeros_like

    This function creates a zeros tensor which has identical shape and dtype 
    with `input`.

    Args:
        x(Variable): The input tensor which specifies shape and dtype. The
            dtype of input can be bool, float16, float32, float64, int32, int64.
        dtype(np.dtype|core.VarDesc.VarType|str, optional): The data type can
            be set bool, float16, float32, float64, int32, int64. The default
            value is None, the dtype is the same as input.
        name(str, optional): The default value is None. Normally there is no
            need for user to set this property. For more information, please
            refer to :ref:`api_guide_Name`.

    Returns:
        out(Variable): The tensor variable storing the output.

    Raise:
        TypeError: If dtype is not bool, float16, float32, float64, int32 or int64.

    Examples:
        .. code-block:: python

        import paddle
        import numpy as np

        paddle.enable_imperative()

        x = paddle.imperative.to_variable(np.array([1,2,3], dtype='float32'))
        out1 = paddle.zeros_like(x) # [1.0, 1.0, 1.0]
        out2 = paddle.zeros_like(x, dtype='int32') # [1, 1, 1]

    """
    return full_like(x=x, fill_value=0, dtype=dtype, name=name)


def eye(num_rows,
        num_columns=None,
        out=None,
        dtype='float32',
        stop_gradient=True,
        name=None):
    """
    **eye**
    This function constructs an identity tensor.

    Args:
        num_rows(int): the number of rows in each batch tensor.
        num_columns(int, optional): the number of columns in each batch tensor.
                          If None, default: num_rows.
        out(Variable, optional): Optional output which can be any created 
            Variable that meets the requirements to store the result of operation.
            if out is None, a new Varibale will be create to store the result.
        dtype(string, optional): The data type of the returned tensor.
                       It should be int32, int64, float16, float32, float64.
        stop_gradient(bool, optional): Whether stop calculating gradients. Default:True.
        name(str, optional): The default value is None.  Normally there is no need for 
            user to set this property.  For more information, please refer to :ref:`api_guide_Name`

    Returns:
        Variable: An identity Tensor or LoDTensor of shape [num_rows, num_columns].

    Examples:
        .. code-block:: python
          import paddle
          data = paddle.eye(3, dtype='int32')
          # [[1, 0, 0]
          #  [0, 1, 0]
          #  [0, 0, 1]]
          data = paddle.eye(2, 3, dtype='int32')
          # [[1, 0, 0]
          #  [0, 1, 0]]
    """

    helper = LayerHelper("eye", **locals())
    if not isinstance(num_rows, int) or num_rows < 0:
        raise TypeError("num_rows should be a non-negative int")
    if num_columns is not None:
        if not isinstance(num_columns, int) or num_columns < 0:
            raise TypeError("num_columns should be a non-negative int")
    else:
        num_columns = num_rows
    if out is None:
        out = helper.create_variable_for_type_inference(dtype=dtype)
    c_dtype = convert_np_dtype_to_dtype_(dtype)
    helper.append_op(
        type='eye',
        inputs={},
        outputs={'Out': [out]},
        attrs={
            'num_rows': num_rows,
            'num_columns': num_columns,
            'dtype': c_dtype
        },
        stop_gradient=True)
    out.stop_gradient = stop_gradient
    return out


def full(shape, fill_value, dtype=None, name=None):
    """
	:alias_main: paddle.full
	:alias: paddle.full,paddle.tensor.full,paddle.tensor.creation.full

    This Op return a Tensor with the `fill_value` which size is same as `shape`
    
    Args:
        shape(list|tuple|Variable): Shape of the Tensor to be created.
                The data type is ``int32`` or ``int64`` . If ``shape`` is a list or tuple,
                the elements of it should be integers or Tensors with shape [1].
                If ``shape`` is an Variable, it should be an 1-D Tensor .
        fill_value(bool|float16|float32|float64|int32|int64|Variable): The constant value
            used to initialize the Tensor to be created. If fill_value is an Variable, it must be an 1-D Tensor.
        dtype(np.dtype|core.VarDesc.VarType|str, optional): Data type of the output tensor
            which can be float16, float32, float64, int32, int64, if dytpe is `None`, the data
            type of created tensor is `float32`
        name(str, optional): The default value is None.  Normally there is no need for user to set this
            property.  For more information, please refer to :ref:`api_guide_Name`.
    
    Returns:
        Variable: Tensor which is created according to shape and dtype.

    Raises:
        TypeError: The `dtype` must be one of None, bool, float16, float32, float64, int32 and int64.
        TypeError: The `shape` must be one of Variable, list tuple.
    
    Examples:
        .. code-block:: python

          import paddle

          paddle.enable_imperative()  # Now we are in imperative mode
          data1 = paddle.full(shape=[2,1], fill_value=0, dtype='int64') # data1=[[0],[0]]

          # attr shape is a list which contains Variable Tensor.
          positive_2 = paddle.fill_constant([1], "int32", 2)
          data3 = paddle.full(shape=[1, positive_2], dtype='float32', fill_value=1.5) # data3=[1.5, 1.5]

          # attr shape is an Variable Tensor.
          shape = paddle.fill_constant([2], "int32", 2) # shape=[2,2]
          data4 = paddle.full(shape=shape, dtype='bool', fill_value=True) # data4=[[True,True],[True,True]]
          
          # attr value is an Variable Tensor.
          val = paddle.fill_constant([1], "float32", 2.0) # val=[2.0]
          data5 = paddle.full(shape=[2,1], fill_value=val, dtype='float32') #data5=[[2.0],[2.0]]
    """

    helper = LayerHelper("full", **locals())

    if dtype is None:
        dtype = 'float32'

    return fill_constant(shape=shape, dtype=dtype, value=fill_value, name=name)


def arange(start=0, end=None, step=1, dtype=None, name=None):
    """
	:alias_main: paddle.arange
	:alias: paddle.arange,paddle.tensor.arange,paddle.tensor.creation.arange

    Return evenly spaced values within a given interval.

    Values are generated into the half-open interval [start, stop) with the step.
    (the interval including start but excluding stop).

    If dtype is float32 or float64, we advise adding a small epsilon to end to
    avoid floating point rounding errors when comparing against end.

    Parameters:
        start(float|int|Variable): Start of interval. The interval includes
            this value. If end is None, the half-open interval is [0, start).
            If start is Variable, it is a 1-D Tensor with shape [1], and it's
            data type should be one of int32, int64, float32, float64. Default
            is 0.
        end(float|int|Variable, optional): End of interval. The interval does
            not include this value. When end is Variable, it is a 1-D Tensor
            with shape [1], and it's data type should be one of int32, int64,
            float32, float64. If end is None, the half-open interval is [0, start).
            Default is None.
        step(float|int|Variable, optional): Spacing between values. For any
            out, this is the istance between two adjacent values, out[i+1] - out[i].
            When end is Variable, it is a 1-D Tensor with shape [1], and it's
            data type should be one of int32, int64, float32, float64. Default is 1.
        dtype(str|np.dtype|core.VarDesc.VarType, optional): The data type of
            the output tensor, can be float32, float64, int32, int64. If dtype
            is `None` , the data type of out tensor is `int64` . Defaule is None
        name(str, optional): Normally there is no need for user to set this property.
            For more information, please refer to :ref:`api_guide_Name` .
            Default is None.

    Returns: a 1-D Tensor which is evenly spaced values within a given interval.
        Its data type is set by dtype.
    
    Return type: Variable

    Raises:
        TypeError: If dtype is not float32, float64, int32 or int64.

    examples:

        .. code-block:: python

        import paddle
        import numpy as np

        paddle.enable_imperative()

        out1 = paddle.arange(5)
        # [0, 1, 2, 3, 4]

        out2 = paddle.arange(3, 9, 2.0)
        # [3, 5, 7]

        # use 4.999 instead of 5.0 to avoid floating point rounding errors
        out3 = paddle.arange(4.999, dtype='float32')
        # [0., 1., 2., 3., 4.]

        start_var = paddle.imperative.to_variable(np.array([3]))
        out4 = paddle.arange(start_var, 7)
        # [3, 4, 5, 6]
             
    """
    if dtype is None:
        dtype = 'int64'
    if end is None:
        end = start
        start = 0

    return paddle.fluid.layers.range(start, end, step, dtype, name)


def _tril_triu_op(helper):
    """Base op of tril_op and triu_op
    """
    op_type = helper.layer_type
    x = helper.kwargs.get('x', None)

    assert x is not None, 'x cannot be None in {}'.format(op_type)
    check_variable_and_dtype(x, 'x', ['float32', 'float64', 'int32', 'int64'],
                             op_type)
    if len(x.shape) < 2:
        raise ValueError("x shape in {} must be at least 2-D".format(op_type))
    diagonal = helper.kwargs.get('diagonal', 0)
    if not isinstance(diagonal, (int, )):
        raise TypeError("diagonal in {} must be a python Int".format(op_type))
    name = helper.kwargs.get('name', None)

    if name is None:
        out = helper.create_variable_for_type_inference(dtype=x.dtype)
    else:
        out = helper.create_variable(
            name=name, dtype=x.dtype, persistable=False)

    helper.append_op(
        type="tril_triu",
        inputs={"X": x},
        attrs={
            "diagonal": diagonal,
            "lower": True if op_type == 'tril' else False,
        },
        outputs={"Out": out}, )

    return out


def tril(x, diagonal=0, name=None):
    """
	:alias_main: paddle.tril
	:alias: paddle.tril,paddle.tensor.tril,paddle.tensor.creation.tril

    This op returns the lower triangular part of a matrix (2-D tensor) or batch
    of matrices :attr:`x`, the other elements of the result tensor are set 
    to 0. The lower triangular part of the matrix is defined as the elements 
    on and below the diagonal.

    Args:
        x (Variable): The input variable x which is a Tensor.
            Support data types: ``float64``, ``float32``, ``int32``, ``int64``.
        diagonal (int, optional): The diagonal to consider, default value is 0.
            If :attr:`diagonal` = 0, all elements on and below the main diagonal are
            retained. A positive value includes just as many diagonals above the main
            diagonal, and similarly a negative value excludes just as many diagonals below
            the main diagonal. The main diagonal are the set of indices
            :math:`\{(i, i)\}` for :math:`i \in [0, \min\{d_{1}, d_{2}\} - 1]` where
            :math:`d_{1}, d_{2}` are the dimensions of the matrix.
        name (str, optional): The default value is None. Normally there is no need for
            user to set this property. For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Variable: Tensor, results of lower triangular operation by the specified diagonal of input tensor x,
        it's data type is the same as x's Tensor.

    Raises:
        TypeError: diagonal is not a int type.
        ValueError: dimension of :attr:`x` is less than 2.

    Examples:
        .. code-block:: python

            import numpy as np
            import paddle

            data = np.arange(1, 13, dtype="int64").reshape(3,-1)
            # array([[ 1,  2,  3,  4],
            #        [ 5,  6,  7,  8],
            #        [ 9, 10, 11, 12]])

            paddle.enable_imperative()

            x = paddle.imperative.to_variable(data)
            
            tril1 = tensor.tril(x)
            # array([[ 1,  0,  0,  0],
            #        [ 5,  6,  0,  0],
            #        [ 9, 10, 11,  0]])

            # example 2, positive diagonal value
            tril2 = tensor.tril(x, diagonal=2)
            # array([[ 1,  2,  3,  0], 
            #        [ 5,  6,  7,  8],
            #        [ 9, 10, 11, 12]])

            # example 3, negative diagonal value
            tril3 = tensor.tril(x, diagonal=-1)
            # array([[ 0,  0,  0,  0],
            #        [ 5,  0,  0,  0],
            #        [ 9, 10,  0,  0]])

    """
    if in_dygraph_mode():
        op = getattr(core.ops, 'tril_triu')
        return op(x, 'diagonal', diagonal, "lower", True)

    return _tril_triu_op(LayerHelper('tril', **locals()))


def triu(x, diagonal=0, name=None):
    """
	:alias_main: paddle.triu
	:alias: paddle.triu,paddle.tensor.triu,paddle.tensor.creation.triu

    This op returns the upper triangular part of a matrix (2-D tensor) or batch of matrices
    :attr:`x`, the other elements of the result tensor are set to 0.
    The upper triangular part of the matrix is defined as the elements on and
    above the diagonal.

    Args:
        x (Variable): The input variable x which is a Tensor.
            Support data types: ``float64``, ``float32``, ``int32``, ``int64``.
        diagonal (int, optional): The diagonal to consider, default value is 0.
            If :attr:`diagonal` = 0, all elements on and above the main diagonal are
            retained. A positive value excludes just as many diagonals above the main
            diagonal, and similarly a negative value includes just as many diagonals below
            the main diagonal. The main diagonal are the set of indices
            :math:`\{(i, i)\}` for :math:`i \in [0, \min\{d_{1}, d_{2}\} - 1]` where
            :math:`d_{1}, d_{2}` are the dimensions of the matrix.
        name (str, optional): The default value is None. Normally there is no need for
            user to set this property. For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Variable: Tensor, results of upper triangular operation by the specified diagonal of input tensor x,
        it's data type is the same as x's Tensor.

    Raises:
        TypeError: diagonal is not a int type.
        ValueError: dimension of :attr:`x` is less than 2.

    Examples:
        .. code-block:: python

            import numpy as np
            import paddle

            data = np.arange(1, 13, dtype="int64").reshape(3,-1)
            # array([[ 1,  2,  3,  4],
            #        [ 5,  6,  7,  8],
            #        [ 9, 10, 11, 12]])

            paddle.enable_imperative()

            # example 1, default diagonal
            x = paddle.imperative.to_variable(data)
            triu1 = paddle.tensor.triu(x)
            # array([[ 1,  2,  3,  4],
            #        [ 0,  6,  7,  8],
            #        [ 0,  0, 11, 12]])

            # example 2, positive diagonal value
            triu2 = tensor.triu(x, diagonal=2)
            # array([[0, 0, 3, 4],
            #        [0, 0, 0, 8],
            #        [0, 0, 0, 0]])

            # example 3, negative diagonal value
            triu3 = tensor.triu(x, diagonal=-1)
            # array([[ 1,  2,  3,  4],
            #        [ 5,  6,  7,  8],
            #        [ 0, 10, 11, 12]])

    """
    if in_dygraph_mode():
        op = getattr(core.ops, 'tril_triu')
        return op(x, 'diagonal', diagonal, "lower", False)

    return _tril_triu_op(LayerHelper('triu', **locals()))


def meshgrid(*args, **kwargs):
    """
	:alias_main: paddle.meshgrid
	:alias: paddle.meshgrid,paddle.tensor.meshgrid,paddle.tensor.creation.meshgrid

    This op takes a list of N tensors as input *args, each of which is 1-dimensional 
    vector, and creates N-dimensional grids.
    
    Args:
        *args(Variable|list of Variable) : tensors (tuple(list) of tensor): the shapes of input k tensors are (N1,), 
            (N2,),..., (Nk,). Support data types: ``float64``, ``float32``, ``int32``, ``int64``.
        **kwargs (optional): Currently, we only accept name in **kwargs 
            The default value is None. Normally there is no need for
            user to set this property. For more information, please refer to :ref:`api_guide_Name`.
 
    Returns:
         Variable: k tensors. The shape of each tensor is (N1, N2, ..., Nk)

    Examples:
      .. code-block:: python

          import paddle
          import paddle.fluid as fluid
          import numpy as np

          x = fluid.data(name='x', shape=[100], dtype='int32')
          y = fluid.data(name='y', shape=[200], dtype='int32')

          input_1 = np.random.randint(0, 100, [100, ]).astype('int32')
          input_2 = np.random.randint(0, 100, [200, ]).astype('int32')

          exe = fluid.Executor(place=fluid.CPUPlace())
          grid_x, grid_y = paddle.tensor.meshgrid(x, y)
          res_1, res_2 = exe.run(fluid.default_main_program(),
                                 feed={'x': input_1,
                                       'y': input_2},
                                 fetch_list=[grid_x, grid_y])
     
          #the shape of res_1 is (100, 200)
          #the shape of res_2 is (100, 200)

      .. code-block:: python

          #example 2: in dygraph mode

          import paddle
          import numpy as np
          
          paddle.enable_imperative()

          input_3 = np.random.randint(0, 100, [100, ]).astype('int32')
          input_4 = np.random.randint(0, 100, [200, ]).astype('int32')
          tensor_3 = paddle.imperative.to_variable(input_3)
          tensor_4 = paddle.imperative.to_variable(input_4)
          grid_x, grid_y = paddle.tensor.meshgrid(tensor_3, tensor_4)

          #the shape of grid_x is (100, 200)
          #the shape of grid_y is (100, 200)

    """

    if len(args) == 1 and isinstance(args[0], (list, tuple)):
        args = args[0]
    if in_dygraph_mode():
        num = len(args)
        out = core.ops.meshgrid(list(args), num)
        return out

    name = kwargs.get("name", None)
    helper = LayerHelper('meshgrid', **locals())

    if not isinstance(args, (list, tuple)):
        raise TypeError("The type of input args in meshgrid should be list.")

    for id, input_ in enumerate(args):
        check_dtype(input_.dtype, 'create data type',
                    ['float16', 'float32', 'float64', 'int32', 'int64'],
                    'meshgrid')

    num = len(args)
    out = [
        helper.create_variable_for_type_inference(dtype=args[i].dtype)
        for i in range(num)
    ]
    helper.append_op(
        type='meshgrid', inputs={'X': list(args)}, outputs={'Out': out})

    return out
