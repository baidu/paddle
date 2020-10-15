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

import numpy as np
from paddle.common_ops_import import *
from ..fluid.layer_helper import LayerHelper
from ..fluid.data_feeder import check_variable_and_dtype, check_type
from ..fluid.framework import in_dygraph_mode, _varbase_creator

from ..fluid.layers import transpose  #DEFINE_ALIAS

__all__ = [
    'matmul',
    'dot',
    #       'einsum',
    'norm',
    'transpose',
    'dist',
    't',
    'cross',
    'cholesky',
    #       'tensordot',
    'bmm',
    'histogram',
    'mv'
]


def matmul(x, y, transpose_x=False, transpose_y=False, name=None):
    """
    Applies matrix multiplication to two tensors. `matmul` follows 
    the complete broadcast rules, 
    and its behavior is consistent with `np.matmul`.

    Currently, the input tensors' number of dimensions can be any, `matmul` can be used to
    achieve the `dot`, `matmul` and `batchmatmul`.

    The actual behavior depends on the shapes of :math:`x`, :math:`y` and the
    flag values of :attr:`transpose_x`, :attr:`transpose_y`. Specifically:

    - If a transpose flag is specified, the last two dimensions of the tensor
      are transposed. If the tensor is ndim-1 of shape, the transpose is invalid. If the tensor 
      is ndim-1 of shape :math:`[D]`, then for :math:`x` it is treated as :math:`[1, D]`, whereas 
      for :math:`y` it is the opposite: It is treated as :math:`[D, 1]`.

    The multiplication behavior depends on the dimensions of `x` and `y`. Specifically:

    - If both tensors are 1-dimensional, the dot product result is obtained.

    - If both tensors are 2-dimensional, the matrix-matrix product is obtained.

    - If the `x` is 1-dimensional and the `y` is 2-dimensional, 
      a `1` is prepended to its dimension in order to conduct the matrix multiply. 
      After the matrix multiply, the prepended dimension is removed.
      
    - If the `x` is 2-dimensional and `y` is 1-dimensional, 
      the matrix-vector product is obtained.

    - If both arguments are at least 1-dimensional and at least one argument 
      is N-dimensional (where N > 2), then a batched matrix multiply is obtained. 
      If the first argument is 1-dimensional, a 1 is prepended to its dimension 
      in order to conduct the batched matrix multiply and removed after. 
      If the second argument is 1-dimensional, a 1 is appended to its 
      dimension for the purpose of the batched matrix multiple and removed after. 
      The non-matrix (exclude the last two dimensions) dimensions are 
      broadcasted according the broadcast rule. 
      For example, if input is a (j, 1, n, m) tensor and the other is a (k, m, p) tensor, 
      out will be a (j, k, n, p) tensor.

    Args:
        x (Tensor): The input tensor which is a Tensor.
        y (Tensor): The input tensor which is a Tensor.
        transpose_x (bool): Whether to transpose :math:`x` before multiplication.
        transpose_y (bool): Whether to transpose :math:`y` before multiplication.
        name(str|None): A name for this layer(optional). If set None, the layer
            will be named automatically.

    Returns:
        Tensor: The output Tensor.

    Examples:

    .. code-block:: python

        import paddle
        import numpy as np

        paddle.disable_static()
        # vector * vector
        x_data = np.random.random([10]).astype(np.float32)
        y_data = np.random.random([10]).astype(np.float32)
        x = paddle.to_tensor(x_data)
        y = paddle.to_tensor(y_data)
        z = paddle.matmul(x, y)
        print(z.numpy().shape)
        # [1]

        # matrix * vector
        x_data = np.random.random([10, 5]).astype(np.float32)
        y_data = np.random.random([5]).astype(np.float32)
        x = paddle.to_tensor(x_data)
        y = paddle.to_tensor(y_data)
        z = paddle.matmul(x, y)
        print(z.numpy().shape)
        # [10]

        # batched matrix * broadcasted vector
        x_data = np.random.random([10, 5, 2]).astype(np.float32)
        y_data = np.random.random([2]).astype(np.float32)
        x = paddle.to_tensor(x_data)
        y = paddle.to_tensor(y_data)
        z = paddle.matmul(x, y)
        print(z.numpy().shape)
        # [10, 5]

        # batched matrix * batched matrix
        x_data = np.random.random([10, 5, 2]).astype(np.float32)
        y_data = np.random.random([10, 2, 5]).astype(np.float32)
        x = paddle.to_tensor(x_data)
        y = paddle.to_tensor(y_data)
        z = paddle.matmul(x, y)
        print(z.numpy().shape)
        # [10, 5, 5]

        # batched matrix * broadcasted matrix
        x_data = np.random.random([10, 1, 5, 2]).astype(np.float32)
        y_data = np.random.random([1, 3, 2, 5]).astype(np.float32)
        x = paddle.to_tensor(x_data)
        y = paddle.to_tensor(y_data)
        z = paddle.matmul(x, y)
        print(z.numpy().shape)
        # [10, 3, 5, 5]

    """
    op_type = 'matmul_v2'
    if in_dygraph_mode():
        op = getattr(core.ops, op_type)
        return op(x, y, 'trans_x', transpose_x, 'trans_y', transpose_y)

    attrs = {
        'trans_x': transpose_x,
        'trans_y': transpose_y,
    }

    def __check_input(x, y):
        var_names = {'x': x, 'y': y}
        for name, val in var_names.items():
            check_variable_and_dtype(
                val, name, ['float16', 'float32', 'float64'], 'matmul')

    __check_input(x, y)

    helper = LayerHelper('matmul_v2', **locals())
    out = helper.create_variable_for_type_inference(dtype=x.dtype)
    helper.append_op(
        type='matmul_v2',
        inputs={'X': x,
                'Y': y},
        outputs={'Out': out},
        attrs=attrs)
    return out


def norm(x, p='fro', axis=None, keepdim=False, name=None):
    """
	:alias_main: paddle.norm
	:alias: paddle.norm,paddle.tensor.norm,paddle.tensor.linalg.norm

    Returns the matrix norm (Frobenius) or vector norm (the 1-norm, the Euclidean
    or 2-norm, and in general the p-norm for p > 0) of a given tensor.

    Args:
        x (Tensor): The input tensor could be N-D tensor, and the input data
            type could be float32 or float64.
        p (float|string, optional): Order of the norm. Supported values are `fro`, `0`, `1`, `2`,
            `inf`, `-inf` and any positive real number yielding the corresponding p-norm. Not supported: ord < 0 and nuclear norm. 
            Default value is `fro`.
        axis (int|list|tuple, optional): The axis on which to apply norm operation. If axis is int
            or list(int)/tuple(int)  with only one element, the vector norm is computed over the axis.
            If `axis < 0`, the dimension to norm operation is rank(input) + axis.
            If axis is a list(int)/tuple(int) with two elements, the matrix norm is computed over the axis.
            Defalut value is `None`.
        keepdim (bool, optional): Whether to reserve the reduced dimension in the
            output Tensor. The result tensor will have fewer dimension
            than the :attr:`input` unless :attr:`keepdim` is true, default
            value is False.
        name (str, optional): The default value is None. Normally there is no need for
            user to set this property. For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor: results of norm operation on the specified axis of input tensor,
        it's data type is the same as input's Tensor.
 
    Examples:
        .. code-block:: python
            
            import paddle
            import numpy as np
            paddle.disable_static()
            shape=[2, 3, 4]
            np_input = np.arange(24).astype('float32') - 12
            np_input = np_input.reshape(shape)
            x = paddle.to_tensor(np_input)
            #[[[-12. -11. -10.  -9.] [ -8.  -7.  -6.  -5.] [ -4.  -3.  -2.  -1.]]
            # [[  0.   1.   2.   3.] [  4.   5.   6.   7.] [  8.   9.  10.  11.]]]

            # compute frobenius norm along last two dimensions.
            out_fro = paddle.norm(x, p='fro', axis=[0,1])
            # out_fro.numpy() [17.435596 16.911535 16.7332   16.911535]

            # compute 2-order vector norm along last dimension.
            out_pnorm = paddle.norm(x, p=2, axis=-1)
            #out_pnorm.numpy(): [[21.118711  13.190906   5.477226]
            #                    [ 3.7416575 11.224972  19.131126]]

            # compute 2-order  norm along [0,1] dimension.
            out_pnorm = paddle.norm(x, p=2, axis=[0,1])
            #out_pnorm.numpy(): [17.435596 16.911535 16.7332   16.911535]

            # compute inf-order  norm
            out_pnorm = paddle.norm(x, p=np.inf)
            #out_pnorm.numpy()  = [12.]
            out_pnorm = paddle.norm(x, p=np.inf, axis=0)
            #out_pnorm.numpy(): [[12. 11. 10. 9.] [8. 7. 6. 7.] [8. 9. 10. 11.]]

            # compute -inf-order  norm
            out_pnorm = paddle.norm(x, p=-np.inf)
            #out_pnorm.numpy(): [0.]
            out_pnorm = paddle.norm(x, p=-np.inf, axis=0)
            #out_pnorm.numpy(): [[0. 1. 2. 3.] [4. 5. 6. 5.] [4. 3. 2. 1.]]
    """

    def frobenius_norm(input, dim=None, keepdim=False, name=None):
        """
        The frobenius norm OP is to calculate the frobenius norm of certain two dimensions of Tensor `input`.
        Args:
          input (Variable): Tensor, data type float32, float64.
          dim (list, optional): None for last two dimensions.
          keepdim (bool, optional): Whether keep the dimensions as the `input`, Default False.
        """
        if dim is not None and not (isinstance(dim, list) and len(dim) == 2):
            raise ValueError(
                "The dim of frobenius norm op should be None or two elements list!"
            )
        if in_dygraph_mode():
            if dim is None:
                return core.ops.frobenius_norm(input, 'keep_dim', keepdim,
                                               'reduce_all', True)
            return core.ops.frobenius_norm(input, 'dim', dim, 'keep_dim',
                                           keepdim, 'reduce_all', False)
        attrs = {'dim': dim, 'keep_dim': keepdim, 'reduce_all': False}
        if dim is None:
            attrs['reduce_all'] = True
        check_variable_and_dtype(input, 'input', ['float32', 'float64'],
                                 'frobenius_norm')

        helper = LayerHelper('frobenius_norm', **locals())
        out = helper.create_variable_for_type_inference(
            dtype=helper.input_dtype())

        helper.append_op(
            type='frobenius_norm',
            inputs={'X': input},
            outputs={'Out': out},
            attrs=attrs)
        return out

    def vector_norm(input,
                    porder=None,
                    axis=None,
                    keepdim=False,
                    asvector=False,
                    name=None):
        """
        Calculate the p-order vector norm for certain  dimension of Tensor `input`.
        Args:
          input (Variable): Tensor, data type float32, float64.
          porder (float, optional): None for porder=2.0.
          axis (int, optional): None for last dimension.
          keepdim (bool, optional): Whether keep the dimensions as the `input`, Default False.
        """
        if in_dygraph_mode():
            if axis is None: axis = -1
            return core.ops.p_norm(input, 'porder', porder, 'axis', axis,
                                   'keepdim', keepdim, 'asvector', asvector)
        if porder is not None:
            check_type(porder, 'porder', (float, int), 'p_norm')
        if axis is not None:
            check_type(axis, 'axis', (int), 'p_norm')
        check_variable_and_dtype(input, 'input', ['float32', 'float64'],
                                 'p_norm')

        attrs = {
            'axis': axis if axis is not None else -1,
            'porder': float(porder) if porder is not None else 2.0,
            'keepdim': keepdim,
            'asvector': asvector,
            'epsilon': 1e-12,
        }
        helper = LayerHelper('p_norm', **locals())
        out = helper.create_variable_for_type_inference(
            dtype=helper.input_dtype())

        helper.append_op(
            type='p_norm',
            inputs={'X': input},
            outputs={'Out': out},
            attrs=attrs)
        return out

    def inf_norm(input,
                 porder=None,
                 axis=axis,
                 keepdim=False,
                 asvector=False,
                 name=None):
        helper = LayerHelper('frobenius_norm', **locals())
        out = helper.create_variable_for_type_inference(
            dtype=helper.input_dtype())
        helper.append_op(type='abs', inputs={'X': input}, outputs={'Out': out})
        reduce_out = helper.create_variable_for_type_inference(
            dtype=helper.input_dtype())

        reduce_all = True if axis == None or axis == [] or asvector == True else False
        axis = axis if axis != None and axis != [] else [0]

        reduce_type = 'reduce_max' if porder == np.float(
            'inf') else 'reduce_min'
        helper.append_op(
            type=reduce_type,
            inputs={'X': out},
            outputs={'Out': reduce_out},
            attrs={'dim': axis,
                   'keep_dim': keepdim,
                   'reduce_all': reduce_all})

        return reduce_out

    def p_matrix_norm(input, porder=1., axis=axis, keepdim=False, name=None):
        block = LayerHelper('norm', **locals())
        out = block.create_variable_for_type_inference(
            dtype=block.input_dtype())
        abs_out = block.create_variable_for_type_inference(
            dtype=block.input_dtype())
        block.append_op(
            type='abs', inputs={'X': input}, outputs={'Out': abs_out})
        pow_out = block.create_variable_for_type_inference(
            dtype=block.input_dtype())

        block.append_op(
            type='pow',
            inputs={'X': abs_out},
            outputs={'Out': pow_out},
            attrs={'factor': porder})
        sum_out = block.create_variable_for_type_inference(
            dtype=block.input_dtype())
        block.append_op(
            type='reduce_sum',
            inputs={'X': pow_out},
            outputs={'Out': sum_out},
            attrs={
                'dim': axis,
                'keep_dim': keepdim,
                'reduce_all': True if axis is None else False
            })
        porder
        block.append_op(
            type='pow',
            inputs={'X': sum_out},
            outputs={'Out': out},
            attrs={'factor': float(1. / porder)})
        return out

    if axis is None and p is not None:
        if isinstance(p, str):
            if p == "fro":
                return frobenius_norm(x, dim=axis, keepdim=keepdim, name=name)
            else:
                raise ValueError(
                    "only valid string values are 'fro', found {}".format(p))
        elif isinstance(p, (int, float)):
            return vector_norm(
                x,
                porder=p,
                axis=axis,
                keepdim=keepdim,
                asvector=True,
                name=name)
        else:
            raise ValueError("only valid p type is string or float, found {}".
                             format(type(p)))

    if isinstance(axis, tuple):
        axis = list(axis)
    if isinstance(axis, list) and len(axis) == 1:
        axis = axis[0]

    #calculate vector norm, where axis is int or list with only one integer
    if isinstance(axis, int):
        if isinstance(p, str):
            if p == "fro":
                return vector_norm(
                    x,
                    porder=2,
                    axis=axis,
                    keepdim=keepdim,
                    asvector=False,
                    name=name)

            else:
                raise ValueError(
                    "only valid string values are 'fro', found {}".format(p))
        elif isinstance(p, (int, float)):
            return vector_norm(
                x,
                axis=axis,
                porder=p,
                keepdim=keepdim,
                asvector=False,
                name=name)
        else:
            raise ValueError(
                "unspport p for p-order vector norm. except float, found {}".
                format(p))
    #calculate matrix norm, where axis is list with two integers
    elif isinstance(axis, list) and len(axis) == 2:
        if p == "fro":
            return frobenius_norm(x, dim=axis, keepdim=keepdim, name=name)
        elif p == np.inf or p == -np.inf:
            return inf_norm(x, porder=p, axis=axis, keepdim=keepdim, name=name)
        elif p == 0:
            raise ValueError(
                "just suport axis type int or list (length of list <=1) if p = 0, found {}".
                format(axis))
        else:
            return p_matrix_norm(
                x, porder=p, axis=axis, keepdim=keepdim, name=name)
    else:
        raise ValueError(
            "except axis type int or list (length of list <=2), found {}".
            format(axis))


def dist(x, y, p=2):
    """
	:alias_main: paddle.dist
	:alias: paddle.dist,paddle.tensor.dist,paddle.tensor.linalg.dist

    This OP returns the p-norm of (x - y). It is not a norm in a strict sense, only as a measure
    of distance. The shapes of x and y must be broadcastable. The definition is as follows, for
    details, please refer to the `numpy's broadcasting <https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html>`_:

    - Each input has at least one dimension.
    - Match the two input dimensions from back to front, the dimension sizes must either be equal, one of them is 1, or one of them does not exist.

    Where, z = x - y, the shapes of x and y are broadcastable, then the shape of z can be
    obtained as follows:

    1. If the number of dimensions of x and y are not equal, prepend 1 to the dimensions of the
    tensor with fewer dimensions.

    For example, The shape of x is [8, 1, 6, 1], the shape of y is [7, 1, 5], prepend 1 to the
    dimension of y.

    x (4-D Tensor):  8 x 1 x 6 x 1

    y (4-D Tensor):  1 x 7 x 1 x 5

    2. Determine the size of each dimension of the output z: choose the maximum value from the
    two input dimensions.

    z (4-D Tensor):  8 x 7 x 6 x 5

    If the number of dimensions of the two inputs are the same, the size of the output can be
    directly determined in step 2. When p takes different values, the norm formula is as follows:

    When p = 0, defining $0^0=0$, the zero-norm of z is simply the number of non-zero elements of z.

    .. math::

        ||z||_{0}=\lim_{p \\rightarrow 0}\sum_{i=1}^{m}|z_i|^{p}

    When p = inf, the inf-norm of z is the maximum element of z.

    .. math::

        ||z||_\infty=\max_i |z_i|

    When p = -inf, the negative-inf-norm of z is the minimum element of z.

    .. math::

        ||z||_{-\infty}=\min_i |z_i|

    Otherwise, the p-norm of z follows the formula,

    .. math::

        ||z||_{p}=(\sum_{i=1}^{m}|z_i|^p)^{\\frac{1}{p}}

    Args:
        x (Variable): 1-D to 6-D Tensor, its data type is float32 or float64.
        y (Variable): 1-D to 6-D Tensor, its data type is float32 or float64.
        p (float, optional): The norm to be computed, its data type is float32 or float64. Default: 2.

    Returns:
        Variable: Tensor that is the p-norm of (x - y).

    Examples:
        .. code-block:: python

            import paddle
            import paddle.fluid as fluid
            import numpy as np

            with fluid.dygraph.guard():
                x = fluid.dygraph.to_variable(np.array([[3, 3],[3, 3]]).astype(np.float32))
                y = fluid.dygraph.to_variable(np.array([[3, 3],[3, 1]]).astype(np.float32))
                out = paddle.dist(x, y, 0)
                print(out.numpy()) # out = [1.]

                out = paddle.dist(x, y, 2)
                print(out.numpy()) # out = [2.]

                out = paddle.dist(x, y, float("inf"))
                print(out.numpy()) # out = [2.]

                out = paddle.dist(x, y, float("-inf"))
                print(out.numpy()) # out = [0.]
    """
    check_variable_and_dtype(x, 'dtype', ['float32', 'float64'], 'dist')
    check_variable_and_dtype(y, 'dtype', ['float32', 'float64'], 'dist')
    check_type(p, 'p', (float, int), 'dist')
    helper = LayerHelper("dist", **locals())
    out = helper.create_variable_for_type_inference(x.dtype)

    inputs = {"X": [x], "Y": [y]}
    outputs = {'Out': [out]}
    attrs = {"p": float(p)}
    helper.append_op(
        type='dist', inputs=inputs, outputs={'Out': out}, attrs=attrs)
    return out


def dot(x, y, name=None):
    """
    This operator calculates inner product for vectors.
   
    .. note::
       Support 1-d and 2-d Tensor. When it is 2d, the first dimension of this matrix 
       is the batch dimension, which means that the vectors of multiple batches are dotted. 

    Parameters:
        x(Tensor): 1-D or 2-D ``Tensor``. Its dtype should be ``float32``, ``float64``, ``int32``, ``int64``
        y(Tensor): 1-D or 2-D ``Tensor``. Its dtype soulde be ``float32``, ``float64``, ``int32``, ``int64``
        name(str, optional): Name of the output. Default is None. It's used to print debug info for developers. Details: :ref:`api_guide_Name`

    Returns:
        Variable: the calculated result Tensor.

    Examples:

    .. code-block:: python

        import paddle
        import numpy as np

        paddle.disable_static()
        x_data = np.random.uniform(0.1, 1, [10]).astype(np.float32)
        y_data = np.random.uniform(1, 3, [10]).astype(np.float32)
        x = paddle.to_tensor(x_data)
        y = paddle.to_tensor(y_data)
        z = paddle.dot(x, y)
        print(z.numpy())

    """
    op_type = 'dot'
    # skip var type check in dygraph mode to improve efficiency
    if in_dygraph_mode():
        op = getattr(core.ops, op_type)
        return op(x, y)

    assert x is not None, 'x cannot be None in {}'.format(op_type)
    assert y is not None, 'y cannot be None in {}'.format(op_type)

    check_variable_and_dtype(x, 'x', ['float32', 'float64', 'int32', 'int64'],
                             op_type)
    check_variable_and_dtype(y, 'y', ['float32', 'float64', 'int32', 'int64'],
                             op_type)

    helper = LayerHelper(op_type, **locals())
    if name is None:
        out = helper.create_variable_for_type_inference(dtype=x.dtype)
    else:
        out = helper.create_variable(
            name=name, dtype=x.dtype, persistable=False)
    helper.append_op(
        type="dot", inputs={'X': x,
                            'Y': y}, attrs={}, outputs={"Out": out})
    return out


def t(input, name=None):
    """
	:alias_main: paddle.t
	:alias: paddle.t,paddle.tensor.t,paddle.tensor.linalg.t

    Transpose <=2-D tensor. 
    0-D and 1-D tensors are returned as it is and 2-D tensor is equal to 
    the fluid.layers.transpose function which perm dimensions set 0 and 1.
    
    Args:
        input (Variable): The input Tensor. It is a N-D (N<=2) Tensor of data types float16, float32, float64, int32.
        name(str, optional): The default value is None.  Normally there is no need for 
            user to set this property.  For more information, please refer to :ref:`api_guide_Name`
    Returns:
        Variable: A transposed n-D Tensor, with data type being float16, float32, float64, int32, int64.
    
    For Example:
        .. code-block:: text
        # Example 1 (0-D tensor)
         x = tensor([0.79])
         paddle.t(x) = tensor([0.79])
         # Example 2 (1-D tensor)
         x = tensor([0.79, 0.84, 0.32])
         paddle.t(x) = tensor([0.79, 0.84, 0.32])
        
         # Example 3 (2-D tensor)
         x = tensor([0.79, 0.84, 0.32],
                    [0.64, 0.14, 0.57])
         paddle.t(x) = tensor([0.79, 0.64],
                              [0.84, 0.14],
                              [0.32, 0.57])
    
     Examples:
        .. code-block:: python
            import paddle
            import paddle.fluid as fluid
            x = fluid.data(name='x', shape=[2, 3],
                            dtype='float32')
            x_transposed = paddle.t(x)
            print x_transposed.shape
            #(3L, 2L)
    """
    if len(input.shape) > 2:
        raise ValueError(
            "Input(input) only support N-D (N<=2) tensor, but received "
            "length of Input(input) is %s. Perhaps you can use paddle."
            "tensor.transpose() instead." % len(input.shape))
    if in_dygraph_mode():
        if len(input.shape) == 1:
            return input
        # 2-D tensor
        perm = [1, 0]
        out, _ = core.ops.transpose2(input, 'axis', perm)
        return out

    check_variable_and_dtype(
        input, 'input', ['float16', 'float32', 'float64', 'int32', 'int64'],
        'transpose')

    helper = LayerHelper('t', **locals())
    out = helper.create_variable_for_type_inference(input.dtype)
    input_shape = helper.create_variable_for_type_inference(input.dtype)
    if len(input.shape) == 1:
        out = input
    else:
        helper.append_op(
            type='transpose2',
            inputs={'X': [input]},
            outputs={'Out': [out],
                     'XShape': [input_shape]},
            attrs={'axis': [1, 0]})
    return out


def cross(x, y, axis=None, name=None):
    """
    Computes the cross product between two tensors along an axis.
    
    Inputs must have the same shape, and the length of their axes should be equal to 3.
    If `axis` is not given, it defaults to the first axis found with the length 3.
    
    Args:
        x (Tensor): The first input tensor.
        y (Tensor): The second input tensor.
        axis (int, optional): The axis along which to compute the cross product. It defaults to the first axis found with the length 3.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor. A Tensor with same data type as `x`.
        
    Examples:
        .. code-block:: python

            import paddle

            x = paddle.to_tensor([[1.0, 1.0, 1.0],
                                  [2.0, 2.0, 2.0],
                                  [3.0, 3.0, 3.0]])
            y = paddle.to_tensor([[1.0, 1.0, 1.0],
                                  [1.0, 1.0, 1.0],
                                  [1.0, 1.0, 1.0]])

            z1 = paddle.cross(x, y)
            # [[-1. -1. -1.]
            #  [ 2.  2.  2.]
            #  [-1. -1. -1.]]

            z2 = paddle.cross(x, y, axis=1)
            # [[0. 0. 0.]
            #  [0. 0. 0.]
            #  [0. 0. 0.]]
    """
    if in_dygraph_mode():
        if axis is not None:
            return core.ops.cross(x, y, 'dim', axis)
        else:
            return core.ops.cross(x, y)

    helper = LayerHelper("cross", **locals())
    out = helper.create_variable_for_type_inference(x.dtype)
    attrs = dict()
    attrs['dim'] = axis

    helper.append_op(
        type='cross',
        inputs={'X': x,
                'Y': y},
        outputs={'Out': out},
        attrs=attrs)
    return out


def cholesky(x, upper=False, name=None):
    """
    Computes the Cholesky decomposition of one symmetric positive-definite
    matrix or batches of symmetric positive-definite matrice. 
    
    If `upper` is `True`, the decomposition has the form :math:`A = U^{T}U` ,
    and the returned matrix :math:`U` is upper-triangular. Otherwise, the
    decomposition has the form  :math:`A = LL^{T}` , and the returned matrix
    :math:`L` is lower-triangular.

    Args:
        x (Variable): The input tensor. Its shape should be `[*, M, M]`,
            where * is zero or more batch dimensions, and matrices on the
            inner-most 2 dimensions all should be symmetric positive-definite.
            Its data type should be float32 or float64.
        upper (bool): The flag indicating whether to return upper or lower
            triangular matrices. Default: False.

    Returns:
        Variable: A Tensor with same shape and data type as `x`. It represents \
            triangular matrices generated by Cholesky decomposition.
        
    Examples:
        .. code-block:: python

            import paddle
            import numpy as np

            paddle.disable_static()
            a = np.random.rand(3, 3)
            a_t = np.transpose(a, [1, 0])
            x_data = np.matmul(a, a_t) + 1e-03
            x = paddle.to_tensor(x_data)
            out = paddle.cholesky(x, upper=False)
            print(out.numpy())
            # [[1.190523   0.         0.        ]
            #  [0.9906703  0.27676893 0.        ]
            #  [1.25450498 0.05600871 0.06400121]]

    """
    if in_dygraph_mode():
        return core.ops.cholesky(x, "upper", upper)
    check_variable_and_dtype(x, 'dtype', ['float32', 'float64'], 'cholesky')
    check_type(upper, 'upper', bool, 'cholesky')
    helper = LayerHelper('cholesky', **locals())
    out = helper.create_variable_for_type_inference(dtype=x.dtype)
    helper.append_op(
        type='cholesky',
        inputs={'X': [x]},
        outputs={'Out': out},
        attrs={'upper': upper})
    return out


def bmm(x, y, name=None):
    """
	:alias_main: paddle.bmm
	:alias: paddle.bmm,paddle.tensor.bmm,paddle.tensor.linalg.bmm

    Applies batched matrix multiplication to two tensors.

    Both of the two input tensors must be three-dementional and share the same batch size.

    if x is a (b, m, k) tensor, y is a (b, k, n) tensor, the output will be a (b, m, n) tensor.

    Args:
        x (Tensor): The input Tensor.
        y (Tensor): The input Tensor.
        name(str|None): A name for this layer(optional). If set None, the layer
            will be named automatically.

    Returns:
        Tensor: The product Tensor.

    Examples:
        import paddle

        # In imperative mode:
        # size x: (2, 2, 3) and y: (2, 3, 2)
        x = paddle.to_tensor([[[1.0, 1.0, 1.0],
                               [2.0, 2.0, 2.0]],
                              [[3.0, 3.0, 3.0],
                               [4.0, 4.0, 4.0]]])
        y = paddle.to_tensor([[[1.0, 1.0],[2.0, 2.0],[3.0, 3.0]],
                              [[4.0, 4.0],[5.0, 5.0],[6.0, 6.0]]])
        out = paddle.bmm(x, y)
        #output size: (2, 2, 2)
        #output value:
        #[[[6.0, 6.0],[12.0, 12.0]],[[45.0, 45.0],[60.0, 60.0]]]
        out_np = out.numpy()
    """
    x_shape = x.shape
    y_shape = y.shape
    if not len(x_shape) == len(y_shape) == 3:
        raise ValueError(
            "x and y should be 3-dimensional. But received x's dimention: {}, y's dimention: {}".
            format(x_shape, y_shape))
    if x_shape[2] != y_shape[1]:
        raise ValueError(
            "x's width must be equal with y's height. But received x's shape: {}, y's shape: {}".
            format(x_shape, y_shape))
    if x_shape[0] != y_shape[0]:
        raise ValueError(
            "x's batch (shape[0]) must be equal with y's batch (shape[0]). But received x's shape: {}, y's shape: {}".
            format(x_shape, y_shape))
    helper = LayerHelper('bmm', **locals())
    if in_dygraph_mode():
        return core.ops.bmm(x, y)
    out = helper.create_variable_for_type_inference(dtype=x.dtype)
    helper.append_op(type='bmm', inputs={'X': x, 'Y': y}, outputs={'Out': out})
    return out


def histogram(input, bins=100, min=0, max=0):
    """
    Computes the histogram of a tensor. The elements are sorted into equal width bins between min and max. 
    If min and max are both zero, the minimum and maximum values of the data are used.

    Args:
        input (Tensor): A Tensor(or LoDTensor) with shape :math:`[N_1, N_2,..., N_k]` . The data type of the input Tensor
            should be float32, float64, int32, int64.
        bins (int): number of histogram bins
        min (int): lower end of the range (inclusive)
        max (int): upper end of the range (inclusive)

    Returns:
        Tensor: data type is int64, shape is (nbins,).

    Examples:
        .. code-block:: python

            import paddle

            inputs = paddle.to_tensor([1, 2, 1])
            result = paddle.histogram(inputs, bins=4, min=0, max=3)
            print(result) # [0, 2, 1, 0]
    """
    if in_dygraph_mode():
        return core.ops.histogram(input, "bins", bins, "min", min, "max", max)

    helper = LayerHelper('histogram', **locals())
    check_variable_and_dtype(
        input, 'X', ['int32', 'int64', 'float32', 'float64'], 'histogram')
    out = helper.create_variable_for_type_inference(VarDesc.VarType.INT64)
    helper.append_op(
        type='histogram',
        inputs={'X': input},
        outputs={'Out': out},
        attrs={'bins': bins,
               'min': min,
               'max': max})
    return out


def mv(x, vec, name=None):
    """
    Performs a matrix-vector product of the matrix x and the vector vec.

    Args:
        x (Tensor): A tensor with shape :math:`[M, N]` , The data type of the input Tensor x
            should be one of float32, float64.
        vec (Tensor): A tensor with shape :math:`[N]` , The data type of the input Tensor x
            should be one of float32, float64.
        name(str, optional): The default value is None.  Normally there is no need for user to set this
            property.  For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor: The tensor which is producted by x and vec.

    Examples:
        .. code-block:: python

            # x: [M, N], vec: [N]
            # paddle.mv(x, vec)  # out: [M]

            import numpy as np
            import paddle

            paddle.disable_static()
            x_data = np.array([[2, 1, 3], [3, 0, 1]]).astype("float64")
            x = paddle.to_tensor(x_data)
            vec_data = np.array([3, 5, 1])
            vec = paddle.to_tensor(vec_data).astype("float64")
            out = paddle.mv(x, vec)
            paddle.enable_static()
    """
    if in_dygraph_mode():
        out = core.ops.mv(x, vec)
        return out

    def __check_input(x, vec):
        var_names = {'x': x, 'vec': vec}
        for name, val in var_names.items():
            check_variable_and_dtype(val, name, ['float32', 'float64'], 'mv')
        x_shape = list(x.shape)
        vec_shape = list(vec.shape)
        if len(x_shape) != 2:
            raise ValueError(
                "x should be 2-dimensional. But received x's dimention: {}".
                format(x_shape))
        if len(vec_shape) != 1:
            raise ValueError(
                "vec should be 1-dimensional. But received vec's dimention: {}".
                format(vec_shape))

    __check_input(x, vec)

    helper = LayerHelper('mv', **locals())
    out = helper.create_variable_for_type_inference(dtype=x.dtype)
    helper.append_op(
        type='mv', inputs={'X': x,
                           'Vec': vec}, outputs={'Out': out})
    return out
