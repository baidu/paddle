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
"""
math functions
"""
from __future__ import print_function

from paddle.common_ops_import import *
from ..fluid import layers
from ..fluid.framework import core, _varbase_creator, in_dygraph_mode, Variable
from ..fluid.layer_helper import LayerHelper
from ..fluid.data_feeder import check_variable_and_dtype, check_type, check_dtype, convert_dtype
from ..fluid.layers.layer_function_generator import _generate_doc_string_
import sys

# TODO: define math functions
# yapf: disable
from ..fluid.layers import abs    #DEFINE_ALIAS
from ..fluid.layers import acos    #DEFINE_ALIAS
from ..fluid.layers import asin    #DEFINE_ALIAS
from ..fluid.layers import ceil    #DEFINE_ALIAS
from ..fluid.layers import cos    #DEFINE_ALIAS
from ..fluid.layers import sinh    #DEFINE_ALIAS
from ..fluid.layers import cosh    #DEFINE_ALIAS
from ..fluid.layers import cumsum    #DEFINE_ALIAS
from ..fluid.layers import elementwise_add    #DEFINE_ALIAS
from ..fluid.layers import elementwise_div    #DEFINE_ALIAS
from ..fluid.layers import elementwise_floordiv    #DEFINE_ALIAS
from ..fluid.layers import elementwise_max    #DEFINE_ALIAS
from ..fluid.layers import elementwise_min    #DEFINE_ALIAS
from ..fluid.layers import elementwise_mod    #DEFINE_ALIAS
from ..fluid.layers import elementwise_mul    #DEFINE_ALIAS
from ..fluid.layers import elementwise_pow    #DEFINE_ALIAS
from ..fluid.layers import elementwise_sub    #DEFINE_ALIAS
from ..fluid.layers import exp    #DEFINE_ALIAS
from ..fluid.layers import floor    #DEFINE_ALIAS
from ..fluid.layers import log    #DEFINE_ALIAS
from ..fluid.layers import reciprocal    #DEFINE_ALIAS
from ..fluid.layers import reduce_max    #DEFINE_ALIAS
from ..fluid.layers import reduce_min    #DEFINE_ALIAS
from ..fluid.layers import reduce_prod    #DEFINE_ALIAS
from ..fluid.layers import reduce_sum    #DEFINE_ALIAS
from ..fluid.layers import round    #DEFINE_ALIAS
from ..fluid.layers import rsqrt    #DEFINE_ALIAS
from ..fluid.layers import scale    #DEFINE_ALIAS
from ..fluid.layers import sign    #DEFINE_ALIAS
from ..fluid.layers import square    #DEFINE_ALIAS
from ..fluid.layers import stanh    #DEFINE_ALIAS
from ..fluid.layers import atan    #DEFINE_ALIAS
from ..fluid.layers import erf    #DEFINE_ALIAS

from ..fluid.layers import increment    #DEFINE_ALIAS
from ..fluid.layers import multiplex    #DEFINE_ALIAS
from ..fluid.layers import sums    #DEFINE_ALIAS

__all__ = [
        'abs',
        'acos',
        'asin',
        'atan',
        'ceil',
        'cos',
        'cosh',
        'cumsum',
        'elementwise_add',
        'elementwise_div',
        'elementwise_floordiv',
        'elementwise_max',
        'elementwise_min',
        'elementwise_mod',
        'elementwise_sub',
        'exp',
        'floor',
        'increment',
        'log',
        'mul',
        'multiplex',
        'power',
        'reciprocal',
        'reduce_max',
        'reduce_min',
        'reduce_prod',
        'reduce_sum',
        'round',
        'rsqrt',
        'scale',
        'sign',
        'sin',
        'sinh',
        'sqrt',
        'square',
        'stanh',
        'sum',
        'sums',
        'tanh',
        'elementwise_sum',
        'max',
        'min',
        'mm',
        'div',
        'multiply',
        'add',
        'atan',
        'logsumexp',
        'inverse',
        'log1p',
        'erf',
        'addcmul',
        'addmm',
        'clamp',
        'trace',
        'kron'
]


# yapf: enable.


def generate_op_noattr(op_type):
    """Register the Python layer for an Operator without Attribute..

    Args:
       op_type: The name of the operator to be created.

    This function takes in the operator type (sin, tanh etc) and
    creates the operator functionality.

    """
    op_proto = OpProtoHolder.instance().get_op_proto(op_type)

    def func(x, name=None, out=None):
        if in_dygraph_mode():
            op = getattr(core.ops, op_type)
            return op(x)

        check_variable_and_dtype(x, 'x', ['float16', 'float32', 'float64'],
                                 op_type)
        helper = LayerHelper(op_type, **locals())

        if name and out:
            warnings.warn(
                "Both name and out parameters have been set in fluid.tensor.math.%s(), only out will take effect to specify the result storage. "
                "You can discard either one to solve this warning." % op_type,
                category=UserWarning,
                stacklevel=2)
        if not out:
            out = helper.create_variable_for_type_inference(dtype=x.dtype)
        helper.append_op(type=op_type, inputs={"X": x}, outputs={"Out": out})
        return out

    func.__name__ = op_type
    func.__doc__ = _generate_doc_string_(
        op_proto,
        additional_args_lines=[
            "name(str, optional): The default value is None.  Normally there is no need for user to set this property.  For more information, please refer to :ref:`api_guide_Name`.\n    "
            "out(Variable, optional): The default value is None. Optional output can be any created Variable that meets the requirements to store the result of operation. if out is None, a new Varibale will be create to store the result."
        ])
    func.__doc__ = func.__doc__ + """

Return type
  Variable
Examples:
    .. code-block:: python

        import numpy as np

        import paddle
        import paddle.fluid as fluid

        inputs = fluid.data(name="x", shape = [None, 4], dtype='float32')
        output = paddle.%s(inputs)

        exe = fluid.Executor(fluid.CPUPlace())
        exe.run(fluid.default_startup_program())

        #input.shape=1X4, batch_size=1
        img = np.array([[1.0, 2.0, 3.0, 4.0]]).astype(np.float32)
        res = exe.run(fluid.default_main_program(), feed={'x':img}, fetch_list=[output])
        print(res)
""" % op_type
    return func

def power(x, exponent, name=None):
    """
	:alias_main: paddle.power
	:alias: paddle.power,paddle.tensor.power,paddle.tensor.math.power

    :math:``out = x^{exponent}``

    Args:
        x(Tensor): A ``Tensor`` with type ``float32`` or ``float64``.
        exponent(float32|Tensor): A scalar with type ``float32`` or a ``Tensor`` with shape [1] and type ``float32``.
        name(str, optional): The default value is None. Normally there is no need for user to set this property. 
            For more information, please refer to :ref:`api_guide_Name` .

    Returns:
        Tensor: A ``Tensor`` with type same as ``x``.

    Examples:

        .. code-block:: python

            import paddle
            import numpy as np

            paddle.enable_imperative()
            
            # example 1: exponent is a float
            x_data = np.array([1, 2, 3])
            exponent = 2
            x = paddle.imperative.to_variable(x_data)
            res = paddle.power(x, exponent)
            print(res.numpy()) # [1 4 9]
            
            # example 2: exponent is a Variable
            exponent = paddle.fill_constant(shape=[1], value=2, dtype='float32')
            res = paddle.power(x, exponent)
            print(res.numpy()) # [1 4 9]
    """

    # in dynamic graph mode
    if in_dygraph_mode():
        # exponent is scalar
        if isinstance(exponent, (int, long, float)):
            return core.ops.pow(x, 'factor', exponent)
        # exponent is Tensor
        else:
            return _elementwise_op_in_dygraph(
                x, exponent, axis=-1, act=None, op_name='elementwise_pow')
    # in static graph mode
    else:
        # exponent is scalar
        if isinstance(exponent, (int, long, float)):
            helper = LayerHelper('pow', **locals())
            inputs = {'X': x}
            attrs = {'factor': exponent}
            out = helper.create_variable_for_type_inference(dtype=x.dtype)
            helper.append_op(
                type='pow', inputs=inputs, outputs={'Out': out}, attrs=attrs)
            return out
        # exponent is Tensor
        else:
            y = exponent
            return _elementwise_op(LayerHelper('elementwise_pow', **locals()))

__ops__noattr__ = [
    'atan',
    'sin',
    'sqrt',
    'tanh',
]

for _OP in set(__ops__noattr__):
    globals()[_OP] = generate_op_noattr(_OP)


@dygraph_only
def _elementwise_op_in_dygraph(x,
                               y,
                               axis=-1,
                               act=None,
                               use_mkldnn=False,
                               op_name=None):
    op = getattr(core.ops, op_name)
    out = op(x, y, 'axis', axis, 'use_mkldnn', use_mkldnn)

    return dygraph_utils._append_activation_in_dygraph(
        out, act, use_mkldnn=use_mkldnn)


def _elementwise_op(helper):
    op_type = helper.layer_type
    original_op_type = helper.kwargs.get('original_op_type', op_type)
    x = helper.kwargs.get('x', None)
    y = helper.kwargs.get('y', None)

    assert x is not None, 'x cannot be None in {}'.format(original_op_type)
    assert y is not None, 'y cannot be None in {}'.format(original_op_type)
    check_variable_and_dtype(
        x, 'x', ['float16', 'float32', 'float64', 'int32', 'int64'],
        original_op_type)
    check_variable_and_dtype(
        y, 'y', ['float16', 'float32', 'float64', 'int32', 'int64'],
        original_op_type)

    axis = helper.kwargs.get('axis', -1)
    use_mkldnn = helper.kwargs.get('use_mkldnn', False)
    name = helper.kwargs.get('name', None)
    out = helper.kwargs.get('out', None)
    if out is None:
        if name is None:
            out = helper.create_variable_for_type_inference(dtype=x.dtype)
        else:
            out = helper.create_variable(
                name=name, dtype=x.dtype, persistable=False)

    helper.append_op(
        type=op_type,
        inputs={'X': x,
                'Y': y},
        outputs={'Out': out},
        attrs={'axis': axis,
               'use_mkldnn': use_mkldnn})
    return helper.append_activation(out)


def add(x, y, alpha=1, out=None, name=None):
    """
Examples:

    .. code-block:: python

        import paddle
        import paddle.fluid as fluid
        import numpy as np

        def gen_data():
            return {
                "x": np.array([2, 3, 4]).astype('float32'),
                "y": np.array([1, 5, 2]).astype('float32')
            }

        x = fluid.data(name="x", shape=[3], dtype='float32')
        y = fluid.data(name="y", shape=[3], dtype='float32')
        z1 = paddle.add(x, y)
        z2 = paddle.add(x, y, alpha=10)
        # z = x + y

        place = fluid.CPUPlace()
        exe = fluid.Executor(place)
        z_value = exe.run(feed=gen_data(),
                            fetch_list=[z1.name, z2.name])

        print(z_value[0]) # [3., 8., 6.]
        print(z_value[1]) # [12. 53. 24.]


    .. code-block:: python

        import paddle
        import paddle.fluid as fluid
        import numpy as np

        def gen_data():
            return {
                "x": np.ones((2, 3, 4, 5)).astype('float32'),
                "y": np.zeros((4, 5)).astype('float32')
            }

        x = fluid.data(name="x", shape=[2, 3, 4, 5], dtype='float32')
        y = fluid.data(name="y", shape=[4, 5], dtype='float32')
        z = paddle.add(x, y, name='z')
        # z = x + y

        place = fluid.CPUPlace()
        exe = fluid.Executor(place)

        z_value = exe.run(feed=gen_data(),
                            fetch_list=[z.name])

        print(z_value[0])
        print(z_value[0].shape) # z.shape=[2,3,4,5]


    ..  code-block:: python

        import paddle
        import paddle.fluid as fluid
        import numpy as np

        def gen_data():
            return {
                "x": np.random.randint(1, 5, size=[2, 3, 4, 5]).astype('float32'),
                "y": np.random.randint(1, 5, size=[5]).astype('float32')
            }

        x = fluid.data(name="x", shape=[2,3,4,5], dtype='float32')
        y = fluid.data(name="y", shape=[5], dtype='float32')
        z = paddle.add(x, y)
        # z = x / y

        place = fluid.CPUPlace()
        exe = fluid.Executor(place)

        z_value = exe.run(feed=gen_data(),
                            fetch_list=[z.name])
        print(z_value[0])
        print(z_value[0].shape) # z.shape=[2,3,4,5]


    ..  code-block:: python

        import paddle
        import paddle.fluid as fluid
        import numpy as np

        x = fluid.data(name="x", shape=[3], dtype="float32")
        y = fluid.data(name='y', shape=[3], dtype='float32')

        output = fluid.data(name="output", shape=[3], dtype="float32")
        z = paddle.add(x, y, out=output)

        place = fluid.CPUPlace()
        exe = fluid.Executor(place)
        data1 = np.array([2, 3, 4], dtype='float32')
        data2 = np.array([1, 5, 2], dtype='float32')
        z_value = exe.run(feed={'x': data1,
                                'y': data2},
                                fetch_list=[z])
        print(z_value[0]) # [3. 8. 6.]


    ..  code-block:: python

        import paddle
        import paddle.fluid as fluid
        import numpy as np

        with fluid.dygraph.guard():
            np_x = np.array([2, 3, 4]).astype('float64')
            np_y = np.array([1, 5, 2]).astype('float64')
            x = fluid.dygraph.to_variable(np_x)
            y = fluid.dygraph.to_variable(np_y)
            z = paddle.add(x, y, alpha=-0.5)
            np_z = z.numpy()
            print(np_z)  # [1.5, 0.5, 3. ]

    """
    op_type = 'elementwise_add'
    axis = -1
    act = None
    if alpha != 1:
        y = scale(y, scale=alpha)
    if in_dygraph_mode():
        return _elementwise_op_in_dygraph(
            x, y, axis=axis, act=act, op_name=op_type)

    original_op_type = 'add'
    if name and out:
        warnings.warn(
            "Both name and out parameters have been set in paddle.tensor.%s, only out will take effect to specify the result storage. "
            "You can discard either one to solve this warning." %
            original_op_type,
            category=UserWarning,
            stacklevel=2)
    return _elementwise_op(LayerHelper(op_type, **locals()))


def div(x, y, out=None, name=None):
    """
Examples:

    .. code-block:: python

        import paddle
        import paddle.fluid as fluid
        import numpy as np

        def gen_data():
            return {
                "x": np.array([2, 3, 4]).astype('float32'),
                "y": np.array([1, 5, 2]).astype('float32')
            }

        x = fluid.data(name="x", shape=[3], dtype='float32')
        y = fluid.data(name="y", shape=[3], dtype='float32')
        z = paddle.div(x, y)
        # z = x / y

        place = fluid.CPUPlace()
        exe = fluid.Executor(place)
        z_value = exe.run(feed=gen_data(),
                            fetch_list=[z.name])

        print(z_value) # [2., 0.6, 2.]


    .. code-block:: python

        import paddle
        import paddle.fluid as fluid
        import numpy as np

        def gen_data():
            return {
                "x": np.ones((2, 3, 4, 5)).astype('float32'),
                "y": np.zeros((4, 5)).astype('float32')
            }

        x = fluid.data(name="x", shape=[2, 3, 4, 5], dtype='float32')
        y = fluid.data(name="y", shape=[4, 5], dtype='float32')
        z = paddle.div(x, y, name='z')
        # z = x / y

        place = fluid.CPUPlace()
        exe = fluid.Executor(place)

        z_value = exe.run(feed=gen_data(),
                            fetch_list=[z.name])

        print(z_value[0])
        print(z_value[0].shape) # z.shape=[2,3,4,5]


    ..  code-block:: python

        import paddle
        import paddle.fluid as fluid
        import numpy as np

        def gen_data():
            return {
                "x": np.random.randint(1, 5, size=[2, 3, 4, 5]).astype('float32'),
                "y": np.random.randint(1, 5, size=[5]).astype('float32')
            }

        x = fluid.data(name="x", shape=[2,3,4,5], dtype='float32')
        y = fluid.data(name="y", shape=[5], dtype='float32')
        output = fluid.data(name="output", shape=[2,3,4,5], dtype="float32")
        z = paddle.div(x, y, out=output)
        # z = x / y

        place = fluid.CPUPlace()
        exe = fluid.Executor(place)

        z_value = exe.run(feed=gen_data(),
                            fetch_list=[z.name])
        print(z_value[0])
        print(z_value[0].shape) # z.shape=[2,3,4,5]


    ..  code-block:: python

        import paddle
        import paddle.fluid as fluid
        import numpy as np

        with fluid.dygraph.guard(fluid.CPUPlace()):
            np_x = np.array([2, 3, 4]).astype('float64')
            np_y = np.array([1, 5, 2]).astype('float64')
            x = fluid.dygraph.to_variable(np_x)
            y = fluid.dygraph.to_variable(np_y)
            z = paddle.div(x, y)
            np_z = z.numpy()
            print(np_z)  # [2., 0.6, 2.]

    """
    op_type = 'elementwise_div'
    axis = -1
    act = None
    if in_dygraph_mode():
        return _elementwise_op_in_dygraph(
            x, y, axis=axis, act=act, op_name=op_type)

    original_op_type = 'div'
    if name and out:
        warnings.warn(
            "Both name and out parameters have been set in paddle.tensor.%s, only out will take effect to specify the result storage. "
            "You can discard either one to solve this warning." %
            original_op_type,
            category=UserWarning,
            stacklevel=2)
    return _elementwise_op(LayerHelper(op_type, **locals()))


def multiply(x, y, axis=-1, name=None):
    """
	:alias_main: paddle.multiply
	:alias: paddle.multiply,paddle.tensor.multiply,paddle.tensor.math.multiply

Examples:

    .. code-block:: python

        import paddle
        import numpy as np

        paddle.enable_imperative()
        x_data = np.array([[1, 2], [3, 4]], dtype=np.float32)
        y_data = np.array([[5, 6], [7, 8]], dtype=np.float32)
        x = paddle.imperative.to_variable(x_data)
        y = paddle.imperative.to_variable(y_data)
        res = paddle.multiply(x, y)
        print(res.numpy()) # [[5, 12], [21, 32]]

        x_data = np.array([[[1, 2, 3], [1, 2, 3]]], dtype=np.float32)
        y_data = np.array([1, 2], dtype=np.float32)
        x = paddle.imperative.to_variable(x_data)
        y = paddle.imperative.to_variable(y_data)
        res = paddle.multiply(x, y, axis=1)
        print(res.numpy()) # [[[1, 2, 3], [2, 4, 6]]]

    """
    op_type = 'elementwise_mul'
    act = None
    if in_dygraph_mode():
        return _elementwise_op_in_dygraph(
            x, y, axis=axis, act=act, op_name=op_type)

    return _elementwise_op(LayerHelper(op_type, **locals()))


for func in [
        add,
        div,
        multiply,
]:
    proto_dict = {'add': 'elementwise_add', 'div': 'elementwise_div', 'multiply': 'elementwise_mul'}
    op_proto = OpProtoHolder.instance().get_op_proto(proto_dict[func.__name__])
    if func.__name__ in ['add']:
        alias_main = ':alias_main: paddle.%(func)s' % {'func': func.__name__}
        alias = ':alias: paddle.%(func)s, paddle.tensor.%(func)s, paddle.tensor.math.%(func)s' % {'func': func.__name__}

        additional_args_lines = [
            "alpha (int|float, optional): The alpha factor of the input. Default is 1. If alpha is not 1, the equation becomes Out = X + alpha * Y.",
            "out (Variable, optinal): The Variable that stores results of the operation. Default is None. If out is None, \
            a new Variable will be created to store the results."
                                                                 ,
            "name (string, optional): Name of the output. \
            Default is None. It's used to print debug info for developers. Details: \
            :ref:`api_guide_Name` "
        ]
    else:
        additional_args_lines = [
            "name (string, optional): Name of the output. \
            Default is None. It's used to print debug info for developers. Details: \
            :ref:`api_guide_Name` "
        ]

    func.__doc__ = alias_main + """\n""" + alias + """\n""" + _generate_doc_string_(
        op_proto,
        additional_args_lines=additional_args_lines,
        skip_attrs_set={"x_data_format", "y_data_format", "axis",
            "use_quantizer", "Scale_x", "Scale_y", "Scale_out"
        }) + """\n""" + str(func.__doc__)

def sum(input, dim=None, dtype=None, keep_dim=False, name=None):
    """
	:alias_main: paddle.sum
	:alias: paddle.sum,paddle.tensor.sum,paddle.tensor.math.sum

    Computes the sum of tensor elements over the given dimension.

    Args:
        input (Variable): The input variable which is a Tensor, the data type is float32,
            float64, int32, int64.
        dim (list|int, optional): The dimensions along which the sum is performed. If
            :attr:`None`, sum all elements of :attr:`input` and return a
            Tensor variable with a single element, otherwise must be in the
            range :math:`[-rank(input), rank(input))`. If :math:`dim[i] < 0`,
            the dimension to reduce is :math:`rank + dim[i]`.
        dtype(str, optional): The dtype of output tensor. The default value is None, the dtype
            of output is the same as input tensor.
        keep_dim (bool, optional): Whether to reserve the reduced dimension in the
            output Tensor. The result tensor will have one fewer dimension
            than the :attr:`input` unless :attr:`keep_dim` is true, default
            value is False.
        name(str, optional): The default value is None.  Normally there is no need for
            user to set this property.  For more information, please refer to :ref:`api_guide_Name`

    Returns:
        Variable: Tensor, results of summation operation on the specified dim of input tensor,
        it's data type is the same as input's Tensor.

    Raises:
        ValueError, the :attr:`dtype` must be float64 or int64.

    Examples:
        .. code-block:: python

            import paddle
            import paddle.fluid as fluid
            # x is a Tensor variable with following elements:
            #    [[0.2, 0.3, 0.5, 0.9]
            #     [0.1, 0.2, 0.6, 0.7]]
            # Each example is followed by the corresponding output tensor.
            x = fluid.data(name='x', shape=[2, 4], dtype='float32')
            out1 = paddle.sum(x)  # [3.5]
            out2 = paddle.sum(x, dim=0)  # [0.3, 0.5, 1.1, 1.6]
            out3 = paddle.sum(x, dim=-1)  # [1.9, 1.6]
            out4 = paddle.sum(x, dim=1, keep_dim=True)  # [[1.9], [1.6]]

            # y is a Tensor variable with shape [2, 2, 2] and elements as below:
            #      [[[1, 2], [3, 4]],
            #      [[5, 6], [7, 8]]]
            # Each example is followed by the corresponding output tensor.
            y = fluid.data(name='y', shape=[2, 2, 2], dtype='float32')
            out5 = paddle.sum(y, dim=[1, 2]) # [10, 26]
            out6 = paddle.sum(y, dim=[0, 1]) # [16, 20]

    """
    if dim is not None and not isinstance(dim, list):
        dim = [dim]
    attrs = {
        'dim': dim if dim != None and dim != [] else [0],
        'keep_dim': keep_dim,
        'reduce_all': True if dim == None or dim == [] else False,
    }
    dtype_flag = False
    if dtype is not None:
        if dtype in ['float64', 'int64']:
            if (convert_dtype(input.dtype) == "float32" and dtype == "float64") or \
               (convert_dtype(input.dtype) == "int32" and dtype == "int64"):
                attrs.update({
                    'in_dtype': input.dtype,
                    'out_dtype': convert_np_dtype_to_dtype_(dtype)
                })
                dtype_flag = True
        else:
            raise ValueError(
                "The value of 'dtype' in sum op must be float64, int64, but received of {}".
                format(dtype))

    if in_dygraph_mode():
        reduce_all = True if dim == None or dim == [] else False
        dim = dim if dim != None and dim != [] else [0]
        if dtype_flag:
            return core.ops.reduce_sum(input, 'dim', dim, 'keep_dim', keep_dim,
                                       'reduce_all', reduce_all, 'in_dtype',
                                       input.dtype, 'out_dtype',
                                       convert_np_dtype_to_dtype_(dtype))
        else:
            return core.ops.reduce_sum(input, 'dim', dim, 'keep_dim', keep_dim,
                                       'reduce_all', reduce_all)
    check_variable_and_dtype(
        input, 'input', ['float32', 'float64', 'int32', 'int64'], 'reduce_sum')
    helper = LayerHelper('sum', **locals())
    if dtype_flag:
        out = helper.create_variable_for_type_inference(
            dtype=convert_np_dtype_to_dtype_(dtype))
    else:
        out = helper.create_variable_for_type_inference(dtype=input.dtype)
    helper.append_op(
        type='reduce_sum',
        inputs={'X': input},
        outputs={'Out': out},
        attrs=attrs)
    return out


@templatedoc(op_type="sum")
def elementwise_sum(inputs, name=None):
    """
	:alias_main: paddle.elementwise_sum
	:alias: paddle.elementwise_sum,paddle.tensor.elementwise_sum,paddle.tensor.math.elementwise_sum

    ${comment}

    Case 1:
    ::
        Input:
            Input. Shape = [2, 3]
            Input = [[1, 2, 3],
                     [4, 5, 6]]

        Output:
            The output. Shape = [2, 3]
            Output = [[1, 2, 3],
                      [4, 5, 6]]

    Case 2:
    ::
        Input:
            First input:
            Input1. Shape = [2, 3]
            Input1 = [[1, 2, 3],
                      [4, 5, 6]]

        The second input:
            Input2. Shape = [2, 3]
            Input2 = [[7, 8, 9],
                      [10, 11, 12]]

        Output:
            The output. Shape = [2, 3]
            Output = [[8, 10, 12],
                      [14, 16, 18]]

    Args:
        inputs (Variable|list(Variable)):  A Varaible list. The shape and data type of the list elementsshould be consistent.
            Variable can be multi-dimensional Tensoror LoDTensor, and data types can be: float32, float64, int32, int64.
        name(str, optional): The default value is None. Normally there is no need for
            user to set this property. For more information, please refer to :ref:`api_guide_Name`

    Returns:
        Variable: the sum of input :math:`inputs` . its shape and data types are consistent with :math:`inputs` .

    Examples:
        .. code-block:: python

            import paddle
            import paddle.fluid as fluid

            input0 = fluid.layers.fill_constant(shape=[2, 3], dtype='int64', value=5)
            input1 = fluid.layers.fill_constant(shape=[2, 3], dtype='int64', value=3)
            sum = paddle.elementwise_sum([input0, input1])

            # You can print out 'sum' via executor.
            out = fluid.layers.Print(sum, message="the sum of input0 and input1: ")
            exe = fluid.Executor(fluid.CPUPlace())
            exe.run(fluid.default_main_program())

            # The printed result is:
            # 1570701754	the sum of input0 and input1: 	The place is:CPUPlace
            # Tensor[elementwise_sum_0.tmp_0]
            #    shape: [2,3,]
            #    dtype: l
            #    data: 8,8,8,8,8,8,

            # the sum of input0 and input1 is 2-D Tensor with shape [2,3].
            # dtype is the corresponding C++ data type, which may vary in different environments.
            # Eg: if the data type of tensor is int64, then the corresponding C++ data type is int64_t,
            #       so the dtype value is typeid(int64_t).Name(), which is 'x' on MacOS, 'l' on Linux,
            #       and '__int64' on Windows. They both represent 64-bit integer variables.
    """

    helper = LayerHelper('elementwise_sum', **locals())
    check_type(inputs, 'inputs', (Variable, tuple, list), 'elementwise_sum')
    if isinstance(inputs, list) or isinstance(inputs, tuple):
        if len(inputs) > 0:
            for input in inputs:
                check_variable_and_dtype(input, "inputs", \
                   ['float32', 'float64', 'int32', 'int64'], 'elementwise_sum')
    else:
        check_variable_and_dtype(inputs, "inputs", \
                ['float32', 'float64', 'int32', 'int64'], 'elementwise_sum')


    out = helper.create_variable_for_type_inference(
        dtype=helper.input_dtype('inputs'))
    helper.append_op(
        type='sum',
        inputs={'X': inputs},
        outputs={'Out': out},
        attrs={'use_mkldnn': False})

    return out


def mm(input, mat2, out=None, name=None):
    """
	:alias_main: paddle.mm
	:alias: paddle.mm,paddle.tensor.mm,paddle.tensor.math.mm

    Applies matrix multiplication to two tensors.

    Currently, the input tensors' rank can be any, but when the rank of any
    inputs is bigger than 3, this two inputs' rank should be equal.


    Also note that if the raw tensor :math:`x` or :math:`mat2` is rank-1 and
    nontransposed, the prepended or appended dimension :math:`1` will be
    removed after matrix multiplication.

    Args:
        x (Variable): The input variable which is a Tensor or LoDTensor.
        mat2 (Variable): The input variable which is a Tensor or LoDTensor.
        out(Variable, optional): Optional output which can be any created
            Variable that meets the requirements to store the result of operation.
            if out is None, a new Varibale will be create to store the result.
        name(str, optional): The default value is None. Normally there is no need for
            user to set this property. For more information, please refer to :ref:`api_guide_Name`

    Returns:
        Variable: The product Tensor (or LoDTensor) variable.

    Examples:
        .. code-block:: python

            # Examples to clarify shapes of the inputs and output
            # x: [B, ..., M, K], mat2: [B, ..., K, N]
            # fluid.layers.matmul(x, mat2)  # out: [B, ..., M, N]

            # x: [B, M, K], mat2: [B, K, N]
            # fluid.layers.matmul(x, mat2)  # out: [B, M, N]

            # x: [B, M, K], mat2: [K, N]
            # fluid.layers.matmul(x, mat2)  # out: [B, M, N]

            # x: [M, K], mat2: [K, N]
            # fluid.layers.matmul(x, mat2)  # out: [M, N]

            # x: [B, M, K], mat2: [K]
            # fluid.layers.matmul(x, mat2)  # out: [B, M]

            # x: [K], mat2: [K]
            # fluid.layers.matmul(x, mat2)  # out: [1]

            import paddle
            import paddle.fluid as fluid
            x = fluid.data(name='x', shape=[2, 3], dtype='float32')
            mat2 = fluid.data(name='mat2', shape=[3, 2], dtype='float32')
            out = paddle.mm(x, mat2) # out shape is [2, 2]
    """
    if in_dygraph_mode():
        if out is None:
            out = _varbase_creator(dtype=input.dtype)
        core.ops.matmul(input, mat2, out)
        return out

    def __check_input(x, y):
        var_names = {'x': x, 'y': y}
        for name, val in var_names.items():
            check_variable_and_dtype(val, name,
                                     ['float16', 'float32', 'float64'], 'mm')
        x_shape = list(x.shape)
        y_shape = list(y.shape)
        if len(x_shape) == 1:
            x_shape = [1] + x_shape
        if len(y_shape) == 1:
            y_shape = y_shape + [1]

        # check the inner 2 dimensions
        if x_shape[-1] != y_shape[-2]:
            if not ((x_shape[-1] == -1) or (y_shape[-2] == -1)):
                raise ValueError(
                    "After performing an optional transpose, Input X's width should be "
                    "equal to Y's width for multiplication "
                    "prerequisites. But received X's shape: %s, Y's shape: %s\n"
                    % (x_shape, y_shape))

        if len(y_shape) > 2 and len(x_shape) > 2:
            for i, dim_x in enumerate(x_shape[:-2]):
                # don't check neg shape
                if dim_x < 0 or y_shape[i] < 0:
                    continue
                if dim_x != y_shape[i]:
                    raise ValueError(
                        "When the matrix is larger than 2 dimensions, the higher "
                        "dimensional values of the two matrices need to be equal. "
                        "But received x_shape[%d] != y_shape[%d]. X's shape: %s, "
                        "Y's shape: %s.\n" % (i, i, x_shape, y_shape))

    __check_input(input, mat2)

    helper = LayerHelper('mm', **locals())
    if out is None:
        out = helper.create_variable_for_type_inference(dtype=input.dtype)
    helper.append_op(
        type='matmul', inputs={'X': input,
                               'Y': mat2}, outputs={'Out': out})
    return out


def addmm(input, x, y, alpha=1.0, beta=1.0, name=None):
    """
	:alias_main: paddle.addmm
	:alias: paddle.addmm,paddle.tensor.addmm,paddle.tensor.math.addmm

    **addmm**

    This operator is used to perform matrix multiplication for input $x$ and $y$.
    $input$ is added to the final result.
    The equation is:

    ..  math::
        Out = alpha * x * y + beta * input

    $Input$, $x$ and $y$ can carry the LoD (Level of Details) information, or not. But the output only shares the LoD information with input $input$.

    Args:
        input (Variable): The input Tensor/LoDTensor to be added to the final result.
        x (Variable): The first input Tensor/LoDTensor for matrix multiplication.
        y (Variable): The second input Tensor/LoDTensor for matrix multiplication.
        alpha (float): Coefficient of $x*y$.
        beta (float): Coefficient of $input$.
        name (str, optional): Name of the output. Normally there is no need for user to set this property. For more information, please refer to :ref:`api_guide_Name`. Default is None.

    Returns:
        Variable(Tensor/LoDTensor): The output Tensor/LoDTensor of addmm op.

    Examples:
        ..  code-block:: python

            import numpy as np
            import paddle
            import paddle.fluid as fluid

            input = fluid.data(name='input', shape=[2, 2], dtype='float32')
            x = fluid.data(name='x', shape=[2, 2], dtype='float32')
            y = fluid.data(name='y', shape=[2, 2], dtype='float32')
            out = paddle.addmm( input=input, x=x, y=y, alpha=5.0, beta=0.5 )

            data_x = np.ones((2, 2)).astype(np.float32)
            data_y = np.ones((2, 2)).astype(np.float32)
            data_input = np.ones((2, 2)).astype(np.float32)

            place =  fluid.CUDAPlace(0) if fluid.core.is_compiled_with_cuda() else fluid.CPUPlace()
            exe = fluid.Executor(place)
            results = exe.run(fluid.default_main_program(),
                              fetch_list=[out], feed={"input": data_input, 'x': data_x, "y": data_y})
            print( np.array(results[0]) )
            # [[10.5 10.5]
            # [10.5 10.5]]
    """
    if in_dygraph_mode():
        out = core.ops.addmm(input, x, y, "Alpha", alpha, "Beta", beta)
        return out

    inputs = {'Input': input, "X": x, "Y": y}
    attrs = {'Alpha': alpha, 'Beta': beta}

    helper = LayerHelper("addmm", **locals())
    check_variable_and_dtype(x, 'Input', ['float32', 'float64'], 'addmm')
    check_variable_and_dtype(x, 'X', ['float32', 'float64'], 'addmm')
    check_variable_and_dtype(y, 'Y', ['float32', 'float64'], 'addmm')
    out = helper.create_variable_for_type_inference(dtype=x.dtype)

    helper.append_op(
        type="addmm", inputs=inputs, attrs=attrs, outputs={"Out": out})
    return out


def logsumexp(x, dim=None, keepdim=False, out=None, name=None):
    """
	:alias_main: paddle.logsumexp
	:alias: paddle.logsumexp,paddle.tensor.logsumexp,paddle.tensor.math.logsumexp

    This operator calculates the log of the sum of exponentials of the input Tensor.

    .. math::
       logsumexp(x) = \log\sum exp(x)


    Parameters:
       x (Variable): Input LoDTensor or Tensor. Must be one of the following types: float32, float64.
       dim (list|int, optional): The dimensions along which the sum is performed. If :attr:`None`,
         sum all elements of :attr:`input` and return a Tensor variable with a single element,
         otherwise must be in the range :math:`[-rank(input), rank(input))`. If :math:`dim[i] < 0`,
         the dimension to reduce is :math:`rank + dim[i]`.
       keep_dim (bool, optional): Whether to reserve the reduced dimension in the output Tensor.
         The result tensor will have one fewer dimension than the :attr:`input` unless :attr:`keep_dim`
         is true, default value is False.
       out (Variable), optional):  Enable user to explicitly specify an output variable to save result.
       name (str, optional): The default value is None.  Normally there is no need for user to
         set this property.  For more information, please refer to :ref:`api_guide_Name`

    Returns:
       Variable: The calcuated result Tensor/LoDTensor.

    Examples:

    .. code-block:: python

        import paddle
        import paddle.fluid as fluid
        import numpy as np

        with fluid.dygraph.guard():
          np_x = np.random.uniform(0.1, 1, [10]).astype(np.float32)
          x = fluid.dygraph.to_variable(np_x)
          print(paddle.logsumexp(x).numpy())

    ..  code-block:: python

        import paddle
        import paddle.fluid as fluid
        import numpy as np

        with fluid.dygraph.guard():
            np_x = np.random.uniform(0.1, 1, [2, 3, 4]).astype(np.float32)
            x = fluid.dygraph.to_variable(np_x)
            print(paddle.logsumexp(x, dim=1).numpy())
            print(paddle.logsumexp(x, dim=[0, 2]).numpy())

    """
    op_type = 'logsumexp'
    assert x is not None, 'x cannot be None in {}'.format(op_type)

    # reduce_sum does not support float16
    check_variable_and_dtype(x, 'x', ['float32', 'float64'], op_type)

    exp_out = layers.exp(x)
    sum_out = layers.reduce_sum(exp_out, dim, keepdim)

    if out is not None:
        check_variable_and_dtype(out, 'out', [x.dtype], op_type)
        helper = LayerHelper(op_type, **locals())
        helper.append_op(type="log", inputs={"X": sum_out}, outputs={"Out": out})
        return out

    return layers.log(sum_out, name)


def inverse(input, out=None, name=None):
    """
	:alias_main: paddle.inverse
	:alias: paddle.inverse,paddle.tensor.inverse,paddle.tensor.math.inverse

    Takes the inverse of the square matrix. A square matrix is a matrix with
    the same number of rows and columns. The input can be a square matrix
    (2-D Tensor) or batches of square matrices.

    Args:
        input (Variable): The input Variable which holds a Tensor. The last two
            dimensions should be equal. When the number of dimensions is
            greater than 2, it is treated as batches of square matrix. The data
            type can be float32 and float64.
        out (Variable, optional): Optional output which can be any created
            Variable that meets the requirements to store the result of operation.
            If out is None, a new Varibale will be create to store the result.
        name (str, optional): The default value is None. Normally there is no need for
            user to set this property. For more information,
            please refer to :ref:`api_guide_Name`

    Returns:
        Variable: A Tensor holds the inverse of input. The shape and data type
            is the same as input.

    Examples:
        .. code-block:: python

            import numpy as np
            import paddle
            import paddle.fluid as fluid

            mat_np = np.array([[2, 0], [0, 2]]).astype("float32")

            # example for static graph
            input = fluid.data("input", shape=[2, 2], dtype="float32")
            out = paddle.inverse(input)

            place = fluid.CPUPlace()
            exe = fluid.Executor(place)
            results = exe.run(feed={"input": mat_np },
                              fetch_list=[out.name])
            print(results[0]) # [[0.5, 0], [0, 0.5]]

            # example for dynamic graph
            with fluid.dygraph.guard():
                mat = fluid.dygraph.to_variable(mat_np)
                inv = paddle.inverse(mat)
                print(inv) # [[0.5, 0], [0, 0.5]]
    """
    if in_dygraph_mode():
        return core.ops.inverse(input)

    def _check_input(input):
        check_variable_and_dtype(input, 'input',
                                 ['float32', 'float64'], 'inverse')
        if len(input.shape) < 2:
            raise ValueError(
                "The input of inverse is expected to be a Tensor whose number "
                "of dimensions is no less than 2. But reviced: %d, "
                "input's shape: %s." % (len(input.shape), input.shape))

        if out is not None:
            check_variable_and_dtype(out, 'out', input.dtype, 'inverse')

    _check_input(input)

    helper = LayerHelper('inverse', **locals())
    if out is None:
        out = helper.create_variable_for_type_inference(dtype=input.dtype)
    helper.append_op(
        type='inverse', inputs={'Input': [input] }, outputs={'Output': [out]})
    return out


def max(input, dim=None, keep_dim=False, out=None, name=None):
    """
	:alias_main: paddle.max
	:alias: paddle.max,paddle.tensor.max,paddle.tensor.math.max

    Computes the maximum of tensor elements over the given dimension.

    Args:
        input (Variable): The input variable which is a Tensor, the data type is float32,
            float64, int32, int64.
        dim (list|int, optional): The dimension along which the maximum is computed.
            If :attr:`None`, compute the maximum over all elements of
            :attr:`input` and return a Tensor variable with a single element,
            otherwise must be in the range :math:`[-rank(input), rank(input))`.
            If :math:`dim[i] < 0`, the dimension to reduce is :math:`rank + dim[i]`.
        keep_dim (bool, optional): Whether to reserve the reduced dimension in the
            output Tensor. The result tensor will have one fewer dimension
            than the :attr:`input` unless :attr:`keep_dim` is true, default
            value is False.
        out(Variable, optional): Optional output which can be any created
            Variable that meets the requirements to store the result of operation.
            if out is None, a new Varibale will be create to store the result.
        name(str, optional): The default value is None.  Normally there is no need for
            user to set this property.  For more information, please refer to :ref:`api_guide_Name`

    Returns:
        Variable: Tensor, results of maximum on the specified dim of input tensor,
        it's data type is the same as input's Tensor.

    Examples:
        .. code-block:: python
            import paddle
            import paddle.fluid as fluid

            # x is a Tensor variable with following elements:
            #    [[0.2, 0.3, 0.5, 0.9]
            #     [0.1, 0.2, 0.6, 0.7]]
            # Each example is followed by the corresponding output tensor.
            x = fluid.data(name='x', shape=[2, 4], dtype='float32')
            paddle.max(x)  # [0.9]
            paddle.max(x, dim=0)  # [0.2, 0.3, 0.6, 0.9]
            paddle.max(x, dim=-1)  # [0.9, 0.7]
            paddle.max(x, dim=1, keep_dim=True)  # [[0.9], [0.7]]
            # y is a Tensor variable with shape [2, 2, 2] and elements as below:
            #      [[[1.0, 2.0], [3.0, 4.0]],
            #      [[5.0, 6.0], [7.0, 8.0]]]
            # Each example is followed by the corresponding output tensor.
            y = fluid.data(name='y', shape=[2, 2, 2], dtype='float32')
            paddle.max(y, dim=[1, 2]) # [4.0, 8.0]
            paddle.max(y, dim=[0, 1]) # [7.0, 8.0]
    """

    helper = LayerHelper('max', **locals())
    if out is None:
        out = helper.create_variable_for_type_inference(
            dtype=helper.input_dtype())
    if dim is not None and not isinstance(dim, list):
        dim = [dim]

    check_variable_and_dtype(
        input, 'input', ['float32', 'float64', 'int32', 'int64'], 'max')

    reduce_all = True if dim == None or dim == [] else False
    dim = dim if dim != None and dim != [] else [0]

    if in_dygraph_mode():
        return core.ops.reduce_max(input, 'dim', dim, 'keep_dim', keep_dim,
                                   'reduce_all', reduce_all)
    helper.append_op(
        type='reduce_max',
        inputs={'X': input},
        outputs={'Out': out},
        attrs={
            'dim': dim,
            'keep_dim': keep_dim,
            'reduce_all': reduce_all
        })
    return out


def min(input, dim=None, keep_dim=False, out=None, name=None):
    """
	:alias_main: paddle.min
	:alias: paddle.min,paddle.tensor.min,paddle.tensor.math.min

    Computes the minimum of tensor elements over the given dimension.

    Args:
        input (Variable): The input variable which is a Tensor, the data type is float32,
            float64, int32, int64.
        dim (list|int, optional): The dimensions along which the minimum is computed.
            If :attr:`None`, compute the minimum over all elements of
            :attr:`input` and return a Tensor variable with a single element,
            otherwise must be in the range :math:`[-rank(input), rank(input))`.
            If :math:`dim[i] < 0`, the dimension to reduce is :math:`rank + dim[i]`.
        keep_dim (bool, optional): Whether to reserve the reduced dimension in the
            output Tensor. The result tensor will have one fewer dimension
            than the :attr:`input` unless :attr:`keep_dim` is true, default
            value is False.
        out(Variable, optional): Optional output which can be any created
            Variable that meets the requirements to store the result of operation.
            if out is None, a new Varibale will be create to store the result.
        name(str, optional): The default value is None.  Normally there is no need for
            user to set this property.  For more information, please refer to :ref:`api_guide_Name`

    Returns:
        Variable: Tensor, result of minimum on the specified dim of input tensor,
        it's data type is the same as input's Tensor.

    Examples:
        .. code-block:: python
            import paddle
            import paddle.fluid as fluid
            # x is a Tensor variable with following elements:
            #    [[0.2, 0.3, 0.5, 0.9]
            #     [0.1, 0.2, 0.6, 0.7]]
            # Each example is followed by the corresponding output tensor.
            x = fluid.data(name='x', shape=[2, 4], dtype='float32')
            paddle.min(x)  # [0.1]
            paddle.min(x, dim=0)  # [0.1, 0.2, 0.5, 0.7]
            paddle.min(x, dim=-1)  # [0.2, 0.1]
            paddle.min(x, dim=1, keep_dim=True)  # [[0.2], [0.1]]
            # y is a Tensor variable with shape [2, 2, 2] and elements as below:
            #      [[[1.0, 2.0], [3.0, 4.0]],
            #      [[5.0, 6.0], [7.0, 8.0]]]
            # Each example is followed by the corresponding output tensor.
            y = fluid.data(name='y', shape=[2, 2, 2], dtype='float32')
            paddle.min(y, dim=[1, 2]) # [1.0, 5.0]
            paddle.min(y, dim=[0, 1]) # [1.0, 2.0]
    """

    helper = LayerHelper('min', **locals())
    if out is None:
        out = helper.create_variable_for_type_inference(
            dtype=helper.input_dtype())
    if dim is not None and not isinstance(dim, list):
        dim = [dim]

    check_variable_and_dtype(
        input, 'input', ['float32', 'float64', 'int32', 'int64'], 'max')

    reduce_all = True if dim == None or dim == [] else False
    dim = dim if dim != None and dim != [] else [0]

    if in_dygraph_mode():
        return core.ops.reduce_min(input, 'dim', dim, 'keep_dim', keep_dim,
                                   'reduce_all', reduce_all)
    helper.append_op(
        type='reduce_min',
        inputs={'X': input},
        outputs={'Out': out},
        attrs={
            'dim': dim,
            'keep_dim': keep_dim,
            'reduce_all': reduce_all
        })
    return out


def log1p(x, name=None):
    """
	:alias_main: paddle.log1p
	:alias: paddle.log1p,paddle.tensor.log1p,paddle.tensor.math.log1p

    Calculates the natural log of the given input tensor, element-wise.
    .. math::
        Out = \\ln(x+1)
    Args:
        x (Variable): Input LoDTensor or Tensor. Must be one of the following types: float32, float64.
        name(str, optional): The default value is None.  Normally there is no need for 
            user to set this property.  For more information, please refer to :ref:`api_guide_Name`
    Returns:
        Variable: The natural log of the input LoDTensor or Tensor computed element-wise.

    Examples:
        .. code-block:: python
            import paddle
            import paddle.fluid as fluid
            import numpy as np
            # Graph Organizing
            x = fluid.data(name="x", shape=[2,1], dtype="float32")
            res = paddle.log1p(x)
            # Create an executor using CPU as an example
            exe = fluid.Executor(fluid.CPUPlace())
            # Execute
            x_i = np.array([[0], [1]]).astype(np.float32)
            res_val, = exe.run(fluid.default_main_program(), feed={'x':x_i}, fetch_list=[res])
            print(res_val) # [[0.], [0.6931472]]
    """

    if in_dygraph_mode():
        return core.ops.log1p(x)

    check_variable_and_dtype(x, 'x', ['float32', 'float64'], "log1p")
    inputs = {'X': [x]}
    helper = LayerHelper('log1p', **locals())
    dtype = helper.input_dtype(input_param_name='x')
    out = helper.create_variable_for_type_inference(dtype)
    helper.append_op(type="log1p", inputs={"X": x}, outputs={"Out": out})
    return out


def addcmul(input, tensor1, tensor2, value=1.0, out=None, name=None):
    """
	:alias_main: paddle.addcmul
	:alias: paddle.addcmul,paddle.tensor.addcmul,paddle.tensor.math.addcmul

    Calculate the element-wise multiplication of tensor1 and tensor2,
    then multiply the result by value, and add it to input. The shape of input,
    tensor1, tensor2 should be broadcastable.
    The equation is:
    ..  math::
        out = input + value * tensor1 * tensor2
    Args:
        input(Variable): The input to be added. A Tensor with type float32, float64, int32, int64.
        tensor1(Variable): The tensor to be multiplied. A Tensor with type float32, float64, int32, int64.
        tensor2(Variable): The tensor to be multiplied. A Tensor with type float32, float64, int32, int64.
        value(int|float): The multiplier for tensor1*tensor2. For float32 and float64 type input, value must be float, otherwise an integer.
        out(Variable, Optional): The variable that specifies the output of the
            operator, which can be Variable that has been created in the
            program. The default value is None, and a new Variable will be
            created to save the output. Default: None.
        name(str, Optional): For details, please refer to :ref:`api_guide_Name`.
                        Generally, no setting is required. Default: None.
    Returns:
        out(Variable): The output result. A Tensor with the same data type as input's.
    Examples:
        .. code-block:: python
          import paddle
          import paddle.fluid as fluid
          input = fluid.data(name='input', dtype='float32', shape=[3, 4])
          tensor1 = fluid.data(name='tenosr1', dtype='float32', shape=[1, 4])
          tensor2 = fluid.data(name='tensor2', dtype='float32', shape=[3, 4])
          data = paddle.addcmul(input, tensor1, tensor2, value=1.0)
    """

    check_variable_and_dtype(input, 'input', ['float32', 'float64', 'int32', 'int64'], 'addcmul')
    check_variable_and_dtype(tensor1, 'tensor1', ['float32', 'float64', 'int32', 'int64'], 'addcmul')
    check_variable_and_dtype(tensor2, 'tensor2', ['float32', 'float64', 'int32', 'int64'], 'addcmul')
    if convert_dtype(input.dtype) in ['float32', 'float64']:
        check_type(value, 'value', float, 'addcmul')
    if convert_dtype(input.dtype) in ['int32', 'int64']:
        check_type(value, 'value', int, 'addcmul')

    if out is not None:
        layers.assign(layers.elementwise_add(input, layers.elementwise_mul(tensor1, tensor2) * value), out)
    else:
        out = layers.elementwise_add(input, layers.elementwise_mul(tensor1, tensor2) * value)
    return out


def clamp(input, min=None, max=None, output=None, name=None):
    """
	:alias_main: paddle.clamp
	:alias: paddle.clamp,paddle.tensor.clamp,paddle.tensor.math.clamp

    **clampe layer**

    This operator clamps all elements in input into the range [ min, max ] and return
    a resulting tensor as the following equation:

    .. math::

        Out = MIN(MAX(x, min), max)

    Args:
        input (Variable): An input N-D Tensor or LoDTensor
            with data type float32, float64.
        min (float32|Variable): The lower bound with type ``float32`` or a ``Tensor``
            with shape [1] and type ``int32``, ``float32``, ``float64``.
        max (float32|Variable): The upper bound with type ``float32`` or a ``Tensor``
            with shape [1] and type ``int32``, ``float32``, ``float64``.
        output (Variable, optional): A tensor or LoDTensor. If :attr:`output` is None,
            a new tensor will be created as :attr:`output`. Default: None.
        name (str, optional): The default value is None. Normally there is no
            need for user to set this property. For more information, please
            refer to :ref:`api_guide_Name`.

    Returns:
        Variable: A Tensor or LodTensor with the same data type and data shape as input's.

    Examples:
        .. code-block:: python

            import paddle
            import paddle.fluid as fluid
            import numpy as np

            in1 = np.array([[1.2,3.5],
                            [4.5,6.4]]).astype('float32')
            with fluid.dygraph.guard():
                x1 = fluid.dygraph.to_variable(in1)
                out1 = paddle.tensor.clamp(x1, min=3.5, max=5.0)
                out2 = paddle.tensor.clamp(x1, min=2.5)
                print(out1.numpy())
                # [[3.5, 3.5]
                # [4.5, 5.0]]
                print(out2.numpy())
                # [[2.5, 3.5]
                # [[4.5, 6.4]
    """

    assert min is not None or max is not None, "either min or max should be defined."

    if min is not None:
        check_type(min, 'min', (float, Variable), 'clamp')
        if isinstance(min, Variable):
            check_dtype(min.dtype, 'min', ['float32', 'float64', 'int32'],
                        'clamp', '(When the type of min in clamp is Variable.)')
    if max is not None:
        check_type(max, 'max', (float, Variable), 'clamp')
        if isinstance(max, Variable):
            check_dtype(max.dtype, 'max', ['float32', 'float64', 'int32'],
                        'clamp', '(When the type of max in clamp is Variable.)')

    inputs = {'X': input}
    attrs = {'min': sys.float_info.min, 'max': sys.float_info.max}

    if isinstance(min, Variable):
        min.stop_gradient = True
        inputs['Min'] = min
    elif min is not None:
        attrs['min'] = min

    if isinstance(max, Variable):
        max.stop_gradient = True
        inputs['Max'] = max
    elif max is not None:
        attrs['max'] = max

    helper = LayerHelper('clamp', **locals())
    if output is None:
        output = helper.create_variable_for_type_inference(
            dtype=helper.input_dtype())
    helper.append_op(
        type='clip', inputs=inputs, outputs={'Out': [output]}, attrs=attrs)

    return output

def trace(x, offset=0, axis1=0, axis2=1, name=None):
    """
	:alias_main: paddle.trace
	:alias: paddle.trace,paddle.tensor.trace,paddle.tensor.math.trace

    This OP computes the sum along diagonals of the input tensor x.

    If ``x`` is 2D, returns the sum of diagonal.

    If ``x`` has larger dimensions, then returns an tensor of diagonals sum, diagonals be taken from
    the 2D planes specified by axis1 and axis2. By default, the 2D planes formed by the first and second axes
    of the input tensor x.

    The argument ``offset`` determines where diagonals are taken from input tensor x:

    - If offset = 0, it is the main diagonal.
    - If offset > 0, it is above the main diagonal.
    - If offset < 0, it is below the main diagonal.

    Args:
        x(Variable): The input tensor x. Must be at least 2-dimensional. The input data type should be float32, float64, int32, int64.
        offset(int, optional): Which diagonals in input tensor x will be taken. Default: 0 (main diagonals).
        axis1(int, optional): The first axis with respect to take diagonal. Default: 0.
        axis2(int, optional): The second axis with respect to take diagonal. Default: 1.
        name (str, optional): Normally there is no need for user to set this property. For more information, please refer to :ref:`api_guide_Name`. Default: None.

    Returns:
        Variable: the output data type is the same as input data type.

    Examples:
        .. code-block:: python

            import paddle
            import numpy as np

            case1 = np.random.randn(2, 3).astype('float32')
            case2 = np.random.randn(3, 10, 10).astype('float32')
            case3 = np.random.randn(3, 10, 5, 10).astype('float32')

            paddle.enable_imperative()

            case1 = paddle.imperative.to_variable(case1)
            case2 = paddle.imperative.to_variable(case2)
            case3 = paddle.imperative.to_variable(case3)
            data1 = paddle.trace(case1) # data1.shape = [1]
            data2 = paddle.trace(case2, offset=1, axis1=1, axis2=2) # data2.shape = [3]
            data3 = paddle.trace(case3, offset=-3, axis1=1, axis2=-1) # data2.shape = [3, 5]
    """
    inputs = {'Input': [x]}
    attrs = {'offset': offset, 'axis1': axis1, 'axis2': axis2}

    def __check_input(input, offset, dim1, dim2):
        check_dtype(x.dtype, 'Input',
                    ['int32', 'int64', 'float16', 'float32', 'float64'],
                    'trace')

        input_shape = list(x.shape)
        assert len(input_shape) >= 2,                     \
                "The x must be at least 2-dimensional, "   \
                "But received Input x's dimensional: %s.\n" %  \
                len(input_shape)

        axis1_ = axis1 if axis1 >= 0 else len(input_shape) + axis1
        axis2_ = axis2 if axis2 >= 0 else len(input_shape) + axis2

        assert axis1_ < len(input_shape),     \
            "The argument axis1 is out of range (expected to be in range of [%d, %d], but got %d).\n"  \
            % (-(len(input_shape)), len(input_shape) - 1, axis1)

        assert axis2_ < len(input_shape),   \
            "The argument axis2 is out of range (expected to be in range of [%d, %d], but got %d).\n"   \
            % (-(len(input_shape)), len(input_shape) - 1, axis2)


        assert  axis1_ != axis2_,   \
               "axis1 and axis2 cannot be the same axis." \
                "But received axis1 = %d, axis2 = %d\n"%(axis1, axis2)

    if not in_dygraph_mode():
        __check_input(input, offset, axis1, axis2)
    helper = LayerHelper('trace', **locals())

    out = helper.create_variable_for_type_inference(dtype=x.dtype)

    helper.append_op(
        type='trace',
        inputs={'Input': [x]},
        attrs={'offset': offset,
               'axis1': axis1,
               'axis2': axis2},
        outputs={'Out': [out]})
    return out

@templatedoc(op_type="kron")
def kron(x, y, out=None, name=None):
    """
	:alias_main: paddle.kron
	:alias: paddle.kron,paddle.tensor.kron,paddle.tensor.math.kron

${comment}

    Args:
        x (Variable): the fist operand of kron op, data type: float16, float32,
            float64, int32 or int64.
        y (Variable): the second operand of kron op, data type: float16,
            float32, float64, int32 or int64. Its data type should be the same
            with x.
        out (Variable, optional): Optional output which can be any created
            Variable that meets the requirements to store the result of
            operation. If out is None, a new Varibale will be create to store
            the result. Defaults to None.
        name(str, optional): The default value is None.  Normally there is no
            need for user to set this property.  For more information, please
            refer to :ref:`api_guide_Name`.

    Returns:
        Variable: The output of kron op, data type: float16, float32, float64, int32 or int64. Its data is the same with x.

    Examples:
        .. code-block:: python

          import paddle
          from paddle import fluid
          import paddle.fluid.dygraph as dg
          import numpy as np

          a = np.arange(1, 5).reshape(2, 2).astype(np.float32)
          b = np.arange(1, 10).reshape(3, 3).astype(np.float32)

          place = fluid.CPUPlace()
          with dg.guard(place):
              a_var = dg.to_variable(a)
              b_var = dg.to_variable(b)
              c_var = paddle.kron(a_var, b_var)
              c_np = c_var.numpy()
          print(c_np)

          #[[ 1.  2.  3.  2.  4.  6.]
          # [ 4.  5.  6.  8. 10. 12.]
          # [ 7.  8.  9. 14. 16. 18.]
          # [ 3.  6.  9.  4.  8. 12.]
          # [12. 15. 18. 16. 20. 24.]
          # [21. 24. 27. 28. 32. 36.]]
    """
    if in_dygraph_mode():
        return core.ops.kron(x, y)

    helper = LayerHelper('kron', **locals())
    check_variable_and_dtype(x, 'x', ['float16', 'float32', 'float64', 'int32', 'int64'], 'kron')
    check_variable_and_dtype(y, 'y', ['float16', 'float32', 'float64', 'int32', 'int64'], 'kron')

    if out is None:
        out = helper.create_variable_for_type_inference(dtype=x.dtype)
    else:
        check_variable_and_dtype(out, 'out', ['float16', 'float32', 'float64', 'int32', 'int64'], 'kron')
    helper.append_op(type="kron", inputs={"X": x, "Y": y}, outputs={"Out": out})
    return out
