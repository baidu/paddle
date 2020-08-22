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
"""
math functions
"""
from __future__ import print_function

from paddle.common_ops_import import *
from ..fluid import layers
from ..fluid.framework import core, _varbase_creator, in_dygraph_mode, Variable
from ..fluid.layer_helper import LayerHelper
from ..fluid.data_feeder import check_variable_and_dtype, check_type, check_dtype, convert_dtype
from ..fluid.layers.layer_function_generator import _generate_doc_string_, generate_activation_fn, generate_layer_fn
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
from ..fluid.layers import elementwise_add    #DEFINE_ALIAS
from ..fluid.layers import elementwise_div    #DEFINE_ALIAS
from ..fluid.layers import elementwise_floordiv    #DEFINE_ALIAS
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
from ..fluid.layers import sqrt    #DEFINE_ALIAS
from ..fluid.layers import sin    #DEFINE_ALIAS
from ..fluid.layers import increment    #DEFINE_ALIAS
from ..fluid.layers import multiplex    #DEFINE_ALIAS
from ..fluid.layers import sums    #DEFINE_ALIAS
from ..fluid import layers

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
        'elementwise_mod',
        'elementwise_pow',
        'elementwise_sub',
        'exp',
        'floor',
        'increment',
        'log',
        'mul',
        'multiplex',
        'prod',
        'pow',
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
        'maximum',
        'min',
        'minimum',
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
        'clip',
        'trace',
        'kron'
]
# yapf: enable.

@templatedoc()
def pow(input, exponent, name=None):
    """
	:alias_main: paddle.pow
	:alias: paddle.pow,paddle.tensor.pow,paddle.tensor.math.pow

    This is Pow Activation Operator.

    :math:`out = input^{exponent}`

    Args:
        input(Variable): A ``Tensor`` or ``LoDTensor`` . The data type is ``float32`` or ``float64``.
        exponent(float32|Variable): A scalar with type ``float32`` or a ``Tensor`` with shape [1] and type ``float32``.
        name(str, optional): The default value is None. Normally there is no need for user to set this property.
            For more information, please refer to :ref:`api_guide_Name` .

    Returns:
        Variable: A ``Tensor`` or ``LoDTensor``. The data type is same as ``input``.

    Examples:

        .. code-block:: python

            import paddle
            import paddle.fluid as fluid

            x = fluid.data(name="x", shape=[32,32], dtype="float32")

            # example 1: argument exponent is float
            y_1 = paddle.pow(x, 2.0)
            # y_1 is x^{2.0}

            # example 2: argument exponent is Variable
            exponent_tensor = fluid.layers.fill_constant([1], "float32", 3.0)
            y_2 = paddle.pow(x, exponent_tensor)
            # y_2 is x^{3.0}
    """
    if in_dygraph_mode():
        return core.ops.pow(input, "exponent", exponent)

    helper = LayerHelper('pow', **locals())
    inputs = {'X': input}
    attrs = {}
    if isinstance(exponent, Variable):
        exponent.stop_gradient = True
        inputs['FactorTensor'] = exponent
    else:
        attrs['factor'] = exponent

    out = helper.create_variable_for_type_inference(dtype=input.dtype)
    check_dtype(
        out.dtype, out.name,
        convert_dtype(input.dtype), 'pow',
        '(The out data type in pow must be the same with input data type.)')

    helper.append_op(
        type='pow', inputs=inputs, outputs={'Out': out}, attrs=attrs)
    return out


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


def add(x, y, name=None):
    """
Examples:

    ..  code-block:: python

        import paddle
        import numpy as np

        paddle.disable_static()
        np_x = np.array([2, 3, 4]).astype('float64')
        np_y = np.array([1, 5, 2]).astype('float64')
        x = paddle.to_variable(np_x)
        y = paddle.to_variable(np_y)
        z = paddle.add(x, y)
        np_z = z.numpy()
        print(np_z)  # [3., 8., 6. ]

    """
    op_type = 'elementwise_add'
    axis = -1
    if in_dygraph_mode():
        return _elementwise_op_in_dygraph(
            x, y, axis=axis, op_name=op_type)

    return _elementwise_op(LayerHelper(op_type, **locals()))


def div(x, y, name=None):
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
        z = paddle.div(x, y)
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

    return _elementwise_op(LayerHelper(op_type, **locals()))


def multiply(x, y, axis=-1, name=None):
    """
	:alias_main: paddle.multiply
	:alias: paddle.multiply,paddle.tensor.multiply,paddle.tensor.math.multiply

Examples:

    .. code-block:: python

        import paddle
        import numpy as np

        paddle.disable_static()
        x_data = np.array([[1, 2], [3, 4]], dtype=np.float32)
        y_data = np.array([[5, 6], [7, 8]], dtype=np.float32)
        x = paddle.to_variable(x_data)
        y = paddle.to_variable(y_data)
        res = paddle.multiply(x, y)
        print(res.numpy()) # [[5, 12], [21, 32]]

        x_data = np.array([[[1, 2, 3], [1, 2, 3]]], dtype=np.float32)
        y_data = np.array([1, 2], dtype=np.float32)
        x = paddle.to_variable(x_data)
        y = paddle.to_variable(y_data)
        res = paddle.multiply(x, y, axis=1)
        print(res.numpy()) # [[[1, 2, 3], [2, 4, 6]]]

    """
    op_type = 'elementwise_mul'
    act = None
    if in_dygraph_mode():
        return _elementwise_op_in_dygraph(
            x, y, axis=axis, act=act, op_name=op_type)

    return _elementwise_op(LayerHelper(op_type, **locals()))

def maximum(x, y, axis=-1, name=None):
    """
Examples:

    .. code-block:: python

        import paddle
        import numpy as np

        paddle.disable_static()
  
        x_data = np.array([[1, 2], [3, 4]], dtype=np.float32)
        y_data = np.array([[5, 6], [7, 8]], dtype=np.float32)
        x = paddle.to_variable(x_data)
        y = paddle.to_variable(y_data)
        res = paddle.maximum(x, y)
        print(res.numpy())
        #[[5. 6.]
        # [7. 8.]]

        x_data = np.array([[[1, 2, 3], [1, 2, 3]]], dtype=np.float32)
        y_data = np.array([1, 2], dtype=np.float32)
        x = paddle.to_variable(x_data)
        y = paddle.to_variable(y_data)
        res = paddle.maximum(x, y, axis=1)
        print(res.numpy())
        #[[[1. 2. 3.]
        #  [2. 2. 3.]]]

        x_data = np.array([2, 3, 5], dtype=np.float32)
        y_data = np.array([1, 4, np.nan], dtype=np.float32)
        x = paddle.to_variable(x_data)
        y = paddle.to_variable(y_data)
        res = paddle.maximum(x, y)
        print(res.numpy())
        #[ 2.  4. nan]

        x_data = np.array([5, 3, np.inf], dtype=np.float32)
        y_data = np.array([1, 4, 5], dtype=np.float32)
        x = paddle.to_variable(x_data)
        y = paddle.to_variable(y_data)
        res = paddle.maximum(x, y)
        print(res.numpy())
        #[ 5.  4. inf]
    """
    op_type = 'elementwise_max'
    act = None
    if in_dygraph_mode():
        return _elementwise_op_in_dygraph(
            x, y, axis=axis, act=act, op_name=op_type)
    return _elementwise_op(LayerHelper(op_type, **locals()))

def minimum(x, y, axis=-1, name=None):
    """
Examples:

    .. code-block:: python

        import paddle
        import numpy as np
        paddle.disable_static()
  
        x_data = np.array([[1, 2], [3, 4]], dtype=np.float32)
        y_data = np.array([[5, 6], [7, 8]], dtype=np.float32)
        x = paddle.to_variable(x_data)
        y = paddle.to_variable(y_data)
        res = paddle.minimum(x, y)
        print(res.numpy())
        #[[1. 2.]
        # [3. 4.]]

        x_data = np.array([[[1, 2, 3], [1, 2, 3]]], dtype=np.float32)
        y_data = np.array([1, 2], dtype=np.float32)
        x = paddle.to_variable(x_data)
        y = paddle.to_variable(y_data)
        res = paddle.minimum(x, y, axis=1)
        print(res.numpy())
        #[[[1. 1. 1.]
        #  [2. 2. 2.]]]

        x_data = np.array([2, 3, 5], dtype=np.float32)
        y_data = np.array([1, 4, np.nan], dtype=np.float32)
        x = paddle.to_variable(x_data)
        y = paddle.to_variable(y_data)
        res = paddle.minimum(x, y)
        print(res.numpy())
        #[ 1.  3. nan]

        x_data = np.array([5, 3, np.inf], dtype=np.float32)
        y_data = np.array([1, 4, 5], dtype=np.float32)
        x = paddle.to_variable(x_data)
        y = paddle.to_variable(y_data)
        res = paddle.minimum(x, y)
        print(res.numpy())
        #[1. 3. 5.]
    """
    op_type = 'elementwise_min'
    act = None
    if in_dygraph_mode():
        return _elementwise_op_in_dygraph(
            x, y, axis=axis, act=act, op_name=op_type)
    return _elementwise_op(LayerHelper(op_type, **locals()))

for func in [
        add,
        div,
        maximum,
        minimum,
        multiply
]:
    proto_dict = {'add': 'elementwise_add', 'div': 'elementwise_div', 'maximum': 'elementwise_max', 'minimum': 'elementwise_min', 'multiply': 'elementwise_mul'}
    op_proto = OpProtoHolder.instance().get_op_proto(proto_dict[func.__name__])

    additional_args_lines = [
        "name (string, optional): Name of the output. \
        Default is None. It's used to print debug info for developers. Details: \
        :ref:`api_guide_Name` "
    ]

    func.__doc__ = _generate_doc_string_(
        op_proto,
        additional_args_lines=additional_args_lines,
        skip_attrs_set={"x_data_format", "y_data_format", "axis",
            "use_quantizer", "mkldnn_data_type", "Scale_x", "Scale_y", "Scale_out"
        }) + """\n""" + str(func.__doc__)


def sum(x, axis=None, dtype=None, keepdim=False, name=None):
    """
    Computes the sum of tensor elements over the given dimension.

    Args:
        x (Tensor): An N-D Tensor, the data type is float32, float64, int32 or int64.
        axis (int|list|tuple, optional): The dimensions along which the sum is performed. If
            :attr:`None`, sum all elements of :attr:`x` and return a
            Tensor variable with a single element, otherwise must be in the
            range :math:`[-rank(x), rank(x))`. If :math:`axis[i] < 0`,
            the dimension to reduce is :math:`rank + axis[i]`.
        dtype (str, optional): The dtype of output Tensor. The default value is None, the dtype
            of output is the same as input Tensor `x`.
        keepdim (bool, optional): Whether to reserve the reduced dimension in the
            output Tensor. The result Tensor will have one fewer dimension
            than the :attr:`x` unless :attr:`keepdim` is true, default
            value is False.
        name (str, optional): The default value is None. Normally there is no need for
            user to set this property.  For more information, please refer to :ref:`api_guide_Name`

    Returns:
        Tensor: Results of summation operation on the specified axis of input Tensor `x`,
        it's data type is the same as `x`.

    Raises:
        ValueError: The :attr:`dtype` must be float64 or int64.
        TypeError: The type of :attr:`axis` must be int, list or tuple.

    Examples:
        .. code-block:: python

            import numpy as np
            import paddle
            paddle.disable_static()

            # x is a Tensor variable with following elements:
            #    [[0.2, 0.3, 0.5, 0.9]
            #     [0.1, 0.2, 0.6, 0.7]]
            # Each example is followed by the corresponding output tensor.
            x_data = np.array([[0.2, 0.3, 0.5, 0.9],[0.1, 0.2, 0.6, 0.7]]).astype('float32')
            x = paddle.to_variable(x_data)
            out1 = paddle.sum(x)  # [3.5]
            out2 = paddle.sum(x, axis=0)  # [0.3, 0.5, 1.1, 1.6]
            out3 = paddle.sum(x, axis=-1)  # [1.9, 1.6]
            out4 = paddle.sum(x, axis=1, keepdim=True)  # [[1.9], [1.6]]

            # y is a Tensor variable with shape [2, 2, 2] and elements as below:
            #      [[[1, 2], [3, 4]],
            #      [[5, 6], [7, 8]]]
            # Each example is followed by the corresponding output tensor.
            y_data = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]).astype('float32')
            y = paddle.to_variable(y_data)
            out5 = paddle.sum(y, axis=[1, 2]) # [10, 26]
            out6 = paddle.sum(y, axis=[0, 1]) # [16, 20]
    """
    if axis is not None and not isinstance(axis, (list, tuple)):
        axis = [axis]

    if not axis:
        reduce_all_flag = True
    else:
        if len(axis) == len(x.shape):
            reduce_all_flag = True
        else:
            reduce_all_flag = False

    attrs = {
        'dim': axis if axis != None and axis != [] and axis != () else [0],
        'keep_dim': keepdim,
        'reduce_all': reduce_all_flag
    }
    dtype_flag = False
    if dtype is not None:
        if dtype in ['float64', 'int64']:
            if (convert_dtype(x.dtype) == "float32" and dtype == "float64") or \
               (convert_dtype(x.dtype) == "int32" and dtype == "int64"):
                attrs.update({
                    'in_dtype': x.dtype,
                    'out_dtype': convert_np_dtype_to_dtype_(dtype)
                })
                dtype_flag = True
        else:
            raise ValueError(
                "The value of 'dtype' in sum op must be float64, int64, but received of {}".
                format(dtype))

    if in_dygraph_mode():
        axis = axis if axis != None and axis != [] else [0]
        if dtype_flag:
            return core.ops.reduce_sum(x, 'dim', axis, 'keep_dim', keepdim,
                                       'reduce_all', reduce_all_flag, 'in_dtype',
                                       x.dtype, 'out_dtype',
                                       convert_np_dtype_to_dtype_(dtype))
        else:
            return core.ops.reduce_sum(x, 'dim', axis, 'keep_dim', keepdim,
                                       'reduce_all', reduce_all_flag)
    check_variable_and_dtype(
        x, 'x', ['float32', 'float64', 'int32', 'int64'], 'sum')
    check_type(axis, 'axis', (int, list, tuple, type(None)), 'sum')

    helper = LayerHelper('sum', **locals())
    if dtype_flag:
        out = helper.create_variable_for_type_inference(
            dtype=convert_np_dtype_to_dtype_(dtype))
    else:
        out = helper.create_variable_for_type_inference(dtype=x.dtype)
    helper.append_op(
        type='reduce_sum',
        inputs={'X': x},
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


def mm(input, mat2, name=None):
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
    out = helper.create_variable_for_type_inference(dtype=input.dtype)
    helper.append_op(
        type='matmul', inputs={'X': input,
                               'Y': mat2}, outputs={'Out': out})
    return out


def addmm(input, x, y, beta=1.0, alpha=1.0, name=None):
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
        beta (float): Coefficient of $input$.
        alpha (float): Coefficient of $x*y$.
        name (str, optional): Name of the output. Normally there is no need for user to set this property. For more information, please refer to :ref:`api_guide_Name`. Default is None.

    Returns:
        Variable(Tensor/LoDTensor): The output Tensor/LoDTensor of addmm op.

    Examples:
        ..  code-block:: python

            import numpy as np
            import paddle

            data_x = np.ones((2, 2)).astype(np.float32)
            data_y = np.ones((2, 2)).astype(np.float32)
            data_input = np.ones((2, 2)).astype(np.float32)

            paddle.disable_static()

            x = paddle.to_variable(data_x)
            y = paddle.to_variable(data_y)
            input = paddle.to_variable(data_input)

            out = paddle.tensor.addmm( input=input, x=x, y=y, beta=0.5, alpha=5.0 )

            print( out.numpy() )
            # [[10.5 10.5]
            # [10.5 10.5]]
    """
    input_shape = input.shape
    x_shape = x.shape
    y_shape = y.shape
    if not len(input_shape) == len(x_shape) == len(y_shape) == 2:
        raise ValueError("The dimention of input, x, y should be 2 but receive input's shape: {}, x's shape: {}, y's shape: {}".format(input_shape, x_shape, y_shape))
    if input_shape[0] != x_shape[0]:
        if input_shape[0] != 1:
            raise ValueError( "When x's dimension[0] is not equal with input's dimension[0], input's dimension[0] must be 1 but got {}".format(input_shape[0]))
        if input_shape[1] != y_shape[1] and input_shape[1] != 1:
            raise ValueError( "When y's dimension[1] is not equal with input's dimension[1], input's dimension[1] must be 1 but got {}".format(input_shape[1]))
    if input_shape[1] != y_shape[1]:
        if input_shape[1] != 1:
            raise ValueError( "When y's dimension[1] is not equal with input's dimension[1], input's dimension[1] must be 1 but got {}".format(input_shape[1]))
        if input_shape[0] != x_shape[0] and input_shape[0] != 1:
            raise ValueError( "When x's dimension[0] is not equal with input's dimension[0], input's dimension[0] must be 1 but got {}".format(input_shape[0]))
    if x_shape[1] != y_shape[0]:
        raise ValueError("The input Variable x's width must be equal with Variable y' height. But received x's shape = {}, y's shape = {}.".format(x_shape, y_shape))



    if in_dygraph_mode():
        out = core.ops.addmm(input, x, y, "Alpha", alpha, "Beta", beta)
        return out

    inputs = {'Input': input, "X": x, "Y": y}
    attrs = {'Alpha': alpha, 'Beta': beta}

    helper = LayerHelper("addmm", **locals())
    check_variable_and_dtype(input, 'Input', ['float32', 'float64'], 'addmm')
    check_variable_and_dtype(x, 'X', ['float32', 'float64'], 'addmm')
    check_variable_and_dtype(y, 'Y', ['float32', 'float64'], 'addmm')
    out = helper.create_variable_for_type_inference(dtype=x.dtype)

    helper.append_op(
        type="addmm", inputs=inputs, attrs=attrs, outputs={"Out": out})
    return out


def logsumexp(x, dim=None, keepdim=False, name=None):
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

    return layers.log(sum_out, name)



def inverse(x, name=None):
    """
    Takes the inverse of the square matrix. A square matrix is a matrix with
    the same number of rows and columns. The input can be a square matrix
    (2-D Tensor) or batches of square matrices.

    Args:
        x (Variable): The input tensor. The last two
            dimensions should be equal. When the number of dimensions is
            greater than 2, it is treated as batches of square matrix. The data
            type can be float32 and float64.
        name (str, optional): The default value is None. Normally there is no need for
            user to set this property. For more information,
            please refer to :ref:`api_guide_Name`

    Returns:
        Variable: A Tensor holds the inverse of x. The shape and data type
                        is the same as x.

    Examples:
        .. code-block:: python

            import numpy as np
            import paddle

            mat_np = np.array([[2, 0], [0, 2]]).astype("float32")
            paddle.disable_static()
            mat = paddle.to_variable(mat_np)
            inv = paddle.inverse(mat)
            print(inv) # [[0.5, 0], [0, 0.5]]

    """
    if in_dygraph_mode():
        return core.ops.inverse(x)

    def _check_input(x):
        check_variable_and_dtype(x, 'x',
                                 ['float32', 'float64'], 'inverse')
        if len(x.shape) < 2:
            raise ValueError(
                "The input of inverse is expected to be a Tensor whose number "
                "of dimensions is no less than 2. But reviced: %d, "
                "x's shape: %s." % (len(x.shape), x.shape))
    _check_input(x)
    helper = LayerHelper('inverse', **locals())
    out = helper.create_variable_for_type_inference(dtype=x.dtype)
    helper.append_op(
        type='inverse', inputs={'Input': [x] }, outputs={'Output': [out]})
    return out


def max(x, axis=None, keepdim=False, name=None):
    """

    Computes the maximum of tensor elements over the given axis.

    Args:
        x(Tensor): A tensor, the data type is float32,
            float64, int32, int64.
        axis(list|int, optional): The axis along which the maximum is computed.
            If :attr:`None`, compute the maximum over all elements of
             `x` and return a Tensor variable with a single element,
            otherwise must be in the range :math:`[-x.ndim(x), x.ndim(x))`.
            If :math:`axis[i] < 0`, the axis to reduce is :math:`x.ndim + axis[i]`.
        keepdim(bool, optional): Whether to reserve the reduced dimension in the
            output Tensor. The result tensor will have one fewer dimension
            than the `x` unless :attr:`keepdim` is true, default
            value is False.
        name(str, optional): The default value is None.  Normally there is no need for
            user to set this property.  For more information, please refer to :ref:`api_guide_Name`

    Returns:
        Tensor, results of maximum on the specified axis of input tensor,
        it's data type is the same as `x`.

    Examples:
        .. code-block:: python

            import numpy as np
            import paddle

            paddle.disable_static()

            # data_x is a variable with shape [2, 4]
            # the axis is a int element
            data_x = np.array([[0.2, 0.3, 0.5, 0.9],
                               [0.1, 0.2, 0.6, 0.7]])
            x = paddle.to_variable(data_x)
            result1 = paddle.max(x)
            print(result1.numpy())
            #[0.9]
            result2 = paddle.max(x, axis=0)
            print(result2.numpy()) 
            #[0.2 0.3 0.6 0.9]
            result3 = paddle.max(x, axis=-1)
            print(result3.numpy())
            #[0.9 0.7]
            result4 = paddle.max(x, axis=1, keepdim=True)
            print(result4.numpy())
            #[[0.9]
            # [0.7]]

            # data_y is a variable with shape [2, 2, 2]
            # the axis is list 
            data_y = np.array([[[1.0, 2.0], [3.0, 4.0]],
                               [[5.0, 6.0], [7.0, 8.0]]])
            y = paddle.to_variable(data_y)
            result5 = paddle.max(y, axis=[1, 2])
            print(result5.numpy())
            #[4. 8.]
            result6 = paddle.max(y, axis=[0, 1])
            print(result6.numpy())
            #[7. 8.]
    """

    if axis is not None and not isinstance(axis, list):
        if isinstance(axis, tuple):
            axis = list(axis)
        elif isinstance(axis, int):
            axis= [axis]
        else:
            raise TypeError(
                "The type of axis must be int, list or tuple, but received {}".format(type(axis)))

    reduce_all = True if axis == None or axis == [] else False
    axis = axis if axis != None and axis != [] else [0]
    if in_dygraph_mode():
        return core.ops.reduce_max(x, 'dim', axis, 'keep_dim', keepdim,
                                   'reduce_all', reduce_all)

    helper = LayerHelper('max', **locals())
    check_variable_and_dtype(
        x, 'x', ['float32', 'float64', 'int32', 'int64'], 'max')

    out = helper.create_variable_for_type_inference(
            dtype=helper.input_dtype())
    helper.append_op(
        type='reduce_max',
        inputs={'X': x},
        outputs={'Out': out},
        attrs={
            'dim': axis,
            'keep_dim': keepdim,
            'reduce_all': reduce_all
        })
    return out

def min(x, axis=None, keepdim=False, name=None):
    """

    Computes the minimum of tensor elements over the given axis

    Args:
        x(Tensor): A tensor, the data type is float32, float64, int32, int64.
        axis(list|int, optional): The axis along which the minimum is computed.
            If :attr:`None`, compute the minimum over all elements of
            `x` and return a Tensor variable with a single element,
            otherwise must be in the range :math:`[-x.ndim, x.ndim)`.
            If :math:`axis[i] < 0`, the axis to reduce is :math:`x.ndim + axis[i]`.
        keepdim(bool, optional): Whether to reserve the reduced dimension in the
            output Tensor. The result tensor will have one fewer dimension
            than the `x` unless :attr:`keepdim` is true, default
            value is False.
        name(str, optional): The default value is None.  Normally there is no need for 
            user to set this property.  For more information, please refer to :ref:`api_guide_Name`

    Returns:
        Tensor, results of minimum on the specified axis of input tensor,
        it's data type is the same as input's Tensor.

    Examples:
        .. code-block:: python

            import numpy as np
            import paddle

            paddle.disable_static()

            # data_x is a variable with shape [2, 4]
            # the axis is a int element
            data_x = np.array([[0.2, 0.3, 0.5, 0.9],
                            [0.1, 0.2, 0.6, 0.7]])
            x = paddle.to_variable(data_x)
            result1 = paddle.min(x)
            print(result1.numpy())
            #[0.1]
            result2 = paddle.min(x, axis=0)
            print(result2.numpy())
            #[0.1 0.2 0.5 0.7]
            result3 = paddle.min(x, axis=-1)
            print(result3.numpy()) 
            #[0.2 0.1]
            result4 = paddle.min(x, axis=1, keepdim=True)
            print(result4.numpy())
            #[[0.2]
            # [0.1]]

            # data_y is a variable with shape [2, 2, 2]
            # the axis is list 
            data_y = np.array([[[1.0, 2.0], [3.0, 4.0]],
                               [[5.0, 6.0], [7.0, 8.0]]])
            y = paddle.to_variable(data_y)
            result5 = paddle.min(y, axis=[1, 2])
            print(result5.numpy()) 
            #[1. 5.]
            result6 = paddle.min(y, axis=[0, 1])
            print(result6.numpy())
            #[1. 2.]
    """

    if axis is not None and not isinstance(axis, list):
        if isinstance(axis, tuple):
            axis = list(axis)
        elif isinstance(axis, int):
            axis= [axis]
        else:
            raise TypeError(
                "The type of axis must be int, list or tuple, but received {}".format(type(axis)))
    reduce_all = True if axis == None or axis == [] else False
    axis = axis if axis != None and axis != [] else [0]
    if in_dygraph_mode():
        return core.ops.reduce_min(x, 'dim', axis, 'keep_dim', keepdim,
                                   'reduce_all', reduce_all)

    helper = LayerHelper('min', **locals())
    check_variable_and_dtype(
        x, 'x', ['float32', 'float64', 'int32', 'int64'], 'min')

    out = helper.create_variable_for_type_inference(
            dtype=helper.input_dtype())
    helper.append_op(
        type='reduce_min',
        inputs={'X': x},
        outputs={'Out': out},
        attrs={
            'dim': axis,
            'keep_dim': keepdim,
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


def addcmul(input, tensor1, tensor2, value=1.0, name=None):
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

    out = layers.elementwise_add(input, layers.elementwise_mul(tensor1, tensor2) * value)
    return out


def clip(x, min=None, max=None, name=None):
    """
        :alias_main: paddle.clip
        :alias: paddle.clip,paddle.tensor.clip,paddle.tensor.math.clip

    **clip layer**

    This operator clip all elements in input into the range [ min, max ] and return
    a resulting tensor as the following equation:

    .. math::

        Out = MIN(MAX(x, min), max)

    Args:
        x (Tensor): An N-D Tensor with data type float32 or float64.
        min (float32|Tensor): The lower bound with type ``float32`` or a ``Tensor``
            with shape [1] and type ``int32``, ``float32``, ``float64``.
        max (float32|Tensor): The upper bound with type ``float32`` or a ``Tensor``
            with shape [1] and type ``int32``, ``float32``, ``float64``.
        name (str, optional): The default value is None. Normally there is no
            need for user to set this property. For more information, please
            refer to :ref:`api_guide_Name`.

    Returns:
        Tensor: A Tensor with the same data type and data shape as input.

    Examples:
        .. code-block:: python

            import paddle
            import numpy as np

            paddle.disable_static()
            x = np.array([[1.2,3.5], [4.5,6.4]]).astype('float32')
            x1 = paddle.to_variable(x)
            out1 = paddle.clip(x1, min=3.5, max=5.0)
            out2 = paddle.clip(x1, min=2.5)
            print(out1.numpy())
            # [[3.5, 3.5]
            # [4.5, 5.0]]
            print(out2.numpy())
            # [[2.5, 3.5]
            # [[4.5, 6.4]
    """

    assert min is not None or max is not None, "either min or max should be defined."

    if in_dygraph_mode():
        min = sys.float_info.min if min is None else min
        max = sys.float_info.max if max is None else max
        return core.ops.clip(x, "min", min, "max", max)

    if min is not None:
        check_type(min, 'min', (float, int, Variable), 'clip')
        if isinstance(min, Variable):
            check_dtype(min.dtype, 'min', ['float32', 'float64', 'int32'],
                        'clip', '(When the type of min in clip is Variable.)')
    if max is not None:
        check_type(max, 'max', (float, int, Variable), 'clip')
        if isinstance(max, Variable):
            check_dtype(max.dtype, 'max', ['float32', 'float64', 'int32'],
                        'clip', '(When the type of max in clip is Variable.)')

    check_variable_and_dtype(x, 'x', ['float16', 'float32', 'float64'], 'clip')

    inputs = {'X': x}
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

    helper = LayerHelper('clip', **locals())
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

            paddle.disable_static()

            case1 = paddle.to_variable(case1)
            case2 = paddle.to_variable(case2)
            case3 = paddle.to_variable(case3)
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
def kron(x, y, name=None):
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

    out = helper.create_variable_for_type_inference(dtype=x.dtype)
    helper.append_op(type="kron", inputs={"X": x, "Y": y}, outputs={"Out": out})
    return out


def cumsum(x, axis=None, dtype=None, name=None):
    """
    The cumulative sum of the elements along a given axis. The first element of the result is the same of the first element of the input. 

    Args:
        x (Tensor): Input of cumsum operator, the Tensor needed to be cumsumed. 
        axis (int, optional): The dimension to accumulate along. -1 means the last dimension. The default (None) is to compute the cumsum over the flattened array.
        dtype (str, optional): The data type of the output tensor, can be float32, float64, int32, int64. If specified, the input tensor is casted to dtype before the operation is performed. This is useful for preventing data type overflows. The default value is None. 
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor, the result of cumsum operator, output of cumsum operator. 

    Examples:
        .. code-block:: python
            
            import paddle
            from paddle import to_variable
            import numpy as np

            paddle.disable_static()
            data_np = np.arange(12).reshape(3, 4)
            data = to_variable(data_np)

            y = paddle.cumsum(data)
            print(y.numpy())
            # [ 0  1  3  6 10 15 21 28 36 45 55 66]

            y = paddle.cumsum(data, axis=0)
            print(y.numpy())
            # [[ 0  1  2  3]
            #  [ 4  6  8 10]
            #  [12 15 18 21]]
            
            y = paddle.cumsum(data, axis=-1)
            print(y.numpy())
            # [[ 0  1  3  6]
            #  [ 4  9 15 22]
            #  [ 8 17 27 38]]

            y = paddle.cumsum(data, dtype='float64')
            print(y.dtype)
            # VarType.FP64
    """
    if axis is None:
        flatten = True
    else:
        flatten = False
    if dtype is not None and x.dtype != convert_np_dtype_to_dtype_(dtype):
        x = layers.cast(x, dtype)

    if in_dygraph_mode():
        if axis is None:
            return core.ops.cumsum(x, 'flatten', flatten)
        else:
            return core.ops.cumsum(x, 'axis', axis, 'flatten', flatten)

    check_type(x, 'x', (Variable), 'cumsum')
    locals_var = locals().copy()
    kwargs = dict()
    for name, val in locals_var.items():
        if val is not None:
            kwargs[name] = val
    _cum_sum_ = generate_layer_fn('cumsum')
    return _cum_sum_(**kwargs)

def prod(x, axis=None, keepdim=False, dtype=None, name=None):
    """
    Compute the product of tensor elements over the given axis.

    Args:
        x(Tensor): An N-D Tensor, the data type is float32, float64, int32 or int64.
        axis(int|list|tuple, optional): The axis along which the product is computed. If :attr:`None`, 
            multiply all elements of `x` and return a Tensor with a single element, 
            otherwise must be in the range :math:`[-x.ndim, x.ndim)`. If :math:`axis[i]<0`, 
            the axis to reduce is :math:`x.ndim + axis[i]`. Default is None.
        dtype(str|np.dtype, optional): The desired date type of returned tensor, can be float32, float64, 
            int32, int64. If specified, the input tensor is casted to dtype before operator performed. 
            This is very useful for avoiding data type overflows. The default value is None, the dtype 
            of output is the same as input Tensor `x`.
        keepdim(bool, optional): Whether to reserve the reduced dimension in the output Tensor. The result 
            tensor will have one fewer dimension than the input unless keep_dim is true. Default is False.
        name(string, optional): The default value is None. Normally there is no need for user to set this property.
            For more information, please refer to :ref:`api_guide_Name` .

    Returns:
        Tensor, result of product on the specified dim of input tensor.

    Raises:
        ValueError: The :attr:`dtype` must be float32, float64, int32 or int64.
        TypeError: The type of :attr:`axis` must be int, list or tuple.

    Examples:
        .. code-block:: python

            import paddle
            import numpy as np

            paddle.disable_static()

            # the axis is a int element
            data_x = np.array([[0.2, 0.3, 0.5, 0.9],
                         [0.1, 0.2, 0.6, 0.7]]).astype(np.float32)
            x = paddle.to_tensor(data_x)
            out1 = paddle.prod(x)
            print(out1.numpy())
            # [0.0002268]

            out2 = paddle.prod(x, -1)
            print(out2.numpy())
            # [0.027  0.0084]

            out3 = paddle.prod(x, 0)
            print(out3.numpy())
            # [0.02 0.06 0.3  0.63]
            print(out3.numpy().dtype)
            # float32

            out4 = paddle.prod(x, 0, keepdim=True)
            print(out4.numpy())
            # [[0.02 0.06 0.3  0.63]]

            out5 = paddle.prod(x, 0, dtype='int64')
            print(out5.numpy())
            # [0 0 0 0]
            print(out5.numpy().dtype)
            # int64

            # the axis is list
            data_y = np.array([[[1.0, 2.0], [3.0, 4.0]],
                               [[5.0, 6.0], [7.0, 8.0]]])
            y = paddle.to_tensor(data_y)
            out6 = paddle.prod(y, [0, 1])
            print(out6.numpy())
            # [105. 384.]

            out7 = paddle.prod(y, (1, 2))
            print(out7.numpy())
            # [  24. 1680.]

    """
    if dtype is not None:
        check_dtype(dtype, 'dtype', ['float32', 'float64', 'int32', 'int64'], 'prod')
        if x.dtype != convert_np_dtype_to_dtype_(dtype):
            x = layers.cast(x, dtype)

    return layers.reduce_prod(input=x, dim=axis, keep_dim=keepdim, name=name)


def tanh(x, name=None):
    """

    .. math::

        tanh(x) = (exp(x) - exp(-x))/(exp(x) + exp(-x))

    Args:
        x (Tensor): The input Tensor with data type float32, float16.
        name (str, optional): Name for the operation (optional, default is None).
            For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        A Tensor with the same data type and shape as ``x`` .

    Examples:

        .. code-block:: python

        import paddle
        import numpy as np

        paddle.disable_static()

        x = paddle.to_variable(np.array([-1, 0.5, 1]))
        out = paddle.tanh(x) # [-0.761594, 0.462117, 0.761594]


    """
    if in_dygraph_mode():
        return core.ops.tanh(x)

    check_variable_and_dtype(x, 'x', ['float16', 'float32', 'float64'], 'tanh')
    helper = LayerHelper('tanh', **locals())
    out = helper.create_variable_for_type_inference(x.dtype)
    helper.append_op(type='tanh', inputs={'X': x}, outputs={'Out': out})
    return out
