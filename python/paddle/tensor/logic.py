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

from ..fluid.layer_helper import LayerHelper
from ..fluid.data_feeder import check_type, check_variable_and_dtype
from ..fluid.layers.layer_function_generator import templatedoc
from .. import fluid
from ..fluid.framework import in_dygraph_mode
from paddle.common_ops_import import *

# TODO: define logic functions of a tensor  
from ..fluid.layers import is_empty  #DEFINE_ALIAS
from ..fluid.layers import isfinite  #DEFINE_ALIAS
from ..fluid.layers import logical_and  #DEFINE_ALIAS
from ..fluid.layers import logical_not  #DEFINE_ALIAS
from ..fluid.layers import logical_or  #DEFINE_ALIAS
from ..fluid.layers import logical_xor  #DEFINE_ALIAS

__all__ = [
    'equal',
    'equal_all',
    'greater_equal',
    'greater_than',
    'is_empty',
    'isfinite',
    'less_equal',
    'less_than',
    'logical_and',
    'logical_not',
    'logical_or',
    'logical_xor',
    'not_equal',
    'allclose',
    #       'isnan'
]


def equal_all(x, y, name=None):
    """
	:alias_main: paddle.equal_all
	:alias: paddle.equal_all,paddle.tensor.equal_all,paddle.tensor.logic.equal_all

    This OP returns the truth value of :math:`x == y`. True if two inputs have the same elements, False otherwise.

    **NOTICE**: The output of this OP has no gradient.

    Args:
        x(Tensor): Tensor, data type is float32, float64, int32, int64.
        y(Tensor): Tensor, data type is float32, float64, int32, int64.
        name(str, optional): The default value is None.  Normally there is no need for
            user to set this property.  For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor: output Tensor, data type is bool, value is [False] or [True].

    Examples:
        .. code-block:: python

          import paddle

          paddle.disable_static()
          x = paddle.to_tensor([1, 2, 3])
          y = paddle.to_tensor([1, 2, 3])
          z = paddle.to_tensor([1, 4, 3])
          result1 = paddle.equal_all(x, y)
          print(result1.numpy()) # result1 = [True ]
          result2 = paddle.equal_all(x, z)
          print(result2.numpy()) # result2 = [False ]
    """

    helper = LayerHelper("equal_all", **locals())
    out = helper.create_variable_for_type_inference(dtype='bool')
    helper.append_op(
        type='equal_all', inputs={'X': [x],
                                  'Y': [y]}, outputs={'Out': [out]})
    return out


@templatedoc()
def allclose(x, y, rtol=1e-05, atol=1e-08, equal_nan=False, name=None):
    """
    ${comment}

    Args:
        x(Tensor): ${input_comment}.
        y(Tensor): ${other_comment}.
        rtol(rtoltype, optional): ${rtol_comment}.
        atol(atoltype, optional): ${atol_comment}.
        equal_nan(equalnantype, optional): ${equal_nan_comment}.
        name (str, optional): Name for the operation. For more information, please
            refer to :ref:`api_guide_Name`. Default: None.

    Returns:
        Tensor: ${out_comment}.

    Raises:
        TypeError: The data type of ``x`` must be one of float32, float64.
        TypeError: The data type of ``y`` must be one of float32, float64.
        TypeError: The type of ``rtol`` must be float.
        TypeError: The type of ``atol`` must be float.
        TypeError: The type of ``equal_nan`` must be bool.

    Examples:
        .. code-block:: python

          import paddle

          paddle.disable_static()

          x = paddle.to_tensor([10000., 1e-07])
          y = paddle.to_tensor([10000.1, 1e-08])
          result1 = paddle.allclose(x, y, rtol=1e-05, atol=1e-08,
                                  equal_nan=False, name="ignore_nan")
          np_result1 = result1.numpy()
          # [False]
          result2 = paddle.allclose(x, y, rtol=1e-05, atol=1e-08,
                                      equal_nan=True, name="equal_nan")
          np_result2 = result2.numpy()
          # [False]

          x = paddle.to_tensor([1.0, float('nan')])
          y = paddle.to_tensor([1.0, float('nan')])
          result1 = paddle.allclose(x, y, rtol=1e-05, atol=1e-08,
                                  equal_nan=False, name="ignore_nan")
          np_result1 = result1.numpy()
          # [False]
          result2 = paddle.allclose(x, y, rtol=1e-05, atol=1e-08,
                                      equal_nan=True, name="equal_nan")
          np_result2 = result2.numpy()
          # [True]
    """

    if in_dygraph_mode():
        return core.ops.allclose(x, y, 'rtol', rtol, 'atol', atol, 'equal_nan',
                                 equal_nan)

    check_variable_and_dtype(x, "input", ['float32', 'float64'], 'allclose')
    check_variable_and_dtype(y, "input", ['float32', 'float64'], 'allclose')
    check_type(rtol, 'rtol', float, 'allclose')
    check_type(atol, 'atol', float, 'allclose')
    check_type(equal_nan, 'equal_nan', bool, 'allclose')

    helper = LayerHelper("allclose", **locals())
    out = helper.create_variable_for_type_inference(dtype='bool')

    inputs = {'Input': x, 'Other': y}
    outputs = {'Out': out}
    attrs = {'rtol': rtol, 'atol': atol, 'equal_nan': equal_nan}
    helper.append_op(
        type='allclose', inputs=inputs, outputs=outputs, attrs=attrs)

    return out


@templatedoc()
def equal(x, y, name=None):
    """
	:alias_main: paddle.equal
	:alias: paddle.equal,paddle.tensor.equal,paddle.tensor.logic.equal

    This layer returns the truth value of :math:`x == y` elementwise.
    **NOTICE**: The output of this OP has no gradient.

    Args:
        x(Tensor): Tensor, data type is float32, float64, int32, int64.
        y(Tensor): Tensor, data type is float32, float64, int32, int64.
        name(str, optional): The default value is None.  Normally there is no need for
            user to set this property.  For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor: output Tensor, it's shape is the same as the input's Tensor,
        and the data type is bool. The result of this op is stop_gradient. 

    Examples:
        .. code-block:: python

          import paddle

          paddle.disable_static()
          x = paddle.to_tensor([1, 2, 3])
          y = paddle.to_tensor([1, 3, 2])
          result1 = paddle.equal(x, y)
          print(result1.numpy())  # result1 = [True False False]
    """
    out = fluid.layers.equal(x, y, name=name, cond=None)
    return out


@templatedoc()
def greater_equal(x, y, name=None):
    """
    :alias_main: paddle.greater_equal
	:alias: paddle.greater_equal,paddle.tensor.greater_equal,paddle.tensor.logic.greater_equal

    This OP returns the truth value of :math:`x >= y` elementwise, which is equivalent function to the overloaded operator `>=`.
    **NOTICE**: The output of this OP has no gradient.

    Args:
        x(Tensor): First input to compare which is N-D tensor. The input data type should be float32, float64, int32, int64.
        y(Tensor): Second input to compare which is N-D tensor. The input data type should be float32, float64, int32, int64.
        name(str, optional): The default value is None.  Normally there is no need for
            user to set this property.  For more information, please refer to :ref:`api_guide_Name`.
    Returns:
        Tensor, the output data type is bool: The tensor storing the output, the output shape is same as input :attr:`x`.

    Examples:
        .. code-block:: python
            import paddle

            paddle.disable_static()
            x = paddle.to_tensor([1, 2, 3])
            y = paddle.to_tensor([1, 3, 2])
            result1 = paddle.greater_equal(x, y)
            print(result1.numpy())  # result1 = [True False True]
    """
    out = fluid.layers.greater_equal(x, y, name=name, cond=None)
    return out


@templatedoc()
def greater_than(x, y, name=None):
    """
    :alias_main: paddle.greater_than
	:alias: paddle.greater_than,paddle.tensor.greater_than,paddle.tensor.logic.greater_than

    This OP returns the truth value of :math:`x > y` elementwise, which is equivalent function to the overloaded operator `>`.
    **NOTICE**: The output of this OP has no gradient.

    Args:
        x(Tensor): First input to compare which is N-D tensor. The input data type should be float32, float64, int32, int64.
        y(Tensor): Second input to compare which is N-D tensor. The input data type should be float32, float64, int32, int64.
        name(str, optional): The default value is None.  Normally there is no need for
            user to set this property.  For more information, please refer to :ref:`api_guide_Name`.
    Returns:
        Tensor, the output data type is bool: The tensor storing the output, the output shape is same as input :attr:`x` .

    Examples:
        .. code-block:: python
            import paddle

            paddle.disable_static()
            x = paddle.to_tensor([1, 2, 3])
            y = paddle.to_tensor([1, 3, 2])
            result1 = paddle.greater_than(x, y)
            print(result1.numpy())  # result1 = [False False True]
    """
    out = fluid.layers.greater_than(x, y, name=name, cond=None)
    return out


@templatedoc()
def less_equal(x, y, name=None):
    """
    :alias_main: paddle.less_equal
	:alias: paddle.less_equal,paddle.tensor.less_equal,paddle.tensor.logic.less_equal

    This OP returns the truth value of :math:`x <= y` elementwise, which is equivalent function to the overloaded operator `<=`.
    **NOTICE**: The output of this OP has no gradient.

    Args:
        x(Tensor): First input to compare which is N-D tensor. The input data type should be float32, float64, int32, int64.
        y(Tensor): Second input to compare which is N-D tensor. The input data type should be float32, float64, int32, int64.
        name(str, optional): The default value is None.  Normally there is no need for
            user to set this property.  For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor, the output data type is bool: The tensor storing the output, the output shape is same as input :attr:`x`.

    Examples:
        .. code-block:: python
            import paddle

            paddle.disable_static()
            x = paddle.to_tensor([1, 2, 3])
            y = paddle.to_tensor([1, 3, 2])
            result1 = paddle.less_equal(x, y)
            print(result1.numpy())  # result1 = [True True False]
    """
    out = fluid.layers.less_equal(x, y, name=name, cond=None)
    return out


@templatedoc()
def less_than(x, y, name=None):
    """
    :alias_main: paddle.less_than
	:alias: paddle.less_than,paddle.tensor.less_than,paddle.tensor.logic.less_than

    This OP returns the truth value of :math:`x < y` elementwise, which is equivalent function to the overloaded operator `<`.
    **NOTICE**: The output of this OP has no gradient.

    Args:
        x(Tensor): First input to compare which is N-D tensor. The input data type should be float32, float64, int32, int64.
        y(Tensor): Second input to compare which is N-D tensor. The input data type should be float32, float64, int32, int64.
        name(str, optional): The default value is None.  Normally there is no need for
            user to set this property.  For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor, the output data type is bool: The tensor storing the output, the output shape is same as input :attr:`x`.

    Examples:
        .. code-block:: python
            import paddle

            paddle.disable_static()
            x = paddle.to_tensor([1, 2, 3])
            y = paddle.to_tensor([1, 3, 2])
            result1 = paddle.less_than(x, y)
            print(result1.numpy())  # result1 = [False True False]
    """
    out = fluid.layers.less_than(x, y, force_cpu=False, name=name, cond=None)
    return out


@templatedoc()
def not_equal(x, y, name=None):
    """
    :alias_main: paddle.not_equal
	:alias: paddle.not_equal,paddle.tensor.not_equal,paddle.tensor.logic.not_equal

    This OP returns the truth value of :math:`x != y` elementwise, which is equivalent function to the overloaded operator `!=`.
    **NOTICE**: The output of this OP has no gradient.

    Args:
        x(Tensor): First input to compare which is N-D tensor. The input data type should be float32, float64, int32, int64.
        y(Tensor): Second input to compare which is N-D tensor. The input data type should be float32, float64, int32, int64.
        name(str, optional): The default value is None.  Normally there is no need for
            user to set this property.  For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor, the output data type is bool: The tensor storing the output, the output shape is same as input :attr:`x`.

    Examples:
        .. code-block:: python

            import paddle

            paddle.disable_static()
            x = paddle.to_tensor([1, 2, 3])
            y = paddle.to_tensor([1, 3, 2])
            result1 = paddle.not_equal(x, y)
            print(result1.numpy())  # result1 = [False True True]
    """
    out = fluid.layers.not_equal(x, y, name=name, cond=None)
    return out
