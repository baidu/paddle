#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from paddle.common_ops_import import *
from ..helper import is_complex, is_real, complex_variable_exists
from ....fluid.framework import ComplexVariable
from ....fluid import layers

__all__ = [
    'reshape',
    'transpose',
]


def reshape(x, shape, inplace=False, name=None):
    """
    To change the shape of ``x`` without changing its data.

    There are some tricks when specifying the target shape.

    1. -1 means the value of this dimension is inferred from the total element
    number of x and remaining dimensions. Thus one and only one dimension can
    be set -1.

    2. 0 means the actual dimension value is going to be copied from the
    corresponding dimension of x. The index of 0s in shape can not exceed
    the dimension of x.

    Here are some examples to explain it.

    1. Given a 3-D tensor x with a shape [2, 4, 6], and the target shape
    is [6, 8], the reshape operator will transform x into a 2-D tensor with
    shape [6, 8] and leaving x's data unchanged.

    2. Given a 3-D tensor x with a shape [2, 4, 6], and the target shape
    specified is [2, 3, -1, 2], the reshape operator will transform x into a
    4-D tensor with shape [2, 3, 4, 2] and leaving x's data unchanged. In this
    case, one dimension of the target shape is set to -1, the value of this
    dimension is inferred from the total element number of x and remaining
    dimensions.

    3. Given a 3-D tensor x with a shape [2, 4, 6], and the target shape
    is [-1, 0, 3, 2], the reshape operator will transform x into a 4-D tensor
    with shape [2, 4, 3, 2] and leaving x's data unchanged. In this case,
    besides -1, 0 means the actual dimension value is going to be copied from
    the corresponding dimension of x.

    Args:
        x(ComplexVariable): the input. A ``Tensor`` or ``LoDTensor`` , data 
            type: ``complex64`` or ``complex128``.
        shape(list|tuple|Variable): target shape. At most one dimension of 
            the target shape can be -1. If ``shape`` is a list or tuple, the 
            elements of it should be integers or Tensors with shape [1] and 
            data type ``int32``. If ``shape`` is an Variable, it should be 
            an 1-D Tensor of data type ``int32``.
        inplace(bool, optional): If ``inplace`` is True, the output of 
            ``reshape`` is the same ComplexVariable as the input. Otherwise, 
            the input and output of ``reshape`` are different 
            ComplexVariables. Defaults to False. Note that if ``x``is more 
            than one OPs' input, ``inplace`` must be False.
        name(str, optional): The default value is None. Normally there is no 
            need for user to set this property. For more information, please 
            refer to :ref:`api_guide_Name` .

    Returns:
        ComplexVariable: A ``Tensor`` or ``LoDTensor``. The data type is same as ``x``. It is a new ComplexVariable if ``inplace`` is ``False``, otherwise it is ``x``.
        
    Raises:
        ValueError: If more than one elements of ``shape`` is -1.
        ValueError: If the element of ``shape`` is 0, the corresponding dimension should be less than or equal to the dimension of ``x``.
        ValueError: If the elements in ``shape`` is negative except -1.

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            import paddle.complex as cpx
            import paddle.fluid.dygraph as dg
            import numpy as np
            
            x_np = np.random.randn(2, 3, 4) + 1j * np.random.randn(2, 3, 4)
            
            place = fluid.CPUPlace()
            with dg.guard(place):
                x_var = dg.to_variable(x_np)
                y_var = cpx.reshape(x_var, (2, -1))
                y_np = y_var.numpy()
                print(y_np.shape)
                # (2, 12)
    """
    complex_variable_exists([x], "reshape")
    if inplace:
        x.real = fluid.layers.reshape(x.real, shape, inplace=inplace, name=name)
        x.imag = fluid.layers.reshape(x.imag, shape, inplace=inplace, name=name)
        return x
    out_real = fluid.layers.reshape(x.real, shape, inplace=inplace, name=name)
    out_imag = fluid.layers.reshape(x.imag, shape, inplace=inplace, name=name)
    return ComplexVariable(out_real, out_imag)


def transpose(x, perm, name=None):
    """
    Permute the data dimensions for complex number :attr:`input` according to `perm`. 
    
    See :ref:`api_fluid_layers_transpose` for the real number API. 
    
    Args:
        x (ComplexVariable): The input n-D ComplexVariable with data type 
            complex64 or complex128.
        perm (list): Permute the input according to the value of perm.
        name (str): The name of this layer. It is optional.

    Returns:
        ComplexVariable: A transposed n-D ComplexVariable, with the same data type as :attr:`input`.

    Examples:
        .. code-block:: python
 
            import paddle
 
            x = paddle.to_tensor([[1.0 + 1.0j, 2.0 + 1.0j], [3.0+1.0j, 4.0+1.0j], [5.0+1.0j, 6.0+1.0j]])
            x_transposed = paddle.complex.transpose(x, [1, 0])
            print(x_transposed.numpy())
            #[[1.+1.j 3.+1.j 5.+1.j]
            # [2.+1.j 4.+1.j 6.+1.j]]

    """
    complex_variable_exists([x], "transpose")
    real = layers.transpose(x.real, perm, name)
    imag = layers.transpose(x.imag, perm, name)
    return ComplexVariable(real, imag)
