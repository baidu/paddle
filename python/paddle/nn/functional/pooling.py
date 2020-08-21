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

# TODO: define pooling functions

from ...fluid.layers import pool2d  #DEFINE_ALIAS
from ...fluid.layers import pool3d  #DEFINE_ALIAS
from ...fluid.layers import adaptive_pool2d  #DEFINE_ALIAS
from ...fluid.layers import adaptive_pool3d  #DEFINE_ALIAS
from ...fluid.layers import utils, LayerHelper
from ...fluid import core
from ...fluid.framework import in_dygraph_mode, convert_np_dtype_to_dtype_
from ...fluid.data_feeder import check_type, check_variable_and_dtype
from ...fluid.layers import unsqueeze, squeeze, LayerHelper

__all__ = [
    'pool2d', 'pool3d', 'adaptive_pool2d', 'adaptive_pool3d', 'avg_pool1d',
    'max_pool1d', 'adaptive_avg_pool1d', 'adaptive_max_pool1d'
]


def check_input(x, dimension):
    if len(x.shape) != dimension:
        raise ValueError("Excepted Input X is 3-D tensor, but received {}-D {}".
                         format(len(x.shape), type(x)))


def update_padding1d(padding, pool_type='avg'):
    def is_list_or_tuple(ele):
        if isinstance(ele, list) or isinstance(ele, tuple):
            return True
        return False

    if is_list_or_tuple(padding):
        if padding.__len__() == 1 and not is_list_or_tuple(padding[0]):
            return [0, padding[0]]
        else:
            raise ValueError(
                "{}_pool1d() argument 'padding' should contain one int (got {})".
                format(pool_type, padding.__len__()))
    else:
        padding = [0, padding]

    return padding


def avg_pool1d(x,
               kernel_size,
               stride=None,
               padding=0,
               count_include_pad=True,
               ceil_mode=False,
               name=None):
    """

    This operation applies a 1D average pooling over an input signal composed
    of several input planes, based on the input, output_size, return_indices parameters.
    Input(X) and output(Out) are in NCL format, where N is batch
    size, C is the number of channels, L is the length of the feature.
    The output tensor shape will be [N, C, output_size].

    The output value of the layer with input size (N, C, L),
    output (N, C, L_{out}) and kernel_size k can be precisely described as
    For average pool1d:

    ..  math::

       Output(N_i, C_i, l) &= mean(Input[N_i, C_i, stride \times l:stride \times l+k])


    Args:
        x (Tensor): The input tensor of pooling operator which is a 3-D tensor with
                          shape [N, C, L]. The format of input tensor is `"NCL"` or
                          `"NHL"`, where `N` is batch size, `C` is the number of channels,
                          `L` is the length of the feature. The data type if float32 or float64.
        kernel_size (int|list|tuple): The pool kernel size. If pool kernel size is a tuple or list,
            it must contain one integers.
        stride (int|list|tuple): The pool stride size. If pool stride size is a tuple or list,
            it must contain one integers.
        padding (string|int|list|tuple): The pool padding. If `pool_padding` is a string, either 'VALID' or
            'SAME' which is the padding algorithm. If pool padding size is a tuple or list,
            it could be the following forms: `[pad_left, pad_right]`. If padding is non-zero,
            then the input is implicitly zero-padded on both sides for padding number of points.
        count_include_pad (bool): Whether to exclude padding points in average pooling
                          mode, default is `true`.
        ceil_mode (bool): ${ceil_mode_comment}Whether to use the ceil function to calculate output height and width.
            If it is set to False, the floor function will be used. Default False
        name(str, optional): For detailed information, please refer
                             to :ref:`api_guide_Name`. Usually name is no need to set and
                             None by default.

    Returns:
        Tensor: The output tensor of pooling result. The data type is same as input tensor.

    Raises:
        ValueError: If `padding` is a string, but not "SAME" or "VALID".
        ValueError: If `padding` is "VALID", but `ceil_mode` is True.
        ValueError: If `padding` is a list or tuple but its length greater than 1.
        ShapeError: If the input is not a 3-D.
        ShapeError: If the output's shape calculated is not greater than 0.


    Examples:

        .. code-block:: python

          import paddle
          import paddle.nn.functional as F
          paddle.disable_static()

          data = paddle.to_tensor(np.random.uniform(-1, 1, [1, 3, 32]).astype(np.float32))
          pool_out = F.avg_pool1d(data, kernel_size=2, stride=2, padding=0)
          # pool_out shape: [1, 3, 16]

    """
    """NCL to NCHW"""
    data_format = "NCHW"
    check_variable_and_dtype(x, 'input', ['float32', 'float64'], 'avg_pool1d')
    check_input(x, 3)
    x = unsqueeze(x, [2])
    kernel_size = utils.convert_to_list(kernel_size, 1, 'pool_size')
    kernel_size = [1] + kernel_size
    if stride is None:
        stride = kernel_size
    else:
        stride = utils.convert_to_list(stride, 1, 'pool_stride')
        stride = [1] + stride

    padding_algorithm = "EXPLICIT"
    if isinstance(padding, str):
        padding = padding.upper()
        if padding not in ["SAME", "VALID"]:
            raise ValueError(
                "Unknown Attr(padding): '%s'. It can only be 'SAME' or 'VALID'."
                % str(padding))
        if padding == "VALID":
            padding_algorithm = "VALID"
            padding = [0]
            if ceil_mode != False:
                raise ValueError(
                    "When Attr(padding) is \"VALID\", Attr(ceil_mode) must be False. "
                    "Received ceil_mode: True.")
        elif padding == "SAME":
            padding_algorithm = "SAME"
            padding = [0]

    padding = update_padding1d(padding, "avg")

    if in_dygraph_mode():
        output = core.ops.pool2d(
            x, 'pooling_type', 'avg', 'ksize', kernel_size, 'global_pooling',
            False, 'strides', stride, 'paddings', padding, 'padding_algorithm',
            padding_algorithm, 'use_cudnn', not count_include_pad, 'ceil_mode',
            ceil_mode, 'use_mkldnn', False, 'exclusive', True, 'data_format',
            data_format)
        return squeeze(output, [2])

    op_type = 'pool2d'
    helper = LayerHelper(op_type, **locals())
    dtype = helper.input_dtype()
    pool_out = helper.create_variable_for_type_inference(dtype)

    helper.append_op(
        type=op_type,
        inputs={"X": x},
        outputs={"Out": pool_out},
        attrs={
            "pooling_type": 'avg',
            "ksize": kernel_size,
            "global_pooling": False,
            "strides": stride,
            "paddings": padding,
            "padding_algorithm": padding_algorithm,
            "use_cudnn": True,
            "ceil_mode": ceil_mode,
            "use_mkldnn": False,
            "exclusive": not count_include_pad,
            "data_format": data_format,
        })

    return squeeze(pool_out, [2])


def max_pool1d(x,
               kernel_size,
               stride=None,
               padding=0,
               return_indices=False,
               ceil_mode=False,
               name=None):
    """

    Applies a 1D max pooling over an input signal composed of several input planes based
    on the input, output_size, return_indices parameters.
    Input(X) and output(Out) are in NCL format, where N is batch
    size, C is the number of channels, L is the length of the feature.

    The output value of the layer with input size (N, C, L),
    output (N, C, L_{out}) and kernel_size k can be precisely described as
    For average pool1d:

    ..  math::

       Output(N_i, C_i, l) &=  max(Input[N_i, C_i, stride \times l:stride \times l+k])}

    Args:
        x (Tensor): The input tensor of pooling operator which is a 3-D tensor with
                          shape [N, C, L]. The format of input tensor is `"NCL"` or
                          `"NHL"`, where `N` is batch size, `C` is the number of channels,
                          `L` is the length of the feature. The data type if float32 or float64.
        kernel_size (int|list|tuple): The pool kernel size. If pool kernel size is a tuple or list,
            it must contain one integers.
        stride (int|list|tuple): The pool stride size. If pool stride size is a tuple or list,
            it must contain one integers.
        padding (string|int|list|tuple): The pool padding. If `pool_padding` is a string, either 'VALID' or
            'SAME' which is the padding algorithm. If pool padding size is a tuple or list,
            it could be the following forms: `[pad_left, pad_right]`.
        return_indices (bool): Whether return the max indices along with the outputs. default is `False`.
        ceil_mode (bool): Whether to use the ceil function to calculate output height and width. False is the default.
            If it is set to False, the floor function will be used. Default False.
        name(str, optional): For detailed information, please refer
                             to :ref:`api_guide_Name`. Usually name is no need to set and
                             None by default.

    Returns:
        Tensor: The output tensor of pooling result. The data type is same as input tensor.

    Raises:
        TypeError: If `use_cudnn` is not a bool value.
        ValueError: If `padding` is a string, but not "SAME" or "VALID".
        ValueError: If `padding` is "VALID", but `ceil_mode` is True.
        ValueError: If `padding` is a list or tuple but its length greater than 1.
        ShapeError: If the input is not a 3-D.
        ShapeError: If the output's shape calculated is not greater than 0.


    Examples:

        .. code-block:: python

          import paddle
          import paddle.nn.functional as F
          paddle.disable_static()

          data = paddle.to_tensor(np.random.uniform(-1, 1, [1, 3, 32]).astype(np.float32))
          pool_out = F.max_pool1d(data, kernel_size=2, stride=2, padding=0)
          # pool_out shape: [1, 3, 16]

          pool_out, indices = F.max_pool1d(data, kernel_size=2, stride=2, padding=0, return_indices=True)
          # pool_out shape: [1, 3, 16],  indices shape: [1, 3, 16]

    """
    """NCL to NCHW"""
    data_format = "NCHW"
    check_variable_and_dtype(x, 'input', ['float32', 'float64'], 'max_pool1d')
    check_input(x, 3)
    x = unsqueeze(x, [2])
    kernel_size = [1] + utils.convert_to_list(kernel_size, 1, 'pool_size')
    if stride is None:
        stride = kernel_size
    else:
        stride = [1] + utils.convert_to_list(stride, 1, 'pool_stride')

    padding_algorithm = "EXPLICIT"
    if isinstance(padding, str):
        padding = padding.upper()
        if padding not in ["SAME", "VALID"]:
            raise ValueError(
                "Unknown Attr(padding): '%s'. It can only be 'SAME' or 'VALID'."
                % str(padding))
        if padding == "VALID":
            padding_algorithm = "VALID"
            padding = [0]
            if ceil_mode != False:
                raise ValueError(
                    "When Attr(padding) is \"VALID\", Attr(ceil_mode) must be False. "
                    "Received ceil_mode: True.")
        elif padding == "SAME":
            padding_algorithm = "SAME"
            padding = [0]

    padding = update_padding1d(padding, 'max')

    if in_dygraph_mode():
        pool_out = core.ops.max_pool2d_with_index(
            x, 'ksize', kernel_size, 'global_pooling', False, 'strides', stride,
            'paddings', padding, 'padding_algorithm', padding_algorithm,
            'use_cudnn', True, 'ceil_mode', ceil_mode, 'use_mkldnn', False,
            'exclusive', True, 'data_format', data_format)
        return (squeeze(pool_out[0], [2]), squeeze(
            pool_out[1], [2])) if return_indices else squeeze(pool_out[0], [2])

    op_type = 'max_pool2d_with_index'
    helper = LayerHelper(op_type, **locals())
    dtype = helper.input_dtype()
    pool_out = helper.create_variable_for_type_inference(dtype)
    mask = helper.create_variable_for_type_inference(dtype)
    outputs = {"Out": pool_out, "Mask": mask}

    helper.append_op(
        type=op_type,
        inputs={"X": x},
        outputs=outputs,
        attrs={
            "pooling_type": 'max',
            "ksize": kernel_size,
            "global_pooling": False,
            "strides": stride,
            "paddings": padding,
            "padding_algorithm": padding_algorithm,
            "use_cudnn": True,
            "ceil_mode": ceil_mode,
            "use_mkldnn": False,
            "exclusive": True,
            "data_format": data_format,
        })

    return (squeeze(pool_out, [2]),
            squeeze(mask, [2])) if return_indices else squeeze(pool_out, [2])


def adaptive_avg_pool1d(x, output_size, name=None):
    """

        This operation applies a 1D adaptive average pooling over an input signal composed
        of several input planes, based on the input, output_size, return_indices parameters.
        Input(X) and output(Out) are in NCL format, where N is batch
        size, C is the number of channels, L is the length of the feature.
        The output tensor shape will be [N, C, output_size].

        For average adaptive pool1d:

        ..  math::

           lstart &= floor(i * L_{in} / L_{out})

           lend &= ceil((i + 1) * L_{in} / L_{out})

           Output(i) &= \\frac{sum(Input[lstart:lend])}{(lstart - lend)}

        Args:
            x (Tensor): The input tensor of pooling operator, which is a 3-D tensor
                              with shape [N, C, L].  The format of input tensor is NCL,
                              where N is batch size, C is the number of channels, L is the
                              length of the feature. The data type is float32 or float64.
            output_size (int|list|tuple): The pool kernel size. If pool kernel size is a tuple or list,
                it must contain one int.
            name(str, optional): For detailed information, please refer
                                 to :ref:`api_guide_Name`. Usually name is no need to set and
                                 None by default.

        Returns:
            Tensor: The output tensor of adaptive average pooling result. The data type is same
                      as input tensor.

        Raises:
            ValueError: 'pool_size' should be a integer or list or tuple with length as 1.

        Examples:
            .. code-block:: python

              # average adaptive pool1d
              # suppose input data in shape of [N, C, L], `output_size` is m or [m],
              # output shape is [N, C, m], adaptive pool divide L dimension
              # of input data into m grids averagely and performs poolings in each
              # grid to get output.
              # adaptive max pool performs calculations as follow:
              #
              #     for i in range(m):
              #         lstart = floor(i * L / m)
              #         lend = ceil((i + 1) * L / m)
              #         output[:, :, i] = sum(input[:, :, lstart: lend])/(lstart - lend)
              #
              import paddle
              import paddle.nn.functional as F
              paddle.disable_static()

              data = paddle.to_tensor(np.random.uniform(-1, 1, [1, 3, 32]).astype(np.float32))
              pool_out = F.adaptive_average_pool1d(data, output_size=16)
              # pool_out shape: [1, 3, 16])
        """
    pool_type = 'avg'
    check_variable_and_dtype(x, 'input', ['float32', 'float64'],
                             'adaptive_pool2d')
    check_input(x, 3)
    check_type(output_size, 'pool_size', (int), 'adaptive_pool1d')

    pool_size = [1] + utils.convert_to_list(output_size, 1, 'pool_size')

    l_type = "pool2d"
    x = unsqueeze(x, [2])
    if in_dygraph_mode():
        pool_out = core.ops.pool2d(x, 'pooling_type', pool_type, 'ksize',
                                   pool_size, 'adaptive', True)
        return squeeze(pool_out, [2])

    helper = LayerHelper(l_type, **locals())
    dtype = helper.input_dtype()
    pool_out = helper.create_variable_for_type_inference(dtype)

    outputs = {"Out": pool_out}
    helper.append_op(
        type=l_type,
        inputs={"X": x},
        outputs=outputs,
        attrs={
            "pooling_type": pool_type,
            "ksize": pool_size,
            "adaptive": True,
        })

    return squeeze(pool_out, [2])


def adaptive_max_pool1d(x, output_size, return_indices=False, name=None):
    """
        This operation applies a 1D adaptive max pooling over an input signal composed
        of several input planes, based on the input, output_size, return_indices parameters.
        Input(X) and output(Out) are in NCL format, where N is batch
        size, C is the number of channels, L is the length of the feature.
        The output tensor shape will be [N, C, output_size].

        For max adaptive pool1d:

        ..  math::

           lstart &= floor(i * L_{in} / L_{out})

           lend &= ceil((i + 1) * L_{in} / L_{out})

           Output(i) &= max(Input[lstart:lend])}

        Args:
            x (Tensor): The input tensor of pooling operator, which is a 3-D tensor
                              with shape [N, C, L].  The format of input tensor is NCL,
                              where N is batch size, C is the number of channels, L is the
                              length of the feature. The data type is float32 or float64.
            output_size (int|list|tuple): The pool kernel size. If pool kernel size is a tuple or list,
                it must contain one int.
            return_indices (bool): If true, the index of max pooling point will be returned along
                with outputs. It cannot be set in average pooling type. Default False.
            name(str, optional): For detailed information, please refer
                                 to :ref:`api_guide_Name`. Usually name is no need to set and
                                 None by default.

        Returns:
            Tensor: The output tensor of adaptive pooling result. The data type is same
                      as input tensor.

        Raises:
            ValueError: 'pool_size' should be a integer or list or tuple with length as 1.

        Examples:
            .. code-block:: python

              # max adaptive pool1d
              # suppose input data in shape of [N, C, L], `output_size` is m or [m],
              # output shape is [N, C, m], adaptive pool divide L dimension
              # of input data into m grids averagely and performs poolings in each
              # grid to get output.
              # adaptive max pool performs calculations as follow:
              #
              #     for i in range(m):
              #         lstart = floor(i * L / m)
              #         lend = ceil((i + 1) * L / m)
              #         output[:, :, i] = max(input[:, :, lstart: lend])
              #
              import paddle
              import paddle.nn.functional as F
              paddle.disable_static()

              data = paddle.to_tensor(np.random.uniform(-1, 1, [1, 3, 32]).astype(np.float32))
              pool_out = F.adaptive_max_pool1d(data, output_size=16)
              # pool_out shape: [1, 3, 16])

              pool_out, indices = F.adaptive_max_pool1d(data, output_size=16, return_indices=True)
              # pool_out shape: [1, 3, 16] indices  shape: [1, 3, 16]


    """
    pool_type = 'max'
    check_variable_and_dtype(x, 'input', ['float32', 'float64'],
                             'adaptive_max_pool1d')
    check_input(x, 3)
    check_type(output_size, 'pool_size', (int), 'adaptive_max_pool1d')
    check_type(return_indices, 'return_indices', bool, 'adaptive_max_pool1d')

    pool_size = [1] + utils.convert_to_list(output_size, 1, 'pool_size')

    l_type = 'max_pool2d_with_index'

    x = unsqueeze(x, [2])
    if in_dygraph_mode():
        pool_out = core.ops.max_pool2d_with_index(
            x, 'pooling_type', pool_type, 'ksize', pool_size, 'adaptive', True)
        return (squeeze(pool_out[0], [2]), squeeze(
            pool_out[1], [2])) if return_indices else squeeze(pool_out[0], [2])

    helper = LayerHelper(l_type, **locals())
    dtype = helper.input_dtype()
    pool_out = helper.create_variable_for_type_inference(dtype)

    mask = helper.create_variable_for_type_inference(dtype)
    outputs = {"Out": pool_out, "Mask": mask}

    helper.append_op(
        type=l_type,
        inputs={"X": x},
        outputs=outputs,
        attrs={
            "pooling_type": pool_type,
            "ksize": pool_size,
            "adaptive": True,
        })

    return (squeeze(pool_out, [2]),
            squeeze(mask, [2])) if return_indices else squeeze(pool_out, [2])
