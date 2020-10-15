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

# TODO: define the common classes to build a neural network
import paddle
from ...fluid.dygraph import BilinearTensorProduct  #DEFINE_ALIAS
from ...fluid.dygraph import Pool2D  #DEFINE_ALIAS
from ...fluid.dygraph import Flatten  #DEFINE_ALIAS
from ...fluid.dygraph import layers
from .. import functional as F
from ...fluid.framework import _dygraph_tracer

__all__ = [
    'BilinearTensorProduct',
    'Pool2D',
    'Embedding',
    'Linear',
    'Upsample',
    'Pad2D',
    'UpsamplingNearest2d',
    'UpsamplingBilinear2d',
    'ReflectionPad1d',
    'ReplicationPad1d',
    'ConstantPad1d',
    'ReflectionPad2d',
    'ReplicationPad2d',
    'ConstantPad2d',
    'ZeroPad2d',
    'ConstantPad3d',
    'ReplicationPad3d',
    'CosineSimilarity',
    'Dropout',
    'Dropout2d',
    'Dropout3d',
    'Bilinear',
    'AlphaDropout',
]


class Linear(layers.Layer):
    """

    Fully-connected linear transformation layer. For each input :math:`X` ,
    the equation is:

    .. math::

        Out = XW + b

    where :math:`W` is the weight and :math:`b` is the bias.

    Linear layer takes only one multi-dimensional tensor as input with the
    shape :math:`[batch\_size, *, in\_features]` , where :math:`*` means any
    number of additional dimensions. It multiplies input tensor with the weight
    (a 2-D tensor of shape :math:`[in\_features, out\_features]` ) and produces
    an output tensor of shape :math:`[batch\_size, *, out\_features]` .
    If :math:`bias\_attr` is not False, the bias (a 1-D tensor of
    shape :math:`[out\_features]` ) will be created and added to the output.

    Parameters:
        in_features (int): The number of input units.
        out_features (int): The number of output units.
        weight_attr (ParamAttr, optional): The attribute for the learnable
            weight of this layer. The default value is None and the weight will be
            initialized to zero. For detailed information, please refer to
            paddle.ParamAttr.
        bias_attr (ParamAttr|bool, optional): The attribute for the learnable bias
            of this layer. If it is set to False, no bias will be added to the output.
            If it is set to None or one kind of ParamAttr, a bias parameter will
            be created according to ParamAttr. For detailed information, please refer
            to paddle.ParamAttr. The default value is None and the bias will be
            initialized to zero.
        name (str, optional): Normally there is no need for user to set this parameter.
            For detailed information, please refer to :ref:`api_guide_Name` .

    Attribute:
        **weight** (Parameter): the learnable weight of this layer.

        **bias** (Parameter): the learnable bias of this layer.

    Shape:
        - input: Multi-dimentional tensor with shape :math:`[batch\_size, *, in\_features]` .
        - output: Multi-dimentional tensor with shape :math:`[batch\_size, *, out\_features]` .

    Examples:
        .. code-block:: python

          import paddle

          # Define the linear layer.
          weight_attr = paddle.ParamAttr(
              name="weight",
              initializer=paddle.nn.initializer.Constant(value=0.5))
          bias_attr = paddle.ParamAttr(
              name="bias",
              initializer=paddle.nn.initializer.Constant(value=1.0))
          linear = paddle.nn.Linear(2, 4, weight_attr=weight_attr, bias_attr=bias_attr)
          # linear.weight: [[0.5 0.5 0.5 0.5]
          #                 [0.5 0.5 0.5 0.5]]
          # linear.bias: [1. 1. 1. 1.]

          x = paddle.randn((3, 2), dtype="float32")
          # x: [[-0.32342386 -1.200079  ]
          #     [ 0.7979031  -0.90978354]
          #     [ 0.40597573  1.8095392 ]]
          y = linear(x)
          # y: [[0.23824859 0.23824859 0.23824859 0.23824859]
          #     [0.9440598  0.9440598  0.9440598  0.9440598 ]
          #     [2.1077576  2.1077576  2.1077576  2.1077576 ]]
    """

    def __init__(self,
                 in_features,
                 out_features,
                 weight_attr=None,
                 bias_attr=None,
                 name=None):
        super(Linear, self).__init__()
        self._dtype = self._helper.get_default_dtype()
        self._weight_attr = weight_attr
        self._bias_attr = bias_attr
        self.name = name
        self.weight = self.create_parameter(
            shape=[in_features, out_features],
            attr=self._weight_attr,
            dtype=self._dtype,
            is_bias=False)
        self.bias = self.create_parameter(
            shape=[out_features],
            attr=self._bias_attr,
            dtype=self._dtype,
            is_bias=True)
        self.name = name

    def forward(self, input):
        out = F.linear(
            x=input, weight=self.weight, bias=self.bias, name=self.name)
        return out


class Upsample(layers.Layer):
    """
    This op resizes a batch of images.

    The input must be a 3-D Tensor of the shape (num_batches, channels, in_w)
    or 4-D (num_batches, channels, in_h, in_w), or a 5-D Tensor of the shape
    (num_batches, channels, in_d, in_h, in_w) or (num_batches, in_d, in_h, in_w, channels),
    Where in_w is width of the input tensor, in_h is the height of the input tensor,
    in_d is the depth of the intput tensor.
    and the resizing only applies on the three dimensions(depth, height and width).

    Supporting resample methods:
        'linear' : Linear interpolation
        'bilinear' : Bilinear interpolation
        'trilinear' : Trilinear interpolation
        'nearest' : Nearest neighbor interpolation
        'bicubic' : Bicubic interpolation

    Linear interpolation is the method of using a line connecting two known quantities
    to determine the value of an unknown quantity between the two known quantities.

    Nearest neighbor interpolation is to perform nearest neighbor interpolation
    in both the 3rd dimension(in height direction) and the 4th dimension(in width
    direction) on input tensor.

    Bilinear interpolation is an extension of linear interpolation for
    interpolating functions of two variables (e.g. H-direction and
    W-direction in this op) on a rectilinear 2D grid. The key idea is
    to perform linear interpolation first in one direction, and then
    again in the other direction.

    Bicubic interpolation is an extension of cubic interpolation for interpolating
    data points on a two-dimensional regular grid. The interpolated surface is
    smoother than corresponding surfaces obtained by bilinear interpolation or
    nearest-neighbor interpolation.

    Trilinear interpolation is an extension of linear interpolation for
    interpolating functions of three variables (e.g. D-direction,
    H-direction and W-direction in this op) on a rectilinear 3D grid.
    The linear interpolation is performed on three directions.
    align_corners and align_mode are optional parameters,the calculation method
    of interpolation can be selected by them.

    Area interpolation is to perform area interpolation
    in both the 3rd dimension(in height direction) , the 4th dimension(in width
    direction) and the 5th dimension(in depth direction) on input tensor. Set to
    area will directly call `paddle.nn.functional.adaptive_avg_pool1d` or
    `paddle.nn.functional.adaptive_avg_pool2d` or `paddle.nn.functional.adaptive_avg_pool3d`.

    Example:

    .. code-block:: text

        For scale_factor:
            if align_corners = True && out_size > 1 :
              scale_factor = (in_size-1.0)/(out_size-1.0)
            else:
              scale_factor = float(in_size/out_size)

        Linear interpolation:
            if:
                align_corners = False , align_mode = 0
                input : (N,C,W_in)
                output: (N,C,W_out) where:
                W_out = (W_{in}+0.5) * scale_{factor} - 0.5
            else:
                input : (N,C,W_in)
                output: (N,C,W_out) where:
                W_out = W_{in} * scale_{factor}

        Nearest neighbor interpolation:
          if:
              align_corners = False
              input : (N,C,H_in,W_in)
              output: (N,C,H_out,W_out) where:
              H_out = floor (H_{in} * scale_{factor})
              W_out = floor (W_{in} * scale_{factor})
          else:
              align_corners = True
              input : (N,C,H_in,W_in)
              output: (N,C,H_out,W_out) where:
              H_out = round(H_{in} * scale_{factor})
              W_out = round(W_{in} * scale_{factor})

        Bilinear interpolation:
          if:
              align_corners = False , align_mode = 0

              input : (N,C,H_in,W_in)
              output: (N,C,H_out,W_out) where:
              H_out = (H_{in}+0.5) * scale_{factor} - 0.5
              W_out = (W_{in}+0.5) * scale_{factor} - 0.5
          else:

              input : (N,C,H_in,W_in)
              output: (N,C,H_out,W_out) where:
              H_out = H_{in} * scale_{factor}
              W_out = W_{in} * scale_{factor}

        Bicubic interpolation:
          if:
              align_corners = False
              input : (N,C,H_in,W_in)
              output: (N,C,H_out,W_out) where:
              H_out = (H_{in}+0.5) * scale_{factor} - 0.5
              W_out = (W_{in}+0.5) * scale_{factor} - 0.5

          else:
              input : (N,C,H_in,W_in)
              output: (N,C,H_out,W_out) where:
              H_out = H_{in} * scale_{factor}
              W_out = W_{in} * scale_{factor}

        Trilinear interpolation:
          if:
              align_corners = False , align_mode = 0
              input : (N,C,D_in,H_in,W_in)
              output: (N,C,D_out,H_out,W_out) where:
              D_out = (D_{in}+0.5) * scale_{factor} - 0.5
              H_out = (H_{in}+0.5) * scale_{factor} - 0.5
              W_out = (W_{in}+0.5) * scale_{factor} - 0.5
          else:
              input : (N,C,D_in,H_in,W_in)
              output: (N,C,D_out,H_out,W_out) where:
              D_out = D_{in} * scale_{factor}
              H_out = H_{in} * scale_{factor}
              W_out = W_{in} * scale_{factor}

    https://en.wikipedia.org/wiki/Linear_interpolation.
    For details of linear interpolation, please refer to Wikipedia:

    For details of nearest neighbor interpolation, please refer to Wikipedia:
    https://en.wikipedia.org/wiki/Nearest-neighbor_interpolation.

    For details of bilinear interpolation, please refer to Wikipedia:
    https://en.wikipedia.org/wiki/Bilinear_interpolation.

    For details of bicubic interpolation, please refer to Wikipedia:
    https://en.wikipedia.org/wiki/Bicubic_interpolation

    For details of trilinear interpolation, please refer to Wikipedia:
    https://en.wikipedia.org/wiki/Trilinear_interpolation.

    Parameters:
        x (Tensor): 3-D, 4-D or 5-D Tensor, its data type is float32, float64, or uint8,
                          its data format is specified by :attr:`data_format`.
        size (list|tuple|Tensor|None): Output shape of image resize
             layer, the shape is (out_w, ) when input is a 3-D Tensor, the shape is (out_h, out_w) 
             when input is a 4-D Tensor and is (out_d, out_h, out_w) when input is a 5-D Tensor. 
             Default: None. If a list, each element can be an integer or a Tensor Variable of shape: [1].
             If a Tensor Variable, its dimensions size should be a 1.
        scale_factor (float|Tensor|list|tuple|None): The multiplier for the input height or width. At
             least one of :attr:`size` or :attr:`scale_factor` must be set.
             And :attr:`size` has a higher priority than :attr:`scale_factor`. Has to match input size if it is either a list or a tuple or a Tensor.
             Default: None.
        mode (str): The resample method. It supports 'linear', 'nearst', 'bilinear',
                       'bicubic' and 'trilinear' currently. Default: 'nearest'
        align_corners(bool) :  An optional bool, If True, the centers of the 4 corner pixels of the
                               input and output tensors are aligned, preserving the values at the
                               corner pixels.
                               Default: False
        align_mode(int)  :  An optional for linear/bilinear/trilinear interpolation. Refer to the formula in the example above,
                            it can be \'0\' for src_idx = scale_factor*(dst_indx+0.5)-0.5 , can be \'1\' for
                            src_idx = scale_factor*dst_index.
        data_format (str, optional): Specify the data format of the input, and the data format of the output
            will be consistent with that of the input. An optional string from:`NCW`, `NWC`, `"NCHW"`, `"NHWC"`, `"NCDHW"`,
            `"NDHWC"`. The default is `"NCHW"`. When it is `"NCHW"`, the data is stored in the order of:
            `[batch_size, input_channels, input_height, input_width]`. When it is `"NCHW"`, the data is stored
            in the order of: `[batch_size, input_channels, input_depth, input_height, input_width]`.
        name(str, optional): The default value is None.
                             Normally there is no need for user to set this property.
                             For more information, please refer to :ref:`api_guide_Name`
    Returns:
        A 3-D Tensor of the shape (num_batches, channels, out_w) or (num_batches, out_w, channels),
        A 4-D Tensor of the shape (num_batches, channels, out_h, out_w) or (num_batches, out_h, out_w, channels),
        or 5-D Tensor of the shape (num_batches, channels, out_d, out_h, out_w) or (num_batches, out_d, out_h, out_w, channels).
    Raises:
        TypeError: size should be a list or tuple or Tensor.
        ValueError: The 'mode' of image_resize can only be 'linear', 'bilinear',
                    'trilinear', 'bicubic', or 'nearest' currently.
        ValueError: 'linear' only support 3-D tensor.
        ValueError: 'bilinear', 'bicubic' and 'nearest' only support 4-D tensor.
        ValueError: 'trilinear' only support 5-D tensor.
        ValueError: One of size and scale_factor must not be None.
        ValueError: size length should be 1 for input 3-D tensor.
        ValueError: size length should be 2 for input 4-D tensor.
        ValueError: size length should be 3 for input 5-D tensor.
        ValueError: scale_factor should be greater than zero.
        TypeError: align_corners should be a bool value
        ValueError: align_mode can only be '0' or '1'
        ValueError: data_format can only be 'NCW', 'NWC', 'NCHW', 'NHWC', 'NCDHW' or 'NDHWC'.

    Examples:
        .. code-block:: python
            import paddle
            import paddle.nn as nn
            import numpy as np
            paddle.disable_static()

            input_data = np.random.rand(2,3,6,10).astype("float32")
            upsample_out  = paddle.nn.Upsample(size=[12,12])

            input = paddle.to_tensor(input_data)
            output = upsample_out(x=input)
            print(output.shape)
            # [2L, 3L, 12L, 12L]

    """

    def __init__(self,
                 size=None,
                 scale_factor=None,
                 mode='nearest',
                 align_corners=False,
                 align_mode=0,
                 data_format='NCHW',
                 name=None):
        super(Upsample, self).__init__()
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode.lower()
        self.align_corners = align_corners
        self.align_mode = align_mode
        self.data_format = data_format
        self.name = name

    def forward(self, x):
        out = F.interpolate(
            x,
            size=self.size,
            scale_factor=self.scale_factor,
            mode=self.mode,
            align_corners=self.align_corners,
            align_mode=self.align_mode,
            data_format=self.data_format,
            name=self.name)

        return out


class UpsamplingNearest2d(layers.Layer):
    """
    This op upsamples a batch of images, using nearest neighbours' pixel values.
    The input must be a 4-D Tensor of the shape (num_batches, channels, in_h, in_w), 
    where in_w is width of the input tensor, in_h is the height of the input tensor.
    And the upsampling only applies on the two dimensions(height and width).

    Nearest neighbor interpolation is to perform nearest neighbor interpolation
    in both the 3rd dimension(in height direction) and the 4th dimension(in width
    direction) on input tensor.
    
    For details of nearest neighbor interpolation, please refer to Wikipedia:
    https://en.wikipedia.org/wiki/Nearest-neighbor_interpolation.
    
        x (Tensor): 4-D Tensor, its data type is float32, float64, or uint8,
                          its data format is specified by :attr:`data_format`.
        size (list|tuple|Tensor|None): Output shape of image resize
             layer, the shape is (out_h, out_w) when input is a 4-D Tensor. 
             Default: None. If a list, each element can be an integer or a Tensor Variable of shape: [1].
             If a Tensor Variable, its dimensions size should be a 1.
        scale_factor (float|int|list|tuple|Tensor|None): The multiplier for the input height or width. At
             least one of :attr:`size` or :attr:`scale_factor` must be set.
             And :attr:`size` has a higher priority than :attr:`scale_factor`.
             Has to match input size if it is either a list or a tuple or a Tensor.
             Default: None.
        data_format (str, optional): Specify the data format of the input, and the data format of the output
            will be consistent with that of the input. An optional string from:`NCW`, `NWC`, `"NCHW"`, `"NHWC"`, `"NCDHW"`,
            `"NDHWC"`. The default is `"NCHW"`. When it is `"NCHW"`, the data is stored in the order of:
            `[batch_size, input_channels, input_height, input_width]`. When it is `"NCHW"`, the data is stored
            in the order of: `[batch_size, input_channels, input_depth, input_height, input_width]`.
        name(str, optional): The default value is None.
                             Normally there is no need for user to set this property.
                             For more information, please refer to :ref:`api_guide_Name`
    Returns:
        A 4-D Tensor of the shape (num_batches, channels, out_h, out_w) or (num_batches, out_h, out_w, channels),
    Raises:
        TypeError: size should be a list or tuple or Tensor.
        ValueError: 'nearest' only support 4-D tensor.
        ValueError: One of size and scale_factor must not be None.
        ValueError: size length should be 2 for input 4-D tensor.
        ValueError: scale_factor should be greater than zero.
        ValueError: data_format can only be 'NCHW', 'NHWC'.
    Examples:
        .. code-block:: python

            import paddle
            import paddle.nn as nn
            import numpy as np
            paddle.disable_static()

            input_data = np.random.rand(2,3,6,10).astype("float32")
            upsample_out  = paddle.nn.UpsamplingNearest2d(size=[12,12])

            input = paddle.to_tensor(input_data)
            output = upsample_out(x=input)
            print(output.shape)
            # [2L, 3L, 12L, 12L]

    """

    def __init__(self,
                 size=None,
                 scale_factor=None,
                 data_format='NCHW',
                 name=None):
        super(UpsamplingNearest2d, self).__init__()
        self.size = size
        self.scale_factor = scale_factor
        self.data_format = data_format
        self.name = name

    def forward(self, x):
        out = F.interpolate(
            x,
            size=self.size,
            scale_factor=self.scale_factor,
            mode='nearest',
            align_corners=False,
            align_mode=0,
            data_format=self.data_format,
            name=self.name)

        return out


class UpsamplingBilinear2d(layers.Layer):
    """
    This op upsamples a batch of images, using bilinear' pixel values.
    The input must be a 4-D Tensor of the shape (num_batches, channels, in_h, in_w), 
    where in_w is width of the input tensor, in_h is the height of the input tensor.
    And the upsampling only applies on the two dimensions(height and width).

    Bilinear interpolation is an extension of linear interpolation for
    interpolating functions of two variables (e.g. H-direction and
    W-direction in this op) on a rectilinear 2D grid. The key idea is
    to perform linear interpolation first in one direction, and then
    again in the other direction.
    
    For details of bilinear interpolation, please refer to Wikipedia:
    https://en.wikipedia.org/wiki/Bilinear_interpolation.
    
        x (Tensor): 4-D Tensor, its data type is float32, float64, or uint8,
                          its data format is specified by :attr:`data_format`.
        size (list|tuple|Tensor|None): Output shape of image resize
             layer, the shape is (out_h, out_w) when input is a 4-D Tensor. 
             Default: None. If a list, each element can be an integer or a Tensor Variable of shape: [1].
             If a Tensor Variable, its dimensions size should be a 1.
        scale_factor (float|int|list|tuple|Tensor|None): The multiplier for the input height or width. At
             least one of :attr:`size` or :attr:`scale_factor` must be set.
             And :attr:`size` has a higher priority than :attr:`scale_factor`.
             Has to match input size if it is either a list or a tuple or a Tensor.
             Default: None.
        data_format (str, optional): Specify the data format of the input, and the data format of the output
            will be consistent with that of the input. An optional string from:`NCW`, `NWC`, `"NCHW"`, `"NHWC"`, `"NCDHW"`,
            `"NDHWC"`. The default is `"NCHW"`. When it is `"NCHW"`, the data is stored in the order of:
            `[batch_size, input_channels, input_height, input_width]`. When it is `"NCHW"`, the data is stored
            in the order of: `[batch_size, input_channels, input_depth, input_height, input_width]`.
        name(str, optional): The default value is None.
                             Normally there is no need for user to set this property.
                             For more information, please refer to :ref:`api_guide_Name`
    Returns:
        A 4-D Tensor of the shape (num_batches, channels, out_h, out_w) or (num_batches, out_h, out_w, channels),
    Raises:
        TypeError: size should be a list or tuple or Tensor.
        ValueError: 'bilinear' only support 4-D tensor.
        ValueError: One of size and scale_factor must not be None.
        ValueError: size length should be 2 for input 4-D tensor.
        ValueError: scale_factor should be greater than zero.
        ValueError: data_format can only be 'NCHW', 'NHWC'.
    Examples:
        .. code-block:: python
            import paddle
            import paddle.nn as nn
            import numpy as np
            paddle.disable_static()

            input_data = np.random.rand(2,3,6,10).astype("float32")
            upsample_out  = paddle.nn.UpsamplingBilinear2d(size=[12,12])

            input = paddle.to_tensor(input_data)
            output = upsample_out(x=input)
            print(output.shape)
            # [2L, 3L, 12L, 12L]
    """

    def __init__(self,
                 size=None,
                 scale_factor=None,
                 data_format='NCHW',
                 name=None):
        super(UpsamplingBilinear2d, self).__init__()
        self.size = size
        self.scale_factor = scale_factor
        self.data_format = data_format
        self.name = name

    def forward(self, x):
        out = F.interpolate(
            x,
            size=self.size,
            scale_factor=self.scale_factor,
            mode='bilinear',
            align_corners=True,
            align_mode=0,
            data_format=self.data_format,
            name=self.name)

        return out


class Pad2D(layers.Layer):
    """
    This interface is used to construct a callable object of the ``Pad2D``  class.
    The Pad2D layer pads the input tensor boundaries according to 'paddings' and 'mode'.
    If mode is 'reflect', paddings[0] and paddings[1] must be no greater
    than height-1. And the width dimension has the same condition.
    Parameters:
        paddings (int | List[int32]): The padding size. If padding is a int, uses the same
            padding in all boundaries, if padding is a List, it must contain four integers,
            (padding_top, padding_bottom, padding_left, padding_right).
            Default is [0, 0, 0, 0].
        mode (str): Three modes: 'constant' (default), 'reflect', 'edge' .
        	When in 'constant' mode, this op uses a constant value to pad the input tensor.
        	When in 'reflect' mode, uses reflection of the input boundaries to pad the input tensor.
        	When in 'edge' mode, uses input boundaries to pad the input tensor.
        	Default is 'constant'
        pad_value (float32): The value to fill the padded areas in 'constant' mode . Default is 0.0
        data_format (str): An string from: "NHWC", "NCHW". Specify the data format of
                           the input data.
                           Default is  "NCHW"
    Returns:
        None
    Examples:
        .. code-block:: text
            Input = [[[[1., 2., 3.],
                       [4., 5., 6.]]]]
            Case 0:
                paddings = [0, 1, 2, 3],
                mode = 'constant'
                pad_value = 0
                Out = [[[[0., 0., 1., 2., 3., 0., 0., 0.],
                         [0., 0., 4., 5., 6., 0., 0., 0.],
                         [0., 0., 0., 0., 0., 0., 0., 0.]]]]
            Case 1:
                paddings = [0, 1, 2, 1],
                mode = 'reflect'
                Out = [[[[3., 2., 1., 2., 3., 2.],
                         [6., 5., 4., 5., 6., 5.],
                         [3., 2., 1., 2., 3., 2.]]]]
            Case 2:
                paddings = [0, 1, 2, 1],
                mode = 'edge'
                Out = [[[[1., 1., 1., 2., 3., 3.],
                         [4., 4., 4., 5., 6., 6.],
                         [4., 4., 4., 5., 6., 6.]]]]
    Code Examples:
        .. code-block:: python
            import paddle.fluid as fluid
            import paddle.nn as nn
            import numpy as np
            data = np.ones((2, 2, 2, 2)).astype('float32')
            my_pad = nn.layer.Pad2D(paddings=[1, 1, 1, 1])
            with fluid.dygraph.guard():
                data = fluid.dygraph.to_variable(data)
                result = my_pad(data)
    """

    def __init__(self,
                 paddings=0,
                 mode='constant',
                 pad_value=0.0,
                 data_format="NCHW"):
        super(Pad2D, self).__init__()
        self._mode = mode
        self._pad_value = pad_value
        self._data_format = data_format
        self._paddings = [paddings] * 4 if isinstance(paddings,
                                                      int) else paddings

    def forward(self, input):
        return paddle.fluid.layers.pad2d(
            input,
            paddings=self._paddings,
            mode=self._mode,
            pad_value=self._pad_value,
            data_format=self._data_format)


class Bilinear(layers.Layer):
    """

    This layer performs bilinear on two inputs.

    .. math::

      out_{i} = x1 * W_{i} * {x2^\mathrm{T}}, i=0,1,...,size-1

      out = out + b

    In this formula:
     - :math:`x1`: the first input contains in1_features elements, shape is [batch_size, in1_features].
     - :math:`x2`: the second input contains in2_features elements, shape is [batch_size, in2_features].
     - :math:`W_{i}`: the i-th learned weight, shape is [in1_features, in2_features], and learned weight's shape is [out_features, in1_features, in2_features].
     - :math:`out_{i}`: the i-th element of out, shape is [batch_size, out_features].
     - :math:`b`: the learned bias, shape is [1, out_features].
     - :math:`x2^\mathrm{T}`: the transpose of :math:`x2`.

    Parameters:
       in1_features (int): The dimension of each first input(`x1`).
       in2_features (int): The dimension of each second input(`x2`).
       out_features (int): The dimension of output of this layer.
       weight_attr (ParamAttr, optional): The parameter attribute for the learnable w, parameters/weights of
       this layer. The default value is None.
       bias_attr (ParamAttr, optional): The parameter attribute for the bias
           of this layer. If it is set to False, no bias will be added to the output units.
           If it is set to None, the bias is initialized zero. The default value is None.
       name (str, optional): The default value is None. Normally there is no need for user
           to set this property. For more information, please refer to :ref:`api_guide_Name`. Default: None.

    Attribute:
        **weight** (Parameter): the learnable weights of this layer.

        **bias** (Parameter): the learnable bias of this layer.

    Returns:
       Tensor: A 2-D Tensor of shape [batch_size, out_features].

    Examples:
       .. code-block:: python

        import paddle
        import numpy

        paddle.disable_static()
        layer1 = numpy.random.random((5, 5)).astype('float32')
        layer2 = numpy.random.random((5, 4)).astype('float32')
        bilinear = paddle.nn.Bilinear(
            in1_features=5, in2_features=4, out_features=1000)
        result = bilinear(paddle.to_tensor(layer1),
                        paddle.to_tensor(layer2))     # result shape [5, 1000]

    """

    def __init__(self,
                 in1_features,
                 in2_features,
                 out_features,
                 weight_attr=None,
                 bias_attr=None,
                 name=None):
        super(Bilinear, self).__init__()
        self._weight_attr = weight_attr
        self._bias_attr = bias_attr
        self._name = name
        self._in1_features = in1_features
        self._in2_features = in2_features
        self._out_features = out_features
        self._dtype = self._helper.get_default_dtype()

        weight_shape = [
            self._out_features, self._in1_features, self._in2_features
        ]
        self.weight = self.create_parameter(
            attr=self._weight_attr,
            shape=weight_shape,
            dtype=self._dtype,
            is_bias=False)
        bias_shape = [1, self._out_features]
        self.bias = self.create_parameter(
            attr=self._bias_attr,
            shape=bias_shape,
            dtype=self._dtype,
            is_bias=True)

    def forward(self, x1, x2):
        return F.bilinear(x1, x2, self.weight, self.bias, self._name)


class Dropout(layers.Layer):
    """
    Dropout is a regularization technique for reducing overfitting by preventing
    neuron co-adaption during training as described in the paper:
    `Improving neural networks by preventing co-adaptation of feature detectors <https://arxiv.org/abs/1207.0580>`_
    The dropout operator randomly sets the outputs of some units to zero, while upscale others
    according to the given dropout probability.

    See ``paddle.nn.functional.dropout`` for more details.

    In dygraph mode, please use ``eval()`` to switch to evaluation mode, where dropout is disabled.

    Parameters:
        p (float | int): Probability of setting units to zero. Default: 0.5
        axis (int | list): The axis along which the dropout is performed. Default None.
        mode(str, optional): ['upscale_in_train'(default) | 'downscale_in_infer']

                               1. upscale_in_train(default), upscale the output at training time

                                  - train: out = input * mask / ( 1.0 - p )
                                  - inference: out = input

                               2. downscale_in_infer, downscale the output at inference

                                  - train: out = input * mask
                                  - inference: out = input * (1.0 - p)
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Shape:
        - input: N-D tensor.
        - output: N-D tensor, the same shape as input.

    Examples:
        .. code-block:: python
            import paddle
            import numpy as np

            paddle.disable_static()
            x = np.array([[1,2,3], [4,5,6]]).astype('float32')
            x = paddle.to_tensor(x)
            m = paddle.nn.Dropout(p=0.5)
            y_train = m(x)
            m.eval()  # switch the model to test phase
            y_test = m(x)
            print(x.numpy())
            print(y_train.numpy())
            print(y_test.numpy())
   """

    def __init__(self, p=0.5, axis=None, mode="upscale_in_train", name=None):
        super(Dropout, self).__init__()

        self.p = p
        self.axis = axis
        self.mode = mode
        self.name = name

    def forward(self, input):
        out = F.dropout(
            input,
            p=self.p,
            axis=self.axis,
            training=self.training,
            mode=self.mode,
            name=self.name)
        return out


class Dropout2d(layers.Layer):
    """
    Randomly zero out entire channels (in the batched input 4d tensor with the shape `NCHW` ,
    a channel is a 2D feature map with the shape `HW`). Each channel will be zeroed out independently
    on every forward call with probability `p` using samples from a Bernoulli distribution.
    Dropout2d will help promote independence between feature maps as described in the paper:
    `Efficient Object Localization Using Convolutional Networks <https://arxiv.org/abs/1411.4280>`_

    See ``paddle.nn.functional.dropout2d`` for more details.

    In dygraph mode, please use ``eval()`` to switch to evaluation mode, where dropout is disabled.

    Parameters:
        p (float, optional): Probability of setting units to zero. Default: 0.5
        data_format (str, optional): Specify the data format of the input, and the data format of the output
                                     will be consistent with that of the input. An optional string from:
                                    `NCHW`, `NHWC`. The default is `NCHW`. When it is `NCHW`, the data is
                                     stored in the order of: [batch_size, input_channels, input_height, input_width].
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Shape:
        - input: 4-D tensor.
        - output: 4-D tensor, the same shape as input.

    Examples:
        .. code-block:: python
            import paddle
            import numpy as np

            paddle.disable_static()
            x = np.random.random(size=(2, 3, 4, 5)).astype('float32')
            x = paddle.to_tensor(x)
            m = paddle.nn.Dropout2d(p=0.5)
            y_train = m(x)
            m.eval()  # switch the model to test phase
            y_test = m(x)
            print(x.numpy())
            print(y_train.numpy())
            print(y_test.numpy())
   """

    def __init__(self, p=0.5, data_format='NCHW', name=None):
        super(Dropout2d, self).__init__()

        self.p = p
        self.data_format = data_format
        self.name = name

    def forward(self, input):
        out = F.dropout2d(
            input,
            p=self.p,
            training=self.training,
            data_format=self.data_format,
            name=self.name)
        return out


class Dropout3d(layers.Layer):
    """
    Randomly zero out entire channels (in the batched input 5d tensor with the shape `NCDHW` ,
    a channel is a 3D feature map with the shape `DHW` ). Each channel will be zeroed out independently
    on every forward call with probability `p` using samples from a Bernoulli distribution.
    Dropout3d will help promote independence between feature maps as described in the paper:
    `Efficient Object Localization Using Convolutional Networks <https://arxiv.org/abs/1411.4280>`_

    See ``paddle.nn.functional.dropout3d`` for more details.

    In dygraph mode, please use ``eval()`` to switch to evaluation mode, where dropout is disabled.

    Parameters:
        p (float | int): Probability of setting units to zero. Default: 0.5
        data_format (str, optional): Specify the data format of the input, and the data format of the output
                                     will be consistent with that of the input. An optional string from:
                                    `NCDHW`, `NDHWC`. The default is `NCDHW`. When it is `NCDHW`, the data is
                                     stored in the order of: [batch_size, input_channels, input_depth, input_height, input_width].
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Shape:
        - input: 5-D tensor.
        - output: 5-D tensor, the same shape as input.

    Examples:
        .. code-block:: python
            import paddle
            import numpy as np

            paddle.disable_static()
            x = np.random.random(size=(2, 3, 4, 5, 6)).astype('float32')
            x = paddle.to_tensor(x)
            m = paddle.nn.Dropout3d(p=0.5)
            y_train = m(x)
            m.eval()  # switch the model to test phase
            y_test = m(x)
            print(x.numpy())
            print(y_train.numpy())
            print(y_test.numpy())
   """

    def __init__(self, p=0.5, data_format='NCDHW', name=None):
        super(Dropout3d, self).__init__()

        self.p = p
        self.data_format = data_format
        self.name = name

    def forward(self, input):
        out = F.dropout3d(
            input,
            p=self.p,
            training=self.training,
            data_format=self.data_format,
            name=self.name)
        return out


class AlphaDropout(layers.Layer):
    """
    Alpha Dropout is a type of Dropout that maintains the self-normalizing property. For an input with
    zero mean and unit standard deviation, the output of Alpha Dropout maintains the original mean and
    standard deviation of the input. Alpha Dropout fits well to SELU activate function by randomly setting
    activations to the negative saturation value.

    For more information, please refer to:
    `Self-Normalizing Neural Networks <https://arxiv.org/abs/1706.02515>`_

    In dygraph mode, please use ``eval()`` to switch to evaluation mode, where dropout is disabled.

    Parameters:
        p (float | int): Probability of setting units to zero. Default: 0.5
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Shape:
        - input: N-D tensor.
        - output: N-D tensor, the same shape as input.

    Examples:
        .. code-block:: python
            import paddle
            import numpy as np

            paddle.disable_static()
            x = np.array([[-1, 1], [-1, 1]]).astype('float32')
            x = paddle.to_tensor(x)
            m = paddle.nn.AlphaDropout(p=0.5)
            y_train = m(x)
            m.eval()  # switch the model to test phase
            y_test = m(x)
            print(x.numpy())
            print(y_train.numpy())
            # [[-0.10721093, 1.6655989 ], [-0.7791938, -0.7791938]] (randomly)
            print(y_test.numpy())
   """

    def __init__(self, p=0.5, name=None):
        super(AlphaDropout, self).__init__()
        self.p = p
        self.name = name

    def forward(self, input):
        out = F.alpha_dropout(
            input, p=self.p, training=self.training, name=self.name)
        return out


class ReflectionPad1d(layers.Layer):
    """
    This interface is used to construct a callable object of the ``ReflectionPad1d`` class.
    Uses reflection of the input boundaries to pad the input tensor.

    Parameters:
        padding (Tensor | List[int32]): The padding size with data type int32. [len(padding)/2] dimensions
            of input will be padded. The pad has the form (pad_left, pad_right).
        data_format (str): An string from: "NCL", "NLC". Specify the data format of the input data.
           Default is  "NCL"
        name (str, optional) : The default value is None.  Normally there is no need for
            user to set this property.  For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        None

    Examples:
        .. code-block:: text

            x = [[[1., 2., 3.],
                  [4., 5., 6.]]]
            padding = [1, 2],
            Out = [[[2. 1. 2. 3. 2. 1.]
                    [5. 4. 5. 6. 5. 4.]]]

    Code Examples:
        .. code-block:: python

            import paddle
            import paddle.nn as nn
            import numpy as np
            paddle.disable_static()

            input_shape = (1, 2, 3)
            pad = [1, 2]
            data = np.arange(np.prod(input_shape), dtype=np.float32).reshape(input_shape) + 1
            my_pad = nn.ReflectionPad1d(padding=pad)
            data = paddle.to_tensor(data)
            result = my_pad(data)
            print(result.numpy())
            # [[[2. 1. 2. 3. 2. 1.]
            #   [5. 4. 5. 6. 5. 4.]]]
    """

    def __init__(self, padding, data_format="NCL", name=None):
        super(ReflectionPad1d, self).__init__()
        self._mode = "reflect"
        self._data_format = data_format
        self._pad = padding
        self._name = name

    def forward(self, x):
        return F.pad(x,
                     pad=self._pad,
                     mode=self._mode,
                     data_format=self._data_format,
                     name=self._name)


class ReplicationPad1d(layers.Layer):
    """
    This interface is used to construct a callable object of the ``ReplicationPad1d`` class.
    Uses input boundaries to pad the input tensor.

    Parameters:
        padding (Tensor | List[int32]): The padding size with data type int32. [len(padding)/2] dimensions
            of input will be padded. The pad has the form (pad_left, pad_right).
        data_format (str): An string from: "NCL", "NLC". Specify the data format of the input data.
           Default is  "NCL"
        name (str, optional) : The default value is None.  Normally there is no need for
            user to set this property.  For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        None

    Examples:
        .. code-block:: text

            x = [[[1., 2., 3.],
                  [4., 5., 6.]]]
            padding = [1, 2],
            Out = [[[2. 1. 2. 3. 2. 1.]
                    [5. 4. 5. 6. 5. 4.]]]

    Code Examples:
        .. code-block:: python

            import paddle
            import paddle.nn as nn
            import numpy as np
            paddle.disable_static()

            input_shape = (1, 2, 3)
            pad = [1, 2]
            data = np.arange(np.prod(input_shape), dtype=np.float32).reshape(input_shape) + 1
            my_pad = nn.ReplicationPad1d(padding=pad)
            data = paddle.to_tensor(data)
            result = my_pad(data)
            print(result.numpy())
            # [[[1. 1. 2. 3. 3. 3.]
            #   [1. 4. 5. 6. 6. 6.]]]
    """

    def __init__(self, padding, data_format="NCL", name=None):
        super(ReplicationPad1d, self).__init__()
        self._mode = "replicate"
        self._data_format = data_format
        self._pad = padding
        self._name = name

    def forward(self, x):
        return F.pad(x,
                     pad=self._pad,
                     mode=self._mode,
                     data_format=self._data_format,
                     name=self._name)


class ConstantPad1d(layers.Layer):
    """
    This interface is used to construct a callable object of the ``ConstantPad1d`` class.
    Uses a constant value to pad the input tensor.

    Parameters:
        padding (Tensor | List[int32]): The padding size with data type int32. [len(padding)/2] dimensions
            of input will be padded. The pad has the form (pad_left, pad_right).
        value (float32): The value to fill the padded areas. Default is 0.0
        data_format (str): An string from: "NCL", "NLC". Specify the data format of the input data.
           Default is  "NCL"
        name (str, optional) : The default value is None.  Normally there is no need for
            user to set this property.  For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        None

    Examples:
        .. code-block:: text

            x = [[[1., 2., 3.],
                  [4., 5., 6.]]]
            padding = [1, 2],
            value = 0.0
            Out = [[[0. 1. 2. 3. 0. 0.]
                    [0. 4. 5. 6. 0. 0.]]]

    Code Examples:
        .. code-block:: python

            import paddle
            import paddle.nn as nn
            import numpy as np
            paddle.disable_static()

            input_shape = (1, 2, 3)
            pad = [1, 2]
            data = np.arange(np.prod(input_shape), dtype=np.float32).reshape(input_shape) + 1
            my_pad = nn.ConstantPad1d(padding=pad)
            data = paddle.to_tensor(data)
            result = my_pad(data)
            print(result.numpy())
            # [[[0. 1. 2. 3. 0. 0.]
            #   [0. 4. 5. 6. 0. 0.]]]
    """

    def __init__(self, padding, value=0.0, data_format="NCL", name=None):
        super(ConstantPad1d, self).__init__()
        self._mode = "constant"
        self._data_format = data_format
        self._pad = padding
        self._value = value
        self._name = name

    def forward(self, x):
        return F.pad(x,
                     pad=self._pad,
                     mode=self._mode,
                     value=self._value,
                     data_format=self._data_format,
                     name=self._name)


class ConstantPad2d(layers.Layer):
    """
    This interface is used to construct a callable object of the ``ConstantPad2d`` class.
    Uses a constant value to pad the input tensor.

    Parameters:
        padding (Tensor | List[int32]): The padding size with data type int32. [len(padding)/2] dimensions
            of input will be padded. The pad has the form (pad_left, pad_right, pad_top, pad_bottom).
        value (float32): The value to fill the padded areas. Default is 0.0
        data_format (str): An string from: "NCHW", "NHWC". Specify the data format of the input data.
           Default is  "NCHW"
        name (str, optional) : The default value is None.  Normally there is no need for
            user to set this property.  For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        None

    Examples:
        .. code-block:: text

            x = [[[[1., 2., 3.],
                   [4., 5., 6.]]]]
            padding = [1, 1, 0, 0]
            value = 0.0
            Out = [[[[0. 1. 2. 3. 0.]
                     [0. 4. 5. 6. 0.]]]]

    Code Examples:
        .. code-block:: python

            import paddle
            import paddle.nn as nn
            import numpy as np
            paddle.disable_static()

            input_shape = (1, 1, 2, 3)
            pad = [1, 0, 1, 2]
            data = np.arange(np.prod(input_shape), dtype=np.float32).reshape(input_shape) + 1
            my_pad = nn.ConstantPad2d(padding=pad)
            data = paddle.to_tensor(data)
            result = my_pad(data)
            print(result.numpy())
            # [[[[0. 0. 0. 0.]
            #    [0. 1. 2. 3.]
            #    [0. 4. 5. 6.]
            #    [0. 0. 0. 0.]
            #    [0. 0. 0. 0.]]]]
    """

    def __init__(self, padding, value=0.0, data_format="NCHW", name=None):
        super(ConstantPad2d, self).__init__()
        self._mode = "constant"
        self._data_format = data_format
        self._pad = padding
        self._value = value
        self._name = name

    def forward(self, x):
        return F.pad(x,
                     pad=self._pad,
                     mode=self._mode,
                     value=self._value,
                     data_format=self._data_format,
                     name=self._name)


class ZeroPad2d(layers.Layer):
    """
    This interface is used to construct a callable object of the ``ZeroPad2d`` class.
    Uses 0 to pad the input tensor.

    Parameters:
        padding (Variable | List[int32]): The padding size with data type int32. [len(padding)/2] dimensions
            of input will be padded. The pad has the form (pad_left, pad_right, pad_top, pad_bottom).
        data_format (str): An string from: "NCHW", "NHWC". Specify the data format of the input data.
           Default is  "NCHW"
        name (str, optional) : The default value is None.  Normally there is no need for
            user to set this property.  For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        None

    Examples:
        .. code-block:: text

            x = [[[[1., 2., 3.],
                   [4., 5., 6.]]]]
            padding = [1, 1, 0, 0]
            Out = [[[[0. 1. 2. 3. 0.]
                     [0. 4. 5. 6. 0.]]]]

    Code Examples:
        .. code-block:: python

            import paddle
            import paddle.nn as nn
            import numpy as np
            paddle.disable_static()

            input_shape = (1, 1, 2, 3)
            pad = [1, 0, 1, 2]
            data = np.arange(np.prod(input_shape), dtype=np.float32).reshape(input_shape) + 1
            my_pad = nn.ZeroPad2d(padding=pad)
            data = paddle.to_tensor(data)
            result = my_pad(data)
            print(result.numpy())
            # [[[[0. 0. 0. 0.]
            #    [0. 1. 2. 3.]
            #    [0. 4. 5. 6.]
            #    [0. 0. 0. 0.]
            #    [0. 0. 0. 0.]]]]
    """

    def __init__(self, padding, data_format="NCHW", name=None):
        super(ZeroPad2d, self).__init__()
        self._mode = "constant"
        self._data_format = data_format
        self._pad = padding
        self._name = name

    def forward(self, x):
        return F.pad(x,
                     pad=self._pad,
                     mode=self._mode,
                     data_format=self._data_format,
                     name=self._name)


class ReplicationPad2d(layers.Layer):
    """
    This interface is used to construct a callable object of the ``ReplicationPad2d`` class.
    Uses input boundaries to pad the input tensor.

    Parameters:
        padding (Tensor | List[int32]): The padding size with data type int32. [len(padding)/2] dimensions
            of input will be padded. The pad has the form (pad_left, pad_right, pad_top, pad_bottom).
        data_format (str): An string from: "NCHW", "NHWC". Specify the data format of the input data.
           Default is  "NCHW"
        name (str, optional) : The default value is None.  Normally there is no need for
            user to set this property.  For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        None

    Examples:
        .. code-block:: text

            x = [[[[1., 2., 3.],
                   [4., 5., 6.]]]]
            padding = [1, 1, 0, 0]
            Out = [[[[1. 1. 2. 3. 3.]
                     [4. 4. 5. 6. 6.]]]]

    Code Examples:
        .. code-block:: python

            import paddle
            import paddle.nn as nn
            import numpy as np
            paddle.disable_static()

            input_shape = (1, 1, 2, 3)
            pad = [1, 0, 1, 2]
            data = np.arange(np.prod(input_shape), dtype=np.float32).reshape(input_shape) + 1
            my_pad = nn.ReplicationPad2d(padding=pad)
            data = paddle.to_tensor(data)
            result = my_pad(data)
            print(result.numpy())
            # [[[[1. 1. 2. 3.]
            #    [1. 1. 2. 3.]
            #    [4. 4. 5. 6.]
            #    [4. 4. 5. 6.]
            #    [4. 4. 5. 6.]]]]
    """

    def __init__(self, padding, data_format="NCHW", name=None):
        super(ReplicationPad2d, self).__init__()
        self._mode = "replicate"
        self._data_format = data_format
        self._pad = padding
        self._name = name

    def forward(self, x):
        return F.pad(x,
                     pad=self._pad,
                     mode=self._mode,
                     data_format=self._data_format,
                     name=self._name)


class ReflectionPad2d(layers.Layer):
    """
    This interface is used to construct a callable object of the ``ReflectionPad2d`` class.
    Uses reflection of the input boundaries to pad the input tensor.

    Parameters:
        padding (Variable | List[int32]): The padding size with data type int32. [len(padding)/2] dimensions
            of input will be padded. The pad has the form (pad_left, pad_right, pad_top, pad_bottom).
        data_format (str): An string from: "NCHW", "NHWC". Specify the data format of the input data.
           Default is  "NCHW"
        name (str, optional) : The default value is None.  Normally there is no need for
            user to set this property.  For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        None

    Examples:
        .. code-block:: text

            x = [[[[1., 2., 3.],
                   [4., 5., 6.]]]]
            padding = [1, 1, 0, 0]
            Out = [[[[2. 1. 2. 3. 2.]
                     [5. 4. 5. 6. 5.]]]]

    Code Examples:
        .. code-block:: python

            import paddle
            import paddle.nn as nn
            import numpy as np
            paddle.disable_static()

            input_shape = (1, 1, 4, 3)
            pad = [1, 0, 1, 2]
            data = np.arange(np.prod(input_shape), dtype=np.float32).reshape(input_shape) + 1
            my_pad = nn.ReflectionPad2d(padding=pad)
            data = paddle.to_tensor(data)
            result = my_pad(data)
            print(result.numpy())
            # [[[[ 5.  4.  5.  6.]
            #    [ 2.  1.  2.  3.]
            #    [ 5.  4.  5.  6.]
            #    [ 8.  7.  8.  9.]
            #    [11. 10. 11. 12.]
            #    [ 8.  7.  8.  9.]
            #    [ 5.  4.  5.  6.]]]]
    """

    def __init__(self, padding, data_format="NCHW", name=None):
        super(ReflectionPad2d, self).__init__()
        self._mode = "reflect"
        self._data_format = data_format
        self._pad = padding
        self._name = name

    def forward(self, x):
        return F.pad(x,
                     pad=self._pad,
                     mode=self._mode,
                     data_format=self._data_format,
                     name=self._name)


class ConstantPad3d(layers.Layer):
    """
    This interface is used to construct a callable object of the ``ConstantPad3d`` class.
    Uses a constant value to pad the input tensor.

    Parameters:
        padding (Tensor | List[int32]): The padding size with data type int32. [len(padding)/2] dimensions
            of input will be padded. The pad has the form (pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back).
        value (float32): The value to fill the padded areas. Default is 0.0
        data_format (str): An string from: "NCDHW", "NDHWC". Specify the data format of the input data.
           Default is  "NCDHW"
        name (str, optional) : The default value is None.  Normally there is no need for
            user to set this property.  For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        None

    Examples:
        .. code-block:: text

            x = [[[[[1., 2., 3.],
                    [4., 5., 6.]]]]]
            padding = [1, 2, 0, 0, 0, 0]
            value = 0.0
            Out = [[[[[0. 1. 2. 3. 0. 0.]
                      [0. 4. 5. 6. 0. 0.]]]]]

    Code Examples:
        .. code-block:: python

            import paddle
            import paddle.nn as nn
            import numpy as np
            paddle.disable_static()

            input_shape = (1, 1, 1, 2, 3)
            pad = [1, 0, 1, 2, 0, 0]
            data = np.arange(np.prod(input_shape), dtype=np.float32).reshape(input_shape) + 1
            my_pad = nn.ConstantPad3d(padding=pad)
            data = paddle.to_tensor(data)
            result = my_pad(data)
            print(result.numpy())
            # [[[[[0. 0. 0. 0.]
            #     [0. 1. 2. 3.]
            #     [0. 4. 5. 6.]
            #     [0. 0. 0. 0.]
            #     [0. 0. 0. 0.]]]]]
    """

    def __init__(self, padding, value=0.0, data_format="NCDHW", name=None):
        super(ConstantPad3d, self).__init__()
        self._mode = "constant"
        self._data_format = data_format
        self._pad = padding
        self._value = value
        self._name = name

    def forward(self, x):
        return F.pad(x,
                     pad=self._pad,
                     mode=self._mode,
                     value=self._value,
                     data_format=self._data_format,
                     name=self._name)


class ReplicationPad3d(layers.Layer):
    """
    This interface is used to construct a callable object of the ``ReplicationPad3d`` class.
    Uses input boundaries to pad the input tensor.

    Parameters:
        padding (Tensor | List[int32]): The padding size with data type int32. [len(padding)/2] dimensions
            of input will be padded. The pad has the form (pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back).
        data_format (str): An string from: "NCDHW", "NDHWC". Specify the data format of the input data.
           Default is  "NCDHW"
        name (str, optional) : The default value is None.  Normally there is no need for
            user to set this property.  For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        None

    Examples:
        .. code-block:: text

            x = [[[[[1., 2., 3.],
                    [4., 5., 6.]]]]]
            padding = [1, 2, 0, 0, 0, 0]
            Out = [[[[[1. 1. 2. 3. 3. 3.]
                      [4. 4. 5. 6. 6. 6.]]]]]

    Code Examples:
        .. code-block:: python

            import paddle
            import paddle.nn as nn
            import numpy as np
            paddle.disable_static()

            input_shape = (1, 1, 1, 2, 3)
            pad = [1, 0, 1, 2, 0, 0]
            data = np.arange(np.prod(input_shape), dtype=np.float32).reshape(input_shape) + 1
            my_pad = nn.ReplicationPad3d(padding=pad)
            data = paddle.to_tensor(data)
            result = my_pad(data)
            print(result.numpy())
            # [[[[[1. 1. 2. 3.]
            #     [1. 1. 2. 3.]
            #     [4. 4. 5. 6.]
            #     [4. 4. 5. 6.]
            #     [4. 4. 5. 6.]]]]]
    """

    def __init__(self, padding, data_format="NCDHW", name=None):
        super(ReplicationPad3d, self).__init__()
        self._mode = "replicate"
        self._data_format = data_format
        self._pad = padding
        self._name = name

    def forward(self, x):
        return F.pad(x,
                     pad=self._pad,
                     mode=self._mode,
                     data_format=self._data_format,
                     name=self._name)


class CosineSimilarity(layers.Layer):
    """
    This interface is used to compute cosine similarity between x1 and x2 along axis.

    Parameters:
        axis (int): Dimension of vectors to compute cosine similarity. Default is 1.
        eps(float): Small value to avoid division by zero. Default is 1e-8.
    Returns:
        None

    Examples:
        .. code-block:: text

            Case 0:
                x1 = [[0.8024077  0.9927354  0.27238318 0.8344984 ]
                     [0.48949873 0.5797396  0.65444374 0.66510963]
                     [0.1031398  0.9614342  0.08365563 0.6796464 ]
                     [0.10760343 0.7461209  0.7726148  0.5801006 ]]
                x2 = [[0.62913156 0.1536727  0.9847992  0.04591406]
                     [0.9098952  0.15715368 0.8671125  0.3156102 ]
                     [0.4427798  0.54136837 0.5276275  0.32394758]
                     [0.3769419  0.8535014  0.48041078 0.9256797 ]]
                axis = 1
                eps = 1e-8
                Out: [0.5275037  0.8368967  0.75037485 0.9245899]

    Code Examples:
        .. code-block:: python

            import paddle
            import paddle.nn as nn
            import numpy as np
            paddle.disable_static()

            np.random.seed(0)
            x1 = np.random.rand(2,3)
            x2 = np.random.rand(2,3)
            x1 = paddle.to_tensor(x1)
            x2 = paddle.to_tensor(x2)

            cos_sim_func = nn.CosineSimilarity(axis=0)
            result = cos_sim_func(x1, x2)
            print(result.numpy())
            # [0.99806249 0.9817672  0.94987036]
    """

    def __init__(self, axis=1, eps=1e-8):
        super(CosineSimilarity, self).__init__()
        self._axis = axis
        self._eps = eps

    def forward(self, x1, x2):
        return F.cosine_similarity(x1, x2, axis=self._axis, eps=self._eps)


class Embedding(layers.Layer):
    """
    **Embedding Layer**

    This interface is used to construct a callable object of the ``Embedding`` class.
    For specific usage, refer to code examples. It implements the function of the Embedding Layer.
    This layer is used to lookup embeddings vector of ids provided by :attr:`x` .
    It automatically constructs a 2D embedding matrix based on the
    input :attr:`num_embeddings` and attr:`embedding_dim`.

    The shape of output Tensor is generated by appending an emb_size dimension to the
    last dimension of the input Tensor shape.

    **Note:** The id in :attr:`x` must satisfy :math:`0 =< id < num_embeddings` ,
    otherwise the program will throw an exception and exit.

    .. code-block:: text

        Case 1:

        input is a Tensor. padding_idx = -1
            input.data = [[1, 3], [2, 4], [4, 127]
            input.shape = [3, 2]
        Given size = [128, 16]
        output is a Tensor:
            out.shape = [3, 2, 16]
            out.data = [[[0.129435295, 0.244512452, ..., 0.436322452],
                        [0.345421456, 0.524563927, ..., 0.144534654]],

                        [[0.345249859, 0.124939536, ..., 0.194353745],
                        [0.945345345, 0.435394634, ..., 0.435345365]],

                        [[0.945345345, 0.435394634, ..., 0.435345365],
                        [0.0,         0.0,         ..., 0.0        ]]]  # padding data
        The input padding_idx is less than 0, it is automatically converted to padding_idx = -1 + 128 = 127
        It will pad all-zero data when ids is 127.

    Parameters:
        num_embeddings (int): Just one element which indicate the size
            of the dictionary of embeddings.
        embedding_dim:  Just one element which indicate the size of each embedding vector respectively.
        padding_idx(int|long|None): padding_idx needs to be in the interval [-num_embeddings, num_embeddings).
            If :math:`padding\_idx < 0`, the :math:`padding\_idx` will automatically be converted
            to :math:`vocab\_size + padding\_idx` . It will output all-zero padding data whenever lookup
            encounters :math:`padding\_idx` in id. And the padding data will not be updated while training.
            If set None, it makes no effect to output. Default: None.
        sparse(bool): The flag indicating whether to use sparse update. This parameter only
            affects the performance of the backwards gradient update. It is recommended to set
            True because sparse update is faster. But some optimizer does not support sparse update,
            such as :ref:`api_optimizer_AdadeltaOptimizer` , :ref:`api_optimizer_AdamaxOptimizer` ,
            :ref:`api_optimizer_DecayedAdagradOptimizer` , :ref:`api_optimizer_FtrlOptimizer` ,
            :ref:`api_optimizer_LambOptimizer` and :ref:`api_optimizer_LarsMomentumOptimizer` .
            In these case, sparse must be False. Default: False.
        weight_attr(ParamAttr): To specify the weight parameter property. Default: None, which means the
            default weight parameter property is used. See usage for details in :ref:`api_ParamAttr` . In addition,
            user-defined or pre-trained word vectors can be loaded with the :attr:`param_attr` parameter.
            The local word vector needs to be transformed into numpy format, and the shape of local word
            vector should be consistent with :attr:`num_embeddings` . Then :ref:`api_initializer_NumpyArrayInitializer`
            is used to load custom or pre-trained word vectors. See code example for details.
        name(str|None): For detailed information, please refer
               to :ref:`api_guide_Name`. Usually name is no need to set and
               None by default.

    Attribute:
        **weight** (Parameter): the learnable weights of this layer.

    Returns:
        None

    Examples:

        .. code-block:: python

            import paddle
            import numpy as np

            x_data = np.arange(3, 6).reshape((3, 1)).astype(np.int64)
            y_data = np.arange(6, 12).reshape((3, 2)).astype(np.float32)
            paddle.disable_static(paddle.CPUPlace())
            x = paddle.to_tensor(x_data, stop_gradient=False)
            y = paddle.to_tensor(y_data, stop_gradient=False)

            embedding = paddle.nn.Embedding(10, 3, sparse=True)

            w0=np.full(shape=(10, 3), fill_value=2).astype(np.float32)
            embedding.weight.set_value(w0)

            adam = paddle.optimizer.Adam(parameters=[embedding.weight], learning_rate=0.01)
            adam.clear_grad()

            # weight.shape = [10, 3]

            # x.data = [[3],[4],[5]]
            # x.shape = [3, 1]

            # out.data = [[2,2,2], [2,2,2], [2,2,2]]
            # out.shape = [3, 1, 3]
            out=embedding(x)
            out.backward()
            adam.step()

    """

    def __init__(self,
                 num_embeddings,
                 embedding_dim,
                 padding_idx=None,
                 sparse=False,
                 weight_attr=None,
                 name=None):
        super(Embedding, self).__init__()
        self._num_embeddings = num_embeddings
        self._embedding_dim = embedding_dim
        self._sparse = sparse
        self._is_distributed = False
        self._padding_idx = -1 if padding_idx is None else padding_idx if padding_idx >= 0 else (
            num_embeddings + padding_idx)

        if self._num_embeddings <= 0:
            raise ValueError("num_embeddings must be gather than 0")

        if self._embedding_dim <= 0:
            raise ValueError("embedding_dim must be gather than 0")

        if self._padding_idx >= num_embeddings or self._padding_idx < -num_embeddings:
            raise ValueError("padding_idx must be within [-{}, {})".format(
                num_embeddings, num_embeddings))

        self._dtype = self._helper.get_default_dtype()
        self._size = [self._num_embeddings, self._embedding_dim]

        self._weight_attr = weight_attr
        self._remote_prefetch = False
        self._name = name
        self.weight = self.create_parameter(
            attr=self._weight_attr,
            shape=self._size,
            dtype=self._dtype,
            is_bias=False)

    def forward(self, x):
        return F.embedding(
            x,
            weight=self.weight,
            padding_idx=self._padding_idx,
            sparse=self._sparse,
            name=self._name)
