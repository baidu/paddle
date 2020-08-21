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

# TODO: define normalization api  
import paddle
import paddle.fluid as fluid
from ...fluid.data_feeder import check_variable_and_dtype, check_type
from ...fluid.layer_helper import LayerHelper
from ...fluid.framework import in_dygraph_mode, core
from ...framework import create_parameter
from ...fluid.layers import l2_normalize  #DEFINE_ALIAS
from ...fluid.layers import lrn  #DEFINE_ALIAS
from ...fluid.initializer import Constant
from ...fluid.param_attr import ParamAttr
from ...fluid import core, dygraph_utils

__all__ = [
    'batch_norm',
    #       'data_norm',
    'instance_norm',
    'l2_normalize',
    'layer_norm',
    'lrn',
    'normalize',
    #       'spectral_norm'
]


def normalize(x, p=2, axis=1, epsilon=1e-12, name=None):
    """
    This op normalizes ``x`` along dimension ``axis`` using :math:`L_p` norm. This layer computes

    .. math::

        y = \frac{x}{ \max\left( \lvert \lvert x \rvert \rvert_p, epsilon\right) }
    
    .. math::
        \lvert \lvert x \rvert \rvert_p = \left(\sum_i {\lvert x_i\rvert^p}  \right)^{1/p}

    where, :math:`\sum_i{\lvert x_i\rvert^p}` is calculated along the ``axis`` dimension.


    Args:
        x (Tensor): The input tensor could be N-D tensor, and the input data type could be float32 or float64.
        p (float|int, optional): The exponent value in the norm formulation. Default: 2
        axis (int, optional): The axis on which to apply normalization. If ``x`` is 1-D tensor, ``axis`` is fixed to 0. If `axis < 0`, \
            the dimension to normalization is `x.ndim + axis`. -1 is the last dimension.
        epsilon (float, optional): Small float added to denominator to avoid dividing by zero. Default is 1e-12.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor, the output has the same shape and data type with ``x``.

    Examples:

        .. code-block:: python

            import numpy as np
            import paddle
            import paddle.nn.functional as F

            paddle.disable_static()
            x = np.arange(6, dtype=np.float32).reshape(2,3)
            x = paddle.to_variable(x)
            y = F.normalize(x)
            print(y.numpy())
            # [[0.         0.4472136  0.8944272 ]
            # [0.42426404 0.5656854  0.7071067 ]]

            y = F.normalize(x, p=1.5)
            print(y.numpy())
            # [[0.         0.40862012 0.81724024]
            # [0.35684016 0.4757869  0.5947336 ]]

            y = F.normalize(x, axis=0)
            print(y.numpy())
            # [[0.         0.24253564 0.37139067]
            # [1.         0.97014254 0.9284767 ]]
    """
    if len(x.shape) == 1:
        axis = 0
    if in_dygraph_mode():
        eps = fluid.dygraph.base.to_variable([epsilon], dtype=x.dtype)
        out = core.ops.p_norm(x, 'axis', axis, 'porder',
                              float(p), 'keepdim', True, 'epsilon', epsilon)
        return x / core.ops.elementwise_max(out, eps)

    check_type(p, 'p', (float, int), 'normalize')
    check_type(axis, 'axis', (int), 'normalize')
    check_variable_and_dtype(x, 'x', ['float32', 'float64'], 'normalize')

    attrs = {
        'axis': axis,
        'porder': float(p),
        'keepdim': True,
        'epsilon': epsilon,
    }
    helper = LayerHelper('p_norm', **locals())
    out = helper.create_variable_for_type_inference(dtype=x.dtype)
    helper.append_op(
        type='p_norm', inputs={'X': x}, outputs={'Out': out}, attrs=attrs)
    eps = out.block.create_var(dtype=out.dtype)
    paddle.fill_constant([1], out.dtype, epsilon, out=eps)
    return paddle.elementwise_div(x, paddle.maximum(out, eps), name=name)


def batch_norm(x,
               running_mean,
               running_var,
               weight=None,
               bias=None,
               training=False,
               momentum=0.9,
               epsilon=1e-05,
               data_format="NCHW",
               name=None):
    """
    Applies Batch Normalization as described in the paper Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift .

    see nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d for detail.
    
    Parameters:
        x(Tesnor): input value.
        running_mean(Tensor): running mean.
        running_var(Tensor): running variance.
        epsilon(float, optional): The small value added to the variance to prevent division by zero. Default: 1e-5.
        momentum(float, optional): The value used for the moving_mean and moving_var computation. Default: 0.9.
        weight(Tensor, optional): The parameter attribute for Parameter `scale` of batch_norm. Default: None.
        bias_attr(ParamAttr, optional): The parameter attribute for the bias of batch_norm. Default: None.
        training(bool, optional): defalut False.
        data_format(str, optional): Specify the input data format. Defalut "NCHW".
        name(str, optional): Default: None.

    Returns:
        None

    Examples:
        .. code-block:: python

          import paddle
          import numpy as np

          paddle.disable_static()
          np.random.seed(123)
          x_data = np.random.random(size=(2, 1, 2, 3)).astype('float32')
          running_mean = np.random.random(size=1).astype('float32')
          running_variance = np.random.random(size=1).astype('float32')
          rm = to_variable(running_mean)
          rv = to_variable(running_variance)
          batch_norm_out = paddle.nn.functional.batch_norm(x, rm, rv)

          print(batch_norm_out.numpy)
    """

    assert len(x.shape) >= 2, "input dim must be larger than 1"
    param_shape = [x.shape[1]]
    if weight is None or weight is False:
        weight = create_parameter(
            dtype=x.dtype, shape=param_shape, default_initializer=Constant(1.0))
        weight.stop_gradient = True

    if bias is None or bias is False:
        bias = create_parameter(
            dtype=x.dtype, shape=param_shape, default_initializer=Constant(0.0))
        bias.stop_gradient = True

    mean_out = running_mean
    variance_out = running_var
    if in_dygraph_mode():
        attrs = ("momentum", momentum, "epsilon", epsilon, "is_test", training,
                 "data_layout", data_format, "use_mkldnn", False,
                 "fuse_with_relu", False, "use_global_stats", training,
                 'trainable_statistics', training)
        batch_norm_out, _, _, _, _, _ = core.ops.batch_norm(
            x, weight, bias, running_mean, running_var, mean_out, variance_out,
            *attrs)

        return dygraph_utils._append_activation_in_dygraph(
            batch_norm_out, act=None)

    check_variable_and_dtype(x, 'input', ['float16', 'float32', 'float64'],
                             'BatchNorm')

    attrs = {
        "momentum": momentum,
        "epsilon": epsilon,
        "is_test": not training,
        "data_layout": data_format,
        "use_mkldnn": False,
        "fuse_with_relu": False,
        "use_global_stats": training,
        "trainable_statistics": training,
    }

    inputs = {
        "X": [x],
        "Scale": [weight],
        "Bias": [bias],
        "Mean": [running_mean],
        "Variance": [running_var]
    }

    saved_mean = self._helper.create_variable_for_type_inference(
        dtype=x.dtype, stop_gradient=True)
    saved_variance = self._helper.create_variable_for_type_inference(
        dtype=x.dtype, stop_gradient=True)
    batch_norm_out = self._helper.create_variable_for_type_inference(x.dtype)

    outputs = {
        "Y": [batch_norm_out],
        "MeanOut": [mean],
        "VarianceOut": [variance],
        "SavedMean": [saved_mean],
        "SavedVariance": [saved_variance]
    }

    self._helper.append_op(
        type="batch_norm", inputs=inputs, outputs=outputs, attrs=attrs)

    # Currently, we don't support inplace in dygraph mode
    return self._helper.append_activation(batch_norm_out, None)


def layer_norm(x,
               normalized_shape,
               weight=None,
               bias=None,
               epsilon=1e-05,
               name=None):
    """
    see more detail in paddle.nn.LayerNorm
    
    Parameters:
        x(Tensor): Input Tensor.
        normalized_shape(int or list or tuple): Input shape from an expected input of
            size :math:`[*, normalized_shape[0], normalized_shape[1], ..., normalized_shape[-1]]`.
            If it is a single integer, this module will normalize over the last dimension
            which is expected to be of that specific size.
        epsilon(float, optional): The small value added to the variance to prevent
            division by zero. Default: 1e-05.
        weight_attr(ParamAttr|bool, optional): The parameter attribute for the learnable
            gain :math:`g`. If False, weight is None. If is None, a default :code:`ParamAttr` would be added as scale. The
            :attr:`param_attr` is initialized as 1 if it is added. Default: None.
        bias_attr(ParamAttr, optional): The parameter attribute for the learnable
            bias :math:`b`. If is False, bias is None. If is None, a default :code:`ParamAttr` would be added as bias. The
            :attr:`bias_attr` is initialized as 0 if it is added. Default: None.
        name(str, optional): parameter name. Default None.

    Returns:
        None

    Examples:

        .. code-block:: python

          import paddle
          import numpy as np

          paddle.disable_static()
          np.random.seed(123)
          x_data = np.random.random(size=(2, 2, 2, 3)).astype('float32')
          x = paddle.to_tensor(x_data) 
          layer_norm = paddle.nn.functional.layer_norm(x, x.shape[1:])
          layer_norm_out = layer_norm(x)

          print(layer_norm_out.numpy)
    """
    input_shape = list(x.shape)
    input_ndim = len(input_shape)
    normalized_ndim = len(normalized_shape)
    begin_norm_axis = input_ndim - normalized_ndim
    if input_ndim < normalized_ndim or input_shape[
            begin_norm_axis:] != normalized_shape:
        str_normalized_shape = str(normalized_shape)
        raise ValueError('Given normalized_shape is ' + str_normalized_shape +
                         ', expected input with shape [*, ' +
                         str_normalized_shape[
                             1:] + ', but got input shape ' + str(input_shape))

    if in_dygraph_mode():
        pre_act, _, _ = core.ops.layer_norm(x, weight, bias, 'epsilon', epsilon,
                                            'begin_norm_axis', begin_norm_axis)
        return dygraph_utils._append_activation_in_dygraph(pre_act, act=None)

    check_variable_and_dtype(x, 'input', ['float32', 'float64'], 'LayerNorm')

    inputs = dict()
    inputs['X'] = [x]
    if weight:
        inputs['Scale'] = [weight]
    if bias:
        inputs['Bias'] = [bias]
    attrs = {"epsilon": epsilon, "begin_norm_axis": begin_norm_axis}

    # create output
    mean_out = self._helper.create_variable_for_type_inference(
        dtype=x.type, stop_gradient=True)
    variance_out = self._helper.create_variable_for_type_inference(
        dtype=x.type, stop_gradient=True)
    layer_norm_out = self._helper.create_variable_for_type_inference(x.type)

    self._helper.append_op(
        type="layer_norm",
        inputs=inputs,
        outputs={
            "Y": layer_norm_out,
            "Mean": mean_out,
            "Variance": variance_out,
        },
        attrs={"epsilon": epsilon,
               "begin_norm_axis": begin_norm_axis})

    return self._helper.append_activation(layer_norm_out, act=None)


def instance_norm(x,
                  running_mean=None,
                  running_var=None,
                  weight=None,
                  bias=None,
                  use_input_stats=True,
                  momentum=0.1,
                  eps=1e-05,
                  data_format="NCHW",
                  name=None):
    """
    See more detail in nn.layer.InstanceNorm2d.

    Parameters:
        num_features(int): Indicate the number of channels of the input ``Tensor``.
        running_mean(Tensor): running mean. Default None.
        running_var(Tensor): running variance. Default None.
        eps(float, optional): A value added to the denominator for
            numerical stability. Default is 1e-5.
        momentum(float, optional): The value used for the moving_mean and moving_var computation. Default: 0.9.
        use_input_stats(bool): Default True.
        weight_attr(ParamAttr|bool, optional): The parameter attribute for Parameter `scale`
             of instance_norm. If it is set to None or one attribute of ParamAttr, instance_norm
	     will create ParamAttr as weight_attr, the name of scale can be set in ParamAttr.
	     If the Initializer of the weight_attr is not set, the parameter is initialized 
	     one. If it is set to False, will not create weight_attr. Default: None.
        bias_attr(ParamAttr|bool, optional): The parameter attribute for the bias of instance_norm.
             If it is set to None or one attribute of ParamAttr, instance_norm
	     will create ParamAttr as bias_attr, the name of bias can be set in ParamAttr. 
	     If the Initializer of the bias_attr is not set, the bias is initialized zero. 
             If it is set to False, will not create bias_attr. Default: None.
        data_format(str, optional): Specify the input data format. Default: NCL.
        name(str, optional): Default None.

    Returns:
        None.

    Examples:

        .. code-block:: python

          import paddle
          import numpy as np

          paddle.disable_static()
          np.random.seed(123)
          x_data = np.random.random(size=(2, 2, 2, 3)).astype('float32')
          x = paddle.to_tensor(x_data) 
          instance_norm_out = paddle.nn.functional.instancenorm(x)

          print(instance_norm_out.numpy)

    """

    def _check_input_dim(self, input):
        if len(input.shape) != 2 and len(input.shape) != 3:
            raise ValueError('expected 2D or 3D input (got {}D input)'.format(
                len(input.shape)))
