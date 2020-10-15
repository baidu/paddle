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

from paddle.fluid.dygraph import layers
from paddle.fluid import core
from paddle.fluid import dygraph_utils
from paddle.fluid import unique_name
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.framework import _varbase_creator
from paddle.fluid.framework import in_dygraph_mode
from paddle.fluid.initializer import Constant
from paddle.fluid.data_feeder import check_variable_and_dtype

__all__ = [
    'FakeQuantMovingAverage', 'FakeQuantAbsMax', 'QuantizedConv2D',
    'QuantizedLinear', 'FakeChannelWiseQuantDequantAbsMax',
    'MovingAverageAbsMaxScale'
]


class FakeQuantMovingAverage(layers.Layer):
    """
    FakeQuantMovingAverage layer does the moving_average_abs_max quant and then dequant.
    Its computational formula is described as below:

    :math:`scale = (moving\_rate*accum+max(abs(x)))/(moving\_rate*state+1)`
    :math:`range = 2^{bit\_length - 1} - 1`
    :math:`Out = round(X / scale * range) * scale / range`
    """

    def __init__(self,
                 name=None,
                 moving_rate=0.9,
                 quant_bits=8,
                 dtype='float32'):
        super(FakeQuantMovingAverage, self).__init__()
        self._moving_rate = moving_rate
        self._quant_bits = quant_bits

        scale_prefix = "{}.scale".format(
            name) if name else 'quant_dequant.scale'
        scale_attr = ParamAttr(
            name=unique_name.generate(scale_prefix),
            initializer=Constant(0.001),
            trainable=False)
        self._scale = self.create_parameter(
            shape=[1], attr=scale_attr, dtype=dtype)
        self._scale.stop_gradient = True

        state_prefix = "{}.state".format(
            name) if name else 'quant_dequant.state'
        state_attr = ParamAttr(
            name=unique_name.generate(state_prefix),
            initializer=Constant(1),
            trainable=False)
        self._state = self.create_parameter(
            shape=[1], attr=state_attr, dtype=dtype)
        self._state.stop_gradient = True

        accum_prefix = "{}.accum".format(
            name) if name else 'quant_dequant.accum'
        accum_attr = ParamAttr(
            name=unique_name.generate(accum_prefix),
            initializer=Constant(1),
            trainable=False)
        self._accum = self.create_parameter(
            shape=[1], attr=accum_attr, dtype=dtype)
        self._accum.stop_gradient = True

    def forward(self, input):
        if in_dygraph_mode():
            attrs = ('moving_rate', self._moving_rate, 'bit_length',
                     self._quant_bits, 'is_test', not self.training)
            quant_out = _varbase_creator(
                type=input.type,
                name="{}.quantized.dequantized".format(input.name),
                shape=input.shape,
                dtype=input.dtype,
                persistable=False)
            state = self._state if self.training else None
            accum = self._accum if self.training else None

            out, _, _, _ = core.ops.fake_quantize_dequantize_moving_average_abs_max(
                input, self._scale, accum, state, quant_out, self._scale, state,
                accum, *attrs)
            return out

        check_variable_and_dtype(input, 'input', ['float32'],
                                 "FakeQuantMovingAverage")
        attrs = {
            'moving_rate': self._moving_rate,
            'bit_length': self._quant_bits,
            'is_test': not self.training
        }
        inputs = {"X": [input], "InScale": [self._scale]}
        quant_out = self._helper.create_variable(
            name="{}.quantized.dequantized".format(input.name),
            dtype=input.dtype,
            type=core.VarDesc.VarType.LOD_TENSOR,
            persistable=False,
            stop_gradient=False)
        outputs = {"Out": [quant_out], "OutScale": [self._scale]}

        if self.training:
            inputs['InState'] = [self._state]
            inputs['InAccum'] = [self._accum]
            outputs['OutState'] = [self._state]
            outputs['OutAccum'] = [self._accum]

        self._helper.append_op(
            type="fake_quantize_dequantize_moving_average_abs_max",
            inputs=inputs,
            outputs=outputs,
            attrs=attrs)

        return quant_out


class FakeQuantAbsMax(layers.Layer):
    """
    FakeQuantAbsMax layer does the abs_max quant and then dequant.
    Its computational formula is described as below:

    :math:`scale = max(abs(X))`
    :math:`range = 2^{bit\_length - 1} - 1`
    :math:`Out = round(X / scale * range) * scale / range`
    """

    def __init__(self,
                 name=None,
                 quant_bits=8,
                 dtype='float32',
                 quant_on_weight=False):
        super(FakeQuantAbsMax, self).__init__()
        self._quant_bits = quant_bits
        self._dtype = dtype
        self._name = name
        scale_prefix = "{}.scale".format(
            name) if name else 'quant_dequant.scale'
        self._scale_name = unique_name.generate(scale_prefix)
        if quant_on_weight:
            scale_attr = ParamAttr(
                name=self._scale_name,
                initializer=Constant(0.0),
                trainable=False)
            self._scale = self.create_parameter(
                shape=[1], attr=scale_attr, dtype=self._dtype)
            self._scale.stop_gradient = True
        else:
            self._scale = None

    def forward(self, input):
        if in_dygraph_mode():
            attrs = ('bit_length', self._quant_bits)
            quant_out = _varbase_creator(
                type=input.type,
                name="{}.quantized.dequantized".format(input.name),
                shape=input.shape,
                dtype=input.dtype,
                persistable=False)
            out_scale = self._scale
            if not out_scale:
                out_scale = _varbase_creator(
                    type=core.VarDesc.VarType.LOD_TENSOR,
                    name=self._scale_name,
                    shape=[1],
                    dtype=self._dtype,
                    persistable=False)
                out_scale.stop_gradient = True
            out, _, = core.ops.fake_quantize_dequantize_abs_max(
                input, quant_out, out_scale, *attrs)
            return out

        check_variable_and_dtype(input, 'input', ['float32'], "FakeQuantAbsMax")
        attrs = {'bit_length': self._quant_bits}
        inputs = {"X": [input]}
        quant_out = self._helper.create_variable(
            name="{}.quantized.dequantized".format(input.name),
            dtype=input.dtype,
            type=core.VarDesc.VarType.LOD_TENSOR,
            persistable=False,
            stop_gradient=False)
        out_scale = self._scale
        if not out_scale:
            out_scale = self._helper.create_variable(
                name=self._scale_name,
                dtype=self._dtype,
                type=core.VarDesc.VarType.LOD_TENSOR,
                persistable=False,
                stop_gradient=True)
        outputs = {"Out": [quant_out], "OutScale": [out_scale]}

        self._helper.append_op(
            type="fake_quantize_dequantize_abs_max",
            inputs=inputs,
            outputs=outputs,
            attrs=attrs)

        return quant_out


class FakeChannelWiseQuantDequantAbsMax(layers.Layer):
    def __init__(self,
                 name=None,
                 channel_num=None,
                 quant_bits=8,
                 quant_axis=0,
                 dtype='float32',
                 quant_on_weight=False):
        assert quant_on_weight == True, "Channel_wise only can be used on weight quantization."
        super(FakeChannelWiseQuantDequantAbsMax, self).__init__()
        self._quant_bits = quant_bits
        self._quant_axis = quant_axis
        self._dtype = dtype
        self._name = name
        self._channel_num = channel_num
        scale_prefix = "{}.scale".format(
            name) if name else 'quant_dequant.scale'
        self._scale_name = unique_name.generate(scale_prefix)
        if quant_on_weight:
            scale_attr = ParamAttr(
                name=self._scale_name,
                initializer=Constant(0.0),
                trainable=False)
            self._scale = self.create_parameter(
                shape=[self._channel_num], attr=scale_attr, dtype=self._dtype)
            self._scale.stop_gradient = True
        else:
            self._scale = None

    def forward(self, input):
        if in_dygraph_mode():
            attrs = ('bit_length', self._quant_bits, 'quant_axis',
                     self._quant_axis)
            quant_out = _varbase_creator(
                type=input.type,
                name="{}.quantized.dequantized".format(input.name),
                shape=input.shape,
                dtype=input.dtype,
                persistable=False)

            out_scale = self._scale
            if out_scale is None:
                out_scale = _varbase_creator(
                    type=core.VarDesc.VarType.LOD_TENSOR,
                    name=self._scale_name,
                    shape=[self._channel_num],
                    dtype=self._dtype,
                    persistable=False)
                out_scale.stop_gradient = True

            out, _, = core.ops.fake_channel_wise_quantize_dequantize_abs_max(
                input, quant_out, out_scale, *attrs)
            return out

        check_variable_and_dtype(input, 'input', ['float32'],
                                 "FakeChannelWiseQuantDequantAbsMax")
        attrs = {'bit_length': self._quant_bits, 'quant_axis': self._quant_axis}
        inputs = {"X": [input]}
        quant_out = self._helper.create_variable(
            name="{}.quantized.dequantized".format(input.name),
            dtype=input.dtype,
            type=core.VarDesc.VarType.LOD_TENSOR,
            persistable=False,
            stop_gradient=False)
        out_scale = self._scale
        if not out_scale:
            out_scale = self._helper.create_variable(
                name=self._scale_name,
                dtype=self._dtype,
                type=core.VarDesc.VarType.LOD_TENSOR,
                persistable=False,
                stop_gradient=True)
        outputs = {"Out": [quant_out], "OutScale": [out_scale]}

        self._helper.append_op(
            type="fake_channel_wise_quantize_dequantize_abs_max",
            inputs=inputs,
            outputs=outputs,
            attrs=attrs)

        return quant_out


def _get_fake_quant_type(quant_type, **kwargs):
    call_args = {
        "name": kwargs.get("name", None),
        "quant_bits": kwargs.get("quant_bits", 8),
        "dtype": kwargs.get("dtype", "float32")
    }

    if quant_type == 'abs_max':
        call_args["quant_on_weight"] = kwargs.get("quant_on_weight", False)
    elif quant_type == 'moving_average_abs_max':
        call_args["moving_rate"] = kwargs.get("moving_rate", 0.9)
    elif quant_type == 'channel_wise_abs_max':
        call_args["quant_on_weight"] = kwargs.get("quant_on_weight", False)
        call_args["channel_num"] = kwargs.get("channel_num", None)
        call_args["quant_axis"] = kwargs.get("quant_axis", 0)
        assert call_args["channel_num"] is not None, (
            "You need to input channel_num"
            "when you use channel_wise_abs_max strategy.")
    fake_quant_map = {
        'abs_max': FakeQuantAbsMax,
        'moving_average_abs_max': FakeQuantMovingAverage,
        'channel_wise_abs_max': FakeChannelWiseQuantDequantAbsMax
    }

    return fake_quant_map[quant_type](**call_args)


class QuantizedConv2D(layers.Layer):
    """
    The computational logic of QuantizedConv2D is the same with Conv2D.
    The only difference is that its inputs are all fake quantized.
    """

    def __init__(self,
                 layer,
                 weight_bits=8,
                 activation_bits=8,
                 moving_rate=0.9,
                 weight_quantize_type='abs_max',
                 activation_quantize_type='abs_max'):
        super(QuantizedConv2D, self).__init__()
        # For Conv2D
        self._groups = getattr(layer, '_groups')
        self._stride = getattr(layer, '_stride')
        self._padding = getattr(layer, '_padding')
        self._dilation = getattr(layer, '_dilation')
        self._act = getattr(layer, '_act')
        self._use_cudnn = getattr(layer, '_use_cudnn')
        self._dtype = getattr(layer, '_dtype')
        self._l_type = getattr(layer, '_l_type')
        self.weight = getattr(layer, 'weight')
        self.bias = getattr(layer, 'bias')
        # For FakeQuant
        self._conv2d_quant_axis = 0
        self._fake_quant_weight = _get_fake_quant_type(
            weight_quantize_type,
            name=self.weight.name,
            moving_rate=moving_rate,
            quant_bits=weight_bits,
            dtype=self._dtype,
            quant_on_weight=True,
            channel_num=self.weight.shape[self._conv2d_quant_axis],
            quant_axis=self._conv2d_quant_axis)
        self._fake_quant_input = _get_fake_quant_type(
            activation_quantize_type,
            name=layer.full_name(),
            moving_rate=moving_rate,
            quant_bits=activation_bits,
            dtype=self._dtype,
            quant_on_weight=False)

    def forward(self, input):
        quant_input = self._fake_quant_input(input)
        quant_weight = self._fake_quant_weight(self.weight)

        if in_dygraph_mode() and self._l_type == 'conv2d':
            attrs = ('strides', self._stride, 'paddings', self._padding,
                     'dilations', self._dilation, 'groups', self._groups
                     if self._groups else 1, 'use_cudnn', self._use_cudnn)
            pre_bias = core.ops.conv2d(quant_input, quant_weight, *attrs)

            pre_act = dygraph_utils._append_bias_in_dygraph(pre_bias, self.bias,
                                                            1)
            return dygraph_utils._append_activation_in_dygraph(pre_act,
                                                               self._act)
        check_variable_and_dtype(quant_input, 'input',
                                 ['float16', 'float32', 'float64'],
                                 'QuantizedConv2D')
        attrs = {
            'strides': self._stride,
            'paddings': self._padding,
            'dilations': self._dilation,
            'groups': self._groups if self._groups else 1,
            'use_cudnn': self._use_cudnn,
            'use_mkldnn': False,
        }
        pre_bias = self._helper.create_variable_for_type_inference(
            dtype=self._dtype)

        self._helper.append_op(
            type=self._l_type,
            inputs={
                'Input': quant_input,
                'Filter': quant_weight,
            },
            outputs={"Output": pre_bias},
            attrs=attrs)

        if self.bias is not None:
            pre_act = self._helper.create_variable_for_type_inference(
                dtype=self._dtype)
            self._helper.append_op(
                type='elementwise_add',
                inputs={'X': [pre_bias],
                        'Y': [self.bias]},
                outputs={'Out': [pre_act]},
                attrs={'axis': 1})
        else:
            pre_act = pre_bias

        return self._helper.append_activation(pre_act, act=self._act)


class QuantizedLinear(layers.Layer):
    """
    The computational logic of QuantizedLinear is the same with Linear.
    The only difference is that its inputs are all fake quantized.
    """

    def __init__(self,
                 layer,
                 weight_bits=8,
                 activation_bits=8,
                 moving_rate=0.9,
                 weight_quantize_type='abs_max',
                 activation_quantize_type='abs_max'):
        super(QuantizedLinear, self).__init__()
        # For Linear
        self._act = getattr(layer, '_act')
        self._dtype = getattr(layer, '_dtype')
        self.weight = getattr(layer, 'weight')
        self.bias = getattr(layer, 'bias')
        # For FakeQuant
        self._linear_quant_axis = 1
        self._fake_quant_weight = _get_fake_quant_type(
            weight_quantize_type,
            name=self.weight.name,
            moving_rate=moving_rate,
            quant_bits=weight_bits,
            dtype=self._dtype,
            quant_on_weight=True,
            channel_num=self.weight.shape[self._linear_quant_axis],
            quant_axis=self._linear_quant_axis)
        self._fake_quant_input = _get_fake_quant_type(
            activation_quantize_type,
            name=layer.full_name(),
            moving_rate=moving_rate,
            quant_bits=activation_bits,
            dtype=self._dtype,
            quant_on_weight=False)

    def forward(self, input):
        quant_input = self._fake_quant_input(input)
        quant_weight = self._fake_quant_weight(self.weight)
        if in_dygraph_mode():
            pre_bias = _varbase_creator(dtype=input.dtype)
            core.ops.matmul(quant_input, quant_weight, pre_bias, 'transpose_X',
                            False, 'transpose_Y', False, "alpha", 1)
            pre_act = dygraph_utils._append_bias_in_dygraph(
                pre_bias, self.bias, axis=len(input.shape) - 1)

            return dygraph_utils._append_activation_in_dygraph(pre_act,
                                                               self._act)

        check_variable_and_dtype(input, 'input',
                                 ['float16', 'float32', 'float64'],
                                 "QuantizedLinear")
        attrs = {
            "transpose_X": False,
            "transpose_Y": False,
            "alpha": 1,
        }
        inputs = {"X": [quant_input], "Y": [quant_weight]}
        mul_out = self._helper.create_variable_for_type_inference(self._dtype)

        self._helper.append_op(
            type="matmul",
            inputs=inputs,
            outputs={"Out": [mul_out]},
            attrs=attrs)
        if self.bias is not None:
            pre_activation = self._helper.create_variable_for_type_inference(
                dtype=self._dtype)
            self._helper.append_op(
                type='elementwise_add',
                inputs={'X': [mul_out],
                        'Y': [self.bias]},
                outputs={'Out': [pre_activation]},
                attrs={'axis': len(input.shape) - 1})
        else:
            pre_activation = mul_out
        return self._helper.append_activation(pre_activation, act=self._act)


class MovingAverageAbsMaxScale(layers.Layer):
    def __init__(self, name=None, moving_rate=0.9, dtype='float32'):
        """
        MovingAverageMaxScale layer is used to calculating the output quantization scale of Layer.
        Its computational formula is described as below:

        :math:`scale = (moving\_rate*accum+max(abs(x)))/(moving\_rate*state+1)`
        :math:`Out = X`
        """
        super(MovingAverageAbsMaxScale, self).__init__()
        self._moving_rate = moving_rate
        self._dtype = dtype

        scale_prefix = '{}.scale'.format(name) if name else 'outscale.scale'
        name = unique_name.generate(scale_prefix)
        scale_attr = ParamAttr(
            name=name, initializer=Constant(1), trainable=False)
        self._scale = self.create_parameter(
            shape=[1], attr=scale_attr, dtype=self._dtype)
        self._scale.stop_gradient = True

        state_prefix = "{}.state".format(name) if name else 'outscale.state'
        state_attr = ParamAttr(
            name=unique_name.generate(state_prefix),
            initializer=Constant(1),
            trainable=False)
        self._state = self.create_parameter(
            shape=[1], attr=state_attr, dtype=self._dtype)
        self._state.stop_gradient = True

        accum_prefix = "{}.accum".format(name) if name else 'outscale.accum'
        accum_attr = ParamAttr(
            name=unique_name.generate(accum_prefix),
            initializer=Constant(1),
            trainable=False)
        self._accum = self.create_parameter(
            shape=[1], attr=accum_attr, dtype=self._dtype)
        self._accum.stop_gradient = True
        MovingAverageAbsMaxScale._has_create = True

    def forward(self, input):
        if in_dygraph_mode():
            attrs = ('moving_rate', self._moving_rate, 'is_test',
                     not self.training)
            state = self._state if self.training else None
            accum = self._accum if self.training else None

            out_scale, _, _ = core.ops.moving_average_abs_max_scale(
                input, accum, state, self._scale, state, accum, *attrs)
            return out_scale

        check_variable_and_dtype(input, 'input', ['float32', 'float64'],
                                 'MovingAverageAbsMaxScale')

        scale_out = self._scale
        attrs = {'moving_rate': self._moving_rate, 'is_test': not self.training}

        inputs = {"X": [input]}
        outputs = {"OutScale": [scale_out]}

        if self.training:
            inputs['InState'] = [self._state]
            inputs['InAccum'] = [self._accum]
            outputs['OutState'] = [self._state]
            outputs['OutAccum'] = [self._accum]

        self._helper.append_op(
            type="moving_average_abs_max_scale",
            inputs=inputs,
            outputs=outputs,
            attrs=attrs)

        return scale_out
