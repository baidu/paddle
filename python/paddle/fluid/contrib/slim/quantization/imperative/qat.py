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

import logging
import numpy as np
import sys
import os
import paddle
from paddle.fluid import dygraph, core, framework
from paddle.fluid.executor import Executor
from paddle.fluid.dygraph.io import INFER_MODEL_SUFFIX, INFER_PARAMS_SUFFIX
from paddle.fluid.dygraph.nn import Conv2D, Linear, BatchNorm, Pool2D, Conv2DTranspose
from paddle.fluid.io import load_inference_model, save_inference_model
from paddle.nn.layer.activation import ReLU, LeakyReLU, Sigmoid, ReLU6, Tanh, Softmax, PReLU
from paddle.fluid.log_helper import get_logger
from . import quant_nn

__all__ = ['ImperativeQuantAware', 'ImperativeCalcOutScale']

_logger = get_logger(
    __name__, logging.INFO, fmt='%(asctime)s-%(levelname)s: %(message)s')

_op_real_in_out_name = {
    "conv2d": [["Input", "Filter"], ["Output"]],
    "conv2d_transpose": [["Input", "Filter"], ["Output"]],
    "pool2d": [["X"], ["Out"]],
    "elementwise_add": [["X", "Y"], ["Out"]],
    "softmax": [["X"], ["Out"]],
    "relu": [["X"], ["Out"]],
    "relu6": [["X"], ["Out"]],
    "leaky_relu": [["X"], ["Out"]],
    "prelu": [["X"], ["Out"]],
    "tanh": [["X"], ["Out"]],
    "batch_norm": [["X"], ["Y"]],
    "sigmoid": [["X"], ["Out"]],
}


class ImperativeQuantAware(object):
    """
    Add the fake quant logic for given quantizable layers, namely add the quant_dequant
    computational logic both for activation inputs and weight inputs.
    """

    def __init__(self,
                 weight_bits=8,
                 activation_bits=8,
                 weight_quantize_type='abs_max',
                 activation_quantize_type='moving_average_abs_max',
                 moving_rate=0.9,
                 quantizable_layer_type=['Conv2D', 'Linear']):
        """
        The constructor for ImperativeQuantAware.

        Args:
            weight_bits(int): quantization bit number for weights,
                whereas the bias is not quantized.
            activation_bits(int): quantization bit number for activations.
            weight_quantize_type(str): quantization type for weights,
                which supports 'abs_max' now. The 'moving_average_abs_max'
                usually is not used for weights, since weights are fixed once the
                model is well trained.
            activation_quantize_type(str): quantization type for activations,
                which supports 'abs_max' and 'moving_average_abs_max' now.
                If using 'abs_max' mode, the quantization scale will be calculated
                dynamically each step in both training and testing period. If using
                'moving_average_abs_max', the static quantization scale will be calculated
                during training and used in inference.
            moving_rate(float): the parameter for 'moving_average_abs_max' quantization.
            quantizable_op_type(list[str]): List the type of layers that will be quantized. 
                Default is ['Conv2D', 'Linear']. The quantizable_op_type in
                QuantizationFreezePass and ConvertToInt8Pass must be the same as this.


        Examples:
        .. code-block:: python

            import paddle
            from paddle.fluid.contrib.slim.quantization \
                import ImperativeQuantAware
            from paddle.vision.models \
                import resnet
            
            model = resnet.resnet50(pretrained=True)

            imperative_qat = ImperativeQuantAware(
                weight_quantize_type='abs_max',
                activation_quantize_type='moving_average_abs_max')
            
            # Add the fake quant logical.
            # The original model will be rewrite.
            imperative_qat.quantize(model)

            # Fine-tune the quantized model
            # ...
            
            # Save quant model for the inference.
            paddle.jit.save(
                layer=model,
                model_path="./resnet50_qat",
                input_spec=[
                    paddle.static.InputSpec(
                    shape=[None, 3, 224, 224], dtype='float32')])
        """
        super(ImperativeQuantAware, self).__init__()
        self._weight_bits = weight_bits
        self._activation_bits = activation_bits
        self._moving_rate = moving_rate

        quant_type = {
            'abs_max', 'moving_average_abs_max', 'channel_wise_abs_max'
        }

        assert activation_quantize_type != 'channel_wise_abs_max', \
            "The activation quantization type does not support 'channel_wise_abs_max'."
        if activation_quantize_type not in quant_type:
            raise ValueError(
                "Unknown activation_quantize_type : '%s'. It can only be "
                "'abs_max' or 'moving_average_abs_max' now." %
                (str(activation_quantize_type)))
        if weight_quantize_type not in quant_type:
            raise ValueError(
                "Unknown weight_quantize_type: '%s'. It can only be "
                "'abs_max' or 'moving_average_abs_max' or 'channel_wise_abs_max' now."
                % (str(weight_quantize_type)))
        self._activation_quantize_type = activation_quantize_type
        self._weight_quantize_type = weight_quantize_type

        self._quant_layers_map = {'Conv2D': Conv2D, 'Linear': Linear}
        self._quantizable_layer_type = tuple(
            self._quant_layers_map[layer]
            if layer in self._quant_layers_map else layer
            for layer in quantizable_layer_type)
        for layer in self._quantizable_layer_type:
            assert not isinstance(
                layer, str), "{} is unspported to be quantized.".format(layer)

    def quantize(self, model):
        """
        According to weights' and activations' quantization types, the model will be added some fake
        quant ops, such as fake_quantize_dequantize_moving_average_abs_max, fake_quantize_dequantize_abs_max
        and so on.

        Args:
            model(fluid.dygraph.Layer): the model to be quantized.
        Returns:
            None
        """
        for name, layer in model.named_sublayers():
            if not isinstance(layer, self._quantizable_layer_type):
                continue
            scopes = name.split('.')
            target = scopes[-1]
            obj = model
            parent = model
            for i in range(len(scopes) - 1):
                obj = getattr(parent, scopes[i])
                parent = obj

            quant_layer = self._get_quantized_counterpart(layer)
            setattr(obj, target, quant_layer)

    def _get_quantized_counterpart(self, layer):
        quant_layers = tuple(self._quant_layers_map.values())
        quantized_counterpart = tuple('Quantized' + k
                                      for k in self._quant_layers_map.keys())

        predicate = lambda value: isinstance(layer, value)
        index_generator = (i for i, v in enumerate(quant_layers)
                           if predicate(v))

        try:
            index = next(index_generator)
        except StopIteration:
            _logger.fatal("The layer {} is unsupported to be quantized.".format(
                layer.full_name()))
            sys.exit(-1)

        quantized_layer = quant_nn.__dict__[quantized_counterpart[index]](
            layer, self._weight_bits, self._activation_bits, self._moving_rate,
            self._weight_quantize_type, self._activation_quantize_type)
        return quantized_layer


class ImperativeCalcOutScale(object):
    def __init__(self,
                 moving_rate=0.9,
                 target_layer_types=[
                     'BatchNorm', 'Conv2D', 'Conv2DTranspose', 'LeakyReLU',
                     'Linear', 'PReLU', 'Pool2D', 'ReLU', 'ReLU6', 'Sigmoid',
                     'Softmax', 'Tanh'
                 ]):
        """
        Add the logic of calculating and setting output quantization scales of some layers.
        These output quantization scales may be used by tensorRT or some other inference engines.

        Args:
            moving_rate(float): The decay coefficient of moving average. The default value is 0.9.
            quantizable_op_type(list[str]): List the type of layers that will be calculated out_scale. 
                Default is ['Conv2D', 'ReLU', 'PReLU', 'LeakyReLU', 'Linear', 'Sigmoid', 'BatchNorm', 'ReLU6', 'Tanh', 'Softmax', 'Conv2DTranspose']
        """
        super(ImperativeCalcOutScale, self).__init__()
        self._moving_rate = moving_rate
        self._out_scale_layers_map = {
            'BatchNorm': BatchNorm,
            'Conv2D': Conv2D,
            'Conv2DTranspose': Conv2DTranspose,
            'LeakyReLU': LeakyReLU,
            'Linear': Linear,
            'PReLU': PReLU,
            'Pool2D': Pool2D,
            'ReLU': ReLU,
            'ReLU6': ReLU6,
            'Sigmoid': Sigmoid,
            'Softmax': Softmax,
            'Tanh': Tanh
        }
        self._out_scale_layer_type = tuple(
            self._out_scale_layers_map[layer]
            if layer in self._out_scale_layers_map else layer
            for layer in target_layer_types)
        for layer in self._out_scale_layer_type:
            assert not isinstance(
                layer, str), "{} is unspported to be out_scaled.".format(layer)
        self._register_hook_handle_list = []
        self._out_scale_dict = {}

    def calc_out_scale(self, model):
        """
        Insert the `moving_average_abs_max_scale` op to calculate output scale of Specific layers in model.

        Args:
            model(fluid.dygraph.Layer): The target model which would be calculate the output quantization scale.

        Returns:
            None
        """
        assert isinstance(
            model, dygraph.Layer), "model must be the instance of dygraph.Layer"
        for _, layer in model.named_sublayers():
            if not isinstance(layer, self._out_scale_layer_type):
                continue
            forward_post_hook_handle = layer.register_forward_post_hook(
                self._forward_post_hook)
            self._register_hook_handle_list.append(forward_post_hook_handle)

    # Get the output var name of the op
    def _get_op_output_names(self, op):
        assert isinstance(
            op, framework.Operator), "The input op should be Operator."
        var_names = []
        name_list = _op_real_in_out_name[op.type][1]
        for name in name_list:
            var_name = op.output(name)
            if isinstance(var_name, list):
                var_names.extend(var_name)
            else:
                var_names.append(var_name)
        return var_names

    def save_quantized_model(self, layer, path, input_spec=None, **config):
        """
        Save the quantized model for the inference.

        Args:
            layer (Layer): The Layer to be saved.
            path (str): The path prefix to save model. The format is ``dirname/file_prefix`` or ``file_prefix``.
            input_spec (list[InputSpec|Tensor], optional): Describes the input of the saved model's forward 
                method, which can be described by InputSpec or example Tensor. If None, all input variables of 
                the original Layer's forward method would be the inputs of the saved model. Default None.
            **configs (dict, optional): Other save configuration options for compatibility. We do not 
                recommend using these configurations, they may be removed in the future. If not necessary, 
                DO NOT use them. Default None.
                The following options are currently supported:
                (1) output_spec (list[Tensor]): Selects the output targets of the saved model.
                By default, all return variables of original Layer's forward method are kept as the 
                output of the saved model. If the provided ``output_spec`` list is not all output variables, 
                the saved model will be pruned according to the given ``output_spec`` list. 

        Returns:
            None
        """

        assert isinstance(
            layer, dygraph.Layer), "model must be the instance of dygraph.Layer"
        with dygraph.guard():
            layer.eval()
            for handle in self._register_hook_handle_list:
                handle.remove()
            for key in self._out_scale_dict:
                self._out_scale_dict[key] = float(self._out_scale_dict[key]
                                                  .numpy())

        paddle.jit.save(layer=layer, path=path, input_spec=input_spec, **config)

        if core.is_compiled_with_cuda():
            place = core.CUDAPlace(0)
        else:
            place = core.CPUPlace()
        exe = Executor(place)

        file_prefix = os.path.basename(path)
        dirname = os.path.dirname(path)
        model_filename = file_prefix + INFER_MODEL_SUFFIX
        params_filename = file_prefix + INFER_PARAMS_SUFFIX

        [inference_program, feed_target_names, fetch_targets] = (
            load_inference_model(
                dirname=dirname,
                executor=exe,
                model_filename=model_filename,
                params_filename=params_filename))

        # Traverse all ops in the program and find out the op matching
        # the Layer in the dynamic graph.
        layer_var_dict = {}
        for block in inference_program.blocks:
            for op in block.ops:
                if op.type in _op_real_in_out_name:
                    output_var_names = self._get_op_output_names(op)
                    for output_var_name in output_var_names:
                        output_var_tensor = block.var(output_var_name)
                        if output_var_tensor.dtype not in [
                                core.VarDesc.VarType.FP64,
                                core.VarDesc.VarType.FP32
                        ]:
                            continue
                        # Because the Layer in dygraph may correspond to multiple ops
                        # in static program after being saved. To ensure correctness,
                        # the outscale collected for output of dygraph Layer can only
                        # be set to the last op in the corresponding ops in static program.
                        #
                        # We can judge the execution order of the ops which corresponding
                        # to dygraph Layer by the name of output. And use dict to save
                        # the corresponding relationship between the dygraph Layer and the
                        # static graph op that needs to set the outscale attribute.
                        dynamic_layer_name, var_name_suffix = output_var_name.split(
                            ".")
                        if dynamic_layer_name in layer_var_dict:
                            if layer_var_dict[dynamic_layer_name][
                                    0] < var_name_suffix:
                                layer_var_dict[dynamic_layer_name] = [
                                    var_name_suffix, op
                                ]
                        else:
                            layer_var_dict[
                                dynamic_layer_name] = [var_name_suffix, op]

        # Because the naming styles of static and dynamic graph are different,
        # in order to avoid mistakes, we unify the name here.
        for (layer_name, var_name_op_list) in layer_var_dict.items():
            if 'prelu' in layer_name:
                layer_name = layer_name.replace('prelu', 'p_re_lu')
            if 'relu' in layer_name:
                layer_name = layer_name.replace('relu', 're_lu')
            if layer_name not in self._out_scale_dict:
                continue
            var_name_op_list[1]._set_attr('out_threshold',
                                          self._out_scale_dict[layer_name])

        # Save the processed program.
        save_inference_model(
            dirname=dirname,
            feeded_var_names=feed_target_names,
            target_vars=fetch_targets,
            executor=exe,
            main_program=inference_program.clone(),
            model_filename=model_filename,
            params_filename=params_filename)

    def _forward_post_hook(self, layer, input, output):
        assert isinstance(
            output, core.VarBase
        ), "Multiple outputs are not currently supported in ImperativeOutScale."
        if output.dtype not in [
                core.VarDesc.VarType.FP32, core.VarDesc.VarType.FP64
        ]:
            return
        if not hasattr(layer, "_out_scale"):
            layer._out_scale = quant_nn.MovingAverageAbsMaxScale(
                output.name, self._moving_rate, output.dtype)
        scale_out = layer._out_scale(output)
        self._out_scale_dict[layer.full_name()] = scale_out
