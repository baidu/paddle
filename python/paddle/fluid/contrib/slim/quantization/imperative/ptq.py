#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
import copy
import numpy as np

import paddle
from paddle.fluid.log_helper import get_logger

from . import utils
from . import ptq_hooks
from . import ptq_config
from .ptq_registry import PTQRegistry

__all__ = ['ImperativePTQ']

_logger = get_logger(
    __name__, logging.INFO, fmt='%(asctime)s-%(levelname)s: %(message)s')


class ImperativePTQ(object):
    """
    Applying static post_training quantization to the dgraph model.
    """

    def __init__(self, quant_config=ptq_config.default_ptq_config):
        """
        Constructor.
        Args:
            algo(str): The algorithm in post_training quantizaion to be used.
            activation_bits(int): quantization bit number for activations.
            weight_bits(int): quantization bit number for weights.
        """
        super(ImperativePTQ, self).__init__()

        assert isinstance(quant_config, ptq_config.PTQConfig)

        self._quant_config = quant_config

    def quantize(self, model, inplace=False):
        """
        Add hook to the leaf layer to calculate the threshold of inputs and outputs.

        Args:
            model(paddle.nn.Layer): The model to be quantized.
        Returns:
            None
        """
        assert isinstance(model, paddle.nn.Layer), \
            "The model must be the instance of paddle.nn.Layer."

        if not inplace:
            model = copy.deepcopy(model)

        for name, layer in model.named_sublayers():
            if PTQRegistry.is_supported_layer(layer) \
                and utils.is_leaf_layer(layer):
                quant_config = copy.deepcopy(self._quant_config)
                layer._quant_config = quant_config

                hook = ptq_hooks.quant_forward_post_hook
                hook_handle = layer.register_forward_post_hook(hook)
                quant_config.hook_handle = hook_handle
                layer._forward_post_hooks.move_to_end(
                    hook_handle._hook_id, last=False)

        return model

    def convert(self, model):
        """
        Process the scales and remove the hooks.

        Args:
            model(paddle.nn.Layer): The model to be quantized.
        Returns:
            None
        """
        assert isinstance(model, paddle.nn.Layer), \
            "The input model must be the instance of paddle.nn.Layer."

        for name, sub_layer in model.named_sublayers():
            if PTQRegistry.is_supported_layer(sub_layer) \
                and utils.is_leaf_layer(sub_layer):

                assert hasattr(sub_layer, "_quant_config")
                quant_config = sub_layer._quant_config
                quant_config.hook_handle.remove()

                quant_config.in_act_quantizer.cal_thresholds()
                quant_config.out_act_quantizer.cal_thresholds()

                # get weight thresholds
                if isinstance(sub_layer, tuple(utils.fake_quant_input_layers)):
                    weights = (sub_layer.weight, )
                    quant_config.wt_quantizer.sample_data(sub_layer, weights)

                # TODO (jc): 
                # save input activation threshold and quant bits

        return model
