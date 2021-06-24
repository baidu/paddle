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

import six
import abc
import copy

import paddle

from .ptq_quantizer import *

__all__ = ['PTQConfig', 'default_ptq_config']


class PTQConfig(object):
    """
    The PTQ config shows how to quantize the inputs and outputs.
    """

    def __init__(self, activation_quantizer, weight_quantizer):
        super(PTQConfig, self).__init__()

        assert isinstance(activation_quantizer, BaseQuantizer)
        assert isinstance(weight_quantizer, BaseQuantizer)

        self.in_act_quantizer = copy.deepcopy(activation_quantizer)
        self.out_act_quantizer = copy.deepcopy(activation_quantizer)
        self.wt_quantizer = copy.deepcopy(weight_quantizer)

        self.hook_handle = None


default_ptq_config = PTQConfig(AbsmaxQuantizer(), AbsmaxQuantizer())
