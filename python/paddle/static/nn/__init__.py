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

__all__ = [
    'fc',
    'batch_norm',
    'embedding',
    'bilinear_tensor_product',
    'case',
    'cond',
    'conv2d',
    'conv2d_transpose',
    'conv3d',
    'conv3d_transpose',
    'create_parameter',
    'crf_decoding',
    'data_norm',
    'deform_conv2d',
    'group_norm',
    'instance_norm',
    'layer_norm',
    'multi_box_head',
    'nce',
    'prelu',
    'py_func',
    'row_conv',
    'spectral_norm',
    'switch_case',
    'while_loop',
]

from .common import fc  #DEFINE_ALIAS
from .common import deform_conv2d  #DEFINE_ALIAS

from ...fluid.layers import batch_norm  #DEFINE_ALIAS
from ...fluid.layers import bilinear_tensor_product  #DEFINE_ALIAS
from ...fluid.layers import case  #DEFINE_ALIAS
from ...fluid.layers import cond  #DEFINE_ALIAS
from ...fluid.layers import conv2d  #DEFINE_ALIAS
from ...fluid.layers import conv2d_transpose  #DEFINE_ALIAS
from ...fluid.layers import conv3d  #DEFINE_ALIAS
from ...fluid.layers import conv3d_transpose  #DEFINE_ALIAS
from ...fluid.layers import create_parameter  #DEFINE_ALIAS
from ...fluid.layers import crf_decoding  #DEFINE_ALIAS
from ...fluid.layers import data_norm  #DEFINE_ALIAS
from ...fluid.layers import group_norm  #DEFINE_ALIAS
from ...fluid.layers import instance_norm  #DEFINE_ALIAS
from ...fluid.layers import layer_norm  #DEFINE_ALIAS
from ...fluid.layers import multi_box_head  #DEFINE_ALIAS
from ...fluid.layers import nce  #DEFINE_ALIAS
from ...fluid.layers import prelu  #DEFINE_ALIAS
from ...fluid.layers import py_func  #DEFINE_ALIAS
from ...fluid.layers import row_conv  #DEFINE_ALIAS
from ...fluid.layers import spectral_norm  #DEFINE_ALIAS
from ...fluid.layers import switch_case  #DEFINE_ALIAS
from ...fluid.layers import while_loop  #DEFINE_ALIAS

from ...fluid.input import embedding  #DEFINE_ALIAS
