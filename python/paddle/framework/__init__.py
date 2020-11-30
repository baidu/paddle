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

# TODO: import framework api under this directory 
__all__ = [
    'create_parameter', 'ParamAttr', 'CPUPlace', 'CUDAPlace', 'CUDAPinnedPlace',
    'get_default_dtype', 'set_default_dtype'
]

__all__ += ['grad', 'LayerList', 'load', 'save', 'no_grad', 'DataParallel']

from . import random
from .random import seed
from .framework import get_default_dtype
from .framework import set_default_dtype

from ..fluid.param_attr import ParamAttr  #DEFINE_ALIAS
# from ..fluid.layers.tensor import create_global_var  #DEFINE_ALIAS
from ..fluid.layers.tensor import create_parameter  #DEFINE_ALIAS
from ..fluid.core import CPUPlace  #DEFINE_ALIAS
from ..fluid.core import CUDAPlace  #DEFINE_ALIAS
from ..fluid.core import CUDAPinnedPlace  #DEFINE_ALIAS
from ..fluid.core import VarBase  #DEFINE_ALIAS

from paddle.fluid import core  #DEFINE_ALIAS
from ..fluid.dygraph.base import no_grad_ as no_grad  #DEFINE_ALIAS
from ..fluid.dygraph.base import grad  #DEFINE_ALIAS
from .io import save
from .io import load
from ..fluid.dygraph.parallel import DataParallel  #DEFINE_ALIAS
