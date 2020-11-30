#  Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
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
from ....fluid import layers
from ....tensor import math

__all__ = [
    'elementwise_add',
    'elementwise_sub',
    'elementwise_mul',
    'elementwise_div',
    'kron',
    'trace',
    'sum',
]

from ....fluid.layers import elementwise_add  #DEFINE_ALIAS
from ....fluid.layers import elementwise_sub  #DEFINE_ALIAS
from ....fluid.layers import elementwise_mul  #DEFINE_ALIAS
from ....fluid.layers import elementwise_div  #DEFINE_ALIAS
from ....tensor.math import trace  #DEFINE_ALIAS
from ....tensor.math import kron  #DEFINE_ALIAS
from ....tensor.math import sum  #DEFINE_ALIAS
