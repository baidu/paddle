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

# TODO: define the initializers to create a Parameter in neural network
from ...fluid.initializer import Bilinear  #DEFINE_ALIAS

from . import constant
from .constant import Constant  #DEFINE_ALIAS

from . import kaiming
from .kaiming import KaimingNormal  #DEFINE_ALIAS
from .kaiming import KaimingUniform  #DEFINE_ALIAS

__all__ = ['Bilinear', ]

__all__ += constant.__all__
__all__ += kaiming.__all__

from . import xavier
from .xavier import XavierNormal  #DEFINE_ALIAS
from .xavier import XavierUniform  #DEFINE_ALIAS

from . import assign
from .assign import Assign  #DEFINE_ALIAS

from . import normal
from .normal import Normal  #DEFINE_ALIAS
from .normal import TruncatedNormal  #DEFINE_ALIAS

from . import uniform
from .uniform import Uniform  #DEFINE_ALIAS

__all__ += xavier.__all__
__all__ += assign.__all__
__all__ += normal.__all__
__all__ += uniform.__all__
