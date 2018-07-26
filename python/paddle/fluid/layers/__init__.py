#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

from . import ops
from .ops import *
from . import nn
from .nn import *
from . import io
from .io import *
from . import tensor
from .tensor import *
from . import control_flow
from .control_flow import *
from . import device
from .device import *
from . import math_op_patch
from .math_op_patch import *
from . import detection
from .detection import *
from . import metric_op
from .metric_op import *
from .learning_rate_scheduler import *

__all__ = []
__all__ += nn.__all__
__all__ += io.__all__
__all__ += tensor.__all__
__all__ += control_flow.__all__
__all__ += ops.__all__
__all__ += device.__all__
__all__ += detection.__all__
__all__ += metric_op.__all__
__all__ += learning_rate_scheduler.__all__
