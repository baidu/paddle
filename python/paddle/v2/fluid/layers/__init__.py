#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserve.
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

import ops
from ops import *
import nn
from nn import *
import io
from io import *
import tensor
from tensor import *
import control_flow
from control_flow import *
import device
from device import *

__all__ = []
__all__ += nn.__all__
__all__ += io.__all__
__all__ += tensor.__all__
__all__ += control_flow.__all__
__all__ += ops.__all__
__all__ += device.__all__
