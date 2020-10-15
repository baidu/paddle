# Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved
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

from .profiler import ProfilerOptions
from .profiler import Profiler
from .profiler import get_profiler
from .deprecated import deprecated
from .lazy_import import try_import
from .install_check import run_check
from ..fluid.framework import unique_name
from ..fluid.framework import load_op_library
from ..fluid.framework import require_version

from . import download

__all__ = ['dump_config', 'deprecated', 'download', 'run_check']

#TODO: define new api under this directory
__all__ += ['unique_name', 'load_op_library', 'require_version']
