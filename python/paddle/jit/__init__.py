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

from __future__ import print_function

from ..fluid.dygraph.jit import save  #DEFINE_ALIAS
from ..fluid.dygraph.jit import load  #DEFINE_ALIAS
from ..fluid.dygraph.jit import TracedLayer  #DEFINE_ALIAS
from ..fluid.dygraph.jit import set_code_level  #DEFINE_ALIAS
from ..fluid.dygraph.jit import set_verbosity  #DEFINE_ALIAS
from ..fluid.dygraph.jit import declarative as to_static  #DEFINE_ALIAS
from ..fluid.dygraph import ProgramTranslator  #DEFINE_ALIAS
from ..fluid.dygraph.io import TranslatedLayer  #DEFINE_ALIAS

from . import dy2static

__all__ = [
    'save', 'load', 'TracedLayer', 'to_static', 'ProgramTranslator',
    'TranslatedLayer', 'set_code_level', 'set_verbosity'
]
