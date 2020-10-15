# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

from . import spawn
from .spawn import spawn

from . import parallel
from .parallel import init_parallel_env
from .parallel import get_rank
from .parallel import get_world_size
from paddle.fluid.dygraph.parallel import prepare_context  #DEFINE_ALIAS
from paddle.fluid.dygraph.parallel import ParallelEnv  #DEFINE_ALIAS
from paddle.distributed.fleet.dataset import *

from . import collective
from .collective import *

# start multiprocess apis
__all__ = ["spawn"]

# dygraph parallel apis
__all__ += [
    "init_parallel_env",
    "get_rank",
    "get_world_size",
    "prepare_context",
    "ParallelEnv",
    "InMemoryDataset",
    "QueueDataset",
]

# collective apis
__all__ += collective.__all__
