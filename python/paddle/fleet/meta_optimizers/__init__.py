#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

from .recompute_optimizer import RecomputeOptimizer
from .gradient_merge_optimizer import GradientMergeOptimizer
from .graph_execution_optimizer import GraphExecutionOptimizer
from .async_optimizer import AsyncOptimizer
from .pipeline_optimizer import PipelineOptimizer
from .localsgd_optimizer import LocalSGDOptimizer

__all__ = [
    'RecomputeOptimizer',
    'GradientMergeOptimizer',
    'AsyncOptimizer',
    'GraphExecutionOptimizer',
    'PipelineOptimizer',
    'LocalSGDOptimizer',
]
