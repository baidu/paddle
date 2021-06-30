#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
import itertools
from paddle.fluid import core
from paddle.fluid.dygraph.layers import Layer
from paddle.fluid import framework
from .meta_parallel_base import MetaParallelBase
from ..utils.hybrid_parallel_util import broadcast_dp_parameters
from ..utils.hybrid_parallel_util import broadcast_input_data
from ..utils.hybrid_parallel_util import broadcast_mp_parameters
from ..utils.log_util import logger

__all__ = []


class TensorParallel(MetaParallelBase):
    def __init__(self, layers, hcg, **kwargs):
        super(TensorParallel, self).__init__(layers, hcg, **kwargs)

        self._need_dp = (self._hcg.get_data_parallel_world_size() > 1)
        self._dp_group = self._hcg.get_data_parallel_group()
        if self._need_dp:
            assert self._dp_group.parallel_context is not None
            self.comm_buffer_size = int(50 * 1024 * 1024)
            self.last_comm_buffer_size = int(1 * 1024 * 1024)
            self.init_reducer()

    def _prepare_for_model(self):
        logger.info("start broadcast mp parameters")
        broadcast_mp_parameters(self._layers, self._hcg)

        logger.info("start broadcast dp parameters")
        broadcast_dp_parameters(self._layers, self._hcg)

        logger.info("mp's parameters is ready")

    def _pre_forward(self, *inputs, **kwargs):
        logger.debug("mp start broadcast input data")
        return broadcast_input_data(self._hcg, *inputs, **kwargs)

    def init_reducer(self):
        layers_param = []
        params_set = set()
        for sublayer in self.sublayers():
            for _, param in sublayer.named_parameters(include_sublayers=False):
                if param is None or param in params_set:
                    continue
                params_set.add(param)
                if not isinstance(param, core.VarBase):
                    raise TypeError("The data type of '%s' must be Varbase" %
                                    param.name)
                if param.trainable:
                    layers_param.append((sublayer, param))

        trainable_parameters = [param for _, param in layers_param]

        assert len(trainable_parameters) > 0, \
            "This model does not have any parameters to train, and " \
            "does not need to use DataParallel"

        def check_layer_sparse(sublayer):
            if isinstance(sublayer, paddle.nn.layer.common.Embedding):
                return sublayer._sparse
            if isinstance(sublayer, paddle.fluid.dygraph.Embedding):
                return sublayer._is_sparse
            return False

        is_sparse_gradient = [
            check_layer_sparse(sublayer) for sublayer, _ in layers_param
        ]

        self.group_indices = core.assign_group_by_size(
            trainable_parameters, is_sparse_gradient,
            [self.last_comm_buffer_size, self.comm_buffer_size])

        self._reducer = core.Reducer(
            trainable_parameters,
            list(reversed(self.group_indices)), is_sparse_gradient,
            self._dp_group.parallel_context,
            [self.last_comm_buffer_size, self.comm_buffer_size], False)

    def _find_varbase(self, obj):
        if isinstance(obj, core.VarBase):
            return [obj]
        if isinstance(obj, (list, tuple)):
            return itertools.chain(*map(self._find_varbase, obj))
        if isinstance(obj, dict):
            return itertools.chain(*map(self._find_varbase, obj.values()))
        return []

    def forward(self, *inputs, **kwargs):
        outputs = self._layers(*inputs, **kwargs)
        if self._need_dp > 1 and framework._dygraph_tracer()._has_grad:
            self._reducer.prepare_for_backward(
                list(self._find_varbase(outputs)))
        return outputs
