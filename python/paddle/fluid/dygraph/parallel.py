# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except jin compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import six
import numpy as np
import warnings
from collections import OrderedDict

from paddle.fluid import core
from paddle.fluid import framework
from paddle.fluid.dygraph import layers
from paddle.fluid.dygraph import parallel_helper
from paddle.fluid.dygraph import to_variable, no_grad
from paddle.utils import deprecated

__all__ = ["prepare_context", "ParallelEnv", "DataParallel"]

ParallelStrategy = core.ParallelStrategy


@deprecated(since="2.0.0", update_to="paddle.distributed.init_parallel_env")
def prepare_context(strategy=None):
    '''
    :api_attr: imperative
    '''
    if strategy is None:
        strategy = ParallelStrategy()
        strategy.nranks = Env().nranks
        strategy.local_rank = Env().local_rank
        strategy.trainer_endpoints = Env().trainer_endpoints
        strategy.current_endpoint = Env().current_endpoint
    if strategy.nranks < 2:
        return
    assert framework.in_dygraph_mode() is True, \
        "dygraph.prepare_context should be used with dygraph mode."
    place = framework._current_expected_place()
    assert place is not None, \
        "dygraph.prepare_context should be used in fluid.dygraph.guard(place) guard."
    if not parallel_helper._is_parallel_ctx_initialized():
        if isinstance(place, core.CUDAPlace):
            parallel_helper._set_parallel_ctx(
                core.NCCLParallelContext(strategy, place))
        else:
            # TODO(Yancey1989): add Gloo Parallel Context to support CPU parallel computation
            assert ("Only support CUDAPlace for now.")
        parallel_helper._init_parallel_ctx()
    return strategy


class ParallelEnv(object):
    """
    .. note::
        This API is not recommended, if you need to get rank and world_size, 
        it is recommended to use ``paddle.distributed.get_rank()`` and 
        ``paddle.distributed.get_world_size()`` .

    This class is used to obtain the environment variables required for 
    the parallel execution of ``paddle.nn.Layer`` in dynamic mode.

    The parallel execution in dynamic mode needs to be started using ``paddle.distributed.launch`` 
    or ``paddle.distributed.spawn`` .

    Examples:
      .. code-block:: python

        import paddle
        import paddle.distributed as dist

        def train():
            # 1. initialize parallel environment
            dist.init_parallel_env()

            # 2. get current ParallelEnv
            parallel_env = dist.ParallelEnv()
            print("rank: ", parallel_env.rank)
            print("world_size: ", parallel_env.world_size)

            # print result in process 1:
            # rank: 1
            # world_size: 2
            # print result in process 2:
            # rank: 2
            # world_size: 2

        if __name__ == '__main__':
            # 1. start by ``paddle.distributed.spawn`` (default)
            dist.spawn(train, nprocs=2)
            # 2. start by ``paddle.distributed.launch``
            # train()
    """

    def __init__(self):
        self._rank = int(os.getenv("PADDLE_TRAINER_ID", "0"))
        self._world_size = int(os.getenv("PADDLE_TRAINERS_NUM", "1"))
        self._device_id = int(os.getenv("FLAGS_selected_gpus", "0"))
        self._trainer_endpoints = os.getenv("PADDLE_TRAINER_ENDPOINTS",
                                            "").split(",")
        self._current_endpoint = os.getenv("PADDLE_CURRENT_ENDPOINT", "")

    @property
    def rank(self):
        """
        Rank of current trainer.

        Its value is equal to the value of the environment variable ``PADDLE_TRAINER_ID`` . The default value is 0.

        Examples:
          .. code-block:: python

            # execute this command in terminal: export PADDLE_TRAINER_ID=0
            import paddle.distributed as dist
            
            env = dist.ParallelEnv()
            print("The rank is %d" % env.rank)
            # The rank is 0
        """
        return self._rank

    @property
    def world_size(self):
        """
        The number of trainers (number of processes participating in current job).

        Its value is equal to the value of the environment variable ``PADDLE_TRAINERS_NUM`` . The default value is 1.

        Examples:
          .. code-block:: python

            # execute this command in terminal: export PADDLE_TRAINERS_NUM=4
            import paddle.distributed as dist
            
            env = dist.ParallelEnv()
            print("The world_size is %d" % env.world_size)
            # The world_size is 4
        """
        return self._world_size

    @property
    def device_id(self):
        """
        The ID of selected GPU card for parallel training.

        Its value is equal to the value of the environment variable ``FLAGS_selected_gpus`` . The default value is 0.

        Examples:
          .. code-block:: python

            # execute this command in terminal: export FLAGS_selected_gpus=1
            import paddle.distributed as dist
            
            env = dist.ParallelEnv()
            print("The device id are %d" % env.device_id)
            # The device id are 1
        """
        return self._device_id

    @property
    def current_endpoint(self):
        """
        The endpoint of current trainer, it is in the form of (node IP + port).

        Its value is equal to the value of the environment variable ``PADDLE_CURRENT_ENDPOINT`` . The default value is "".

        Examples:
          .. code-block:: python
            
            # execute this command in terminal: export PADDLE_CURRENT_ENDPOINT=127.0.0.1:6170
            import paddle.distributed as dist
            
            env = dist.ParallelEnv()
            print("The current endpoint are %s" % env.current_endpoint)
            # The current endpoint are 127.0.0.1:6170
        """
        return self._current_endpoint

    @property
    def trainer_endpoints(self):
        """
        The endpoints of all trainer nodes in the task, 
        which are used to broadcast the NCCL ID when NCCL2 is initialized.

        Its value is equal to the value of the environment variable ``PADDLE_TRAINER_ENDPOINTS`` . The default value is "".

        Examples:
          .. code-block:: python

            # execute this command in terminal: export PADDLE_TRAINER_ENDPOINTS=127.0.0.1:6170,127.0.0.1:6171
            import paddle.distributed as dist
            
            env = dist.ParallelEnv()
            print("The trainer endpoints are %s" % env.trainer_endpoints)
            # The trainer endpoints are ['127.0.0.1:6170', '127.0.0.1:6171']
        """
        return self._trainer_endpoints

    # [aliases] Compatible with old method names
    local_rank = rank
    nranks = world_size
    dev_id = device_id


# NOTE: [ Compatible ] Originally this class name is `Env`. The semantics of the old class names
# are inaccurate and may confuse users, so replace it with `ParallelEnv`, but to be compatible
# with the old examples, here still need to keep this name.
Env = ParallelEnv


def _build_default_parallel_strategy():
    strategy = ParallelStrategy()
    strategy.nranks = ParallelEnv().nranks
    strategy.local_rank = ParallelEnv().local_rank
    strategy.trainer_endpoints = ParallelEnv().trainer_endpoints
    strategy.current_endpoint = ParallelEnv().current_endpoint
    return strategy


def _coalesce_tensors(var_groups):
    from ..layers import nn
    coalesced_grads_and_grad_vars = []
    for group_id, grad_vars in var_groups.items():
        flattened_vars = []
        g_var_shapes = []
        for g_var in grad_vars:
            g_var_shapes.append(g_var.shape)
            flattened_vars.append(
                nn.reshape(
                    x=g_var, shape=[np.prod(g_var.shape)]))
        coalesced_grad = nn.concat(flattened_vars)
        coalesced_grads_and_grad_vars.append(
            [coalesced_grad, grad_vars, g_var_shapes])
    return coalesced_grads_and_grad_vars


@framework.dygraph_only
def _reshape_inplace(x, shape):
    x_shape = framework._varbase_creator(dtype=x.dtype)
    framework._dygraph_tracer().trace_op(
        type="reshape2",
        inputs={'X': x},
        outputs={'Out': x,
                 'XShape': x_shape},
        attrs={'shape': shape})


@framework.dygraph_only
def _split_tensors(coalesced_grads_and_grad_vars):
    for coalesced_grad, origin_grad_vars, grad_shapes in coalesced_grads_and_grad_vars:
        grad_var_len = [np.prod(g_shape) for g_shape in grad_shapes]
        framework._dygraph_tracer().trace_op(
            type='split',
            inputs={'X': coalesced_grad},
            outputs={'Out': origin_grad_vars},
            attrs={'sections': grad_var_len,
                   'axis': 0})
        for g_var, g_shape in zip(origin_grad_vars, grad_shapes):
            _reshape_inplace(x=g_var, shape=g_shape)
            assert g_var.shape == g_shape


def scale_loss(loss):
    if not ParallelEnv().world_size > 1:
        return loss

    loss_scale = to_variable(
        np.array([ParallelEnv().world_size]).astype("float32"))
    loss_scale.stop_gradient = True
    scaled_loss = loss / loss_scale
    return scaled_loss


@no_grad
def apply_collective_grads(parameters):
    if not ParallelEnv().world_size > 1:
        return

    grad_var_set = set()
    grad_vars = []
    sparse_grad_vars = []
    strategy = _build_default_parallel_strategy()
    for param in parameters:
        # NOTE(zcd): The grad_ivar maybe no generated.
        if param.trainable and (param._grad_ivar() is not None):
            g_var = param._grad_ivar()
            if g_var._is_sparse():
                sparse_grad_vars.append(g_var)
                continue
            grad_vars.append(g_var)
            assert g_var not in grad_var_set
            grad_var_set.add(g_var)

    if sparse_grad_vars:
        sparse_grad_vars.sort(key=lambda x: x.name)
        for grad_var in sparse_grad_vars:
            grad_var._allreduce(strategy)

    # FIXME(zcd): the type of the var should be LoDTensor, i.e
    # the gradients should be dense, otherwise, the following
    # logic should be updated.
    # 128 MB as a group
    mega_bytes = 128 * 1024 * 1024
    group_idx = 0
    memory_counter = 0
    grad_var_groups = OrderedDict()
    dtype = grad_vars[0].dtype
    for g_var in grad_vars:
        # NOTE: the dtype of the same group should be the same.
        bytes = np.prod(g_var.shape) * core.size_of_dtype(g_var.dtype)
        if memory_counter < mega_bytes and dtype == g_var.dtype:
            memory_counter += bytes
        else:
            memory_counter = bytes
            group_idx += 1
        grad_var_groups.setdefault(group_idx, []).append(g_var)

    coalesced_grads_and_vars = _coalesce_tensors(grad_var_groups)

    for coalesced_grad, _, _ in coalesced_grads_and_vars:
        coalesced_grad._allreduce(strategy)

    _split_tensors(coalesced_grads_and_vars)


class DataParallel(layers.Layer):
    """
    Run the dygraph module with data parallelism.

    Currently, DataParallel class only supports to run the dynamic graph
    with multi-process. 
    
    Now supports two ways to start training:

    1. start by ``paddle.distributed.spawn`` method, for example:

        ``python demo.py`` (spawn need to be called in ``__main__`` method)
    
    2. start by ``paddle.distributed.launch`` module, for example:
    
        ``python -m paddle.distributed.launch --selected_gpus=0,1 demo.py`` .

    And the content of `demo.py` is the code of examples.

    Args:
        layers(Layer): The module that should be executed by data parallel.
        strategy(ParallelStrategy, optional): (deprecated) The strategy of data parallelism, 
            contains environment configuration related to parallel execution. Default: None.
            
    Returns:
        Layer: The data paralleled module.

    Examples:
        .. code-block:: python

            import paddle
            import paddle.nn as nn
            import paddle.optimizer as opt
            import paddle.distributed as dist

            class LinearNet(nn.Layer):
                def __init__(self):
                    super(LinearNet, self).__init__()
                    self._linear1 = nn.Linear(10, 10)
                    self._linear2 = nn.Linear(10, 1)
                    
                def forward(self, x):
                    return self._linear2(self._linear1(x))

            def train():
                # 1. enable dynamic mode
                paddle.disable_static()
                
                # 2. initialize parallel environment
                dist.init_parallel_env()

                # 3. create data parallel layer & optimizer
                layer = LinearNet()
                dp_layer = paddle.DataParallel(layer)

                loss_fn = nn.MSELoss()
                adam = opt.Adam(
                    learning_rate=0.001, parameters=dp_layer.parameters())

                # 4. run layer
                inputs = paddle.randn([10, 10], 'float32')
                outputs = dp_layer(inputs)
                labels = paddle.randn([10, 1], 'float32')
                loss = loss_fn(outputs, labels)
                
                loss.backward()

                adam.step()
                adam.clear_grad()

            if __name__ == '__main__':
                # 1. start by ``paddle.distributed.spawn`` (default)
                dist.spawn(train, nprocs=2)
                # 2. start by ``paddle.distributed.launch``
                # train()
    """

    def __init__(self, layers, strategy=None):
        super(DataParallel,
              self).__init__(layers.full_name() + "_data_parallel")

        self._layers = layers

        # NOTE(chenweihang): The ParallelStrategy here is not strictly a strategy. 
        # It just stores some environment variables, which can be constructed by 
        # ParallelEnv. Here it is set as an optional argument.
        # This parameter is not removed because of compatibility with 1.x writing.
        if strategy is not None:
            self._strategy = strategy
        else:
            self._strategy = _build_default_parallel_strategy()

    def forward(self, *inputs, **kwargs):
        return self._layers(*inputs, **kwargs)

    @deprecated(
        since="2.0.0", reason="This method does not need to be called anymore.")
    def scale_loss(self, loss):
        """
        Deprecated method, now ``scale_loss`` is an empty method,  
        keep this method just for compatibility.
        """
        return loss

    @deprecated(
        since="2.0.0", reason="This method does not need to be called anymore.")
    def apply_collective_grads(self):
        """
        Deprecated method, now ``apply_collective_grads`` is an empty method, 
        keep this method just for compatibility.
        """
        return

    def state_dict(self,
                   destination=None,
                   include_sublayers=True,
                   structured_name_prefix=""):
        '''
        Get all parameters of self._layers and its sub-layers. And set all the parameters into a dict

        Parameters:
            destination(dict, optional) : If provide, all the parameters will set to this dict . Default: None
            include_sublayers(bool, optional) : If true, also include the parameters from sublayers. Default: True
            structured_name_prefix(str, optional): If not empty str, all the key in state dict will start 
                                                 with structured_name_prefix

        Retruns:
            dict: a dict contains all the parameters of self._layers

        Examples:
            .. code-block:: python

                import paddle.fluid as fluid
                with fluid.dygraph.guard():
                    strategy=fluid.dygraph.prepare_context()
                    emb = fluid.dygraph.Embedding([10, 10])
                    emb = fluid.dygraph.DataParallel(emb, strategy)

                    state_dict = emb.state_dict()
                    fluid.save_dygraph( state_dict, "paddle_dy")

        '''

        return self._layers.state_dict(
            destination=destination,
            include_sublayers=include_sublayers,
            structured_name_prefix=structured_name_prefix)

    @framework.deprecate_stat_dict
    def set_state_dict(self,
                       state_dict,
                       include_sublayers=True,
                       use_structured_name=True):
        '''
        Set parameters of self._layers from state_dict. All the parameters of self._layers will be reset by the tensor in the state_dict

        Parameters:
            state_dict(dict) : Dict contains all the parameters
            include_sublayers(bool, optional) : If true, also include the parameters from sublayers. Default: True
            use_structured_name(bool, optional) : If true, use structured name as key, otherwise, use parameter name as key. 
                                                  Default: True
        Returns:
            None

        Examples:
            .. code-block:: python

                import paddle   

                paddle.disable_static()

                emb = paddle.nn.Embedding(10, 10)
                emb = fluid.dygraph.DataParallel(emb, strategy)

                state_dict = emb.state_dict()
                paddle.save(state_dict, "paddle_dy.pdparams")

                para_state_dict = paddle.load("paddle_dy.pdparams")

                emb.set_state_dict(para_state_dict)

        '''

        self._layers.set_state_dict(
            state_dict,
            include_sublayers=include_sublayers,
            use_structured_name=use_structured_name)

    # [aliases] Compatible with old method names
    set_dict = set_state_dict
    load_dict = set_state_dict
