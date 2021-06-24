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

from types import MethodType

import paddle
import paddle.fluid as fluid
from .meta_parallel_base import MetaParallelBase
from .pp_utils.utils import is_float_tensor, get_tensor_dtype, paddle_2_number, number_2_dtype
from .pp_utils import utils
from .parallel_layers.pp_layers import PipelineLayer

from ..utils.hybrid_parallel_util import broadcast_mp_parameters
from ..utils.hybrid_parallel_util import broadcast_dp_parameters
from ..utils.log_util import logger
from ..meta_optimizers.dygraph_optimizer import HybridParallelOptimizer

__all__ = []


class PipelineParallel(MetaParallelBase):
    def __init__(self, layers, hcg, strategy):
        if not isinstance(layers, PipelineLayer):
            raise TypeError(
                "The Layer should be a derived class of PipelineLayer.")
        super(PipelineParallel, self).__init__(layers, hcg, strategy)
        self.use_pipe_parallel = self._hcg.get_pipe_parallel_world_size() > 1
        self.use_data_parallel = self._hcg.get_data_parallel_world_size() > 1
        self.use_model_parallel = self._hcg.get_model_parallel_world_size() > 1

        self.num_caches = 0
        self.caches = {
            'inputs': [],
            'labels': [],
            'outputs': [],
        }

        self.recv_cache = None
        self.grad_tensors = None

        self.send_meta = True

        self.current_loss = paddle.to_tensor(0.0)
        self.total_loss = None

        self.micro_batch_size = self._strategy.pipeline_configs[
            'micro_batch_size']
        self.accumulate_steps = self._strategy.pipeline_configs[
            'accumulate_steps']

        self.num_stages = self._hcg.get_pipe_parallel_world_size()
        self.stage_id = self._hcg.get_stage_id()
        self.prev_stage_id = self.stage_id - 1
        self.next_stage_id = self.stage_id + 1
        self.pp_group = self._hcg.get_pipe_parallel_group()

        self.is_first_stage = self.stage_id == 0
        self.is_last_stage = (self.stage_id == (self.num_stages - 1))
        self.global_rank = self._hcg.get_global_rank()

        logger.info("Pipeline Info -- num_stages: {}, stage_id: {}".format(
            self.num_stages, self.stage_id))

        if self.use_model_parallel:
            logger.info("start broadcast mp parameters")
            broadcast_mp_parameters(self._layers, self._hcg)

        if self.use_data_parallel:
            logger.info("start broadcast dp parameters")
            broadcast_dp_parameters(self._layers, self._hcg)

    def _init_caches(self, num_caches):
        if self.num_caches >= num_caches:
            return
        self.num_caches = num_caches - self.num_caches
        for key in self.caches:
            self.caches[key].extend([None] * self.num_caches)

    def _reduce_final_loss(self):
        if self.is_last_stage:
            assert self.total_loss is not None, "train_batch() in last stage should obtain vaild loss"
            loss = self.total_loss.clone() / self.accumulate_steps
            paddle.distributed.broadcast(
                loss,
                src=self.global_rank,
                use_calc_stream=True,
                group=self.pp_group)
        else:
            loss = paddle.to_tensor(0.0)
            paddle.distributed.broadcast(
                loss,
                src=self._hcg.get_rank_from_stage(self.num_stages - 1),
                use_calc_stream=True,
                group=self.pp_group)
        return loss

    def train_batch(self, data, optimizer, lr_scheduler=None):
        assert isinstance(optimizer, HybridParallelOptimizer), (
            'optimizer should be HybridParallelOptimizer subclass.')
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        assert fluid.framework._dygraph_tracer()._has_grad, (
            'Please enable the generation of gradients.')

        if self.is_first_stage or self.is_last_stage:
            assert data is not None, (
                "For the first and the last stage, the data_iter must be set.")
        else:
            data = None

        self.data = data
        self._layers.train()

        # store total loss of entire batch
        self.total_loss = None
        self._init_caches(self.accumulate_steps)
        startup_steps = self.num_stages - self.stage_id - 1
        forward_steps = 0
        backward_steps = 0

        # forward
        while (forward_steps < self.accumulate_steps):
            self._forward(cache_id=forward_steps)
            forward_steps += 1

        # backward
        while (backward_steps < self.accumulate_steps):
            self._backward(cache_id=backward_steps)
            backward_steps += 1

        self._layers.allreduce_shared_weight_gradients()

        # optimizer
        self._step()
        self.train_loss = self._reduce_final_loss()
        return self.train_loss

    def _forward(self, cache_id):
        # load data
        self._load_micro_batch(cache_id)
        if self.stage_id != 0:
            self._recv_activations(cache_id)

        if isinstance(self.caches['inputs'][cache_id], tuple):
            inputs = tuple(t for t in self.caches['inputs'][cache_id])
        else:
            inputs = self.caches['inputs'][cache_id]

        outputs = self._layers.forward(inputs)
        self._clear_grads(inputs)

        self.caches['outputs'][cache_id] = outputs

        if self.is_last_stage:
            if self._layers._loss_fn is not None:
                labels = self.caches['labels'][cache_id]
                outputs = self._layers._loss_fn(outputs, labels)

        if self.is_last_stage:
            self.current_loss = outputs
            if isinstance(self.current_loss, paddle.Tensor):
                if self.total_loss is None:
                    self.total_loss = paddle.zeros_like(self.current_loss)
                self.total_loss += self.current_loss.detach()
            else:
                if self.total_loss is None:
                    self.total_loss = [
                        paddle.zeros_like(v) for v in self.current_loss
                    ]
                for idx, v in enumerate(self.current_loss):
                    self.total_loss[idx] += v.detach()

            if self.accumulate_steps > 1:
                self.current_loss = self.current_loss / self.accumulate_steps

            self.caches['outputs'][cache_id] = self.current_loss.clone()

        else:
            self._send_activations(cache_id)

    def _backward(self, cache_id):
        if self.is_last_stage:
            paddle.autograd.backward(self.caches['outputs'][cache_id])
            self._send_gradients(cache_id)
            return
        self._recv_gradients(cache_id)

        outputs = self.caches['outputs'][cache_id]

        grad_tensors = self.grad_tensors
        if isinstance(outputs, tuple):
            out_tensors = [t for t in outputs if is_float_tensor(t)]
            assert len(out_tensors) == len(grad_tensors)
            paddle.autograd.backward(
                tensors=out_tensors, grad_tensors=grad_tensors)
        else:
            paddle.autograd.backward(
                tensors=[outputs], grad_tensors=[grad_tensors])

        grad_tensors = None
        if self.stage_id != 0: self._send_gradients(cache_id)
        self.caches['outputs'][cache_id] = None

    def _broadcast_data(self, data):
        if isinstance(data, paddle.Tensor):
            paddle.distributed.broadcast(
                data,
                src=self._hcg.get_model_parallel_group_src_rank(),
                group=self._hcg.get_model_parallel_group())
        else:
            for d in data:
                assert isinstance(d, paddle.Tensor)
                paddle.distributed.broadcast(
                    d,
                    src=self._hcg.get_model_parallel_group_src_rank(),
                    group=self._hcg.get_model_parallel_group())
        return data

    def _load_micro_batch(self, cache_id):
        inputs = self.data
        begin = cache_id * self.micro_batch_size
        end = begin + self.micro_batch_size

        if self.is_first_stage:
            assert len(inputs) == 2, "length of input should be 2"
            if self.use_model_parallel:
                inputs[0] = self._broadcast_data(inputs[0])
            if isinstance(inputs[0], tuple):
                batch_size = inputs[0][0].shape[0]
                assert self.micro_batch_size * self.accumulate_steps == batch_size, (
                    "batch_size needs to be divisible by micro_batch_size. Currently, "
                    "batch_size = %d, micro_batch_size = %d, accumulate_steps = %d."
                    %
                    (batch_size, self.micro_batch_size, self.accumulate_steps))
                data = [
                    input[begin:end, :].clone().detach() for input in inputs[0]
                ]
                self.caches['inputs'][cache_id] = tuple(data)
            else:
                batch_size = inputs[0].shape[0]
                assert self.micro_batch_size * self.accumulate_steps == batch_size
                self.caches['inputs'][cache_id] = inputs[0][begin:end, :].clone(
                ).detach()
        elif self.is_last_stage:
            assert len(inputs) == 2, "length of input should be 2"
            if self.use_model_parallel:
                inputs[1] = self._broadcast_data(inputs[1])
            if isinstance(inputs[1], tuple):
                batch_size = inputs[1][0].shape[0]
                assert self.micro_batch_size * self.accumulate_steps == batch_size
                data = [
                    input[begin:end, :].clone().detach() for input in inputs[1]
                ]
                self.caches['labels'][cache_id] = tuple(data)
            else:
                batch_size = inputs[1].shape[0]
                assert self.micro_batch_size * self.accumulate_steps == batch_size
                self.caches['labels'][cache_id] = inputs[1][begin:end, :].clone(
                ).detach()
        else:
            # No data input is required for other stages
            inputs = None

    def _send_meta(self, data, peer):
        if isinstance(data, paddle.Tensor):
            tensor_type = paddle.to_tensor([0])
            # send tensor type
            paddle.distributed.send(
                tensor_type, peer, use_calc_stream=True, group=self.pp_group)

            # send len(shape)
            dims = paddle.to_tensor(len(data.shape))
            paddle.distributed.send(
                dims, peer, use_calc_stream=True, group=self.pp_group)

            # send shape
            shape = paddle.to_tensor(data.shape)
            paddle.distributed.send(
                shape, peer, use_calc_stream=True, group=self.pp_group)

            # send dtype
            dtype = paddle.to_tensor(paddle_2_number(data.dtype))
            paddle.distributed.send(
                dtype, peer, use_calc_stream=True, group=self.pp_group)

        elif isinstance(data, tuple):
            tensor_type = paddle.to_tensor([1])
            paddle.distributed.send(
                tensor_type, peer, use_calc_stream=True, group=self.pp_group)
            nums = paddle.to_tensor(len(data))
            paddle.distributed.send(
                nums, peer, use_calc_stream=True, group=self.pp_group)
            for idx, d in enumerate(data):
                assert isinstance(d, paddle.Tensor)
                # send len(shape)
                dims = paddle.to_tensor(len(d.shape))
                paddle.distributed.send(
                    dims, peer, use_calc_stream=True, group=self.pp_group)

                # send shape
                shape = paddle.to_tensor(d.shape)
                paddle.distributed.send(
                    shape, peer, use_calc_stream=True, group=self.pp_group)

                # send dtype
                dtype = paddle.to_tensor(paddle_2_number(d.dtype))
                paddle.distributed.send(
                    dtype, peer, use_calc_stream=True, group=self.pp_group)

    def _recv_meta(self, peer):
        tensor_type = paddle.to_tensor([0])
        paddle.distributed.recv(
            tensor_type, peer, use_calc_stream=True, group=self.pp_group)
        tensor_type = tensor_type.item()

        if tensor_type == 0:
            # recv len(shape)
            dims = paddle.to_tensor([0])
            paddle.distributed.recv(
                dims, peer, use_calc_stream=True, group=self.pp_group)
            dims = dims.item()

            # recv shape
            shape = paddle.to_tensor([0] * dims)
            paddle.distributed.recv(
                shape, peer, use_calc_stream=True, group=self.pp_group)
            shape = shape.numpy().tolist()

            # recv dtype
            dtype = paddle.to_tensor([0])
            paddle.distributed.recv(
                dtype, peer, use_calc_stream=True, group=self.pp_group)
            return self._allocate_cache(
                shape, dtype=number_2_dtype(dtype.item()), num_caches=1)[0]
        elif tensor_type == 1:
            num = paddle.to_tensor([0])
            paddle.distributed.recv(
                num, peer, use_calc_stream=True, group=self.pp_group)
            num = num.item()
            shapes = []
            dtypes = []
            for i in range(num):
                # recv len(shape)
                dims = paddle.to_tensor([0])
                paddle.distributed.recv(
                    dims, peer, use_calc_stream=True, group=self.pp_group)

                # recv shape
                dims = dims.item()
                shape = paddle.to_tensor([0] * dims)
                paddle.distributed.recv(
                    shape, peer, use_calc_stream=True, group=self.pp_group)
                shapes.append(shape.numpy().tolist())

                # recv dtype
                dtype = paddle.to_tensor([0])
                paddle.distributed.recv(
                    dtype, peer, use_calc_stream=True, group=self.pp_group)
                dtypes.append(number_2_dtype(dtype.item()))

            caches = self._allocate_caches(shapes, dtypes, num_caches=1)[0]
            caches = tuple(caches)
            return caches

    def _send_activations(self, cache_id):
        outputs = self.caches['outputs'][cache_id]

        if self.send_meta:
            self.send_meta = False
            self._send_meta(outputs, self.next_stage_id)

        if isinstance(outputs, paddle.Tensor):
            paddle.distributed.send(
                outputs,
                self.next_stage_id,
                use_calc_stream=True,
                group=self.pp_group)
        elif isinstance(outputs, tuple):
            for output in outputs:
                paddle.distributed.send(
                    output,
                    self.next_stage_id,
                    use_calc_stream=True,
                    group=self.pp_group)

    def _send_gradients(self, cache_id):
        inputs = self.caches['inputs'][cache_id]
        if isinstance(inputs, paddle.Tensor):
            assert inputs.grad is not None
            paddle.distributed.send(
                paddle.to_tensor(inputs.grad),
                self.prev_stage_id,
                use_calc_stream=True,
                group=self.pp_group)
        else:
            for idx, d in enumerate(inputs):
                # Skip tensors that will not produce a grad
                if not is_float_tensor(d):
                    assert d.grad is None
                    continue
                paddle.distributed.send(
                    d.grad,
                    self.prev_stage_id,
                    use_calc_stream=True,
                    group=self.pp_group)
        self.caches['inputs'][cache_id] = None

    def _recv_activations(self, cache_id):
        inputs = None
        if self.recv_cache is None:
            self.recv_cache = self._recv_meta(self.prev_stage_id)

        if isinstance(self.recv_cache, paddle.Tensor):
            paddle.distributed.recv(
                self.recv_cache,
                self.prev_stage_id,
                use_calc_stream=True,
                group=self.pp_group)
            inputs = self.recv_cache.clone().detach()
            inputs.stop_gradient = not is_float_tensor(inputs)
        else:
            assert isinstance(self.recv_cache, tuple)
            inputs = [None] * len(self.recv_cache)
            for idx, d in enumerate(self.recv_cache):
                assert isinstance(d, paddle.Tensor)

                paddle.distributed.recv(
                    d,
                    self.prev_stage_id,
                    use_calc_stream=True,
                    group=self.pp_group)
                inputs[idx] = d.clone().detach()

            inputs = tuple(inputs)

            for d in inputs:
                d.stop_gradient = not is_float_tensor(d)

        self.caches['inputs'][cache_id] = inputs

    def _recv_gradients(self, cache_id):
        outputs = self.caches['outputs'][cache_id]
        if self.grad_tensors is None:
            if isinstance(outputs, paddle.Tensor):
                s = list(outputs.shape)
                dtype = get_tensor_dtype(outputs.dtype)
                self.grad_tensors = self._allocate_cache(
                    s, dtype, num_caches=1)[0]
            else:
                sizes = [list(d.shape) for d in outputs if is_float_tensor(d)]
                dtypes = [
                    get_tensor_dtype(d.dtype) for d in outputs
                    if is_float_tensor(d)
                ]
                self.grad_tensors = self._allocate_caches(
                    sizes, dtypes, num_caches=1)[0]

        if isinstance(self.grad_tensors, paddle.Tensor):
            paddle.distributed.recv(
                self.grad_tensors,
                self.next_stage_id,
                use_calc_stream=True,
                group=self.pp_group)
        else:
            assert isinstance(outputs, tuple)
            for d in self.grad_tensors:
                paddle.distributed.recv(
                    d,
                    self.next_stage_id,
                    use_calc_stream=True,
                    group=self.pp_group)

    def _step(self):
        self.optimizer.step()
        self.optimizer.clear_grad()
        if self.lr_scheduler:
            self.lr_scheduler.step()

    def _clear_grads(self, inputs):
        if isinstance(inputs, paddle.Tensor):
            if inputs.grad is not None:
                inputs.clear_gradient()
        else:
            for d in inputs:
                if d.grad is not None:
                    d.clear_gradient()

    def _allocate_zeros(self, shape, dtype):
        return paddle.zeros(shape, dtype)

    def _allocate_cache(self, shape, dtype, num_caches=-1):
        caches = []
        if num_caches == -1:
            num_caches = self.num_caches
        for count in range(num_caches):
            caches.append(self._allocate_zeros(shape, dtype))
        return caches

    def _allocate_caches(self, shapes, dtypes, num_caches=-1):
        caches = []
        if num_caches == -1:
            num_caches = self.num_caches
        for count in range(num_caches):
            cache = []
            for shape, dtype in zip(shapes, dtypes):
                cache.append(self._allocate_zeros(shape, dtype))
            caches.append(cache)
        return caches

    def save_state_dict(self, model_path):
        state_dict = self._layers.state_dict()
        paddle.save(state_dict, model_path)

    def load_state_dict(self, model_path):
        state_dict = paddle.load(self.model_path)
        self._layers.set_state_dict(state_dict)

    def forward(self, *inputs, **kwargs):
        raise RuntimeError("Call train_batch for pipeline instead of forward.")
