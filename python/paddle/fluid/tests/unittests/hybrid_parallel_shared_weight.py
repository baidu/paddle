# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import division
from __future__ import print_function

import unittest
import paddle
import numpy as np
import random
import paddle
import paddle.distributed as dist
import paddle.distributed.fleet as fleet
from paddle.fluid.dygraph.container import Sequential
from paddle.distributed.fleet.meta_parallel import PipelineLayer
from paddle.fluid.dygraph.layers import Layer
import paddle.nn as nn
import paddle.fluid as fluid
from paddle.distributed.fleet.meta_parallel import LayerDesc, SharedLayerDesc


def print_hook_fn(grad):
    print(grad)


def set_random_seed(seed, dp_id, rank_id):
    """Set random seed for reproducability."""
    random.seed(seed)
    np.random.seed(seed + dp_id)
    paddle.seed(seed + dp_id)


batch_size = 8
micro_batch_size = 2
vocab_size = 128
hidden_size = 16


class SimpleNet(Layer):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)

        self.softmax_weight = self.create_parameter(
            shape=[hidden_size, vocab_size])
        self.softmax_bias = self.create_parameter(
            shape=[vocab_size], is_bias=False)

    def forward(self, x1, x2, y1):
        x_emb = self.word_embeddings(x1)
        fc = fluid.layers.matmul(x_emb, self.softmax_weight)
        fc = fluid.layers.elementwise_add(fc, self.softmax_bias)
        projection = fluid.layers.reshape(fc, shape=[-1, vocab_size])

        projection = paddle.matmul(projection, self.word_embeddings.weight)

        loss = fluid.layers.softmax_with_cross_entropy(
            logits=projection, label=y1, soft_label=False)
        return loss.mean()


class EmbeddingPipe(Layer):
    def __init__(self):
        super(EmbeddingPipe, self).__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)

    @property
    def embedding_weight(self):
        return self.word_embeddings.weight

    def forward(self, args):
        x1, x2 = args
        x_emb = self.word_embeddings(x1)
        return x_emb, x2


class MatmulNet(Layer):
    def __init__(self):
        super(MatmulNet, self).__init__()
        self.softmax_weight = self.create_parameter(
            shape=[hidden_size, vocab_size])

    def forward(self, args):
        x1, x2 = args
        fc = fluid.layers.matmul(x1, self.softmax_weight)

        return fc, x2


class BiasNet(Layer):
    def __init__(self):
        super(BiasNet, self).__init__()
        self.softmax_bias = self.create_parameter(shape=[vocab_size])

    def forward(self, args):
        fc, x2 = args
        fc = fluid.layers.elementwise_add(fc, self.softmax_bias)
        projection = fluid.layers.reshape(fc, shape=[-1, vocab_size])
        return projection, x2


class LossNet(Layer):
    def __init__(self):
        super(LossNet, self).__init__()

    def forward(self, args, y1):
        projection = args
        loss = fluid.layers.softmax_with_cross_entropy(
            logits=projection, label=y1[0], soft_label=False)
        return loss.mean()


class SimpleNetPipe(PipelineLayer):
    def __init__(self, **kwargs):
        self.descs = []
        self.descs.append(
            SharedLayerDesc(
                'embed', EmbeddingPipe, shared_weight_attr='embedding_weight'))
        self.descs.append(LayerDesc(MatmulNet))

        self.descs.append(LayerDesc(BiasNet))

        def _logits_helper(embedding, output):
            return paddle.matmul(output[0], embedding.embedding_weight)

        self.descs.append(
            SharedLayerDesc(
                'embed',
                EmbeddingPipe,
                forward_func=_logits_helper,
                shared_weight_attr='embedding_weight'))

        super(SimpleNetPipe, self).__init__(
            layers=self.descs, loss_fn=LossNet(), **kwargs)


class TestDistEmbeddingTraning(unittest.TestCase):
    def setUp(self):
        strategy = fleet.DistributedStrategy()
        self.model_parallel_size = 1
        self.data_parallel_size = 1
        self.pipeline_parallel_size = 2
        strategy.hybrid_configs = {
            "dp_degree": self.data_parallel_size,
            "mp_degree": self.model_parallel_size,
            "pp_degree": self.pipeline_parallel_size,
        }
        strategy.pipeline_configs = {
            "accumulate_steps": batch_size // micro_batch_size,
            "micro_batch_size": micro_batch_size
        }
        fleet.init(is_collective=True, strategy=strategy)

    def test_pp_model(self):
        hcg = fleet.get_hybrid_communicate_group()
        word_size = hcg.get_model_parallel_world_size()
        dp_id = hcg.get_data_parallel_rank()
        pp_id = hcg.get_stage_id()
        rank_id = dist.get_rank()
        set_random_seed(1024, dp_id, rank_id)

        #construct model a
        model_a = SimpleNet()
        scheduler_a = paddle.optimizer.lr.PiecewiseDecay(
            boundaries=[2, 3, 4], values=[0.01, 0.02, 0.03, 0.04], verbose=True)
        optimizer_a = paddle.optimizer.SGD(learning_rate=scheduler_a,
                                           parameters=model_a.parameters())

        model_b = SimpleNetPipe(topology=hcg.topology())

        scheduler_b = paddle.optimizer.lr.PiecewiseDecay(
            boundaries=[2, 3, 4], values=[0.01, 0.02, 0.03, 0.04], verbose=True)
        optimizer_b = paddle.optimizer.SGD(learning_rate=scheduler_b,
                                           parameters=model_b.parameters())
        model_b = fleet.distributed_model(model_b)
        optimizer_b = fleet.distributed_optimizer(optimizer_b)

        param_len = len(model_a.parameters())

        parameters = []
        for param in model_a.parameters():
            parameters.append(param.numpy())

        model_b_params = model_b.parameters()

        if pp_id == 0:
            model_b_params[0].set_value(parameters[2])
            model_b_params[1].set_value(parameters[0])

        else:
            model_b_params[0].set_value(parameters[2])
            model_b_params[1].set_value(parameters[1])

        for step in range(5):
            x1_data = np.random.randint(0, vocab_size, size=[batch_size, 1])
            x2_data = np.random.randint(0, vocab_size, size=[batch_size, 1])
            y1_data = np.random.randint(0, hidden_size, size=[batch_size, 1])

            x1 = paddle.to_tensor(x1_data)
            x2 = paddle.to_tensor(x2_data)
            y1 = paddle.to_tensor(y1_data)

            x1.stop_gradient = True
            x2.stop_gradient = True
            y1.stop_gradient = True

            loss_a = model_a(x1, x2, y1)
            loss_a.backward()

            optimizer_a.step()
            optimizer_a.clear_grad()
            scheduler_a.step()

            loss_b = model_b.train_batch([(x1, x2), (y1, )], optimizer_b,
                                         scheduler_b)

            print("loss", loss_a.numpy(), loss_b.numpy())
            np.testing.assert_allclose(loss_a.numpy(), loss_b.numpy())


if __name__ == "__main__":
    unittest.main()
