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

import paddle
import numpy as np
import random
import paddle.distributed as dist
import paddle.fluid as fluid
import paddle.distributed.fleet as fleet
from paddle.io import DataLoader, Dataset
import unittest


def set_random_seed(seed, dp_id, rank_id):
    """Set random seed for reproducability."""
    random.seed(seed)
    np.random.seed(seed + dp_id)
    paddle.seed(seed + rank_id)


vocab_size = 20
hidden_size = 10
inner_size = 8
output_size = 10
seq_length = 2
batch_size = 4


def parallel_matmul(lm_output, logit_weights, parallel_output):
    hcg = fleet.get_hybrid_communicate_group()
    model_parallel_group = hcg.get_model_parallel_group()
    world_size = hcg.get_model_parallel_world_size()
    rank = hcg.get_model_parallel_rank()

    if world_size > 1:
        input_parallel = paddle.distributed.collective._c_identity(
            lm_output, group=model_parallel_group)

        logits = paddle.matmul(input_parallel, logit_weights, transpose_y=True)

        if parallel_output:
            return logits

        return paddle.distributed.collective._c_concat(
            logits, group=model_parallel_group)
    else:
        logits = paddle.matmul(lm_output, logit_weights, transpose_y=True)
        return logits


class SimpleMPNet(fluid.dygraph.Layer):
    def __init__(self, vocab_size, hidden_size, inner_size, output_size, np_fc1,
                 np_fc2, mp_id):
        super(SimpleMPNet, self).__init__()

        if mp_id == 0:
            init_fc1_data = np_fc1[:, :(inner_size // 2)]
            init_fc2_data = np_fc2[:(inner_size // 2), :]
        else:
            init_fc1_data = np_fc1[:, (inner_size // 2):]
            init_fc2_data = np_fc2[(inner_size // 2):, :]

        self.linear1 = fleet.meta_parallel.ColumnParallelLinear(
            hidden_size,
            inner_size,
            weight_attr=paddle.framework.ParamAttr(
                initializer=paddle.nn.initializer.Assign(init_fc1_data)),
            gather_output=False,
            has_bias=True)

        self.linear2 = fleet.meta_parallel.RowParallelLinear(
            inner_size,
            hidden_size,
            weight_attr=paddle.framework.ParamAttr(
                initializer=paddle.nn.initializer.Assign(init_fc2_data)),
            input_is_parallel=True,
            has_bias=True)

        self.linear3 = paddle.nn.Linear(
            hidden_size,
            output_size,
            weight_attr=paddle.framework.ParamAttr(
                initializer=paddle.nn.initializer.Constant(0.0)),
            bias_attr=paddle.framework.ParamAttr(
                initializer=paddle.nn.initializer.Constant(0.0)))

        self.embedding = fleet.meta_parallel.VocabParallelEmbedding(
            vocab_size,
            hidden_size,
            weight_attr=paddle.nn.initializer.Constant(value=0.5))

    def forward(self, x):
        x = self.embedding(x)
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        x = parallel_matmul(x, self.embedding.weight, False)
        return x


class SimpleDPNet(fluid.dygraph.Layer):
    def __init__(self, vocab_size, hidden_size, inner_size, output_size, np_fc1,
                 np_fc2):

        super(SimpleDPNet, self).__init__()
        self.linear1 = paddle.nn.Linear(
            hidden_size,
            inner_size,
            weight_attr=paddle.framework.ParamAttr(
                initializer=paddle.nn.initializer.Assign(np_fc1)),
            bias_attr=paddle.framework.ParamAttr(
                initializer=paddle.nn.initializer.Constant(0.0)))

        self.linear2 = paddle.nn.Linear(
            inner_size,
            hidden_size,
            weight_attr=paddle.framework.ParamAttr(
                initializer=paddle.nn.initializer.Assign(np_fc2)),
            bias_attr=paddle.framework.ParamAttr(
                initializer=paddle.nn.initializer.Constant(0.0)))

        self.linear3 = paddle.nn.Linear(
            hidden_size,
            output_size,
            weight_attr=paddle.framework.ParamAttr(
                initializer=paddle.nn.initializer.Constant(0.0)),
            bias_attr=paddle.framework.ParamAttr(
                initializer=paddle.nn.initializer.Constant(0.0)))

        self.embedding = paddle.nn.Embedding(
            vocab_size,
            hidden_size,
            weight_attr=paddle.nn.initializer.Constant(value=0.5))

    def forward(self, x):
        x = self.embedding(x)
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        x = paddle.matmul(x, self.embedding.weight, transpose_y=True)
        return x


class TestDistMPTraning(unittest.TestCase):
    def setUp(self):
        strategy = fleet.DistributedStrategy()
        self.model_parallel_size = 2
        self.data_parallel_size = 1
        strategy.hybrid_configs = {
            "dp_degree": self.data_parallel_size,
            "mp_degree": self.model_parallel_size,
            "pp_degree": 1
        }
        fleet.init(is_collective=True, strategy=strategy)

    def train_batch(self, batch, model, optimizer, is_mp):
        output = model(batch)
        loss = output.mean()
        loss.backward()  # do backward
        optimizer.step()  # update parameters
        optimizer.clear_grad()
        return loss

    def build_optimizer(self, model):
        optimizer = paddle.optimizer.SGD(learning_rate=0.001,
                                         parameters=model.parameters())
        return optimizer

    def build_model_optimizer(self):
        hcg = fleet.get_hybrid_communicate_group()
        word_size = hcg.get_model_parallel_world_size()
        mp_id = hcg.get_model_parallel_rank()
        dp_id = hcg.get_data_parallel_rank()
        rank_id = dist.get_rank()
        set_random_seed(1024, dp_id, rank_id)

        np_fc1 = np.random.random_sample((hidden_size, inner_size))
        np_fc2 = np.random.random_sample((inner_size, hidden_size))

        model_a = SimpleMPNet(vocab_size, hidden_size, inner_size, output_size,
                              np_fc1, np_fc2, mp_id)
        optimizer_a = self.build_optimizer(model_a)
        model_a = fleet.distributed_model(model_a)
        optimizer_a = fleet.distributed_optimizer(optimizer_a)

        model_b = SimpleDPNet(vocab_size, hidden_size, inner_size, output_size,
                              np_fc1, np_fc2)
        optimizer_b = self.build_optimizer(model_b)

        return model_a, optimizer_a, model_b, optimizer_b

    def test_mp_model(self):
        model_a, optimizer_a, model_b, optimizer_b = self.build_model_optimizer(
        )

        for _ in range(5):
            np_data = np.random.randint(0, vocab_size, (
                batch_size,
                seq_length, ))
            batch = paddle.to_tensor(np_data)
            loss_a = self.train_batch(batch, model_a, optimizer_a, True)
            loss_b = self.train_batch(batch, model_b, optimizer_b, False)

            np.testing.assert_allclose(
                loss_a.numpy(), loss_b.numpy(), rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
