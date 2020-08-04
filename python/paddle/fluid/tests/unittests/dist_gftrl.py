#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
import paddle.fluid as fluid
import os
import random
from test_dist_base import TestDistRunnerBase, runtime_main

# Fix seed for test
random.seed = 1
fluid.default_startup_program().random_seed = 1
fluid.default_main_program().random_seed = 1


class Dataset(object):
    def __init__(self, feature_num):
        self.feature_num = feature_num

    def train(self, sample_num):
        return self._parse_creator(sample_num)

    def test(self, sample_num):
        return self._parse_creator(sample_num)

    def _parse_creator(self, sample_num):
        def _parse():
            for i in range(sample_num):
                x = random.sample(
                    list(range(self.feature_num)), int(self.feature_num / 3))
                y = random.choice([0, 1])
                yield [x, y]

        return _parse


class TestDistGFtrl(TestDistRunnerBase):
    def get_model(self, lr=0.001, batch_size=2):
        feature_num = 100
        train_sample_num = 300
        test_sample_num = 200

        x = fluid.layers.data(name="x", shape=[1], dtype="int64", lod_level=1)
        y = fluid.layers.data(name="y", shape=[1], dtype="int64")

        lr_embbding = fluid.layers.embedding(
            input=x,
            size=[feature_num, 1],
            is_sparse=True,
            is_distributed=False)
        lr_pool = fluid.layers.sequence_pool(input=lr_embbding, pool_type="sum")

        predict = fluid.layers.sigmoid(lr_pool)
        acc = fluid.layers.accuracy(input=predict, label=y)
        auc_var, batch_auc_var, auc_states = fluid.layers.auc(input=predict,
                                                              label=y)
        cost = fluid.layers.sigmoid_cross_entropy_with_logits(
            lr_pool, fluid.layers.cast(
                y, dtype='float32'))
        avg_cost = fluid.layers.mean(x=cost)

        inference_program = paddle.fluid.default_main_program().clone()

        gftrl_optimizer = fluid.optimizer.GFtrl(learning_rate=lr)
        gftrl_optimizer.minimize(avg_cost)

        dataset = Dataset(feature_num)
        train_reader = paddle.batch(
            dataset.train(train_sample_num), batch_size=batch_size)
        test_reader = paddle.batch(
            dataset.test(test_sample_num), batch_size=batch_size)

        return inference_program, avg_cost, train_reader, test_reader, acc, predict


if __name__ == "__main__":
    runtime_main(TestDistGFtrl)
