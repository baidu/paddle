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

import unittest

import paddle
import paddle.fluid as fluid
import paddle.fluid.core as core
import paddle.distributed.fleet as fleet
import numpy as np
import os


class TestRawProgramOptimizer(unittest.TestCase):
    def setUp(self):
        os.environ["PADDLE_TRAINER_ID"] = "0"
        os.environ["PADDLE_TRAINER_ENDPOINTS"] = "127.0.0.1:36001"

    def mlp(self, input_x, input_y, hid_dim=128, label_dim=2):
        fc_1 = paddle.static.nn.fc(x=input_x, size=hid_dim, activation='tanh')
        fc_2 = paddle.static.nn.fc(x=fc_1, size=hid_dim, activation='tanh')
        prediction = paddle.static.nn.fc(x=[fc_2],
                                         size=label_dim,
                                         activation='softmax')
        cost = paddle.nn.functional.cross_entropy(
            input=prediction, label=input_y)
        avg_cost = paddle.mean(x=cost)
        return avg_cost

    def gen_data(self):
        return {
            "x": np.random.random(size=(128, 32)).astype('float32'),
            "y": np.random.randint(
                2, size=(128, 1)).astype('int64')
        }

    def test_single_gpu(self):
        paddle.enable_static()
        fleet.init(is_collective=True)
        sharding_program = paddle.static.Program()
        sharding_startup_program = paddle.static.Program()
        strategy = fleet.DistributedStrategy()
        strategy.without_graph_optimization = True
        with fluid.program_guard(sharding_program, sharding_startup_program):
            with fluid.unique_name.guard():
                input_x = paddle.static.data(
                    name="x", shape=[None, 32], dtype='float32')
                input_y = paddle.static.data(
                    name="y", shape=[None, 1], dtype='int64')
                cost = self.mlp(input_x=input_x, input_y=input_y)
                output_name = cost.name
                optimizer = fleet.distributed_optimizer(fluid.optimizer.Adam(),
                                                        strategy)
                optimizer.minimize(cost)

        trainer_id = fleet.worker_index()
        exe = paddle.static.Executor(paddle.CUDAPlace(trainer_id))
        rank = fleet.worker_index()
        exe.run(sharding_startup_program)
        exe.run(program=sharding_program, feed=self.gen_data())


if __name__ == "__main__":
    unittest.main()
