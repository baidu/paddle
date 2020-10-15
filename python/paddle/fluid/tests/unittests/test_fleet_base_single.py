# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import numpy as np
import os
cuda_visible_devices = os.getenv('CUDA_VISIBLE_DEVICES')
if cuda_visible_devices is None or cuda_visible_devices == "":
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = cuda_visible_devices.split(',')[0]
import paddle
import paddle.distributed.fleet as fleet
import paddle.distributed.fleet.base.role_maker as role_maker
import paddle.fluid as fluid
import unittest
import paddle.nn as nn


class LinearNet(nn.Layer):
    def __init__(self):
        super(LinearNet, self).__init__()
        self._linear1 = nn.Linear(10, 10)
        self._linear2 = nn.Linear(10, 1)

    def forward(self, x):
        return self._linear2(self._linear1(x))


class TestFleetDygraphSingle(unittest.TestCase):
    def setUp(self):
        os.environ["PADDLE_TRAINER_ENDPOINTS"] = "127.0.0.1:36213"
        os.environ["PADDLE_CURRENT_ENDPOINTS"] = "127.0.0.1:36213"
        os.environ["PADDLE_TRAINERS_NUM"] = "1"
        os.environ["PADDLE_TRAINER_ID"] = "0"

    def test_dygraph_single(self):
        paddle.disable_static()
        fleet.init(is_collective=True)

        layer = LinearNet()
        loss_fn = nn.MSELoss()
        adam = paddle.optimizer.Adam(
            learning_rate=0.001, parameters=layer.parameters())

        adam = fleet.distributed_optimizer(adam)
        dp_layer = fleet.distributed_model(layer)
        for step in range(2):
            inputs = paddle.randn([10, 10], 'float32')
            outputs = dp_layer(inputs)
            labels = paddle.randn([10, 1], 'float32')
            loss = loss_fn(outputs, labels)
            loss = dp_layer.scale_loss(loss)
            loss.backward()
            dp_layer.apply_collective_grads()
            adam.step()
            adam.clear_grad()


class TestFleetBaseSingleRunCollective(unittest.TestCase):
    def setUp(self):
        pass

    def gen_data(self):
        return {
            "x": np.random.random(size=(128, 32)).astype('float32'),
            "y": np.random.randint(
                2, size=(128, 1)).astype('int64')
        }

    def test_single_run_collective_minimize(self):
        input_x = paddle.static.data(name="x", shape=[-1, 32], dtype='float32')
        input_y = paddle.static.data(name="y", shape=[-1, 1], dtype='int64')

        fc_1 = fluid.layers.fc(input=input_x, size=64, act='tanh')
        prediction = fluid.layers.fc(input=fc_1, size=2, act='softmax')
        cost = fluid.layers.cross_entropy(input=prediction, label=input_y)
        avg_cost = paddle.mean(x=cost)

        fleet.init(is_collective=True)
        optimizer = fluid.optimizer.SGD(learning_rate=0.001)
        optimizer = fleet.distributed_optimizer(optimizer)
        optimizer.minimize(avg_cost)

        place = fluid.CUDAPlace(0) if paddle.fluid.is_compiled_with_cuda(
        ) else fluid.CPUPlace()

        exe = fluid.Executor(place)
        exe.run(paddle.static.default_startup_program())

        for i in range(10):
            cost_val = exe.run(feed=self.gen_data(), fetch_list=[avg_cost.name])
            print("cost of step[{}] = {}".format(i, cost_val))


class TestFleetBaseSingleRunPS(unittest.TestCase):
    def setUp(self):
        pass

    def gen_data(self):
        return {
            "x": np.random.random(size=(128, 32)).astype('float32'),
            "y": np.random.randint(
                2, size=(128, 1)).astype('int64')
        }

    def test_single_run_ps_minimize(self):
        input_x = paddle.static.data(name="x", shape=[-1, 32], dtype='float32')
        input_y = paddle.static.data(name="y", shape=[-1, 1], dtype='int64')

        fc_1 = fluid.layers.fc(input=input_x, size=64, act='tanh')
        prediction = fluid.layers.fc(input=fc_1, size=2, act='softmax')
        cost = fluid.layers.cross_entropy(input=prediction, label=input_y)
        avg_cost = paddle.mean(x=cost)

        fleet.init()
        strategy = paddle.distributed.fleet.DistributedStrategy()
        optimizer = fluid.optimizer.SGD(learning_rate=0.01)
        optimizer = fleet.distributed_optimizer(optimizer, strategy=strategy)
        optimizer.minimize(avg_cost)
        if fleet.is_server():
            fleet.init_server()
            fleet.run_server()
        elif fleet.is_worker():
            place = fluid.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(paddle.static.default_startup_program())
            step = 10
            for i in range(step):
                cost_val = exe.run(program=fluid.default_main_program(),
                                   feed=self.gen_data(),
                                   fetch_list=[avg_cost.name])
                print("worker_index: %d, step%d cost = %f" %
                      (fleet.worker_index(), i, cost_val[0]))


if __name__ == "__main__":
    unittest.main()
