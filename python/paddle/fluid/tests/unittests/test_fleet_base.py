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

import unittest
import paddle
import paddle.distributed.fleet as fleet
import paddle.distributed.fleet.base.role_maker as role_maker
import os
import paddle.fluid as fluid
import paddle.nn as nn
import numpy as np


class TestFleetBase(unittest.TestCase):
    def setUp(self):
        os.environ["POD_IP"] = "127.0.0.1"
        os.environ["PADDLE_TRAINER_ENDPOINTS"] = "127.0.0.1:36000"
        os.environ["PADDLE_TRAINERS_NUM"] = "2"
        os.environ["PADDLE_PSERVERS_IP_PORT_LIST"] = \
            "127.0.0.1:36001,127.0.0.2:36002"

    def test_init(self):
        role = role_maker.PaddleCloudRoleMaker(is_collective=True)
        fleet.init(role)

    def test_is_first_worker(self):
        role = role_maker.PaddleCloudRoleMaker(is_collective=True)
        fleet.init(role)
        if fleet.is_first_worker():
            print("test fleet first worker done.")

    def test_worker_index(self):
        role = role_maker.PaddleCloudRoleMaker(is_collective=True)
        fleet.init(role)
        print(fleet.worker_index())

    def test_worker_num(self):
        role = role_maker.PaddleCloudRoleMaker(is_collective=True)
        fleet.init(role)
        print(fleet.worker_num())

    def test_is_worker(self):
        role = role_maker.PaddleCloudRoleMaker(is_collective=True)
        fleet.init(role)
        if fleet.is_worker():
            print("test fleet is worker")

    def test_worker_endpoints(self):
        role = role_maker.PaddleCloudRoleMaker(is_collective=True)
        fleet.init(role)
        self.assertEqual(
            "127.0.0.1:36000", fleet.worker_endpoints(to_string=True))
        self.assertEqual(["127.0.0.1:36000"], fleet.worker_endpoints())

    def test_server_num(self):
        os.environ["TRAINING_ROLE"] = "PSERVER"
        os.environ["PADDLE_PORT"] = "36001"
        os.environ["POD_IP"] = "127.0.0.1"

        role = role_maker.PaddleCloudRoleMaker()
        fleet.init(role)
        os.environ["PADDLE_TRAINERS_NUM"] = "2"
        self.assertEqual(2, fleet.server_num())

    def test_server_index(self):
        os.environ["TRAINING_ROLE"] = "PSERVER"
        os.environ["PADDLE_PORT"] = "36001"
        os.environ["POD_IP"] = "127.0.0.1"

        role = role_maker.PaddleCloudRoleMaker()
        fleet.init(role)
        self.assertEqual(0, fleet.server_index())

    def test_server_endpoints(self):
        os.environ["TRAINING_ROLE"] = "PSERVER"
        os.environ["PADDLE_PORT"] = "36001"
        os.environ["POD_IP"] = "127.0.0.1"

        role = role_maker.PaddleCloudRoleMaker()
        fleet.init(role)
        if fleet.is_server():
            self.assertEqual(
                "127.0.0.1:36001,127.0.0.2:36002",
                fleet.server_endpoints(to_string=True))
            self.assertEqual(["127.0.0.1:36001", "127.0.0.2:36002"],
                             fleet.server_endpoints())

    def test_is_server(self):
        os.environ["TRAINING_ROLE"] = "PSERVER"
        os.environ["PADDLE_PORT"] = "36001"
        os.environ["POD_IP"] = "127.0.0.1"

        role = role_maker.PaddleCloudRoleMaker()
        fleet.init(role)
        self.assertTrue(fleet.is_server())

    def test_util(self):
        role = role_maker.PaddleCloudRoleMaker(is_collective=True)
        fleet.init(role)
        self.assertNotEqual(fleet.util, None)

    def test_barrier_worker(self):
        role = role_maker.PaddleCloudRoleMaker(is_collective=True)
        fleet.init(role)
        if fleet.is_worker():
            fleet.barrier_worker()

    def test_init_worker(self):
        role = role_maker.PaddleCloudRoleMaker(is_collective=True)
        fleet.init(role)

        with self.assertRaises(ValueError):
            if fleet.is_worker():
                fleet.init_worker()

    def test_stop_worker(self):
        role = role_maker.PaddleCloudRoleMaker(is_collective=True)
        fleet.init(role)
        with self.assertRaises(ValueError):
            if fleet.is_worker():
                fleet.stop_worker()

    def test_distributed_optimizer(self):
        role = role_maker.PaddleCloudRoleMaker(is_collective=True)
        fleet.init(role)

        optimizer = paddle.optimizer.SGD(learning_rate=0.001)
        optimizer = fleet.distributed_optimizer(optimizer)

    def test_exception(self):
        import paddle.distributed.fleet as fleet
        self.assertRaises(Exception, fleet.init_worker)


class TestFleetDygraph(unittest.TestCase):
    def setUp(self):
        os.environ[
            "PADDLE_TRAINER_ENDPOINTS"] = "127.0.0.1:36213,127.0.0.1:36214"
        os.environ["PADDLE_CURRENT_ENDPOINTS"] = "127.0.0.1:36213"
        os.environ["PADDLE_TRAINERS_NUM"] = "2"
        os.environ["PADDLE_TRAINER_ID"] = "0"

    def test_dygraph_method(self):
        paddle.disable_static()
        value = np.arange(26).reshape(2, 13).astype("float32")
        a = fluid.dygraph.to_variable(value)
        layer = paddle.nn.Linear(13, 5)
        adam = paddle.optimizer.Adam(
            learning_rate=0.01, parameters=layer.parameters())
        # remove init cause this UT cannot launch distributed task
        adam = fleet.distributed_optimizer(adam)
        dp_layer = fleet.distributed_model(layer)
        lr = 0.001
        adam.set_lr(lr)
        cur_lr = adam.get_lr()
        assert (lr == cur_lr)
        state_dict = adam.state_dict()
        adam.set_state_dict(state_dict)

        final_strategy = fleet._final_strategy()


class TestFleetBaseSingleError(unittest.TestCase):
    def setUp(self):
        os.environ.pop("PADDLE_TRAINER_ENDPOINTS")

    def gen_data(self):
        return {
            "x": np.random.random(size=(128, 32)).astype('float32'),
            "y": np.random.randint(
                2, size=(128, 1)).astype('int64')
        }

    def test_single_run_collective_minimize(self):
        def test_single_error():
            input_x = paddle.static.data(
                name="x", shape=[-1, 32], dtype='float32')
            input_y = paddle.static.data(name="y", shape=[-1, 1], dtype='int64')

            fc_1 = fluid.layers.fc(input=input_x, size=64, act='tanh')
            prediction = fluid.layers.fc(input=fc_1, size=2, act='softmax')
            cost = fluid.layers.cross_entropy(input=prediction, label=input_y)
            avg_cost = paddle.mean(x=cost)
            fleet.init(is_collective=True)

        # in non_distributed mode(use `python` to launch), raise error if has multi cards
        if fluid.core.is_compiled_with_cuda(
        ) and fluid.core.get_cuda_device_count() > 1:
            self.assertRaises(ValueError, test_single_error)
        else:
            test_single_error()


if __name__ == "__main__":
    unittest.main()
