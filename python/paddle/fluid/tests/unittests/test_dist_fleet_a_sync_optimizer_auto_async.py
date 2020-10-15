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
import os
import paddle.distributed.fleet.base.role_maker as role_maker
import time

paddle.enable_static()


class TestFleetGradientMergeMetaOptimizer(unittest.TestCase):
    def setUp(self):
        os.environ["PADDLE_PSERVER_NUMS"] = "2"
        os.environ["PADDLE_TRAINERS_NUM"] = "2"
        os.environ["POD_IP"] = "127.0.0.1"
        os.environ["PADDLE_PORT"] = "36001"
        os.environ["PADDLE_TRAINER_ID"] = "0"
        os.environ["PADDLE_TRAINERS_NUM"] = "2"
        os.environ["PADDLE_PSERVERS_IP_PORT_LIST"] = \
            "127.0.0.1:36001,127.0.0.2:36001"

    def test_a_sync_optimizer3(self):
        os.environ["TRAINING_ROLE"] = "TRAINER"
        import paddle.distributed.fleet as fleet

        main_program = paddle.fluid.Program()
        startup_program = paddle.fluid.Program()

        paddle.fluid.framework.switch_main_program(main_program)
        paddle.fluid.framework.switch_startup_program(startup_program)

        fleet.init(role_maker.PaddleCloudRoleMaker())
        input_x = paddle.fluid.layers.data(
            name="x",
            shape=[-1, 1],
            dtype="int64",
            lod_level=1,
            append_batch_size=False)
        x_embedding = paddle.fluid.layers.embedding(
            is_distributed=False,
            input=input_x,
            size=[1000000000, 100000],
            param_attr=paddle.fluid.ParamAttr(
                name="embedding",
                initializer=paddle.fluid.initializer.Constant(value=0.01)),
            is_sparse=True)
        input_y = paddle.fluid.layers.data(name="y", shape=[1], dtype='int64')

        fc_1 = paddle.fluid.layers.fc(input=x_embedding, size=64, act='tanh')
        fc_2 = paddle.fluid.layers.fc(input=fc_1, size=64, act='tanh')
        prediction = paddle.fluid.layers.fc(input=[fc_2], size=2, act='softmax')
        cost = paddle.fluid.layers.cross_entropy(
            input=prediction, label=input_y)
        avg_cost = paddle.fluid.layers.mean(x=cost)

        strategy = paddle.distributed.fleet.DistributedStrategy()
        strategy.auto = True
        optimizer = paddle.fluid.optimizer.SGD(learning_rate=0.01)
        optimizer = fleet.distributed_optimizer(optimizer, strategy=strategy)
        optimizer.minimize(avg_cost)

        self.assertTrue(fleet._final_strategy().a_sync)
        a_sync_configs = fleet._final_strategy().a_sync_configs
        self.assertTrue(a_sync_configs['k_steps'] == 0)


if __name__ == "__main__":
    unittest.main()
