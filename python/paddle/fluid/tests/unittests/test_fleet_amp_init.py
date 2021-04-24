#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import numpy as np

import paddle
import paddle.distributed.fleet.base.role_maker as role_maker
import paddle.distributed.fleet as fleet
import paddle.nn.functional as F

paddle.enable_static()


def gen_data():
    return {
        "x": np.random.random(size=(128, 32)).astype('float32'),
        "y": np.random.randint(
            2, size=(128, 1)).astype('int64')
    }


def mlp(input_x, input_y, hid_dim=128, label_dim=2):
    fc_1 = paddle.static.nn.fc(x=input_x, size=hid_dim, activation='tanh')
    fc_2 = paddle.static.nn.fc(x=fc_1, size=hid_dim, activation='tanh')
    prediction = paddle.static.nn.fc(x=fc_2,
                                     size=label_dim,
                                     activation='softmax')
    cost = F.cross_entropy(input=prediction, label=input_y)
    avg_cost = paddle.mean(x=cost)
    return avg_cost


class TestFleetAMPInit(unittest.TestCase):
    def test_fleet_amp_init(self):
        if not paddle.is_compiled_with_cuda():
            return

        main_program = paddle.static.Program()
        startup_program = paddle.static.Program()

        role = role_maker.PaddleCloudRoleMaker(is_collective=True)
        fleet.init(role)

        with paddle.static.program_guard(main_program, startup_program):
            input_x = paddle.static.data(
                name="x", shape=[None, 32], dtype='float32')
            input_y = paddle.static.data(
                name="y", shape=[None, 1], dtype='int64')

            cost = mlp(input_x, input_y)
            optimizer = paddle.optimizer.Momentum(
                learning_rate=0.001,
                momentum=0.9,
                weight_decay=paddle.regularizer.L2Decay(1e-4),
                multi_precision=True)

            optimizer = paddle.static.amp.decorate(optimizer)
            optimizer = fleet.distributed_optimizer(optimizer)
            optimizer.minimize(cost)

        loss_scale = optimizer.get_loss_scaling()

        place = paddle.CUDAPlace(0)

        exe = paddle.static.Executor(place)
        exe.run(startup_program)
        optimizer.amp_init(place)

        step = 1
        for i in range(step):
            cost_val = exe.run(program=main_program,
                               feed=gen_data(),
                               fetch_list=[cost.name])

    def test_fleet_amp_meta_optimizer_init(self):
        if not paddle.is_compiled_with_cuda():
            return

        main_program = paddle.static.Program()
        startup_program = paddle.static.Program()

        role = role_maker.PaddleCloudRoleMaker(is_collective=True)
        fleet.init(role)

        with paddle.static.program_guard(main_program, startup_program):
            input_x = paddle.static.data(
                name="x", shape=[None, 32], dtype='float32')
            input_y = paddle.static.data(
                name="y", shape=[None, 1], dtype='int64')

            cost = mlp(input_x, input_y)
            optimizer = paddle.optimizer.Momentum(
                learning_rate=0.001,
                momentum=0.9,
                weight_decay=paddle.regularizer.L2Decay(1e-4),
                multi_precision=True)

            strategy = paddle.distributed.fleet.DistributedStrategy()
            strategy.amp = True
            strategy.amp_configs = {'use_pure_fp16': True}
            strategy.gradient_merge = True
            strategy.gradient_merge_configs = {"k_steps": 2}

            optimizer = fleet.distributed_optimizer(optimizer, strategy)
            optimizer.minimize(cost)

        print(fleet._get_applied_meta_list())

        place = paddle.CUDAPlace(0)

        exe = paddle.static.Executor(place)
        exe.run(startup_program)
        optimizer.amp_init(place)

        step = 3
        for i in range(step):
            cost_val = exe.run(program=main_program,
                               feed=gen_data(),
                               fetch_list=[cost.name])
            print(cost_val)


if __name__ == '__main__':
    unittest.main()
