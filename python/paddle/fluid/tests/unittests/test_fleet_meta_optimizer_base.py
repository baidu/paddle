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
from paddle import fluid
import os
import paddle.distributed.fleet as fleet
import paddle.distributed.fleet.base.role_maker as role_maker
from paddle.distributed.fleet.meta_optimizers.meta_optimizer_base import MetaOptimizerBase


class TestFleetMetaOptimizerBase(unittest.TestCase):
    def net(self, main_prog, startup_prog):
        with fluid.program_guard(main_prog, startup_prog):
            with fluid.unique_name.guard():
                role = role_maker.PaddleCloudRoleMaker(is_collective=True)
                fleet.init(role)
                input_x = paddle.fluid.layers.data(
                    name="x", shape=[32], dtype='float32')
                input_y = paddle.fluid.layers.data(
                    name="y", shape=[1], dtype='int64')

                fc_1 = paddle.fluid.layers.fc(input=input_x,
                                              size=64,
                                              act='tanh')
                fc_2 = paddle.fluid.layers.fc(input=fc_1, size=256, act='tanh')
                prediction = paddle.fluid.layers.fc(input=[fc_2],
                                                    size=2,
                                                    act='softmax')
                cost = paddle.fluid.layers.cross_entropy(
                    input=prediction, label=input_y)
                avg_cost = paddle.fluid.layers.mean(x=cost)

                optimizer = paddle.fluid.optimizer.SGD(learning_rate=0.01)
                opt = MetaOptimizerBase(optimizer)
                opt_ops, params_grads = opt.minimize(avg_cost)
                opt.apply_optimize(avg_cost,
                                   paddle.static.default_startup_program(),
                                   params_grads)
        return None

    net(fluid.default_startup_program(), fluid.default_main_program())


if __name__ == "__main__":
    unittest.main()
