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

paddle.enable_static()


class TestFleetLarsMetaOptimizer(unittest.TestCase):
    def setUp(self):
        os.environ["PADDLE_TRAINER_ID"] = "1"
        os.environ[
            "PADDLE_TRAINER_ENDPOINTS"] = "127.0.0.1:36001,127.0.0.1:36002"

    def net(self, main_prog, startup_prog):
        with fluid.program_guard(main_prog, startup_prog):
            with fluid.unique_name.guard():
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

                strategy = paddle.distributed.fleet.DistributedStrategy()
                strategy.lars = True
                strategy.lars_configs = {
                    "lars_coeff": 0.001,
                    "lars_weight_decay": 0.0005,
                    "epsilon": 0,
                    "exclude_from_weight_decay": ["batch_norm", ".b"],
                }

        return avg_cost, strategy

    def test_lars_optimizer(self):
        role = role_maker.PaddleCloudRoleMaker(is_collective=True)
        fleet.init(role)
        startup_prog = fluid.Program()
        train_prog = fluid.Program()
        avg_cost, strategy = self.net(train_prog, startup_prog)
        optimizer = paddle.fluid.optimizer.Momentum(
            learning_rate=0.01, momentum=0.9)
        optimizer = fleet.distributed_optimizer(optimizer, strategy=strategy)
        optimizer.minimize(avg_cost)

        ops = [op.type for op in avg_cost.block.ops]
        self.assertIn('lars_momentum', ops)

    def test_lars_not_apply_with_adam(self):
        role = role_maker.PaddleCloudRoleMaker(is_collective=True)
        fleet.init(role)
        startup_prog = fluid.Program()
        train_prog = fluid.Program()
        avg_cost, strategy = self.net(train_prog, startup_prog)
        optimizer = paddle.fluid.optimizer.Adam(learning_rate=0.01)
        optimizer = fleet.distributed_optimizer(optimizer, strategy=strategy)
        optimizer.minimize(avg_cost)

        ops = [op.type for op in avg_cost.block.ops]
        self.assertNotIn('lars_momentum', ops)

    def test_lars_exclude_fn(self):
        role = role_maker.PaddleCloudRoleMaker(is_collective=True)
        fleet.init(role)
        startup_prog = fluid.Program()
        train_prog = fluid.Program()
        avg_cost, strategy = self.net(train_prog, startup_prog)
        optimizer = paddle.fluid.optimizer.Momentum(
            learning_rate=0.01, momentum=0.9)

        optimizer = fleet.distributed_optimizer(optimizer, strategy=strategy)
        optimizer.minimize(avg_cost)

        ops_without_wd = [
            op for op in avg_cost.block.ops
            if op.type == 'lars_momentum' and ("batch_norm" in op.attr(
                'op_role_var')[0] or ".b" in op.attr('op_role_var')[0])
        ]
        for op in ops_without_wd:
            self.assertEqual(op.attr('lars_weight_decay'), 0)

    def test_lars_apply_with_amp(self):
        role = role_maker.PaddleCloudRoleMaker(is_collective=True)
        fleet.init(role)
        input_x = paddle.fluid.layers.data(
            name="x", shape=[32], dtype='float32')
        input_y = paddle.fluid.layers.data(name="y", shape=[1], dtype='int64')

        fc_1 = paddle.fluid.layers.fc(input=input_x, size=64, act='tanh')
        fc_2 = paddle.fluid.layers.fc(input=fc_1, size=64, act='tanh')
        prediction = paddle.fluid.layers.fc(input=[fc_2], size=2, act='softmax')
        cost = paddle.fluid.layers.cross_entropy(
            input=prediction, label=input_y)
        avg_cost = paddle.fluid.layers.mean(x=cost)

        strategy = paddle.distributed.fleet.DistributedStrategy()
        strategy.amp = True
        strategy.amp_configs = {
            "init_loss_scaling": 32768,
            "decr_every_n_nan_or_inf": 2,
            "incr_every_n_steps": 1000,
            "incr_ratio": 2.0,
            "use_dynamic_loss_scaling": True,
            "decr_ratio": 0.5,
            "custom_white_list": ['softmax'],
            "custom_black_list": ['tanh'],
        }
        strategy.lars = True
        strategy.lars_configs = {
            "lars_coeff": 0.001,
            "lars_weight_decay": 0.0005,
            "epsilon": 0,
            "exclude_from_weight_decay": ["batch_norm", ".b"],
        }

        optimizer = paddle.fluid.optimizer.Momentum(
            learning_rate=0.01, momentum=0.9)
        optimizer = fleet.distributed_optimizer(optimizer, strategy=strategy)
        optimizer.minimize(avg_cost)

        ops = [op.type for op in avg_cost.block.ops]
        self.assertIn('lars_momentum', ops)
        self.assertIn('cast', ops)
        self.assertIn('check_finite_and_unscale', ops)


if __name__ == "__main__":
    unittest.main()
