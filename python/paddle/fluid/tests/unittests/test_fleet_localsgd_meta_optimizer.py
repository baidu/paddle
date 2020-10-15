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

import paddle
import paddle.fluid as fluid
import paddle.distributed.fleet as fleet
import paddle.distributed.fleet.base.role_maker as role_maker
from fleet_meta_optimizer_base import TestFleetMetaOptimizer

paddle.enable_static()


class TestFleetLocalSGDMetaOptimizer(TestFleetMetaOptimizer):
    def test_localsgd_optimizer(self):
        train_prog, startup_prog = fluid.Program(), fluid.Program()
        avg_cost, strategy = self.net(train_prog, startup_prog)
        self.set_strategy(strategy, 'localsgd')
        self.optimizer(avg_cost, strategy, train_prog, startup_prog)

        ops = [op.type for op in avg_cost.block.ops]
        outs = [
            ''.join(op.output('Out')) for op in avg_cost.block.ops
            if op.type == 'conditional_block'
        ]

        self.assertIn('conditional_block', ops)
        self.assertIn('@SNAPSHOT', ''.join(outs))

    def test_localsgd_amp_optimizer(self):
        train_prog, startup_prog = fluid.Program(), fluid.Program()
        avg_cost, strategy = self.net(train_prog, startup_prog)
        self.set_strategy(strategy, 'localsgd')
        self.set_strategy(strategy, 'amp')
        self.optimizer(avg_cost, strategy, train_prog, startup_prog)

        ops = [op.type for op in avg_cost.block.ops]
        outs = [
            ''.join(op.output('Out')) for op in avg_cost.block.ops
            if op.type == 'conditional_block'
        ]

        self.assertIn('conditional_block', ops)
        self.assertIn('@SNAPSHOT', ''.join(outs))

        # amp
        self.assertIn('cast', ops)
        self.assertIn('check_finite_and_unscale', ops)


class TestFleetAdaptiveLocalSGDMetaOptimizer(TestFleetMetaOptimizer):
    def test_adaptive_localsgd_optimizer(self):
        train_prog, startup_prog = fluid.Program(), fluid.Program()
        avg_cost, strategy = self.net(train_prog, startup_prog)
        self.set_strategy(strategy, 'adaptive_localsgd')
        self.optimizer(avg_cost, strategy, train_prog, startup_prog)

        ops = [op.type for op in avg_cost.block.ops]
        outs = [
            ''.join(op.output('Out')) for op in avg_cost.block.ops
            if op.type == 'conditional_block'
        ]

        self.assertIn('conditional_block', ops)
        self.assertIn('@SNAPSHOT', ''.join(outs))

    def test_localsgd_amp_optimizer(self):
        train_prog, startup_prog = fluid.Program(), fluid.Program()
        avg_cost, strategy = self.net(train_prog, startup_prog)
        self.set_strategy(strategy, 'adaptive_localsgd')
        self.set_strategy(strategy, 'amp')
        self.optimizer(avg_cost, strategy, train_prog, startup_prog)

        ops = [op.type for op in avg_cost.block.ops]
        outs = [
            ''.join(op.output('Out')) for op in avg_cost.block.ops
            if op.type == 'conditional_block'
        ]

        self.assertIn('conditional_block', ops)
        self.assertIn('@SNAPSHOT', ''.join(outs))

        # amp
        self.assertIn('cast', ops)
        self.assertIn('check_finite_and_unscale', ops)


if __name__ == "__main__":
    unittest.main()
