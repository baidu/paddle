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
import paddle.fluid as fluid
import paddle.distributed.fleet as fleet
from paddle.distributed.fleet.meta_optimizers import AMPOptimizer
import os
from fleet_meta_optimizer_base import TestFleetMetaOptimizer

paddle.enable_static()


class TestFleetAMPOptimizer(TestFleetMetaOptimizer):
    def test_amp_optimizer_backward(self):
        """ test amp optimizer backward """
        train_prog, startup_prog = fluid.Program(), fluid.Program()
        avg_cost, strategy = self.net(train_prog, startup_prog)

        opt = fluid.optimizer.MomentumOptimizer(
            learning_rate=0.001, momentum=0.9)
        opt = AMPOptimizer(opt)
        opt.user_defined_strategy = strategy
        params_grads = opt.backward(avg_cost, startup_prog)

        ops = [op.type for op in avg_cost.block.ops]
        self.assertIn('cast', ops)
        self.assertNotIn('check_finite_and_unscale', ops)

    def test_amp_optimizer_backward_gradients(self):
        """ test amp optimizer backward + gradients"""
        train_prog, startup_prog = fluid.Program(), fluid.Program()
        avg_cost, strategy = self.net(train_prog, startup_prog)

        opt = fluid.optimizer.MomentumOptimizer(
            learning_rate=0.001, momentum=0.9)
        opt = AMPOptimizer(opt)
        opt.user_defined_strategy = strategy
        params_grads = opt.backward(avg_cost, startup_prog)
        with fluid.program_guard(train_prog, startup_prog):
            opt.apply_gradients(params_grads)

        ops = [op.type for op in avg_cost.block.ops]
        self.assertIn('cast', ops)
        self.assertIn('check_finite_and_unscale', ops)

    def test_amp_optimizer_backward_optimize(self):
        """ test amp optimizer backward + optimizer """
        train_prog, startup_prog = fluid.Program(), fluid.Program()
        avg_cost, strategy = self.net(train_prog, startup_prog)

        opt = fluid.optimizer.MomentumOptimizer(
            learning_rate=0.001, momentum=0.9)
        opt = AMPOptimizer(opt)
        opt.user_defined_strategy = strategy
        params_grads = opt.backward(avg_cost, startup_prog)
        opt.apply_optimize(avg_cost, startup_prog, params_grads)

        ops = [op.type for op in avg_cost.block.ops]
        self.assertIn('cast', ops)
        self.assertIn('check_finite_and_unscale', ops)

    def test_amp_optimizer(self):
        """ test amp """
        train_prog, startup_prog = fluid.Program(), fluid.Program()
        avg_cost, strategy = self.net(train_prog, startup_prog)
        self.set_strategy(strategy, 'amp')
        self.optimizer(avg_cost, strategy, train_prog, startup_prog)

        ops = [op.type for op in avg_cost.block.ops]
        self.assertIn('cast', ops)
        self.assertIn('check_finite_and_unscale', ops)

    def test_amp_recompute_optimizer(self):
        """ test amp + recompute """
        train_prog, startup_prog = fluid.Program(), fluid.Program()
        avg_cost, strategy = self.net(train_prog, startup_prog)
        self.set_strategy(strategy, 'amp')
        self.set_strategy(strategy, 'recompute')
        self.optimizer(avg_cost, strategy, train_prog, startup_prog)

        strategy = fleet._final_strategy()

        ops = [op.type for op in avg_cost.block.ops]
        outs = [
            op.output('Out')[0] for op in avg_cost.block.ops if op.type == 'mul'
        ]
        self.assertIn('cast', ops)
        self.assertIn('check_finite_and_unscale', ops)

        # recompute
        self.assertIn('subprog', ''.join(outs))

    def test_amp_recompute_lars_optimizer(self):
        """ test amp + recompute """
        train_prog, startup_prog = fluid.Program(), fluid.Program()
        avg_cost, strategy = self.net(train_prog, startup_prog)
        self.set_strategy(strategy, 'amp')
        self.set_strategy(strategy, 'recompute')
        self.set_strategy(strategy, 'lars')
        self.optimizer(avg_cost, strategy, train_prog, startup_prog)

        strategy = fleet._final_strategy()

        ops = [op.type for op in avg_cost.block.ops]
        outs = [
            op.output('Out')[0] for op in avg_cost.block.ops if op.type == 'mul'
        ]
        self.assertIn('cast', ops)
        self.assertIn('check_finite_and_unscale', ops)

        # recompute
        self.assertIn('subprog', ''.join(outs))

        # lars
        self.assertIn('lars_momentum', ops)

    def test_amp_recompute_lamb_optimizer(self):
        train_prog, startup_prog = fluid.Program(), fluid.Program()
        avg_cost, strategy = self.net(train_prog, startup_prog)
        self.set_strategy(strategy, 'amp')
        self.set_strategy(strategy, 'recompute')
        self.set_strategy(strategy, 'lamb')
        self.optimizer(avg_cost, strategy, train_prog, startup_prog, 'adam')

        ops = [op.type for op in avg_cost.block.ops]
        outs = [
            op.output('Out')[0] for op in avg_cost.block.ops if op.type == 'mul'
        ]
        self.assertIn('cast', ops)
        self.assertIn('check_finite_and_unscale', ops)

        # recompute
        self.assertIn('subprog', ''.join(outs))

        # lamb
        self.assertIn('lamb', ops)


if __name__ == "__main__":
    unittest.main()
