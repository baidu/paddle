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
import paddle.distributed.fleet as fleet
import paddle.distributed.fleet.base.role_maker as role_maker
import paddle.fluid.core as core
import paddle.fluid as fluid

from fleet_meta_optimizer_base import TestFleetMetaOptimizer
import paddle.distributed.fleet.meta_optimizers.sharding as sharding

paddle.enable_static()


class TestFleetShardingMetaOptimizer(TestFleetMetaOptimizer):
    def test_sharding_optimizer(self):
        train_prog, startup_prog = paddle.fluid.Program(), paddle.fluid.Program(
        )
        avg_cost, strategy = self.net(train_prog, startup_prog)
        self.set_strategy(strategy, 'sharding')
        self.optimizer(avg_cost, strategy, train_prog, startup_prog)
        parameters = [
            x.name for x in train_prog.list_vars() if x.persistable == True
        ]
        ops = [op.type for op in avg_cost.block.ops]
        vars = [x.name for x in train_prog.list_vars()]
        self.assertIn('@BroadCast', ''.join(vars))
        self.assertEqual(
            set(parameters),
            set([
                "fc_1.b_0", "fc_2.b_0", "fc_2.w_0", "fc_1.b_0_velocity_0",
                "fc_2.b_0_velocity_0", "fc_2.w_0_velocity_0", "learning_rate_0"
            ]))
        self.assertEqual(ops, [
            'fill_constant', 'fill_constant', 'fill_constant',
            'c_sync_calc_stream', 'c_broadcast', 'c_broadcast', 'c_broadcast',
            'c_broadcast', 'c_broadcast', 'c_broadcast', 'c_sync_comm_stream',
            'mul', 'elementwise_add', 'tanh', 'mul', 'elementwise_add', 'tanh',
            'mul', 'elementwise_add', 'softmax', 'cross_entropy2', 'mean',
            'fill_constant', 'scale', 'mean_grad', 'cross_entropy_grad2',
            'softmax_grad', 'elementwise_add_grad', 'mul_grad', 'tanh_grad',
            'elementwise_add_grad', 'mul_grad', 'tanh_grad',
            'elementwise_add_grad', 'mul_grad', 'c_sync_calc_stream',
            'c_allreduce_sum', 'c_allreduce_sum', 'c_allreduce_sum',
            'c_allreduce_sum', 'c_allreduce_sum', 'c_allreduce_sum',
            'c_sync_comm_stream', 'momentum', 'momentum', 'momentum'
        ])

    def test_sharding_amp_optimizer(self):
        train_prog, startup_prog = paddle.fluid.Program(), paddle.fluid.Program(
        )
        avg_cost, strategy = self.net(train_prog, startup_prog)
        self.set_strategy(strategy, 'sharding')
        self.set_strategy(strategy, 'amp')
        self.optimizer(avg_cost, strategy, train_prog, startup_prog)
        ops = [op.type for op in avg_cost.block.ops]
        vars = [x.name for x in train_prog.list_vars()]
        parameters = [
            x.name for x in train_prog.list_vars() if x.persistable == True
        ]
        self.assertIn('@BroadCast', ''.join(vars))
        self.assertIn('cast', ops)
        self.assertIn('check_finite_and_unscale', ops)
        self.assertEqual(
            set(parameters),
            set([
                "fc_1.b_0", "fc_2.b_0", "fc_2.w_0", "fc_1.b_0_velocity_0",
                "fc_2.b_0_velocity_0", "fc_2.w_0_velocity_0", "learning_rate_0",
                "loss_scaling_0", "num_bad_steps_0", "num_good_steps_0"
            ]))
        self.assertEqual(ops, [
            'cast', 'cast', 'cast', 'fill_constant', 'fill_constant',
            'fill_constant', 'c_sync_calc_stream', 'c_broadcast', 'c_broadcast',
            'c_broadcast', 'c_broadcast', 'c_broadcast', 'c_broadcast',
            'c_sync_comm_stream', 'cast', 'mul', 'elementwise_add', 'cast',
            'tanh', 'cast', 'mul', 'elementwise_add', 'cast', 'tanh', 'cast',
            'mul', 'elementwise_add', 'softmax', 'cast', 'cross_entropy2',
            'mean', 'elementwise_mul', 'fill_constant', 'scale',
            'elementwise_mul_grad', 'mean_grad', 'cross_entropy_grad2', 'cast',
            'softmax_grad', 'elementwise_add_grad', 'mul_grad', 'cast',
            'tanh_grad', 'cast', 'elementwise_add_grad', 'mul_grad', 'cast',
            'tanh_grad', 'cast', 'elementwise_add_grad', 'mul_grad',
            'c_sync_calc_stream', 'c_allreduce_sum', 'c_allreduce_sum',
            'c_allreduce_sum', 'c_allreduce_sum', 'c_allreduce_sum',
            'c_allreduce_sum', 'c_sync_comm_stream', 'cast', 'cast', 'cast',
            'check_finite_and_unscale', 'cast', 'c_sync_calc_stream',
            'c_allreduce_max', 'c_sync_comm_stream', 'cast',
            'update_loss_scaling', 'momentum', 'momentum', 'momentum'
        ])

    def test_sharding_recompute_optimizer(self):
        train_prog, startup_prog = paddle.fluid.Program(), paddle.fluid.Program(
        )
        avg_cost, strategy = self.net(train_prog, startup_prog)
        self.set_strategy(strategy, 'sharding')
        self.set_strategy(strategy, 'recompute')
        self.optimizer(avg_cost, strategy, train_prog, startup_prog)

        ops = [op.type for op in avg_cost.block.ops]
        vars = [x.name for x in train_prog.list_vars()]
        parameters = [
            x.name for x in train_prog.list_vars() if x.persistable == True
        ]

        self.assertIn('@BroadCast', ''.join(vars))
        self.assertIn('subprog', ''.join(vars))
        self.assertEqual(
            set(parameters),
            set([
                "fc_1.b_0", "fc_2.b_0", "fc_2.w_0", "fc_1.b_0_velocity_0",
                "fc_2.b_0_velocity_0", "fc_2.w_0_velocity_0", "learning_rate_0"
            ]))

        self.assertEqual(ops, [
            'fill_constant', 'fill_constant', 'fill_constant',
            'c_sync_calc_stream', 'c_broadcast', 'c_broadcast', 'c_broadcast',
            'c_broadcast', 'c_broadcast', 'c_broadcast', 'c_sync_comm_stream',
            'mul', 'elementwise_add', 'tanh', 'mul', 'elementwise_add', 'tanh',
            'mul', 'elementwise_add', 'softmax', 'cross_entropy2', 'mean',
            'fill_constant', 'scale', 'mean_grad', 'cross_entropy_grad2',
            'softmax_grad', 'elementwise_add_grad', 'mul_grad', 'mul',
            'elementwise_add', 'tanh_grad', 'elementwise_add_grad', 'mul_grad',
            'mul', 'elementwise_add', 'tanh_grad', 'elementwise_add_grad',
            'mul_grad', 'c_sync_calc_stream', 'c_allreduce_sum',
            'c_allreduce_sum', 'c_allreduce_sum', 'c_allreduce_sum',
            'c_allreduce_sum', 'c_allreduce_sum', 'c_sync_comm_stream',
            'momentum', 'momentum', 'momentum'
        ])

    def test_sharding_amp_recompute_optimizer(self):
        train_prog, startup_prog = paddle.fluid.Program(), paddle.fluid.Program(
        )
        avg_cost, strategy = self.net(train_prog, startup_prog)
        self.set_strategy(strategy, 'sharding')
        self.set_strategy(strategy, 'recompute')
        self.set_strategy(strategy, 'amp')
        self.optimizer(avg_cost, strategy, train_prog, startup_prog)

        ops = [op.type for op in avg_cost.block.ops]
        vars = [x.name for x in train_prog.list_vars()]
        parameters = [
            x.name for x in train_prog.list_vars() if x.persistable == True
        ]

        self.assertIn('@BroadCast', ''.join(vars))
        self.assertIn('subprog', ''.join(vars))
        self.assertIn('cast', ops)
        self.assertIn('check_finite_and_unscale', ops)

        self.assertEqual(
            set(parameters),
            set([
                "fc_1.b_0", "fc_2.b_0", "fc_2.w_0", "fc_1.b_0_velocity_0",
                "fc_2.b_0_velocity_0", "fc_2.w_0_velocity_0", "learning_rate_0",
                "loss_scaling_0", "num_bad_steps_0", "num_good_steps_0"
            ]))

        self.assertEqual(ops, [
            'cast', 'cast', 'cast', 'fill_constant', 'fill_constant',
            'fill_constant', 'fill_constant', 'fill_constant', 'fill_constant',
            'c_sync_calc_stream', 'c_broadcast', 'c_broadcast', 'c_broadcast',
            'c_broadcast', 'c_broadcast', 'c_broadcast', 'c_broadcast',
            'c_broadcast', 'c_broadcast', 'c_broadcast', 'c_sync_comm_stream',
            'cast', 'cast', 'mul', 'cast', 'elementwise_add', 'cast', 'tanh',
            'cast', 'cast', 'mul', 'elementwise_add', 'cast', 'tanh', 'cast',
            'mul', 'elementwise_add', 'softmax', 'cast', 'cross_entropy2',
            'mean', 'elementwise_mul', 'fill_constant', 'scale',
            'elementwise_mul_grad', 'mean_grad', 'cross_entropy_grad2', 'cast',
            'softmax_grad', 'elementwise_add_grad', 'mul_grad', 'cast', 'cast',
            'cast', 'mul', 'cast', 'elementwise_add', 'cast', 'tanh_grad',
            'cast', 'elementwise_add_grad', 'mul_grad', 'cast', 'cast', 'mul',
            'cast', 'elementwise_add', 'cast', 'tanh_grad', 'cast',
            'elementwise_add_grad', 'mul_grad', 'c_sync_calc_stream',
            'c_allreduce_sum', 'c_allreduce_sum', 'c_allreduce_sum',
            'c_allreduce_sum', 'c_allreduce_sum', 'c_allreduce_sum',
            'c_sync_comm_stream', 'cast', 'cast', 'cast',
            'check_finite_and_unscale', 'cast', 'c_sync_calc_stream',
            'c_allreduce_max', 'c_sync_comm_stream', 'cast',
            'update_loss_scaling', 'momentum', 'momentum', 'momentum'
        ])

    def test_sharding_weight_decay(self):
        train_prog, startup_prog = paddle.fluid.Program(), paddle.fluid.Program(
        )
        avg_cost, strategy = self.net(train_prog, startup_prog)
        self.set_strategy(strategy, 'sharding')
        regularization = paddle.fluid.regularizer.L2Decay(0.0001)
        self.optimizer(
            avg_cost,
            strategy,
            train_prog,
            startup_prog,
            regularization=regularization)
        parameters = [
            x.name for x in train_prog.list_vars() if x.persistable == True
        ]
        ops = [op.type for op in avg_cost.block.ops]
        vars = [x.name for x in train_prog.list_vars()]
        self.assertIn('@BroadCast', ''.join(vars))
        self.assertEqual(
            set(parameters),
            set([
                "fc_1.b_0", "fc_2.b_0", "fc_2.w_0", "fc_1.b_0_velocity_0",
                "fc_2.b_0_velocity_0", "fc_2.w_0_velocity_0", "learning_rate_0"
            ]))

        self.assertEqual(ops, [
            'fill_constant', 'fill_constant', 'fill_constant',
            'c_sync_calc_stream', 'c_broadcast', 'c_broadcast', 'c_broadcast',
            'c_broadcast', 'c_broadcast', 'c_broadcast', 'c_sync_comm_stream',
            'mul', 'elementwise_add', 'tanh', 'mul', 'elementwise_add', 'tanh',
            'mul', 'elementwise_add', 'softmax', 'cross_entropy2', 'mean',
            'fill_constant', 'scale', 'mean_grad', 'cross_entropy_grad2',
            'softmax_grad', 'elementwise_add_grad', 'mul_grad', 'tanh_grad',
            'elementwise_add_grad', 'mul_grad', 'tanh_grad',
            'elementwise_add_grad', 'mul_grad', 'c_sync_calc_stream',
            'c_allreduce_sum', 'c_allreduce_sum', 'c_allreduce_sum',
            'c_allreduce_sum', 'c_allreduce_sum', 'c_allreduce_sum',
            'c_sync_comm_stream', 'scale', 'sum', 'scale', 'sum', 'scale',
            'sum', 'momentum', 'momentum', 'momentum'
        ])

    def test_sharding_gradient_clip(self):
        train_prog, startup_prog = paddle.fluid.Program(), paddle.fluid.Program(
        )
        avg_cost, strategy = self.net(train_prog, startup_prog)
        self.set_strategy(strategy, 'sharding')
        clip = paddle.fluid.clip.GradientClipByGlobalNorm(clip_norm=1.0)
        self.optimizer(
            avg_cost, strategy, train_prog, startup_prog, grad_clip=clip)
        parameters = [
            x.name for x in train_prog.list_vars() if x.persistable == True
        ]
        ops = [op.type for op in avg_cost.block.ops]
        vars = [x.name for x in train_prog.list_vars()]
        self.assertIn('@BroadCast', ''.join(vars))
        self.assertEqual(
            set(parameters),
            set([
                "fc_1.b_0", "fc_2.b_0", "fc_2.w_0", "fc_1.b_0_velocity_0",
                "fc_2.b_0_velocity_0", "fc_2.w_0_velocity_0", "learning_rate_0"
            ]))
        self.assertEqual(ops, [
            'fill_constant', 'fill_constant', 'fill_constant',
            'c_sync_calc_stream', 'c_broadcast', 'c_broadcast', 'c_broadcast',
            'c_broadcast', 'c_broadcast', 'c_broadcast', 'c_sync_comm_stream',
            'mul', 'elementwise_add', 'tanh', 'mul', 'elementwise_add', 'tanh',
            'mul', 'elementwise_add', 'softmax', 'cross_entropy2', 'mean',
            'fill_constant', 'scale', 'mean_grad', 'cross_entropy_grad2',
            'softmax_grad', 'elementwise_add_grad', 'mul_grad', 'tanh_grad',
            'elementwise_add_grad', 'mul_grad', 'tanh_grad',
            'elementwise_add_grad', 'mul_grad', 'c_sync_calc_stream',
            'c_allreduce_sum', 'c_allreduce_sum', 'c_allreduce_sum',
            'c_allreduce_sum', 'c_allreduce_sum', 'c_allreduce_sum',
            'c_sync_comm_stream', 'square', 'reduce_sum', 'square',
            'reduce_sum', 'square', 'reduce_sum', 'sum', 'c_sync_calc_stream',
            'c_allreduce_sum', 'c_sync_comm_stream', 'sqrt', 'fill_constant',
            'elementwise_max', 'elementwise_div', 'elementwise_mul',
            'elementwise_mul', 'elementwise_mul', 'momentum', 'momentum',
            'momentum'
        ])

    def test_sharding_clone_for_test(self):
        train_prog, startup_prog = paddle.fluid.Program(), paddle.fluid.Program(
        )
        avg_cost, strategy = self.net(train_prog, startup_prog)
        self.set_strategy(strategy, 'sharding')
        self.optimizer(avg_cost, strategy, train_prog, startup_prog)
        sharding.utils.comm_analyse(train_prog)
        test_prog = train_prog.clone(for_test=True)
        sharding.utils.add_sync_comm(test_prog, strategy)
        ops = [op.type for op in test_prog.global_block().ops]

        self.assertEqual(ops, [
            'fill_constant', 'fill_constant', 'fill_constant',
            'c_sync_calc_stream', 'c_broadcast', 'c_broadcast', 'c_broadcast',
            'c_broadcast', 'c_broadcast', 'c_broadcast', 'c_sync_comm_stream',
            'mul', 'elementwise_add', 'tanh', 'mul', 'elementwise_add', 'tanh',
            'mul', 'elementwise_add', 'softmax', 'cross_entropy2', 'mean'
        ])


if __name__ == "__main__":
    unittest.main()
