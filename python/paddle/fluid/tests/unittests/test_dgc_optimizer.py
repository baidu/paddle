# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
import paddle.fluid.framework as framework
import paddle.fluid.optimizer as optimizer
import paddle.fluid.regularizer as regularizer
import paddle.fluid.clip as clip
import paddle.compat as cpt
from paddle.fluid.backward import append_backward
paddle.enable_static()


class TestDGCMomentumOptimizer(unittest.TestCase):
    class MockDGCMomentum(optimizer.DGCMomentumOptimizer):
        def get_accumulators(self):
            return self._accumulators

        def get_velocity_str(self):
            return self._u_velocity_acc_str

    def check_dgc_momentum_optimizer(self,
                                     dims=[5, 10, 8],
                                     name="momentum",
                                     regularization=None,
                                     use_recompute=False):
        init_program = framework.Program()
        program = framework.Program()
        block = program.global_block()
        mul_x = block.create_parameter(
            dtype="float32",
            shape=[dims[0], dims[1]],
            lod_level=0,
            name="mul.x",
            optimize_attr={'learning_rate': 1.1},
            regularizer=None if regularization is not None else
            regularizer.L2DecayRegularizer(2e-4))
        mul_y = block.create_var(
            dtype="float32",
            shape=[dims[1], dims[2]],
            lod_level=0,
            name="mul.y")
        mul_out = block.create_var(
            dtype="float32",
            shape=[dims[0], dims[2]],
            lod_level=0,
            name="mul.out")
        block.append_op(
            type="mul",
            inputs={"X": mul_x,
                    "Y": mul_y},
            outputs={"Out": mul_out},
            attrs={"x_num_col_dims": 1})
        learning_rate = 0.01

        dgc_momentum_optimizer = self.MockDGCMomentum(
            learning_rate=learning_rate,
            momentum=0.2,
            rampup_begin_step=0,
            num_trainers=2,
            regularization=regularization,
            grad_clip=clip.GradientClipByNorm(1.0))

        if use_recompute:
            dgc_momentum_optimizer = optimizer.RecomputeOptimizer(
                dgc_momentum_optimizer)
            dgc_momentum_optimizer._set_checkpoints([])
            dgc_momentum_optimizer.get_accumulators = dgc_momentum_optimizer._optimizer.get_accumulators
            dgc_momentum_optimizer.get_velocity_str = dgc_momentum_optimizer._optimizer.get_velocity_str

        mean_out = block.create_var(
            dtype="float32", shape=[1], lod_level=0, name="mean.out")
        block.append_op(
            type="mean", inputs={"X": mul_out}, outputs={"Out": mean_out})
        # params_grads = append_backward(mean_out)
        params_grads = dgc_momentum_optimizer.backward(
            mean_out, startup_program=init_program)

        with framework.program_guard(program, init_program):
            opts = dgc_momentum_optimizer.apply_gradients(params_grads)

        accumulator_count = 1 if name == "momentum" else 2
        self.assertEqual(len(params_grads), 1)
        self.assertEqual(
            len(dgc_momentum_optimizer.get_accumulators()), accumulator_count)

        self.assertEqual(len(opts), 2)
        sgd_op = opts[-1]
        self.assertEqual([op.type for op in opts], ["scale", name])
        self.assertFalse(sgd_op.attr('use_nesterov'))

        # Check accumulators
        accumulators = dgc_momentum_optimizer.get_accumulators()
        self.assertEqual(len(accumulators), accumulator_count)
        self.assertTrue(
            dgc_momentum_optimizer.get_velocity_str() in accumulators)
        velocity_acc = accumulators[dgc_momentum_optimizer.get_velocity_str()]
        self.assertEqual(len(velocity_acc), 1)
        self.assertTrue(mul_x.name in velocity_acc)

        # Check init_program
        # dgc not apply include: lr, dgc(count, nranks, begin step), (u,)
        # dgc apply include: lr, dgc(count, nranks, begin_step), (u,v,k,encode,gather)
        init_ops_count = 5 if name == "momentum" else 9
        init_ops = init_program.global_block().ops
        self.assertEqual(len(init_ops), init_ops_count)
        self.assertEqual(init_ops[0].type, "fill_constant")
        self.assertAlmostEqual(init_ops[0].attr('value'), learning_rate)

        # check dgc op regularization coeff
        train_ops = program.global_block().ops
        for op in train_ops:
            if op.type == "dgc":
                coeff = 2e-4 if regularization is None else 1e-4
                self.assertAlmostEqual(op.attr('regular_coeff'), coeff)
                print("dgc regular_coeff=" + str(coeff))

    def test_tpyeError(self):
        # the type of DGCMomentumOptimizer(grad_clip=) must be 'GradientClipByNorm'
        with self.assertRaises(TypeError):
            dgc_momentum_optimizer = self.MockDGCMomentum(
                learning_rate=0.01,
                momentum=0.2,
                rampup_begin_step=0,
                num_trainers=2,
                grad_clip=clip.GradientClipByGlobalNorm(1.0))

    def test_momentum_without_dgc(self):
        self.check_dgc_momentum_optimizer(
            regularization=regularizer.L1Decay(1e-4))

    def test_momentum_with_dgc(self):
        # 16 * 1024 = 16384, use dgc momentum
        self.check_dgc_momentum_optimizer(
            dims=[16, 1024, 8],
            name="dgc_momentum",
            regularization=regularizer.L2Decay(1e-4))

        # check param.regularizer in dgc
        self.check_dgc_momentum_optimizer(
            dims=[16, 1024, 8], name="dgc_momentum")

    def test_momentum_with_dgc_recompute(self):
        # 16 * 1024 = 16384, use dgc momentum
        self.check_dgc_momentum_optimizer(
            dims=[16, 1024, 8],
            name="dgc_momentum",
            regularization=regularizer.L2Decay(1e-4),
            use_recompute=True)


if __name__ == '__main__':
    unittest.main()
