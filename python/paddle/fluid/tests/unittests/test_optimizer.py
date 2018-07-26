#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

import paddle.fluid.framework as framework
import paddle.fluid.optimizer as optimizer
from paddle.fluid.backward import append_backward


class TestOptimizer(unittest.TestCase):
    def test_sgd_optimizer(self):
        def check_sgd_optimizer(optimizer_attr):
            init_program = framework.Program()
            program = framework.Program()
            block = program.global_block()
            mul_x = block.create_parameter(
                dtype="float32",
                shape=[5, 10],
                lod_level=0,
                name="mul.x",
                optimize_attr=optimizer_attr)
            mul_y = block.create_var(
                dtype="float32", shape=[10, 8], lod_level=0, name="mul.y")
            mul_out = block.create_var(
                dtype="float32", shape=[5, 8], lod_level=0, name="mul.out")
            mean_out = block.create_var(
                dtype="float32", shape=[1], lod_level=0, name="mean.out")
            block.append_op(
                type="mul",
                inputs={"X": mul_x,
                        "Y": mul_y},
                outputs={"Out": mul_out},
                attrs={"x_num_col_dims": 1})
            block.append_op(
                type="mean", inputs={"X": mul_out}, outputs={"Out": mean_out})
            sgd_optimizer = optimizer.SGDOptimizer(learning_rate=0.01)
            opts, _ = sgd_optimizer.minimize(mean_out, init_program)
            return opts

        opts = check_sgd_optimizer({'learning_rate': 1.1})
        self.assertEqual(len(opts), 3)
        self.assertEqual([op.type for op in opts],
                         ["fill_constant", "elementwise_mul", "sgd"])

        opts = check_sgd_optimizer({'learning_rate': 1.0})
        self.assertEqual(len(opts), 1)
        self.assertEqual([op.type for op in opts], ["sgd"])


class TestMomentumOptimizer(unittest.TestCase):
    class MockMomentum(optimizer.MomentumOptimizer):
        def get_accumulators(self):
            return self._accumulators

        def get_velocity_str(self):
            return self._velocity_acc_str

    def test_vanilla_momentum_optimizer(self):
        init_program = framework.Program()
        program = framework.Program()
        block = program.global_block()
        mul_x = block.create_parameter(
            dtype="float32",
            shape=[5, 10],
            lod_level=0,
            name="mul.x",
            optimize_attr={'learning_rate': 1.1})
        mul_y = block.create_var(
            dtype="float32", shape=[10, 8], lod_level=0, name="mul.y")
        mul_out = block.create_var(
            dtype="float32", shape=[5, 8], lod_level=0, name="mul.out")
        block.append_op(
            type="mul",
            inputs={"X": mul_x,
                    "Y": mul_y},
            outputs={"Out": mul_out},
            attrs={"x_num_col_dims": 1})
        learning_rate = 0.01
        momentum_optimizer = self.MockMomentum(
            learning_rate=learning_rate, momentum=0.2)
        mean_out = block.create_var(
            dtype="float32", shape=[1], lod_level=0, name="mean.out")
        block.append_op(
            type="mean", inputs={"X": mul_out}, outputs={"Out": mean_out})
        params_grads = append_backward(mean_out)
        self.assertEqual(len(params_grads), 1)
        self.assertEqual(len(momentum_optimizer.get_accumulators()), 0)
        opts = momentum_optimizer._create_optimization_pass(
            params_grads, mul_out, init_program)
        self.assertEqual(len(opts), 3)
        sgd_op = opts[-1]
        self.assertEqual([op.type for op in opts],
                         ["fill_constant", "elementwise_mul", "momentum"])
        self.assertFalse(sgd_op.attr('use_nesterov'))

        # Check accumulators
        accumulators = momentum_optimizer.get_accumulators()
        self.assertEqual(len(accumulators), 1)
        self.assertTrue(momentum_optimizer.get_velocity_str() in accumulators)
        velocity_acc = accumulators[momentum_optimizer.get_velocity_str()]
        self.assertEqual(len(velocity_acc), 1)
        self.assertTrue(mul_x.name in velocity_acc)

        # Check init_program
        init_ops = init_program.global_block().ops
        self.assertEqual(len(init_ops), 2)
        self.assertEqual(init_ops[0].type, "fill_constant")
        self.assertAlmostEqual(init_ops[0].attr('value'), learning_rate)
        self.assertEqual(init_ops[1].type, "fill_constant")
        self.assertAlmostEqual(init_ops[1].attr('value'), 0.0)

    def test_nesterov_momentum_optimizer(self):
        init_program = framework.Program()
        program = framework.Program()
        block = program.global_block()
        mul_x = block.create_parameter(
            dtype="float32",
            shape=[5, 10],
            lod_level=0,
            name="mul.x",
            optimize_attr={'learning_rate': 1.1})
        mul_y = block.create_var(
            dtype="float32", shape=[10, 8], lod_level=0, name="mul.y")
        mul_out = block.create_var(
            dtype="float32", shape=[5, 8], lod_level=0, name="mul.out")
        block.append_op(
            type="mul",
            inputs={"X": mul_x,
                    "Y": mul_y},
            outputs={"Out": mul_out},
            attrs={"x_num_col_dims": 1})
        mean_out = block.create_var(
            dtype="float32", shape=[1], lod_level=0, name="mean.out")
        block.append_op(
            type="mean", inputs={"X": mul_out}, outputs={"Out": mean_out})
        learning_rate = 0.01
        momentum_optimizer = self.MockMomentum(
            learning_rate=learning_rate, momentum=0.2, use_nesterov=True)
        params_grads = append_backward(mean_out)
        self.assertEqual(len(params_grads), 1)
        self.assertEqual(len(momentum_optimizer.get_accumulators()), 0)
        opts = momentum_optimizer._create_optimization_pass(
            params_grads, mul_out, init_program)
        self.assertEqual(len(opts), 3)
        sgd_op = opts[-1]
        self.assertEqual([op.type for op in opts],
                         ["fill_constant", "elementwise_mul", "momentum"])
        self.assertTrue(sgd_op.attr('use_nesterov'))

        # Check accumulators
        accumulators = momentum_optimizer.get_accumulators()
        self.assertEqual(len(accumulators), 1)
        self.assertTrue(momentum_optimizer.get_velocity_str() in accumulators)
        velocity_acc = accumulators[momentum_optimizer.get_velocity_str()]
        self.assertEqual(len(velocity_acc), 1)
        self.assertTrue(mul_x.name in velocity_acc)

        # Check init_program
        init_ops = init_program.global_block().ops
        self.assertEqual(len(init_ops), 2)
        self.assertEqual(init_ops[0].type, "fill_constant")
        self.assertAlmostEqual(init_ops[0].attr('value'), learning_rate)
        self.assertEqual(init_ops[1].type, "fill_constant")
        self.assertAlmostEqual(init_ops[1].attr('value'), 0.0)


class TestAdagradOptimizer(unittest.TestCase):
    class MockAdagrad(optimizer.AdagradOptimizer):
        def get_accumulators(self):
            return self._accumulators

        def get_moment_str(self):
            return self._moment_acc_str

    def test_adagrad_optimizer(self):
        init_program = framework.Program()
        program = framework.Program()
        block = program.global_block()
        mul_x = block.create_parameter(
            dtype="float32",
            shape=[5, 10],
            lod_level=0,
            name="mul.x",
            optimize_attr={'learning_rate': 1.1})
        mul_y = block.create_var(
            dtype="float32", shape=[10, 8], lod_level=0, name="mul.y")
        mul_out = block.create_var(
            dtype="float32", shape=[5, 8], lod_level=0, name="mul.out")
        block.append_op(
            type="mul",
            inputs={"X": mul_x,
                    "Y": mul_y},
            outputs={"Out": mul_out},
            attrs={"x_num_col_dims": 1})
        mean_out = block.create_var(
            dtype="float32", shape=[1], lod_level=0, name="mean.out")
        block.append_op(
            type="mean", inputs={"X": mul_out}, outputs={"Out": mean_out})
        learning_rate = 0.01
        adagrad_optimizer = self.MockAdagrad(
            learning_rate=learning_rate, epsilon=1.0e-6)
        params_grads = append_backward(mean_out)
        self.assertEqual(len(params_grads), 1)
        self.assertEqual(len(adagrad_optimizer.get_accumulators()), 0)
        opts = adagrad_optimizer._create_optimization_pass(
            params_grads, mul_out, init_program)
        self.assertEqual(len(opts), 3)
        self.assertEqual([op.type for op in opts],
                         ["fill_constant", "elementwise_mul", "adagrad"])

        # Check accumulators
        accumulators = adagrad_optimizer.get_accumulators()
        self.assertEqual(len(accumulators), 1)
        self.assertTrue(adagrad_optimizer.get_moment_str() in accumulators)
        moment_acc = accumulators[adagrad_optimizer.get_moment_str()]
        self.assertEqual(len(moment_acc), 1)
        self.assertTrue(mul_x.name in moment_acc)

        # Check init_program
        init_ops = init_program.global_block().ops
        self.assertEqual(len(init_ops), 2)
        self.assertEqual(init_ops[0].type, "fill_constant")
        self.assertAlmostEqual(init_ops[0].attr('value'), learning_rate)
        self.assertEqual(init_ops[1].type, "fill_constant")
        self.assertAlmostEqual(init_ops[1].attr('value'), 0.0)


class TestAdamOptimizer(unittest.TestCase):
    class MockAdam(optimizer.AdamOptimizer):
        def get_accumulators(self):
            return self._accumulators

        def get_moment1_str(self):
            return self._moment1_acc_str

        def get_moment2_str(self):
            return self._moment2_acc_str

    def test_adam_optimizer(self):
        init_program = framework.Program()
        program = framework.Program()
        block = program.global_block()
        mul_x = block.create_parameter(
            dtype="float32",
            shape=[5, 10],
            lod_level=0,
            name="mul.x",
            optimize_attr={'learning_rate': 1.1})
        mul_y = block.create_var(
            dtype="float32", shape=[10, 8], lod_level=0, name="mul.y")
        mul_out = block.create_var(
            dtype="float32", shape=[5, 8], lod_level=0, name="mul.out")
        block.append_op(
            type="mul",
            inputs={"X": mul_x,
                    "Y": mul_y},
            outputs={"Out": mul_out},
            attrs={"x_num_col_dims": 1})
        mean_out = block.create_var(
            dtype="float32", shape=[1], lod_level=0, name="mean.out")
        block.append_op(
            type="mean", inputs={"X": mul_out}, outputs={"Out": mean_out})
        learning_rate = 0.01
        adam_optimizer = self.MockAdam(
            learning_rate=learning_rate, beta1=0.9, beta2=0.999)
        params_grads = append_backward(mean_out)
        self.assertEqual(len(params_grads), 1)
        self.assertEqual(len(adam_optimizer.get_accumulators()), 0)
        opts = adam_optimizer._create_optimization_pass(params_grads, mul_out,
                                                        init_program)
        self.assertEqual(len(opts), 5)
        self.assertEqual(
            [op.type for op in opts],
            ["fill_constant", "elementwise_mul", "adam", "scale", "scale"])

        # Check accumulators
        accumulators = adam_optimizer.get_accumulators()
        self.assertEqual(len(accumulators), 2)
        self.assertTrue(adam_optimizer.get_moment1_str() in accumulators)
        self.assertTrue(adam_optimizer.get_moment2_str() in accumulators)
        moment1_acc = accumulators[adam_optimizer.get_moment1_str()]
        moment2_acc = accumulators[adam_optimizer.get_moment2_str()]
        self.assertEqual(len(moment1_acc), 1)
        self.assertEqual(len(moment2_acc), 1)
        self.assertTrue(mul_x.name in moment1_acc)
        self.assertTrue(mul_x.name in moment2_acc)

        # Check init_program
        init_ops = init_program.global_block().ops
        self.assertEqual(len(init_ops), 5)
        self.assertEqual(init_ops[0].type, "fill_constant")
        self.assertAlmostEqual(init_ops[0].attr('value'), learning_rate)


class TestAdamaxOptimizer(unittest.TestCase):
    class MockAdamax(optimizer.AdamaxOptimizer):
        def get_accumulators(self):
            return self._accumulators

        def get_moment_str(self):
            return self._moment_acc_str

        def get_inf_norm_str(self):
            return self._inf_norm_acc_str

    def test_adamax_optimizer(self):
        init_program = framework.Program()
        program = framework.Program()
        block = program.global_block()
        mul_x = block.create_parameter(
            dtype="float32",
            shape=[5, 10],
            lod_level=0,
            name="mul.x",
            optimize_attr={'learning_rate': 1.1})
        mul_y = block.create_var(
            dtype="float32", shape=[10, 8], lod_level=0, name="mul.y")
        mul_out = block.create_var(
            dtype="float32", shape=[5, 8], lod_level=0, name="mul.out")
        block.append_op(
            type="mul",
            inputs={"X": mul_x,
                    "Y": mul_y},
            outputs={"Out": mul_out},
            attrs={"x_num_col_dims": 1})
        mean_out = block.create_var(
            dtype="float32", shape=[1], lod_level=0, name="mean.out")
        block.append_op(
            type="mean", inputs={"X": mul_out}, outputs={"Out": mean_out})
        learning_rate = 0.01
        adamax_optimizer = self.MockAdamax(
            learning_rate=learning_rate, beta1=0.9, beta2=0.999)
        params_grads = append_backward(mean_out)
        self.assertEqual(len(params_grads), 1)
        self.assertEqual(len(adamax_optimizer.get_accumulators()), 0)
        opts = adamax_optimizer._create_optimization_pass(params_grads, mul_out,
                                                          init_program)
        self.assertEqual(len(opts), 4)
        self.assertEqual(
            [op.type for op in opts],
            ["fill_constant", "elementwise_mul", "adamax", "scale"])

        # Check accumulators
        accumulators = adamax_optimizer.get_accumulators()
        self.assertEqual(len(accumulators), 2)
        self.assertTrue(adamax_optimizer.get_moment_str() in accumulators)
        self.assertTrue(adamax_optimizer.get_inf_norm_str() in accumulators)
        moment_acc = accumulators[adamax_optimizer.get_moment_str()]
        inf_norm_acc = accumulators[adamax_optimizer.get_inf_norm_str()]
        self.assertEqual(len(moment_acc), 1)
        self.assertEqual(len(inf_norm_acc), 1)
        self.assertTrue(mul_x.name in moment_acc)
        self.assertTrue(mul_x.name in inf_norm_acc)

        # Check init_program
        init_ops = init_program.global_block().ops
        self.assertEqual(len(init_ops), 4)
        self.assertEqual(init_ops[0].type, "fill_constant")
        self.assertAlmostEqual(init_ops[0].attr('value'), learning_rate)


class TestDecayedAdagradOptimizer(unittest.TestCase):
    class MockDecayedAdagrad(optimizer.DecayedAdagradOptimizer):
        def get_accumulators(self):
            return self._accumulators

        def get_moment_str(self):
            return self._moment_acc_str

    def test_decayed_adagrad_optimizer(self):
        init_program = framework.Program()
        program = framework.Program()
        block = program.global_block()
        mul_x = block.create_parameter(
            dtype="float32",
            shape=[5, 10],
            lod_level=0,
            name="mul.x",
            optimize_attr={'learning_rate': 1.1})
        mul_y = block.create_var(
            dtype="float32", shape=[10, 8], lod_level=0, name="mul.y")
        mul_out = block.create_var(
            dtype="float32", shape=[5, 8], lod_level=0, name="mul.out")
        block.append_op(
            type="mul",
            inputs={"X": mul_x,
                    "Y": mul_y},
            outputs={"Out": mul_out},
            attrs={"x_num_col_dims": 1})
        mean_out = block.create_var(
            dtype="float32", shape=[1], lod_level=0, name="mean.out")
        block.append_op(
            type="mean", inputs={"X": mul_out}, outputs={"Out": mean_out})
        learning_rate = 0.01
        decayed_adagrad_optimizer = self.MockDecayedAdagrad(
            learning_rate=learning_rate, decay=0.95, epsilon=1.0e-6)
        params_grads = append_backward(mean_out)
        self.assertEqual(len(params_grads), 1)
        self.assertEqual(len(decayed_adagrad_optimizer.get_accumulators()), 0)
        opts = decayed_adagrad_optimizer._create_optimization_pass(
            params_grads, mul_out, init_program)
        self.assertEqual(len(opts), 3)
        self.assertEqual(
            [op.type for op in opts],
            ["fill_constant", "elementwise_mul", "decayed_adagrad"])

        # Check accumulators
        accumulators = decayed_adagrad_optimizer.get_accumulators()
        self.assertEqual(len(accumulators), 1)
        self.assertTrue(
            decayed_adagrad_optimizer.get_moment_str() in accumulators)
        moment_acc = accumulators[decayed_adagrad_optimizer.get_moment_str()]
        self.assertEqual(len(moment_acc), 1)
        self.assertTrue(mul_x.name in moment_acc)

        # Check init_program
        init_ops = init_program.global_block().ops
        self.assertEqual(len(init_ops), 2)
        self.assertEqual(init_ops[0].type, "fill_constant")
        self.assertAlmostEqual(init_ops[0].attr('value'), learning_rate)
        self.assertEqual(init_ops[1].type, "fill_constant")
        self.assertAlmostEqual(init_ops[1].attr('value'), 0.0)


class TestFtrlOptimizer(unittest.TestCase):
    class MockFtrl(optimizer.FtrlOptimizer):
        def get_accumulators(self):
            return self._accumulators

        def get_squared_str(self):
            return self._squared_acc_str

        def get_linear_str(self):
            return self._linear_acc_str

    def test_ftrl_optimizer(self):
        init_program = framework.Program()
        program = framework.Program()
        block = program.global_block()
        mul_x = block.create_parameter(
            dtype="float32",
            shape=[5, 10],
            lod_level=0,
            name="mul.x",
            optimize_attr={'learning_rate': 1.1})
        mul_y = block.create_var(
            dtype="float32", shape=[10, 8], lod_level=0, name="mul.y")
        mul_out = block.create_var(
            dtype="float32", shape=[5, 8], lod_level=0, name="mul.out")
        block.append_op(
            type="mul",
            inputs={"X": mul_x,
                    "Y": mul_y},
            outputs={"Out": mul_out},
            attrs={"x_num_col_dims": 1})
        mean_out = block.create_var(
            dtype="float32", shape=[1], lod_level=0, name="mean.out")
        block.append_op(
            type="mean", inputs={"X": mul_out}, outputs={"Out": mean_out})
        learning_rate = 0.01
        ftrl_optimizer = self.MockFtrl(
            learning_rate=learning_rate, l1=0.0, l2=0.0, lr_power=-0.5)
        params_grads = append_backward(mean_out)
        self.assertEqual(len(params_grads), 1)
        self.assertEqual(len(ftrl_optimizer.get_accumulators()), 0)
        opts = ftrl_optimizer._create_optimization_pass(params_grads, mul_out,
                                                        init_program)
        self.assertEqual(len(opts), 3)
        self.assertEqual([op.type for op in opts],
                         ["fill_constant", "elementwise_mul", "ftrl"])

        # Check accumulators
        accumulators = ftrl_optimizer.get_accumulators()
        self.assertEqual(len(accumulators), 2)
        self.assertTrue(ftrl_optimizer.get_squared_str() in accumulators)
        self.assertTrue(ftrl_optimizer.get_linear_str() in accumulators)
        squared_acc = accumulators[ftrl_optimizer.get_squared_str()]
        linear_acc = accumulators[ftrl_optimizer.get_linear_str()]
        self.assertEqual(len(squared_acc), 1)
        self.assertEqual(len(linear_acc), 1)
        self.assertTrue(mul_x.name in squared_acc)
        self.assertTrue(mul_x.name in linear_acc)

        # Check init_program
        init_ops = init_program.global_block().ops
        self.assertEqual(len(init_ops), 3)
        self.assertEqual(init_ops[0].type, "fill_constant")
        self.assertAlmostEqual(init_ops[0].attr('value'), learning_rate)


if __name__ == '__main__':
    unittest.main()
