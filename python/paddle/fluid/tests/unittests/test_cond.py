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

from __future__ import print_function

import numpy as np
import os
import unittest

import paddle.fluid as fluid
import paddle.fluid.core as core
import paddle.fluid.layers as layers
import paddle.fluid.framework as framework
from paddle.fluid.backward import append_backward
from paddle.fluid.framework import Program, program_guard
from simple_nets import simple_fc_net_with_inputs, batchnorm_fc_with_inputs

np.random.seed(123)


class TestCondInputOutput(unittest.TestCase):
    def test_return_single_var(self):
        """
        pseudocode:

        if 0.23 < 0.1:
            return 2
        else:
            return -1
        """

        def true_func():
            return layers.fill_constant(shape=[2, 3], dtype='int32', value=2)

        def false_func():
            return layers.fill_constant(shape=[3, 2], dtype='int32', value=-1)

        main_program = Program()
        startup_program = Program()
        with program_guard(main_program, startup_program):
            x = layers.fill_constant(shape=[1], dtype='float32', value=0.1)
            y = layers.fill_constant(shape=[1], dtype='float32', value=0.23)
            pred = layers.less_than(y, x)
            out = layers.cond(pred, true_func, false_func)
            # out is one tensor

        place = fluid.CUDAPlace(0) if core.is_compiled_with_cuda(
        ) else fluid.CPUPlace()
        exe = fluid.Executor(place)
        ret = exe.run(main_program, fetch_list=[out.name])
        self.assertTrue(
            np.allclose(np.asarray(ret), np.full((3, 2), -1, np.int32)))

    def test_return_var_tuple(self):
        """
        pseudocode:

        if True:
            return 1, True
        else:
            return 3, 2
        """

        def true_func():
            return layers.fill_constant(
                shape=[1, 2], dtype='int32', value=1), layers.fill_constant(
                    shape=[2, 3], dtype='bool', value=True)

        def false_func():
            return layers.fill_constant(
                shape=[3, 4], dtype='float32', value=3), layers.fill_constant(
                    shape=[4, 5], dtype='int64', value=2)

        main_program = Program()
        startup_program = Program()
        with program_guard(main_program, startup_program):
            pred = layers.fill_constant(shape=[1], dtype='bool', value=True)
            out = layers.cond(pred, true_func, false_func)
            # out is a tuple containing 2 tensors

        place = fluid.CUDAPlace(0) if core.is_compiled_with_cuda(
        ) else fluid.CPUPlace()
        exe = fluid.Executor(place)
        ret = exe.run(main_program, fetch_list=out)
        self.assertTrue(
            np.allclose(np.asarray(ret[0]), np.full((1, 2), 1, np.int32)))
        self.assertTrue(
            np.allclose(np.asarray(ret[1]), np.full((2, 3), True, np.bool)))

    def test_pass_and_modify_var(self):
        """
        pseudocode:
        for i in range(5):
            a = 7
            if i % 2 == 0:
                a = a * (i + 1)
            else:
                a = a - (i - 1)
        """

        def true_func(a, i):
            a = a * (i + 1)
            return a

        def false_func(a, i):
            a = a - (i - 1)
            return a

        main_program = Program()
        startup_program = Program()
        with program_guard(main_program, startup_program):
            a = layers.fill_constant(shape=[3, 2, 1], dtype='int32', value=7)
            i = fluid.data(name="i", shape=[1], dtype='int32')
            pred = ((i % 2) == 0)
            a = layers.cond(pred, lambda: true_func(a, i),
                            lambda: false_func(a, i))
        place = fluid.CUDAPlace(0) if core.is_compiled_with_cuda(
        ) else fluid.CPUPlace()
        exe = fluid.Executor(place)
        for feed_i in range(5):
            expected_a = 7 * (feed_i + 1) if feed_i % 2 == 0 else 8 - feed_i
            ret = exe.run(main_program,
                          feed={'i': np.full((1), feed_i, np.int32)},
                          fetch_list=[a])
            self.assertTrue(
                np.allclose(
                    np.asarray(ret), np.full((3, 2, 1), expected_a, np.int32)))
            self.assertEqual(1, 2)

    def test_return_none(self):
        """
        pseudocode: test doing nothing in branches
        for i in range(5):
            if i % 2 == 0:
                pass
            else:
                pass
        """

        def true_func():
            pass

        def false_func():
            return None

        main_program = Program()
        startup_program = Program()
        with program_guard(main_program, startup_program):
            i = fluid.data(name="i", shape=[1], dtype='int32')
            pred = ((i % 2) == 0)
            out1 = layers.cond(pred, true_func, false_func)
            out2 = layers.cond(pred, None, false_func)
            out3 = layers.cond(pred, true_func, None)
        place = fluid.CUDAPlace(0) if core.is_compiled_with_cuda(
        ) else fluid.CPUPlace()
        exe = fluid.Executor(place)
        for feed_i in range(5):
            # Test that output is None is runnable
            exe.run(main_program, feed={'i': np.full((1), feed_i, np.int32)})
            self.assertIsNone(out1)
            self.assertIsNone(out2)
            self.assertIsNone(out3)

    def test_wrong_structure_exception(self):
        """
        test returning different number of tensors cannot merge into output
        """

        def func_return_none():
            return None

        def func_return_one_tensor():
            return layers.fill_constant(shape=[2, 7], dtype='int32', value=3)

        def func_return_two_tensors():
            return layers.fill_constant(
                shape=[3, 1], dtype='int32', value=7), layers.fill_constant(
                    shape=[3, 1], dtype='int32', value=8)

        main_program = Program()
        startup_program = Program()
        with program_guard(main_program, startup_program):
            i = fluid.data(name="i", shape=[1], dtype='int32')
            pred = ((i % 2) == 0)
            with self.assertRaises(TypeError):
                out = layers.cond(pred, i, func_return_one_tensor)

            with self.assertRaises(TypeError):
                out = layers.cond(pred, func_return_one_tensor, np.asarray([3]))

            with self.assertRaises(Exception) as e:
                out = layers.cond(pred, func_return_none,
                                  func_return_one_tensor)
            self.assertTrue(
                "Incompatible return values of true_fn and false_fn in cond" in
                str(e.exception))

            with self.assertRaises(Exception) as e:
                out = layers.cond(pred, func_return_two_tensors,
                                  func_return_none)
            self.assertTrue(
                "Incompatible return values of true_fn and false_fn in cond" in
                str(e.exception))

            with self.assertRaises(Exception) as e:
                out = layers.cond(pred, func_return_one_tensor,
                                  func_return_two_tensors)
            self.assertTrue(
                "Incompatible return values of true_fn and false_fn in cond" in
                str(e.exception))

    def test_extremely_simple_net_with_op_in_condition(self):
        main_program = fluid.Program()
        startup_program = fluid.Program()
        with fluid.program_guard(main_program, startup_program):
            a = fluid.layers.fill_constant(
                shape=[1], dtype='float32', value=1.23)
            a.stop_gradient = False
            b = fluid.layers.fill_constant(
                shape=[1], dtype='float32', value=1.25)
            b.stop_gradient = False
            out = layers.cond(a - b < -1.0, lambda: a, lambda: b)
        append_backward(out)

        place = fluid.CUDAPlace(0) if core.is_compiled_with_cuda(
        ) else fluid.CPUPlace()
        exe = fluid.Executor(place)
        ret = exe.run(main_program, fetch_list=[out, a.grad_name, b.grad_name])
        # Note: fill_constant has loss of precision, you have to assertEqual
        # with values doens't lose precision in float-point number.
        self.assertEqual(ret[0][0], 1.25)
        self.assertEqual(ret[1][0], 0.0)
        self.assertEqual(ret[2][0], 1.0)


class TestCondNestedControlFlow(unittest.TestCase):
    def test_cond_inside_cond(self):
        """
        pseudocode:
        for i in range(1, 10):
            a = 2 * i
            if i < 5:
                if i >= 3:
                    return a + a 
                else:
                    return a - a
            else:
                if i < 8:
                    return a * a
                else:
                    return a / a
        """

        def less_than_branch(i, a):
            return layers.cond(i >= 3.0, lambda: layers.elementwise_add(a, a),
                               lambda: layers.elementwise_sub(a, a))

        def greater_equal_branch(i, a):
            return layers.cond(i < 8.0, lambda: layers.elementwise_mul(a, a),
                               lambda: layers.elementwise_div(a, a))

        main_program = Program()
        startup_program = Program()
        with program_guard(main_program, startup_program):
            i = fluid.data(name="i", shape=[1], dtype='float32')
            a = 2.0 * i
            out = layers.cond(i < 5.0, lambda: less_than_branch(i, a),
                              lambda: greater_equal_branch(i, a))
            mean = layers.mean(out)
            append_backward(mean)

        place = fluid.CUDAPlace(0) if core.is_compiled_with_cuda(
        ) else fluid.CPUPlace()
        exe = fluid.Executor(place)
        for feed_i in range(0, 10):
            expected_a = 2.0 * feed_i
            if feed_i < 5:
                expected_ret = expected_a + expected_a if feed_i >= 3 else 0.0
                expected_a_grad = 2.0 if feed_i >= 3 else 0.0
            else:
                expected_ret = expected_a * expected_a if feed_i < 8 else 1.0
                expected_a_grad = 2.0 * expected_a if feed_i < 8 else 0.0
            ret = exe.run(main_program,
                          feed={'i': np.full((1), feed_i, np.float32)},
                          fetch_list=[out.name, a.grad_name])
            self.assertEqual(ret[0][0], expected_ret)
            self.assertEqual(ret[1][0], expected_a_grad)

    def test_cond_op_in_condition(self):
        main_program = fluid.Program()
        startup_program = fluid.Program()

        with fluid.program_guard(main_program, startup_program):
            a = fluid.layers.fill_constant(
                shape=[1], dtype='float32', value=1.23)
            a.stop_gradient = False
            b = fluid.layers.fill_constant(
                shape=[1], dtype='float32', value=1.24)
            b.stop_gradient = False
            out = fluid.layers.cond(
                a < b,
                lambda: fluid.layers.cond(a - b < -1.0, lambda: fluid.layers.elementwise_add(a, b), lambda: fluid.layers.elementwise_mul(a, b)),
                lambda: fluid.layers.cond(a == b, lambda: fluid.layers.elementwise_sub(a, b), lambda: fluid.layers.elementwise_pow(a, b))
            )
            append_backward(out)

        place = fluid.CUDAPlace(0) if core.is_compiled_with_cuda(
        ) else fluid.CPUPlace()
        exe = fluid.Executor(place)
        ret = exe.run(main_program, fetch_list=[out, a.grad_name, b.grad_name])
        # Note: fill_constant has loss of precision, so we assertAlmostEqual.    
        self.assertAlmostEqual(ret[0][0], 1.5252)
        self.assertAlmostEqual(ret[1][0], 1.24)
        self.assertAlmostEqual(ret[2][0], 1.23)


class TestCondBackward(unittest.TestCase):
    def backward_value_helper(self, cond_func, use_cuda, use_parallel_exe):
        """
        Helper function that compares calculated backward value is close to dy/dx
        """
        main_program = Program()
        main_program.random_seed = 123
        startup_program = Program()
        startup_program.random_seed = 123
        with program_guard(main_program, startup_program):
            img = fluid.data(name='image', shape=[-1, 9], dtype='float32')
            img.stop_gradient = False
            label = fluid.data(name='label', shape=[-1, 1], dtype='int64')
            i = fluid.data(name="i", shape=[1], dtype='int32')
            loss = cond_func(i, img, label)
            append_backward(loss)
        place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
        exe = fluid.Executor(place)
        exe.run(startup_program)

        num_devices = 1
        if use_parallel_exe:
            os.environ['CPU_NUM'] = str(2)
            exe = fluid.ParallelExecutor(
                use_cuda=use_cuda,
                main_program=main_program,
                loss_name=loss.name)
            num_devices = exe.device_count

        delta = 0.005
        for feed_i in range(0, 10):
            feed_img = np.random.random(size=[1, 9]).astype(np.float32)
            feed_label = np.random.randint(
                low=0, high=10, size=[1, 1], dtype=np.int64)
            if use_parallel_exe:
                img_grad, loss_value = exe.run(
                    feed={
                        'i': np.full((num_devices), feed_i, np.int32),
                        'image': np.repeat(
                            feed_img, num_devices, axis=0),
                        'label': np.repeat(
                            feed_label, num_devices, axis=0)
                    },
                    fetch_list=[img.grad_name, loss.name])
            else:
                img_grad, loss_value = exe.run(
                    main_program,
                    feed={
                        'i': np.full((1), feed_i, np.int32),
                        'image': feed_img,
                        'label': feed_label
                    },
                    fetch_list=[img.grad_name, loss.name])

            numerical_grad = np.zeros(shape=[num_devices, 9], dtype=np.float32)
            feed_img_delta = np.copy(feed_img)
            for j in range(9):
                feed_img_delta[0][j] = feed_img[0][j] + delta
                if use_parallel_exe:
                    loss_delta = exe.run(feed={
                        'i': np.full((num_devices), feed_i, np.int32),
                        'image': np.repeat(
                            feed_img_delta, num_devices, axis=0),
                        'label': np.repeat(
                            feed_label, num_devices, axis=0)
                    },
                                         fetch_list=[loss.name])
                    multi_device_grad = (
                        loss_delta[0] - loss_value[0]) / delta / num_devices
                    for d in range(num_devices):
                        numerical_grad[d][j] = multi_device_grad[d]
                else:
                    loss_delta = exe.run(main_program,
                                         feed={
                                             'i': np.full((1), feed_i,
                                                          np.int32),
                                             'image': feed_img_delta,
                                             'label': feed_label
                                         },
                                         fetch_list=[loss.name])
                    numerical_grad[0][j] = (
                        loss_delta[0] - loss_value[0]) / delta
                feed_img_delta[0][j] = feed_img[0][j]
            self.assertTrue(
                np.isclose(
                    img_grad, numerical_grad, atol=0.05, rtol=0.05).all())

    def add_optimizer_helper(self, cond_func, use_cuda, use_parallel_exe):
        """
        Test that program is runnable when add optimizer
        """
        main_program = Program()
        startup_program = Program()
        with program_guard(main_program, startup_program):
            img = fluid.data(name='image', shape=[-1, 784], dtype='float32')
            label = fluid.data(name='label', shape=[-1, 1], dtype='int64')
            i = fluid.data(name="i", shape=[1], dtype='int32')
            loss = cond_func(i, img, label)
            optimizer = fluid.optimizer.SGD(learning_rate=0.1)
            optimizer.minimize(loss)

        place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
        exe = fluid.Executor(place)
        exe.run(startup_program)
        if use_parallel_exe:
            os.environ['CPU_NUM'] = str(2)
            exe = fluid.ParallelExecutor(
                use_cuda=use_cuda,
                main_program=main_program,
                loss_name=loss.name)
            num_devices = exe.device_count

        for feed_i in range(0, 10):
            feed_img = np.random.random(size=[16, 784]).astype(np.float32)
            feed_label = np.random.randint(
                low=0, high=10, size=[16, 1], dtype=np.int64)
            if use_parallel_exe:
                exe.run(feed={
                    'i': np.full((num_devices), feed_i, np.int32),
                    'image': np.repeat(
                        feed_img, num_devices, axis=0),
                    'label': np.repeat(
                        feed_label, num_devices, axis=0)
                },
                        fetch_list=[loss.name])
            else:
                exe.run(main_program,
                        feed={
                            'i': np.full((1), feed_i, np.int32),
                            'image': feed_img,
                            'label': feed_label
                        },
                        fetch_list=[loss])

    def test_cond_backward(self):
        def cond_func(i, img, label):
            predicate = ((i % 2) == 0)
            return layers.cond(predicate,
                               lambda: simple_fc_net_with_inputs(img, label, class_num=10),
                               lambda: batchnorm_fc_with_inputs(img, label, class_num=10))

        for use_parallel_exe in [False, True]:
            if use_parallel_exe and os.name == "nt":
                print(
                    "Skip use_parallel_exe=True in Windows because of flaky test when using PE under old Windows machine"
                )
                continue

            self.backward_value_helper(cond_func,
                                       core.is_compiled_with_cuda(),
                                       use_parallel_exe)
            self.add_optimizer_helper(cond_func,
                                      core.is_compiled_with_cuda(),
                                      use_parallel_exe)

    def test_half_nested_cond_backward(self):
        def branch(i, img, label):
            return layers.cond((i % 2) == 0, lambda: simple_fc_net_with_inputs(img, label, class_num=10),
                               lambda: batchnorm_fc_with_inputs(img, label, class_num=10))

        def cond_func_simple_net_at_true(i, img, label):
            return layers.cond(i < 5, lambda: branch(i, img, label),
                               lambda: layers.mean(img))

        def cond_func_simple_net_at_false(i, img, label):
            return layers.cond(i < 5, lambda: layers.mean(img),
                               lambda: branch(i, img, label))

        for use_parallel_exe in [False, True]:
            if use_parallel_exe and os.name == "nt":
                print(
                    "Skip use_parallel_exe=True in Windows because of flaky test when using PE under old Windows machine"
                )
                continue

            self.backward_value_helper(cond_func_simple_net_at_true,
                                       core.is_compiled_with_cuda(),
                                       use_parallel_exe)
            self.add_optimizer_helper(cond_func_simple_net_at_true,
                                      core.is_compiled_with_cuda(),
                                      use_parallel_exe)
            self.backward_value_helper(cond_func_simple_net_at_false,
                                       core.is_compiled_with_cuda(),
                                       use_parallel_exe)
            self.add_optimizer_helper(cond_func_simple_net_at_false,
                                      core.is_compiled_with_cuda(),
                                      use_parallel_exe)

    def test_nested_cond_backward(self):
        def branch(i, img, label, mod_two):
            if mod_two:
                predicate = ((i % 2) == 0)
            else:
                predicate = ((i % 2) != 0)
            return layers.cond(predicate, lambda: simple_fc_net_with_inputs(img, label, class_num=10),
                               lambda: batchnorm_fc_with_inputs(img, label, class_num=10))

        def cond_func(i, img, label):
            return layers.cond(i < 5, lambda: branch(i, img, label, True),
                               lambda: branch(i, img, label, False))

        for use_parallel_exe in [False, True]:
            if use_parallel_exe and os.name == "nt":
                print(
                    "Skip use_parallel_exe=True in Windows because of flaky test when using PE under old Windows machine"
                )
                continue
            self.backward_value_helper(cond_func,
                                       core.is_compiled_with_cuda(),
                                       use_parallel_exe)
            self.add_optimizer_helper(cond_func,
                                      core.is_compiled_with_cuda(),
                                      use_parallel_exe)


class TestCondWithError(unittest.TestCase):
    def test_input_type_error(self):
        main_program = framework.Program()
        startup_program = framework.Program()
        with framework.program_guard(main_program, startup_program):
            pred = fluid.data(name='y', shape=[1], dtype='bool')

            def func():
                return pred

            with self.assertRaises(TypeError):
                layers.cond(None, func, func)

            with self.assertRaises(TypeError):
                layers.cond(pred, func, set())

            with self.assertRaises(TypeError):
                layers.cond(pred, set(), func)

            with self.assertRaises(TypeError):
                layers.cond(pred, func, func, set())


if __name__ == '__main__':
    unittest.main()
