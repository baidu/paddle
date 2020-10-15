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

from __future__ import print_function

import paddle
import paddle.fluid as fluid
import numpy as np
import unittest


class TestFunctionalL1Loss(unittest.TestCase):
    def setUp(self):
        self.input_np = np.random.random(size=(10, 10, 5)).astype(np.float32)
        self.label_np = np.random.random(size=(10, 10, 5)).astype(np.float32)

    def run_imperative(self):
        input = paddle.to_tensor(self.input_np)
        label = paddle.to_tensor(self.label_np)
        dy_result = paddle.nn.functional.l1_loss(input, label)
        expected = np.mean(np.abs(self.input_np - self.label_np))
        self.assertTrue(np.allclose(dy_result.numpy(), expected))
        self.assertTrue(dy_result.shape, [1])

        dy_result = paddle.nn.functional.l1_loss(input, label, reduction='sum')
        expected = np.sum(np.abs(self.input_np - self.label_np))
        self.assertTrue(np.allclose(dy_result.numpy(), expected))
        self.assertTrue(dy_result.shape, [1])

        dy_result = paddle.nn.functional.l1_loss(input, label, reduction='none')
        expected = np.abs(self.input_np - self.label_np)
        self.assertTrue(np.allclose(dy_result.numpy(), expected))
        self.assertTrue(dy_result.shape, [10, 10, 5])

    def run_static(self, use_gpu=False):
        input = paddle.fluid.data(name='input', shape=[10, 10, 5], dtype='float32')
        label = paddle.fluid.data(name='label', shape=[10, 10, 5], dtype='float32')
        result0 = paddle.nn.functional.l1_loss(input, label)
        result1 = paddle.nn.functional.l1_loss(input, label, reduction='sum')
        result2 = paddle.nn.functional.l1_loss(input, label, reduction='none')
        y = paddle.nn.functional.l1_loss(input, label, name='aaa')

        place = fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace()
        exe = fluid.Executor(place)
        exe.run(fluid.default_startup_program())
        static_result = exe.run(
            feed={"input": self.input_np,
                  "label": self.label_np},
            fetch_list=[result0, result1, result2])

        expected = np.mean(np.abs(self.input_np - self.label_np))
        self.assertTrue(np.allclose(static_result[0], expected))
        expected = np.sum(np.abs(self.input_np - self.label_np))
        self.assertTrue(np.allclose(static_result[1], expected))
        expected = np.abs(self.input_np - self.label_np)
        self.assertTrue(np.allclose(static_result[2], expected))

        self.assertTrue('aaa' in y.name)

    def test_cpu(self):
        paddle.disable_static(place=paddle.fluid.CPUPlace())
        self.run_imperative()
        paddle.enable_static()

        with fluid.program_guard(fluid.Program()):
            self.run_static()

    def test_gpu(self):
        if not fluid.core.is_compiled_with_cuda():
            return

        paddle.disable_static(place=paddle.fluid.CUDAPlace(0))
        self.run_imperative()
        paddle.enable_static()

        with fluid.program_guard(fluid.Program()):
            self.run_static(use_gpu=True)

    # test case the raise message
    def test_errors(self):
        def test_value_error():
            input = paddle.fluid.data(
                name='input', shape=[10, 10, 5], dtype='float32')
            label = paddle.fluid.data(
                name='label', shape=[10, 10, 5], dtype='float32')
            loss = paddle.nn.functional.l1_loss(
                input, label, reduction='reduce_mean')

        self.assertRaises(ValueError, test_value_error)


class TestClassL1Loss(unittest.TestCase):
    def setUp(self):
        self.input_np = np.random.random(size=(10, 10, 5)).astype(np.float32)
        self.label_np = np.random.random(size=(10, 10, 5)).astype(np.float32)

    def run_imperative(self):
        input = paddle.to_tensor(self.input_np)
        label = paddle.to_tensor(self.label_np)
        l1_loss = paddle.nn.loss.L1Loss()
        dy_result = l1_loss(input, label)
        expected = np.mean(np.abs(self.input_np - self.label_np))
        self.assertTrue(np.allclose(dy_result.numpy(), expected))
        self.assertTrue(dy_result.shape, [1])

        l1_loss = paddle.nn.loss.L1Loss(reduction='sum')
        dy_result = l1_loss(input, label)
        expected = np.sum(np.abs(self.input_np - self.label_np))
        self.assertTrue(np.allclose(dy_result.numpy(), expected))
        self.assertTrue(dy_result.shape, [1])

        l1_loss = paddle.nn.loss.L1Loss(reduction='none')
        dy_result = l1_loss(input, label)
        expected = np.abs(self.input_np - self.label_np)
        self.assertTrue(np.allclose(dy_result.numpy(), expected))
        self.assertTrue(dy_result.shape, [10, 10, 5])

    def run_static(self, use_gpu=False):
        input = paddle.fluid.data(name='input', shape=[10, 10, 5], dtype='float32')
        label = paddle.fluid.data(name='label', shape=[10, 10, 5], dtype='float32')
        l1_loss = paddle.nn.loss.L1Loss()
        result0 = l1_loss(input, label)
        l1_loss = paddle.nn.loss.L1Loss(reduction='sum')
        result1 = l1_loss(input, label)
        l1_loss = paddle.nn.loss.L1Loss(reduction='none')
        result2 = l1_loss(input, label)
        l1_loss = paddle.nn.loss.L1Loss(name='aaa')
        result3 = l1_loss(input, label)

        place = fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace()
        exe = fluid.Executor(place)
        exe.run(fluid.default_startup_program())
        static_result = exe.run(
            feed={"input": self.input_np,
                  "label": self.label_np},
            fetch_list=[result0, result1, result2])

        expected = np.mean(np.abs(self.input_np - self.label_np))
        self.assertTrue(np.allclose(static_result[0], expected))
        expected = np.sum(np.abs(self.input_np - self.label_np))
        self.assertTrue(np.allclose(static_result[1], expected))
        expected = np.abs(self.input_np - self.label_np)
        self.assertTrue(np.allclose(static_result[2], expected))
        self.assertTrue('aaa' in result3.name)

    def test_cpu(self):
        paddle.disable_static(place=paddle.fluid.CPUPlace())
        self.run_imperative()
        paddle.enable_static()

        with fluid.program_guard(fluid.Program()):
            self.run_static()

    def test_gpu(self):
        if not fluid.core.is_compiled_with_cuda():
            return

        paddle.disable_static(place=paddle.fluid.CUDAPlace(0))
        self.run_imperative()
        paddle.enable_static()

        with fluid.program_guard(fluid.Program()):
            self.run_static(use_gpu=True)

    # test case the raise message
    def test_errors(self):
        def test_value_error():
            loss = paddle.nn.loss.L1Loss(reduction="reduce_mean")

        self.assertRaises(ValueError, test_value_error)


if __name__ == "__main__":
    unittest.main()
