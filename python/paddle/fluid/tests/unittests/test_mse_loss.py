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
import numpy as np
import paddle
import paddle.fluid.core as core
import paddle.fluid as fluid
import paddle.fluid.layers as layers
from paddle.fluid.executor import Executor


class TestMseLoss(unittest.TestCase):
    def test_mse_loss(self):
        input_val = np.random.uniform(0.1, 0.5, (2, 3)).astype("float32")
        label_val = np.random.uniform(0.1, 0.5, (2, 3)).astype("float32")

        sub = input_val - label_val
        np_result = np.mean(sub * sub)

        input_var = layers.create_tensor(dtype="float32", name="input")
        label_var = layers.create_tensor(dtype="float32", name="label")

        output = layers.mse_loss(input=input_var, label=label_var)
        for use_cuda in ([False, True]
                         if core.is_compiled_with_cuda() else [False]):
            place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
            exe = Executor(place)
            result = exe.run(fluid.default_main_program(),
                             feed={"input": input_val,
                                   "label": label_val},
                             fetch_list=[output])

            self.assertTrue(np.isclose(np_result, result).all())


class TestMseInvalidInput(unittest.TestCase):
    def test_error(self):
        def test_invalid_input():
            input = [256, 3]
            label = fluid.data(name='label', shape=[None, 3], dtype='float32')
            loss = fluid.layers.mse_loss(input, label)

        self.assertRaises(TypeError, test_invalid_input)

        def test_invalid_label():
            input = fluid.data(name='input1', shape=[None, 3], dtype='float32')
            label = [256, 3]
            loss = fluid.layers.mse_loss(input, label)

        self.assertRaises(TypeError, test_invalid_label)


class TestNNMseLoss(unittest.TestCase):
    def test_NNMseLoss_mean(self):
        for dim in [[10, 10], [2, 10, 10], [3, 3, 10, 10]]:
            input_np = np.random.uniform(0.1, 0.5, dim).astype("float32")
            label_np = np.random.uniform(0.1, 0.5, dim).astype("float32")
            paddle.enable_static()
            prog = fluid.Program()
            startup_prog = fluid.Program()
            place = fluid.CUDAPlace(0) if fluid.core.is_compiled_with_cuda(
            ) else fluid.CPUPlace()
            with fluid.program_guard(prog, startup_prog):
                input = fluid.layers.data(
                    name='input', shape=dim, dtype='float32')
                label = fluid.layers.data(
                    name='label', shape=dim, dtype='float32')
                mse_loss = paddle.nn.loss.MSELoss()
                ret = mse_loss(input, label)

                exe = fluid.Executor(place)
                static_result = exe.run(
                    prog,
                    feed={"input": input_np,
                          "label": label_np},
                    fetch_list=[ret])

            with fluid.dygraph.guard():
                mse_loss = paddle.nn.loss.MSELoss()
                dy_ret = mse_loss(
                    fluid.dygraph.to_variable(input_np),
                    fluid.dygraph.to_variable(label_np))
                dy_result = dy_ret.numpy()

            sub = input_np - label_np
            expected = np.mean(sub * sub)
            self.assertTrue(np.allclose(static_result, expected))
            self.assertTrue(np.allclose(static_result, dy_result))
            self.assertTrue(np.allclose(dy_result, expected))
            self.assertTrue(dy_result.shape, [1])

    def test_NNMseLoss_sum(self):
        for dim in [[10, 10], [2, 10, 10], [3, 3, 10, 10]]:
            input_np = np.random.uniform(0.1, 0.5, dim).astype("float32")
            label_np = np.random.uniform(0.1, 0.5, dim).astype("float32")
            paddle.enable_static()
            prog = fluid.Program()
            startup_prog = fluid.Program()
            place = fluid.CUDAPlace(0) if fluid.core.is_compiled_with_cuda(
            ) else fluid.CPUPlace()
            with fluid.program_guard(prog, startup_prog):
                input = fluid.layers.data(
                    name='input', shape=dim, dtype='float32')
                label = fluid.layers.data(
                    name='label', shape=dim, dtype='float32')
                mse_loss = paddle.nn.loss.MSELoss(reduction='sum')
                ret = mse_loss(input, label)

                exe = fluid.Executor(place)
                static_result = exe.run(
                    prog,
                    feed={"input": input_np,
                          "label": label_np},
                    fetch_list=[ret])

            with fluid.dygraph.guard():
                mse_loss = paddle.nn.loss.MSELoss(reduction='sum')
                dy_ret = mse_loss(
                    fluid.dygraph.to_variable(input_np),
                    fluid.dygraph.to_variable(label_np))
                dy_result = dy_ret.numpy()

            sub = input_np - label_np
            expected = np.sum(sub * sub)
            self.assertTrue(np.allclose(static_result, expected))
            self.assertTrue(np.allclose(static_result, dy_result))
            self.assertTrue(np.allclose(dy_result, expected))
            self.assertTrue(dy_result.shape, [1])

    def test_NNMseLoss_none(self):
        for dim in [[10, 10], [2, 10, 10], [3, 3, 10, 10]]:
            input_np = np.random.uniform(0.1, 0.5, dim).astype("float32")
            label_np = np.random.uniform(0.1, 0.5, dim).astype("float32")
            paddle.enable_static()
            prog = fluid.Program()
            startup_prog = fluid.Program()
            place = fluid.CUDAPlace(0) if fluid.core.is_compiled_with_cuda(
            ) else fluid.CPUPlace()
            with fluid.program_guard(prog, startup_prog):
                input = fluid.layers.data(
                    name='input', shape=dim, dtype='float32')
                label = fluid.layers.data(
                    name='label', shape=dim, dtype='float32')
                mse_loss = paddle.nn.loss.MSELoss(reduction='none')
                ret = mse_loss(input, label)

                exe = fluid.Executor(place)
                static_result = exe.run(
                    prog,
                    feed={"input": input_np,
                          "label": label_np},
                    fetch_list=[ret])

            with fluid.dygraph.guard():
                mse_loss = paddle.nn.loss.MSELoss(reduction='none')
                dy_ret = mse_loss(
                    fluid.dygraph.to_variable(input_np),
                    fluid.dygraph.to_variable(label_np))
                dy_result = dy_ret.numpy()

            sub = input_np - label_np
            expected = (sub * sub)
            self.assertTrue(np.allclose(static_result, expected))
            self.assertTrue(np.allclose(static_result, dy_result))
            self.assertTrue(np.allclose(dy_result, expected))
            self.assertTrue(dy_result.shape, [1])


class TestNNFunctionalMseLoss(unittest.TestCase):
    def test_NNFunctionalMseLoss_mean(self):
        for dim in [[10, 10], [2, 10, 10], [3, 3, 10, 10]]:
            input_np = np.random.uniform(0.1, 0.5, dim).astype("float32")
            target_np = np.random.uniform(0.1, 0.5, dim).astype("float32")
            paddle.enable_static()
            prog = paddle.static.Program()
            startup_prog = paddle.static.Program()
            place = paddle.CUDAPlace(0) if core.is_compiled_with_cuda(
            ) else paddle.CPUPlace()
            with paddle.static.program_guard(prog, startup_prog):
                input = paddle.fluid.data(name='input', shape=dim, dtype='float32')
                target = paddle.fluid.data(name='target', shape=dim, dtype='float32')
                mse_loss = paddle.nn.functional.mse_loss(input, target, 'mean')

            exe = paddle.static.Executor(place)
            exe.run(startup_prog)
            static_result = exe.run(
                prog,
                feed={"input": input_np,
                      "target": target_np},
                fetch_list=[mse_loss])

            paddle.disable_static()
            dy_ret = paddle.nn.functional.mse_loss(
                paddle.to_tensor(input_np), paddle.to_tensor(target_np), 'mean')
            dy_result = dy_ret.numpy()

            sub = input_np - target_np
            expected = np.mean(sub * sub)
            self.assertTrue(np.allclose(static_result, expected))
            self.assertTrue(np.allclose(static_result, dy_result))
            self.assertTrue(np.allclose(dy_result, expected))
            self.assertTrue(dy_result.shape, [1])

    def test_NNFunctionalMseLoss_sum(self):
        for dim in [[10, 10], [2, 10, 10], [3, 3, 10, 10]]:
            input_np = np.random.uniform(0.1, 0.5, dim).astype("float32")
            target_np = np.random.uniform(0.1, 0.5, dim).astype("float32")
            paddle.enable_static()
            prog = paddle.static.Program()
            startup_prog = paddle.static.Program()
            place = paddle.CUDAPlace(0) if core.is_compiled_with_cuda(
            ) else paddle.CPUPlace()
            with paddle.static.program_guard(prog, startup_prog):
                input = paddle.fluid.data(name='input', shape=dim, dtype='float32')
                target = paddle.fluid.data(name='target', shape=dim, dtype='float32')
                mse_loss = paddle.nn.functional.mse_loss(input, target, 'sum')

                exe = paddle.static.Executor(place)
                exe.run(startup_prog)
                static_result = exe.run(
                    prog,
                    feed={"input": input_np,
                          "target": target_np},
                    fetch_list=[mse_loss])

            paddle.disable_static()
            dy_ret = paddle.nn.functional.mse_loss(
                paddle.to_tensor(input_np), paddle.to_tensor(target_np), 'sum')
            dy_result = dy_ret.numpy()

            sub = input_np - target_np
            expected = np.sum(sub * sub)
            self.assertTrue(np.allclose(static_result, expected))
            self.assertTrue(np.allclose(static_result, dy_result))
            self.assertTrue(np.allclose(dy_result, expected))
            self.assertTrue(dy_result.shape, [1])

    def test_NNFunctionalMseLoss_none(self):
        for dim in [[10, 10], [2, 10, 10], [3, 3, 10, 10]]:
            input_np = np.random.uniform(0.1, 0.5, dim).astype("float32")
            target_np = np.random.uniform(0.1, 0.5, dim).astype("float32")
            paddle.enable_static()
            prog = paddle.static.Program()
            startup_prog = paddle.static.Program()
            place = paddle.CUDAPlace(0) if core.is_compiled_with_cuda(
            ) else paddle.CPUPlace()
            with paddle.static.program_guard(prog, startup_prog):
                input = paddle.fluid.data(name='input', shape=dim, dtype='float32')
                target = paddle.fluid.data(name='target', shape=dim, dtype='float32')
                mse_loss = paddle.nn.functional.mse_loss(input, target, 'none')

                exe = paddle.static.Executor(place)
                exe.run(startup_prog)
                static_result = exe.run(
                    prog,
                    feed={"input": input_np,
                          "target": target_np},
                    fetch_list=[mse_loss])

            paddle.disable_static()
            dy_ret = paddle.nn.functional.mse_loss(
                paddle.to_tensor(input_np), paddle.to_tensor(target_np), 'none')
            dy_result = dy_ret.numpy()

            sub = input_np - target_np
            expected = sub * sub
            self.assertTrue(np.allclose(static_result, expected))
            self.assertTrue(np.allclose(static_result, dy_result))
            self.assertTrue(np.allclose(dy_result, expected))
            self.assertTrue(dy_result.shape, [1])


if __name__ == "__main__":
    unittest.main()
