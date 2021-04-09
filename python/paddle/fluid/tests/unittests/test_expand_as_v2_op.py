#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

from op_test import OpTest
import paddle
import paddle.fluid as fluid
import paddle.fluid.core as core

paddle.enable_static()

DTYPE = "float32" if core.is_compiled_with_rocm() else "float64"


class TestExpandAsOpRank1(OpTest):
    def setUp(self):
        self.op_type = "expand_as_v2"
        x = np.random.rand(100).astype(DTYPE)
        target_tensor = np.random.rand(2, 100).astype(DTYPE)
        self.inputs = {'X': x}
        self.attrs = {'target_shape': target_tensor.shape}
        bcast_dims = [2, 1]
        output = np.tile(self.inputs['X'], bcast_dims)
        self.outputs = {'Out': output}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Out')


class TestExpandAsOpRank2(OpTest):
    def setUp(self):
        self.op_type = "expand_as_v2"
        x = np.random.rand(10, 12).astype(DTYPE)
        target_tensor = np.random.rand(10, 12).astype(DTYPE)
        self.inputs = {'X': x}
        self.attrs = {'target_shape': target_tensor.shape}
        bcast_dims = [1, 1]
        output = np.tile(self.inputs['X'], bcast_dims)
        self.outputs = {'Out': output}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Out')


class TestExpandAsOpRank3(OpTest):
    def setUp(self):
        self.op_type = "expand_as_v2"
        x = np.random.rand(2, 3, 20).astype(DTYPE)
        target_tensor = np.random.rand(2, 3, 20).astype(DTYPE)
        self.inputs = {'X': x}
        self.attrs = {'target_shape': target_tensor.shape}
        bcast_dims = [1, 1, 1]
        output = np.tile(self.inputs['X'], bcast_dims)
        self.outputs = {'Out': output}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Out')


class TestExpandAsOpRank4(OpTest):
    def setUp(self):
        self.op_type = "expand_as_v2"
        x = np.random.rand(1, 1, 7, 16).astype(DTYPE)
        target_tensor = np.random.rand(4, 6, 7, 16).astype(DTYPE)
        self.inputs = {'X': x}
        self.attrs = {'target_shape': target_tensor.shape}
        bcast_dims = [4, 6, 1, 1]
        output = np.tile(self.inputs['X'], bcast_dims)
        self.outputs = {'Out': output}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Out')


class TestExpandAsV2Error(unittest.TestCase):
    def test_errors(self):
        with fluid.program_guard(fluid.Program(), fluid.Program()):
            x1 = fluid.layers.data(name='x1', shape=[4], dtype="uint8")
            x2 = fluid.layers.data(name='x2', shape=[4], dtype="int32")
            self.assertRaises(TypeError, paddle.tensor.expand_as, x1, x2)
            x3 = fluid.layers.data(name='x3', shape=[4], dtype="bool")
            x3.stop_gradient = False
            self.assertRaises(ValueError, paddle.tensor.expand_as, x3, x2)


# Test python API
class TestExpandAsV2API(unittest.TestCase):
    def test_api(self):
        input1 = np.random.random([12, 14]).astype("float32")
        input2 = np.random.random([2, 12, 14]).astype("float32")
        x = fluid.layers.data(
            name='x', shape=[12, 14], append_batch_size=False, dtype="float32")

        y = fluid.layers.data(
            name='target_tensor',
            shape=[2, 12, 14],
            append_batch_size=False,
            dtype="float32")

        out_1 = paddle.expand_as(x, y=y)

        exe = fluid.Executor(place=fluid.CPUPlace())
        res_1 = exe.run(fluid.default_main_program(),
                        feed={"x": input1,
                              "target_tensor": input2},
                        fetch_list=[out_1])
        assert np.array_equal(res_1[0], np.tile(input1, (2, 1, 1)))


if __name__ == "__main__":
    unittest.main()
