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


def bcast(x, target_tensor):
    x_dims = x.shape
    y_dims = target_tensor.shape
    bcast_dims = []
    for i in range(len(x_dims)):
        bcast_dims.append(int(y_dims[i] / x_dims[i]))
    bcast_dims = np.array(bcast_dims).astype("int64")
    return bcast_dims


class TestExpandAsOpRank1(OpTest):
    def setUp(self):
        self.op_type = "expand_as"
        x = np.random.rand(100).astype(DTYPE)
        target_tensor = np.random.rand(200).astype(DTYPE)
        self.inputs = {'X': x, 'target_tensor': target_tensor}
        self.attrs = {}
        bcast_dims = bcast(x, target_tensor)
        output = np.tile(self.inputs['X'], bcast_dims)
        self.outputs = {'Out': output}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Out')


class TestExpandAsOpRank2(OpTest):
    def setUp(self):
        self.op_type = "expand_as"
        x = np.random.rand(10, 12).astype(DTYPE)
        target_tensor = np.random.rand(20, 24).astype(DTYPE)
        self.inputs = {'X': x, 'target_tensor': target_tensor}
        self.attrs = {}
        bcast_dims = bcast(x, target_tensor)
        output = np.tile(self.inputs['X'], bcast_dims)
        self.outputs = {'Out': output}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Out')


class TestExpandAsOpRank3(OpTest):
    def setUp(self):
        self.op_type = "expand_as"
        x = np.random.rand(2, 3, 20).astype(DTYPE)
        target_tensor = np.random.rand(4, 6, 40).astype(DTYPE)
        self.inputs = {'X': x, 'target_tensor': target_tensor}
        self.attrs = {}
        bcast_dims = bcast(x, target_tensor)
        output = np.tile(self.inputs['X'], bcast_dims)
        self.outputs = {'Out': output}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Out')


class TestExpandAsOpRank4(OpTest):
    def setUp(self):
        self.op_type = "expand_as"
        x = np.random.rand(1, 1, 7, 16).astype(DTYPE)
        target_tensor = np.random.rand(4, 6, 14, 32).astype(DTYPE)
        self.inputs = {'X': x, 'target_tensor': target_tensor}
        self.attrs = {}
        bcast_dims = bcast(x, target_tensor)
        output = np.tile(self.inputs['X'], bcast_dims)
        self.outputs = {'Out': output}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Out')


# Test dygraph API
class TestExpandAsDygraphAPI(unittest.TestCase):
    def test_api(self):
        import paddle
        paddle.disable_static()
        np_data_x = np.array([1, 2, 3]).astype('int32')
        np_data_y = np.array([1, 2, 3, 1, 2, 3]).astype('int32')
        data_x = paddle.to_tensor(np_data_x)
        data_y = paddle.to_tensor(np_data_y)
        out = fluid.layers.expand_as(data_x, data_y)
        np_out = out.numpy()
        assert np.array_equal(np_out, np.tile(np_data_x, (2)))
        paddle.enable_static()


# Test python API
class TestExpandAsAPI(unittest.TestCase):
    def test_api(self):
        input1 = np.random.random([12, 14]).astype("float32")
        input2 = np.random.random([48, 14]).astype("float32")
        x = fluid.layers.data(
            name='x', shape=[12, 14], append_batch_size=False, dtype="float32")

        y = fluid.layers.data(
            name='target_tensor',
            shape=[48, 14],
            append_batch_size=False,
            dtype="float32")

        out_1 = fluid.layers.expand_as(x, target_tensor=y)

        exe = fluid.Executor(place=fluid.CPUPlace())
        res_1 = exe.run(fluid.default_main_program(),
                        feed={"x": input1,
                              "target_tensor": input2},
                        fetch_list=[out_1])
        assert np.array_equal(res_1[0], np.tile(input1, (4, 1)))


if __name__ == "__main__":
    unittest.main()
