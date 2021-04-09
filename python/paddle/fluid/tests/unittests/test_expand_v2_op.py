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
import unittest

import numpy as np

import paddle
import paddle.fluid as fluid
from paddle.fluid import compiler, Program, program_guard
import paddle.fluid.core as core
from op_test import OpTest

paddle.enable_static()

DTYPE = "float32" if core.is_compiled_with_rocm() else "float64"


# Situation 1: shape is a list(without tensor)
class TestExpandV2OpRank1(OpTest):
    def setUp(self):
        self.op_type = "expand_v2"
        self.init_data()

        self.inputs = {'X': np.random.random(self.ori_shape).astype(DTYPE)}
        self.attrs = {'shape': self.shape}
        output = np.tile(self.inputs['X'], self.expand_times)
        self.outputs = {'Out': output}

    def init_data(self):
        self.ori_shape = [100]
        self.shape = [100]
        self.expand_times = [1]

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Out')


class TestExpandV2OpRank2_DimExpanding(TestExpandV2OpRank1):
    def init_data(self):
        self.ori_shape = [120]
        self.shape = [2, 120]
        self.expand_times = [2, 1]


class TestExpandV2OpRank2(TestExpandV2OpRank1):
    def init_data(self):
        self.ori_shape = [1, 140]
        self.shape = [12, 140]
        self.expand_times = [12, 1]


class TestExpandV2OpRank3_Corner(TestExpandV2OpRank1):
    def init_data(self):
        self.ori_shape = (2, 10, 5)
        self.shape = (2, 10, 5)
        self.expand_times = (1, 1, 1)


class TestExpandV2OpRank4(TestExpandV2OpRank1):
    def init_data(self):
        self.ori_shape = (2, 4, 5, 7)
        self.shape = (-1, -1, -1, -1)
        self.expand_times = (1, 1, 1, 1)


# Situation 2: shape is a list(with tensor)
class TestExpandV2OpRank1_tensor_attr(OpTest):
    def setUp(self):
        self.op_type = "expand_v2"
        self.init_data()
        expand_shapes_tensor = []
        for index, ele in enumerate(self.expand_shape):
            expand_shapes_tensor.append(("x" + str(index), np.ones(
                (1)).astype('int32') * ele))

        self.inputs = {
            'X': np.random.random(self.ori_shape).astype(DTYPE),
            'expand_shapes_tensor': expand_shapes_tensor,
        }
        self.attrs = {"shape": self.infer_expand_shape}
        output = np.tile(self.inputs['X'], self.expand_times)
        self.outputs = {'Out': output}

    def init_data(self):
        self.ori_shape = [100]
        self.expand_times = [1]
        self.expand_shape = [100]
        self.infer_expand_shape = [-1]

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Out')


class TestExpandV2OpRank2_Corner_tensor_attr(TestExpandV2OpRank1_tensor_attr):
    def init_data(self):
        self.ori_shape = [12, 14]
        self.expand_times = [1, 1]
        self.expand_shape = [12, 14]
        self.infer_expand_shape = [12, -1]


# Situation 3: shape is a tensor
class TestExpandV2OpRank1_tensor(OpTest):
    def setUp(self):
        self.op_type = "expand_v2"
        self.init_data()

        self.inputs = {
            'X': np.random.random(self.ori_shape).astype(DTYPE),
            'Shape': np.array(self.expand_shape).astype("int32"),
        }
        self.attrs = {}
        output = np.tile(self.inputs['X'], self.expand_times)
        self.outputs = {'Out': output}

    def init_data(self):
        self.ori_shape = [100]
        self.expand_times = [2, 1]
        self.expand_shape = [2, 100]

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Out')


# Situation 4: input x is Integer
class TestExpandV2OpInteger(OpTest):
    def setUp(self):
        self.op_type = "expand_v2"
        self.inputs = {
            'X': np.random.randint(
                10, size=(2, 4, 5)).astype("int32")
        }
        self.attrs = {'shape': [2, 4, 5]}
        output = np.tile(self.inputs['X'], (1, 1, 1))
        self.outputs = {'Out': output}

    def test_check_output(self):
        self.check_output()


# Situation 5: input x is Bool
class TestExpandV2OpBoolean(OpTest):
    def setUp(self):
        self.op_type = "expand_v2"
        self.inputs = {'X': np.random.randint(2, size=(2, 4, 5)).astype("bool")}
        self.attrs = {'shape': [2, 4, 5]}
        output = np.tile(self.inputs['X'], (1, 1, 1))
        self.outputs = {'Out': output}

    def test_check_output(self):
        self.check_output()


# Situation 56: input x is Integer
class TestExpandV2OpInt64_t(OpTest):
    def setUp(self):
        self.op_type = "expand_v2"
        self.inputs = {
            'X': np.random.randint(
                10, size=(2, 4, 5)).astype("int64")
        }
        self.attrs = {'shape': [2, 4, 5]}
        output = np.tile(self.inputs['X'], (1, 1, 1))
        self.outputs = {'Out': output}

    def test_check_output(self):
        self.check_output()


class TestExpandV2Error(unittest.TestCase):
    def test_errors(self):
        with program_guard(Program(), Program()):
            x1 = fluid.create_lod_tensor(
                np.array([[-1]]), [[1]], fluid.CPUPlace())
            shape = [2, 2]
            self.assertRaises(TypeError, paddle.tensor.expand, x1, shape)
            x2 = fluid.layers.data(name='x2', shape=[4], dtype="uint8")
            self.assertRaises(TypeError, paddle.tensor.expand, x2, shape)
            x3 = fluid.layers.data(name='x3', shape=[4], dtype="bool")
            x3.stop_gradient = False
            self.assertRaises(ValueError, paddle.tensor.expand, x3, shape)


# Test python API
class TestExpandV2API(unittest.TestCase):
    def test_api(self):
        input = np.random.random([12, 14]).astype("float32")
        x = fluid.layers.data(
            name='x', shape=[12, 14], append_batch_size=False, dtype="float32")

        positive_2 = fluid.layers.fill_constant([1], "int32", 12)
        expand_shape = fluid.layers.data(
            name="expand_shape",
            shape=[2],
            append_batch_size=False,
            dtype="int32")

        out_1 = paddle.expand(x, shape=[12, 14])
        out_2 = paddle.expand(x, shape=[positive_2, 14])
        out_3 = paddle.expand(x, shape=expand_shape)

        g0 = fluid.backward.calc_gradient(out_2, x)

        exe = fluid.Executor(place=fluid.CPUPlace())
        res_1, res_2, res_3 = exe.run(fluid.default_main_program(),
                                      feed={
                                          "x": input,
                                          "expand_shape":
                                          np.array([12, 14]).astype("int32")
                                      },
                                      fetch_list=[out_1, out_2, out_3])
        assert np.array_equal(res_1, np.tile(input, (1, 1)))
        assert np.array_equal(res_2, np.tile(input, (1, 1)))
        assert np.array_equal(res_3, np.tile(input, (1, 1)))


if __name__ == "__main__":
    unittest.main()
