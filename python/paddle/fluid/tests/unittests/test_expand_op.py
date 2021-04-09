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
from paddle.fluid import compiler, Program, program_guard, core
from op_test import OpTest

paddle.enable_static()

DTYPE = "float32" if core.is_compiled_with_rocm() else "float64"


# Situation 1: expand_times is a list(without tensor)
class TestExpandOpRank1(OpTest):
    def setUp(self):
        self.op_type = "expand"
        self.init_data()

        self.inputs = {'X': np.random.random(self.ori_shape).astype(DTYPE)}
        self.attrs = {'expand_times': self.expand_times}
        output = np.tile(self.inputs['X'], self.expand_times)
        self.outputs = {'Out': output}

    def init_data(self):
        self.ori_shape = [100]
        self.expand_times = [2]

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Out')


class TestExpandOpRank2_Corner(TestExpandOpRank1):
    def init_data(self):
        self.ori_shape = [120]
        self.expand_times = [2]


class TestExpandOpRank2(TestExpandOpRank1):
    def init_data(self):
        self.ori_shape = [12, 14]
        self.expand_times = [2, 3]


class TestExpandOpRank3_Corner(TestExpandOpRank1):
    def init_data(self):
        self.ori_shape = (2, 10, 5)
        self.expand_times = (1, 1, 1)


class TestExpandOpRank3(TestExpandOpRank1):
    def init_data(self):
        self.ori_shape = (2, 4, 15)
        self.expand_times = (2, 1, 4)


class TestExpandOpRank4(TestExpandOpRank1):
    def init_data(self):
        self.ori_shape = (2, 4, 5, 7)
        self.expand_times = (3, 2, 1, 2)


# Situation 2: expand_times is a list(with tensor)
class TestExpandOpRank1_tensor_attr(OpTest):
    def setUp(self):
        self.op_type = "expand"
        self.init_data()
        expand_times_tensor = []
        for index, ele in enumerate(self.expand_times):
            expand_times_tensor.append(("x" + str(index), np.ones(
                (1)).astype('int32') * ele))

        self.inputs = {
            'X': np.random.random(self.ori_shape).astype(DTYPE),
            'expand_times_tensor': expand_times_tensor,
        }
        self.attrs = {"expand_times": self.infer_expand_times}
        output = np.tile(self.inputs['X'], self.expand_times)
        self.outputs = {'Out': output}

    def init_data(self):
        self.ori_shape = [100]
        self.expand_times = [2]
        self.infer_expand_times = [-1]

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Out')


class TestExpandOpRank2_Corner_tensor_attr(TestExpandOpRank1_tensor_attr):
    def init_data(self):
        self.ori_shape = [12, 14]
        self.expand_times = [1, 1]
        self.infer_expand_times = [1, -1]


class TestExpandOpRank2_attr_tensor(TestExpandOpRank1_tensor_attr):
    def init_data(self):
        self.ori_shape = [12, 14]
        self.expand_times = [2, 3]
        self.infer_expand_times = [-1, 3]


# Situation 3: expand_times is a tensor
class TestExpandOpRank1_tensor(OpTest):
    def setUp(self):
        self.op_type = "expand"
        self.init_data()

        self.inputs = {
            'X': np.random.random(self.ori_shape).astype(DTYPE),
            'ExpandTimes': np.array(self.expand_times).astype("int32"),
        }
        self.attrs = {}
        output = np.tile(self.inputs['X'], self.expand_times)
        self.outputs = {'Out': output}

    def init_data(self):
        self.ori_shape = [100]
        self.expand_times = [2]

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Out')


class TestExpandOpRank2_tensor(TestExpandOpRank1_tensor):
    def init_data(self):
        self.ori_shape = [12, 14]
        self.expand_times = [2, 3]


# Situation 4: input x is Integer
class TestExpandOpInteger(OpTest):
    def setUp(self):
        self.op_type = "expand"
        self.inputs = {
            'X': np.random.randint(
                10, size=(2, 4, 5)).astype("int32")
        }
        self.attrs = {'expand_times': [2, 1, 4]}
        output = np.tile(self.inputs['X'], (2, 1, 4))
        self.outputs = {'Out': output}

    def test_check_output(self):
        self.check_output()


# Situation 5: input x is Bool
class TestExpandOpBoolean(OpTest):
    def setUp(self):
        self.op_type = "expand"
        self.inputs = {'X': np.random.randint(2, size=(2, 4, 5)).astype("bool")}
        self.attrs = {'expand_times': [2, 1, 4]}
        output = np.tile(self.inputs['X'], (2, 1, 4))
        self.outputs = {'Out': output}

    def test_check_output(self):
        self.check_output()


# Situation 56: input x is Integer
class TestExpandOpInt64_t(OpTest):
    def setUp(self):
        self.op_type = "expand"
        self.inputs = {
            'X': np.random.randint(
                10, size=(2, 4, 5)).astype("int64")
        }
        self.attrs = {'expand_times': [2, 1, 4]}
        output = np.tile(self.inputs['X'], (2, 1, 4))
        self.outputs = {'Out': output}

    def test_check_output(self):
        self.check_output()


class TestExpandError(unittest.TestCase):
    def test_errors(self):
        with program_guard(Program(), Program()):
            x1 = fluid.create_lod_tensor(
                np.array([[-1]]), [[1]], fluid.CPUPlace())
            expand_times = [2, 2]
            self.assertRaises(TypeError, fluid.layers.expand, x1, expand_times)
            x2 = fluid.layers.data(name='x2', shape=[4], dtype="uint8")
            self.assertRaises(TypeError, fluid.layers.expand, x2, expand_times)
            x3 = fluid.layers.data(name='x3', shape=[4], dtype="bool")
            x3.stop_gradient = True
            self.assertRaises(ValueError, fluid.layers.expand, x3, expand_times)


# Test python API
class TestExpandAPI(unittest.TestCase):
    def test_api(self):
        input = np.random.random([12, 14]).astype("float32")
        x = fluid.layers.data(
            name='x', shape=[12, 14], append_batch_size=False, dtype="float32")

        positive_2 = fluid.layers.fill_constant([1], "int32", 2)
        expand_times = fluid.layers.data(
            name="expand_times", shape=[2], append_batch_size=False)

        out_1 = fluid.layers.expand(x, expand_times=[2, 3])
        out_2 = fluid.layers.expand(x, expand_times=[positive_2, 3])
        out_3 = fluid.layers.expand(x, expand_times=expand_times)

        g0 = fluid.backward.calc_gradient(out_2, x)

        exe = fluid.Executor(place=fluid.CPUPlace())
        res_1, res_2, res_3 = exe.run(fluid.default_main_program(),
                                      feed={
                                          "x": input,
                                          "expand_times":
                                          np.array([1, 3]).astype("int32")
                                      },
                                      fetch_list=[out_1, out_2, out_3])
        assert np.array_equal(res_1, np.tile(input, (2, 3)))
        assert np.array_equal(res_2, np.tile(input, (2, 3)))
        assert np.array_equal(res_3, np.tile(input, (1, 3)))


class TestExpandDygraphAPI(unittest.TestCase):
    def test_expand_times_is_tensor(self):
        with paddle.fluid.dygraph.guard():
            a = paddle.rand([2, 5])
            b = paddle.fluid.layers.expand(a, expand_times=[2, 3])
            c = paddle.fluid.layers.expand(
                a, expand_times=paddle.to_tensor(
                    [2, 3], dtype='int32'))
            self.assertTrue(
                np.array_equal(b.numpy(), np.tile(a.numpy(), [2, 3])))
            self.assertTrue(
                np.array_equal(c.numpy(), np.tile(a.numpy(), [2, 3])))


if __name__ == "__main__":
    unittest.main()
