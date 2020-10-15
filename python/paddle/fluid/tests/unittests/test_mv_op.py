#Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
import paddle.fluid.layers as layers
import paddle.fluid.core as core
from paddle.static import program_guard, Program
from op_test import OpTest


class TestMVOp(OpTest):
    def setUp(self):
        self.op_type = "mv"
        self.init_config()
        self.inputs = {'X': self.x, 'Vec': self.vec}
        self.outputs = {'Out': np.dot(self.x, self.vec)}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X', 'Vec'], 'Out')

    def init_config(self):
        self.x = np.random.random((2, 100)).astype("float64")
        self.vec = np.random.random((100)).astype("float64")


class TestMVAPI(unittest.TestCase):
    def test_dygraph_api_out(self):
        paddle.disable_static()

        self.x_data = np.random.random((5, 100)).astype("float64")
        self.x = paddle.to_tensor(self.x_data)
        self.vec_data = np.random.random((100)).astype("float64")
        self.vec = paddle.to_tensor(self.vec_data)
        z = paddle.mv(self.x, self.vec)
        np_z = z.numpy()
        z_expected = np.array(np.dot(self.x_data, self.vec_data))
        self.assertTrue(np.allclose(np_z, z_expected))

        paddle.enable_static()

    def test_static_graph(self):
        for x_stop_gradient in [False, True]:
            for vec_stop_gradient in [False, True]:

                paddle.enable_static()

                train_program = Program()
                startup_program = Program()

                self.input_x = np.random.rand(5, 100).astype("float64")
                self.input_vec = np.random.rand(100).astype("float64")

                with program_guard(train_program, startup_program):
                    data_x = paddle.static.data(
                        "x", shape=[5, 100], dtype="float64")
                    data_vec = paddle.static.data(
                        "vec", shape=[100], dtype="float64")

                    data_x.stop_gradient = x_stop_gradient
                    data_vec.stop_gradient = vec_stop_gradient

                    result_vec = paddle.mv(data_x, data_vec)

                    self.place = paddle.CPUPlace()
                    exe = paddle.static.Executor(self.place)
                    res, = exe.run(
                        feed={"x": self.input_x,
                              "vec": self.input_vec},
                        fetch_list=[result_vec])
                    z_expected = np.array(np.dot(self.input_x, self.input_vec))
                    self.assertTrue(np.allclose(res, z_expected))


class TestMVError(unittest.TestCase):
    def test_input(self):
        def test_shape():
            paddle.enable_static()

            self.input_x = np.random.rand(5, 100).astype("float64")
            self.input_vec = np.random.rand(100).astype("float64")

            data_x = paddle.static.data("x", shape=[5, 100], dtype="float64")
            data_vec = paddle.static.data(
                "vec", shape=[100, 2], dtype="float64")
            result_vec = paddle.mv(data_x, data_vec)

        self.assertRaises(ValueError, test_shape)


if __name__ == '__main__':
    unittest.main()
