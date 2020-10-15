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

import unittest
import numpy as np
from op_test import OpTest
import paddle
import paddle.fluid as fluid
from paddle.fluid import Program, program_guard


class TestNonZeroAPI(unittest.TestCase):
    def test_nonzero_api_as_tuple(self):
        data = np.array([[True, False], [False, True]])
        with program_guard(Program(), Program()):
            x = fluid.layers.data(name='x', shape=[-1, 2])
            y = paddle.nonzero(x, as_tuple=True)
            self.assertEqual(type(y), tuple)
            self.assertEqual(len(y), 2)
            z = fluid.layers.concat(list(y), axis=1)
            exe = fluid.Executor(fluid.CPUPlace())

            res, = exe.run(feed={'x': data},
                           fetch_list=[z.name],
                           return_numpy=False)
        expect_out = np.array([[0, 0], [1, 1]])
        self.assertTrue(np.allclose(expect_out, np.array(res)))

        data = np.array([True, True, False])
        with program_guard(Program(), Program()):
            x = fluid.layers.data(name='x', shape=[-1])
            y = paddle.nonzero(x, as_tuple=True)
            self.assertEqual(type(y), tuple)
            self.assertEqual(len(y), 1)
            z = fluid.layers.concat(list(y), axis=1)
            exe = fluid.Executor(fluid.CPUPlace())
            res, = exe.run(feed={'x': data},
                           fetch_list=[z.name],
                           return_numpy=False)
        expect_out = np.array([[0], [1]])
        self.assertTrue(np.allclose(expect_out, np.array(res)))

    def test_nonzero_api(self):
        data = np.array([[True, False], [False, True]])
        with program_guard(Program(), Program()):
            x = fluid.layers.data(name='x', shape=[-1, 2])
            y = paddle.nonzero(x)
            exe = fluid.Executor(fluid.CPUPlace())
            res, = exe.run(feed={'x': data},
                           fetch_list=[y.name],
                           return_numpy=False)
        expect_out = np.array([[0, 0], [1, 1]])
        self.assertTrue(np.allclose(expect_out, np.array(res)))

        data = np.array([True, True, False])
        with program_guard(Program(), Program()):
            x = fluid.layers.data(name='x', shape=[-1])
            y = paddle.nonzero(x)
            exe = fluid.Executor(fluid.CPUPlace())
            res, = exe.run(feed={'x': data},
                           fetch_list=[y.name],
                           return_numpy=False)
        expect_out = np.array([[0], [1]])
        self.assertTrue(np.allclose(expect_out, np.array(res)))

    def test_dygraph_api(self):
        data_x = np.array([[True, False], [False, True]])
        with fluid.dygraph.guard():
            x = fluid.dygraph.to_variable(data_x)
            z = paddle.nonzero(x)
            np_z = z.numpy()
        expect_out = np.array([[0, 0], [1, 1]])


if __name__ == "__main__":
    unittest.main()
