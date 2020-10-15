#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
import paddle
import paddle.fluid as fluid
import numpy as np
from op_test import OpTest


class TestIndexSampleOp(OpTest):
    def setUp(self):
        self.op_type = "index_sample"
        self.config()
        xnp = np.random.random(self.x_shape).astype(self.x_type)
        indexnp = np.random.randint(
            low=0, high=self.x_shape[1],
            size=self.index_shape).astype(self.index_type)
        self.inputs = {'X': xnp, 'Index': indexnp}
        index_array = []
        for i in range(self.index_shape[0]):
            for j in indexnp[i]:
                index_array.append(xnp[i, j])
        index_array = np.array(index_array).astype(self.x_type)
        out = np.reshape(index_array, self.index_shape)
        self.outputs = {'Out': out}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Out')

    def config(self):
        """
        For multi-dimension input
        """
        self.x_shape = (10, 20)
        self.x_type = "float64"
        self.index_shape = (10, 10)
        self.index_type = "int32"


class TestCase1(TestIndexSampleOp):
    def config(self):
        """
        For one dimension input
        """
        self.x_shape = (100, 1)
        self.x_type = "float64"
        self.index_shape = (100, 1)
        self.index_type = "int32"


class TestCase2(TestIndexSampleOp):
    def config(self):
        """
        For int64_t index type
        """
        self.x_shape = (10, 100)
        self.x_type = "float64"
        self.index_shape = (10, 10)
        self.index_type = "int64"


class TestCase3(TestIndexSampleOp):
    def config(self):
        """
        For int index type
        """
        self.x_shape = (10, 100)
        self.x_type = "float64"
        self.index_shape = (10, 10)
        self.index_type = "int32"


class TestCase4(TestIndexSampleOp):
    def config(self):
        """
        For int64 index type
        """
        self.x_shape = (10, 100)
        self.x_type = "float64"
        self.index_shape = (10, 10)
        self.index_type = "int64"


class TestIndexSampleShape(unittest.TestCase):
    def test_shape(self):
        paddle.enable_static()
        # create x value
        x_shape = (2, 5)
        x_type = "float64"
        x_np = np.random.random(x_shape).astype(x_type)

        # create index value
        index_shape = (2, 3)
        index_type = "int32"
        index_np = np.random.randint(
            low=0, high=x_shape[1], size=index_shape).astype(index_type)

        x = fluid.data(name='x', shape=[-1, 5], dtype='float64')
        index = fluid.data(name='index', shape=[-1, 3], dtype='int32')
        output = paddle.index_sample(x=x, index=index)

        place = fluid.CPUPlace()
        exe = fluid.Executor(place=place)
        exe.run(fluid.default_startup_program())

        feed = {'x': x_np, 'index': index_np}
        res = exe.run(feed=feed, fetch_list=[output])


class TestIndexSampleDynamic(unittest.TestCase):
    def test_result(self):
        with fluid.dygraph.guard():
            x = paddle.to_tensor(
                [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0],
                 [9.0, 10.0, 11.0, 12.0]],
                dtype='float32')
            index = paddle.to_tensor(
                [[0, 1, 2], [1, 2, 3], [0, 0, 0]], dtype='int32')
            out_z1 = paddle.index_sample(x, index)

            except_output = np.array(
                [[1.0, 2.0, 3.0], [6.0, 7.0, 8.0], [9.0, 9.0, 9.0]])
            assert out_z1.numpy().all() == except_output.all()


if __name__ == "__main__":
    paddle.enable_static()
    unittest.main()
