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
import paddle.fluid as fluid
import paddle


class TestGatherNdOpWithEmptyIndex(OpTest):
    #Index has empty element, which means copy entire tensor

    def setUp(self):
        self.op_type = "gather_nd"
        xnp = np.random.random((5, 20)).astype("float64")
        self.inputs = {'X': xnp, 'Index': np.array([[], []]).astype("int32")}
        self.outputs = {
            'Out': np.vstack((xnp[np.newaxis, :], xnp[np.newaxis, :]))
        }

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Out')


class TestGatherNdOpWithIndex1(OpTest):
    def setUp(self):
        self.op_type = "gather_nd"
        xnp = np.random.random((5, 20)).astype("float64")
        self.inputs = {'X': xnp, 'Index': np.array([1]).astype("int32")}
        self.outputs = {'Out': self.inputs["X"][self.inputs["Index"]]}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Out')


class TestGatherNdOpWithLowIndex(OpTest):
    #Index has low rank, X has high rank

    def setUp(self):
        self.op_type = "gather_nd"
        xnp = np.random.uniform(0, 100, (10, 10)).astype("float64")
        index = np.array([[1], [2]]).astype("int64")

        self.inputs = {'X': xnp, 'Index': index}

        self.outputs = {'Out': xnp[tuple(index.T)]}  #[[14, 25, 1], [76, 22, 3]]

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Out')


class TestGatherNdOpIndex1(OpTest):
    #Index has low rank, X has high rank

    def setUp(self):
        self.op_type = "gather_nd"
        xnp = np.random.uniform(0, 100, (10, 10)).astype("float64")
        index = np.array([1, 2]).astype("int64")

        self.inputs = {'X': xnp, 'Index': index}

        self.outputs = {'Out': xnp[tuple(index.T)]}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Out')


class TestGatherNdOpWithSameIndexAsX(OpTest):
    #Index has same rank as X's rank

    def setUp(self):
        self.op_type = "gather_nd"
        xnp = np.random.uniform(0, 100, (10, 10)).astype("float64")
        index = np.array([[1, 1], [2, 1]]).astype("int64")

        self.inputs = {'X': xnp, 'Index': index}
        self.outputs = {'Out': xnp[tuple(index.T)]}  #[25, 22]

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Out')


class TestGatherNdOpWithHighRankSame(OpTest):
    #Both Index and X have high rank, and Rank(Index) = Rank(X)

    def setUp(self):
        self.op_type = "gather_nd"
        shape = (20, 9, 8, 1, 31)
        xnp = np.random.rand(*shape).astype("float64")
        index = np.vstack([np.random.randint(0, s, size=150) for s in shape]).T

        self.inputs = {'X': xnp, 'Index': index.astype("int32")}
        self.outputs = {'Out': xnp[tuple(index.T)]}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Out')


class TestGatherNdOpWithHighRankDiff(OpTest):
    #Both Index and X have high rank, and Rank(Index) < Rank(X)

    def setUp(self):
        self.op_type = "gather_nd"
        shape = (20, 9, 8, 1, 31)
        xnp = np.random.rand(*shape).astype("float64")
        index = np.vstack([np.random.randint(0, s, size=1000) for s in shape]).T
        index_re = index.reshape([10, 5, 20, 5])

        self.inputs = {'X': xnp, 'Index': index_re.astype("int32")}
        self.outputs = {'Out': xnp[tuple(index.T)].reshape([10, 5, 20])}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Out')


#Test Python API
class TestGatherNdOpAPI(unittest.TestCase):
    def test_case1(self):
        x1 = fluid.layers.data(
            name='x1', shape=[30, 40, 50, 60], dtype='float32')
        index1 = fluid.layers.data(name='index1', shape=[2, 4], dtype='int32')
        output1 = fluid.layers.gather_nd(x1, index1)

    def test_case2(self):
        x2 = fluid.layers.data(name='x2', shape=[30, 40, 50], dtype='float32')
        index2 = fluid.layers.data(name='index2', shape=[2, 2], dtype='int64')
        output2 = fluid.layers.gather_nd(x2, index2)

    def test_case3(self):
        x3 = fluid.layers.data(name='x3', shape=[3, 4, 5], dtype='float32')
        index3 = fluid.layers.data(name='index3', shape=[2, 1], dtype='int32')
        output3 = fluid.layers.gather_nd(x3, index3, name="gather_nd_layer")


#Test Raise Index Error
class TestGatherNdOpRaise(unittest.TestCase):
    def test_check_raise(self):
        def check_raise_is_test():
            try:
                x = fluid.layers.data(
                    name='x', shape=[3, 4, 5], dtype='float32')
                index = fluid.layers.data(
                    name='index', shape=[2, 10], dtype='int32')
                output = fluid.layers.gather_nd(x, index)
            except Exception as e:
                t = \
                "Input(Index).shape[-1] should be no greater than Input(X).rank"
                if t in str(e):
                    raise IndexError

        self.assertRaises(IndexError, check_raise_is_test)


class TestGatherNdError(unittest.TestCase):
    def test_error(self):
        with paddle.static.program_guard(paddle.static.Program(),
                                         paddle.static.Program()):

            shape = [8, 9, 6]
            x = paddle.fluid.data(shape=shape, dtype='float32', name='x')
            index = paddle.fluid.data(shape=shape, dtype='bool', name='index')
            index_float = paddle.fluid.data(
                shape=shape, dtype='float32', name='index_float')
            np_x = np.random.random(shape).astype('float32')
            np_index = np.array(np.random.randint(2, size=shape, dtype=bool))

            def test_x_type():
                paddle.gather_nd(np_x, index)

            self.assertRaises(TypeError, test_x_type)

            def test_index_type():
                paddle.gather_nd(x, np_index)

            self.assertRaises(TypeError, test_index_type)

            def test_index_dtype():
                paddle.gather_nd(x, index_float)

            self.assertRaises(TypeError, test_index_dtype)


class TestGatherNdAPI2(unittest.TestCase):
    def test_static(self):
        with fluid.program_guard(fluid.Program(), fluid.Program()):
            data1 = fluid.layers.data('data1', shape=[-1, 2], dtype='float64')
            index = fluid.layers.data('index', shape=[-1, 1], dtype='int32')
            out = paddle.gather_nd(data1, index)
            place = fluid.CPUPlace()
            exe = fluid.Executor(place)
            input = np.array([[1, 2], [3, 4], [5, 6]])
            index_1 = np.array([[1]])
            result, = exe.run(feed={"data1": input,
                                    "index": index_1},
                              fetch_list=[out])
            expected_output = np.array([[3, 4]])
        self.assertTrue(np.allclose(result, expected_output))

    def test_imperative(self):
        paddle.disable_static()
        input_1 = np.array([[1, 2], [3, 4], [5, 6]])
        index_1 = np.array([[1]])
        input = fluid.dygraph.to_variable(input_1)
        index = fluid.dygraph.to_variable(index_1)
        output = paddle.fluid.layers.gather(input, index)
        output_np = output.numpy()
        expected_output = np.array([3, 4])
        self.assertTrue(np.allclose(output_np, expected_output))
        paddle.enable_static()


if __name__ == "__main__":
    unittest.main()
