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

import paddle.fluid.core as core
import unittest
import numpy as np
from op_test import OpTest
import paddle
import paddle.fluid as fluid
from paddle.fluid import Program, program_guard


def generate_compatible_shapes4(dim_X, dim_Y, transpose_X, transpose_Y):
    BATCH_SIZE = 2
    M = 3
    N = 4
    K = 5
    if (dim_X == 1 and transpose_X) or (dim_Y == 1 and transpose_Y):
        K = 1
    if dim_X == 1:
        if transpose_X:
            shape_X = [M]
        else:
            shape_X = [K]
    if dim_Y == 1:
        if transpose_Y:
            shape_Y = [N]
        else:
            shape_Y = [K]
    if dim_X >= 2:
        if transpose_X:
            shape_X = [K, M]
        else:
            shape_X = [M, K]
    if dim_X == 3:
        shape_X = [BATCH_SIZE] + shape_X
    if dim_Y >= 2:
        if transpose_Y:
            shape_Y = [N, K]
        else:
            shape_Y = [K, N]
    if dim_Y == 3:
        shape_Y = [BATCH_SIZE] + shape_Y
    return shape_X, shape_Y


def reference_matmul(X, Y, transpose_X=False, transpose_Y=False):
    """Reference forward implementation using np.matmul."""
    # np.matmul does not support the transpose flags, so we manually
    # transpose X and Y appropriately.
    if transpose_X:
        if X.ndim == 1:
            X = X.reshape((X.size, 1))
        elif X.ndim == 2:
            X = X.T
        else:
            dim = [i for i in range(len(X.shape))]
            dim[-1], dim[len(X.shape) - 2] = dim[len(X.shape) - 2], dim[-1]
            X = np.transpose(X, tuple(dim))
    if transpose_Y:
        if Y.ndim == 1:
            Y = Y.reshape((1, Y.size))
        else:
            dim = [i for i in range(len(Y.shape))]
            dim[-1], dim[len(Y.shape) - 2] = dim[len(Y.shape) - 2], dim[-1]
            Y = np.transpose(Y, tuple(dim))

    Out = np.matmul(X, Y)
    if not Out.shape:
        # We do not support 0-dimensional Tensors (scalars). So where
        # np.matmul outputs a scalar, we must convert to a Tensor of
        # shape (1, ) instead.
        # Everywhere else, we are compatible with np.matmul.
        Out = np.array([Out], dtype="float32")
    return Out


class Generator(object):
    def setUp(self):
        self.op_type = "matmul"
        X = np.random.random(self.shape_X).astype("float32")
        Y = np.random.random(self.shape_Y).astype("float32")
        Out = reference_matmul(X, Y, self.transpose_X, self.transpose_Y)
        self.inputs = {'X': X, 'Y': Y}
        self.attrs = {
            'transpose_X': self.transpose_X,
            'transpose_Y': self.transpose_Y
        }
        self.outputs = {'Out': Out}

    def test_check_output(self):
        self.check_output()

    def test_check_grad_normal(self):
        self.check_grad(['X', 'Y'], 'Out', max_relative_error=1e-3)

    def test_check_grad_ignore_x(self):
        self.check_grad(
            ['Y'], 'Out', max_relative_error=1e-3, no_grad_set=set("X"))

    def test_check_grad_ignore_y(self):
        self.check_grad(
            ['X'], 'Out', max_relative_error=1e-3, no_grad_set=set('Y'))


class TestMatmulOpError(unittest.TestCase):
    def test_errors(self):
        with program_guard(Program(), Program()):
            # The inputs type of matmul_op must be Variable.
            input1 = 12
            self.assertRaises(TypeError, fluid.layers.matmul, input1, input1)
            # The inputs dtype of matmul_op must be float32, float64.
            input2 = fluid.layers.data(
                name='input2', shape=[10, 10], dtype="int32")
            self.assertRaises(TypeError, fluid.layers.matmul, input2, input2)
            input3 = fluid.layers.data(
                name='input3', shape=[2, 2], dtype="float16")
            fluid.layers.matmul(input3, input3)


# Negative dimension generation
def generate_negative_dims(in_shape):
    from itertools import combinations
    size = len(in_shape)
    indexs = list()
    shapes = list()
    for i in range(size):
        indexs.extend(list(combinations([j for j in range(size)], i + 1)))
    for idx in indexs:
        shapes.append(
            [in_shape[i] if i not in idx else -1 for i in range(size)])
    return shapes


# Build program with inputs sizes that contain negative numbers
def test_negative_dims_program(obj):
    for shape_x in generate_negative_dims(obj.shape_X):
        for shape_y in generate_negative_dims(obj.shape_Y):
            X = np.random.random(obj.shape_X).astype("float32")
            Y = np.random.random(obj.shape_Y).astype("float32")
            Ref = reference_matmul(X, Y, obj.transpose_X, obj.transpose_Y)
            with program_guard(Program(), Program()):
                x = fluid.data(name='x', shape=shape_x, dtype='float32')
                y = fluid.data(name='y', shape=shape_y, dtype='float32')
                output = fluid.layers.matmul(x, y, obj.transpose_X,
                                             obj.transpose_Y)
                obj.assertEqual(len(Ref.shape), len(output.shape))
                for idx in range(len(Ref.shape)):
                    if output.shape[idx] != -1:
                        obj.assertEqual(Ref.shape[idx], output.shape[idx])
                exe = fluid.Executor(fluid.CPUPlace())
                res, = exe.run(fluid.default_main_program(),
                               feed={'x': X,
                                     'y': Y},
                               fetch_list=[output])
                np.allclose(res, Ref, atol=1e-5)


# Generate program api cases for all negative possibilities
def api_test(dim_x, dim_y, trans_x, trans_y):
    test_name = ('TestMatMulAPI_dimX_{}_dim_Y_{}_transX_{}_transY_{}'.format(
        dim_x, dim_y, trans_x, trans_y))
    shape_x, shape_y = generate_compatible_shapes4(dim_x, dim_y, trans_x,
                                                   trans_y)
    globals()[test_name] = type(test_name, (unittest.TestCase, ), {
        'shape_X': shape_x,
        'shape_Y': shape_y,
        'transpose_X': trans_x,
        'transpose_Y': trans_y,
        'test_propram': test_negative_dims_program,
    })


# Generate operators cases for all possibilities
def inject_test(dim_x, dim_y, trans_x, trans_y):
    test_name = ('TestMatMulOp_dimX_{}_dim_Y_{}_transX_{}_transY_{}'.format(
        dim_x, dim_y, trans_x, trans_y))
    shape_x, shape_y = generate_compatible_shapes4(dim_x, dim_y, trans_x,
                                                   trans_y)
    globals()[test_name] = type(test_name, (Generator, OpTest), {
        'shape_X': shape_x,
        'shape_Y': shape_y,
        'transpose_X': trans_x,
        'transpose_Y': trans_y,
    })


for dim_X in (1, 2, 3):
    for dim_Y in (1, 2, 3):
        for transose_x in (False, True):
            for transose_y in (False, True):
                inject_test(dim_X, dim_Y, transose_x, transose_y)
                api_test(dim_X, dim_Y, transose_x, transose_y)


# Test case more batch_size and N, M, K
def generate_compatible_shapes5(dim_X, dim_Y, transpose_X, transpose_Y,
                                batch_size):
    BATCH_SIZE = 2
    M = 3
    N = 4
    K = 5
    if (dim_X == 1 and transpose_X) or (dim_Y == 1 and transpose_Y):
        K = 1
    if dim_X == 1:
        if transpose_X:
            shape_X = [M]
        else:
            shape_X = [K]
    if dim_Y == 1:
        if transpose_Y:
            shape_Y = [N]
        else:
            shape_Y = [K]
    if dim_X >= 2:
        if transpose_X:
            shape_X = [K, M]
        else:
            shape_X = [M, K]
    if dim_X == 3:
        shape_X = [BATCH_SIZE] + shape_X
    if dim_Y >= 2:
        if transpose_Y:
            shape_Y = [N, K]
        else:
            shape_Y = [K, N]
    if dim_Y == 3:
        shape_Y = [BATCH_SIZE] + shape_Y
    return shape_X, shape_Y


# Test case n-dim
def generate_compatible_shapes3(dim, transpose_X, transpose_Y):
    M = 2
    N = 4
    K = 3
    shape_X = [2 for _ in range(dim - 2)]
    shape_Y = [2 for _ in range(dim - 2)]

    if transpose_X:
        shape_X += [K, M]
    else:
        shape_X += [M, K]

    if transpose_Y:
        shape_Y += [N, K]
    else:
        shape_Y += [K, N]

    return shape_X, shape_Y


# # Test case n-dim
for dim in [4]:
    for transpose_X in [False, True]:
        for transpose_Y in [False, True]:
            test_name = (
                'TestMatMulOp_dimX_{}_dim_Y_{}_transX_{}_transY_{}'.format(
                    dim, dim, transpose_X, transpose_Y))
            shape_X, shape_Y = generate_compatible_shapes3(dim, transpose_X,
                                                           transpose_Y)
            globals()[test_name] = type(test_name, (Generator, OpTest), {
                'shape_X': shape_X,
                'shape_Y': shape_Y,
                'transpose_X': transpose_X,
                'transpose_Y': transpose_Y,
            })


class API_TestMm(unittest.TestCase):
    def test_out(self):
        with fluid.program_guard(fluid.Program()):
            x = fluid.data(name="x", shape=[2], dtype="float64")
            y = fluid.data(name='y', shape=[2], dtype='float64')
            res = fluid.data(name="output", shape=[1], dtype="float64")
            result = paddle.mm(x, y)
            exe = fluid.Executor(fluid.CPUPlace())
            data1 = np.random.rand(2)
            data2 = np.random.rand(2)
            np_res = exe.run(feed={'x': data1, 'y': data2}, fetch_list=[result])
            expected_result = np.matmul(
                data1.reshape(1, 2), data2.reshape(2, 1))

        self.assertTrue(
            np.allclose(
                np_res, expected_result, atol=1e-5),
            "two value is\
            {}\n{}, check diff!".format(np_res, expected_result))

    def test_dygraph_without_out(self):
        device = fluid.CPUPlace()
        with fluid.dygraph.guard(device):
            input_array1 = np.random.rand(3, 4).astype("float64")
            input_array2 = np.random.rand(4, 3).astype("float64")
            data1 = fluid.dygraph.to_variable(input_array1)
            data2 = fluid.dygraph.to_variable(input_array2)
            out = paddle.mm(data1, data2)
            expected_result = np.matmul(input_array1, input_array2)
        self.assertTrue(np.allclose(expected_result, out.numpy()))


class Test_API_Matmul(unittest.TestCase):
    def test_dygraph_without_out(self):
        device = fluid.CPUPlace()
        with fluid.dygraph.guard(device):
            input_array1 = np.random.rand(3, 4).astype("float64")
            input_array2 = np.random.rand(4, 3).astype("float64")
            data1 = fluid.dygraph.to_variable(input_array1)
            data2 = fluid.dygraph.to_variable(input_array2)
            out = paddle.matmul(data1, data2)
            expected_result = np.matmul(input_array1, input_array2)
        self.assertTrue(np.allclose(expected_result, out.numpy()))


class API_TestMmError(unittest.TestCase):
    def test_errors(self):
        def test_error1():
            with fluid.program_guard(fluid.Program(), fluid.Program()):
                data1 = fluid.data(name="data1", shape=[10, 2], dtype="float32")
                data2 = fluid.data(name="data2", shape=[3, 10], dtype="float32")
                paddle.mm(data1, data2)

        self.assertRaises(ValueError, test_error1)

        def test_error2():
            with fluid.program_guard(fluid.Program(), fluid.Program()):
                data1 = fluid.data(
                    name="data1", shape=[-1, 10, 2], dtype="float32")
                data2 = fluid.data(
                    name="data2", shape=[-1, 2, 10], dtype="float32")
                paddle.mm(data1, data2)

        test_error2()

        def test_error3():
            with fluid.program_guard(fluid.Program(), fluid.Program()):
                data1 = fluid.data(
                    name="data1", shape=[10, 10, 2], dtype="float32")
                data2 = fluid.data(
                    name="data2", shape=[3, 2, 10], dtype="float32")
                paddle.mm(data1, data2)

        self.assertRaises(ValueError, test_error3)


if __name__ == "__main__":
    unittest.main()
