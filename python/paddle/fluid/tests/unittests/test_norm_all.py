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


def p_norm(x, axis, porder, keepdims=False):
    r = []
    if axis is None:
        x = x.flatten()
        if porder == np.inf:
            r = np.amax(np.abs(x), keepdims=keepdims)
        elif porder == -np.inf:
            r = np.amin(np.abs(x), keepdims=keepdims)
        else:
            r = np.linalg.norm(x, ord=porder, keepdims=keepdims)
    elif isinstance(axis, list or tuple) and len(axis) == 2:
        if porder == np.inf:
            axis = tuple(axis)
            r = np.amax(np.abs(x), axis=axis, keepdims=keepdims)
        elif porder == -np.inf:
            axis = tuple(axis)
            r = np.amin(np.abs(x), axis=axis, keepdims=keepdims)
        elif porder == 0:
            axis = tuple(axis)
            r = x.astype(bool)
            r = np.sum(r, axis, keepdims=keepdims)
        elif porder == 1:
            axis = tuple(axis)
            r = np.sum(np.abs(x), axis, keepdims=keepdims)
        else:
            axis = tuple(axis)
            xp = np.power(np.abs(x), porder)
            s = np.sum(xp, axis=axis, keepdims=keepdims)
            r = np.power(s, 1.0 / porder)
    else:
        if isinstance(axis, list):
            axis = tuple(axis)
        r = np.linalg.norm(
            x, ord=porder, axis=axis, keepdims=keepdims).astype(x.dtype)

    return r


def frobenius_norm(x, axis=None, keepdims=False):
    if isinstance(axis, list): axis = tuple(axis)
    if axis is None: x = x.reshape(1, x.size)
    r = np.linalg.norm(
        x, ord='fro', axis=axis, keepdims=keepdims).astype(x.dtype)
    return r


class TestFrobeniusNormOp(OpTest):
    def setUp(self):
        self.op_type = "frobenius_norm"
        self.init_test_case()
        x = (np.random.random(self.shape) + 1.0).astype(self.dtype)
        norm = frobenius_norm(x, self.axis, self.keepdim)
        self.reduce_all = (len(self.axis) == len(self.shape))
        self.inputs = {'X': x}
        self.attrs = {
            'dim': list(self.axis),
            'keep_dim': self.keepdim,
            'reduce_all': self.reduce_all
        }
        self.outputs = {'Out': norm}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Out')

    def init_test_case(self):
        self.shape = [2, 3, 4, 5]
        self.axis = (1, 2)
        self.keepdim = False
        self.dtype = "float64"


class TestFrobeniusNormOp2(TestFrobeniusNormOp):
    def init_test_case(self):
        self.shape = [5, 5, 5]
        self.axis = (0, 1)
        self.keepdim = True
        self.dtype = "float32"

    def test_check_grad(self):
        self.check_grad(['X'], 'Out')


class TestPnormOp(OpTest):
    def setUp(self):
        self.op_type = "p_norm"
        self.init_test_case()
        x = (np.random.random(self.shape) + 0.5).astype(self.dtype)
        norm = p_norm(x, self.axis, self.porder, self.keepdim)
        self.inputs = {'X': x}
        self.attrs = {
            'epsilon': self.epsilon,
            'axis': self.axis,
            'keepdim': self.keepdim,
            'porder': float(self.porder)
        }
        self.outputs = {'Out': norm}
        self.gradient = self.calc_gradient()

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Out')

    def init_test_case(self):
        self.shape = [2, 3, 4, 5]
        self.axis = 1
        self.epsilon = 1e-12
        self.porder = 2.0
        self.keepdim = False
        self.dtype = "float64"

    def calc_gradient(self):
        self.attrs = {
            'epsilon': self.epsilon,
            'axis': self.axis,
            'keepdim': self.keepdim,
            'porder': float(self.porder)
        }
        x = self.inputs["X"]
        porder = self.attrs["porder"]
        axis = self.attrs["axis"]
        if porder == 0:
            grad = np.zeros(x.shape).astype(x.dtype)
        elif porder in [float("inf"), float("-inf")]:
            norm = p_norm(x, axis=axis, porder=porder, keepdims=True)
            x_abs = np.abs(x)
            grad = np.sign(x)
            grad[x_abs != norm] = 0.0
        else:
            norm = p_norm(x, axis=axis, porder=porder, keepdims=True)
            grad = np.power(norm, 1 - porder) * np.power(
                np.abs(x), porder - 1) * np.sign(x)

        numel = 1
        for s in x.shape:
            numel *= s
        numel /= x.shape[axis]
        return [grad.astype(x.dtype) * 1 / numel]


class TestPnormOp2(TestPnormOp):
    def init_test_case(self):
        self.shape = [3, 20, 3]
        self.axis = 2
        self.epsilon = 1e-12
        self.porder = 2.0
        self.keepdim = True
        self.dtype = "float32"

    def test_check_grad(self):
        self.check_grad(['X'], 'Out')


class TestPnormOp3(TestPnormOp):
    def init_test_case(self):
        self.shape = [3, 20, 3]
        self.axis = 2
        self.epsilon = 1e-12
        self.porder = np.inf
        self.keepdim = True
        self.dtype = "float32"

    def test_check_grad(self):
        self.check_grad(['X'], 'Out', user_defined_grads=self.gradient)


class TestPnormOp4(TestPnormOp):
    def init_test_case(self):
        self.shape = [3, 20, 3]
        self.axis = 2
        self.epsilon = 1e-12
        self.porder = -np.inf
        self.keepdim = True
        self.dtype = "float32"

    def test_check_grad(self):
        self.check_grad(['X'], 'Out', user_defined_grads=self.gradient)


class TestPnormOp5(TestPnormOp):
    def init_test_case(self):
        self.shape = [3, 20, 3]
        self.axis = 2
        self.epsilon = 1e-12
        self.porder = 0
        self.keepdim = True
        self.dtype = "float32"

    def test_check_grad(self):
        self.check_grad(['X'], 'Out', user_defined_grads=self.gradient)


def run_fro(self, p, axis, shape_x, dtype, keep_dim, check_dim=False):
    with fluid.program_guard(fluid.Program()):
        data = fluid.data(name="X", shape=shape_x, dtype=dtype)
        out = paddle.norm(x=data, p=p, axis=axis, keepdim=keep_dim)
        place = fluid.CPUPlace()
        exe = fluid.Executor(place)
        np_input = (np.random.rand(*shape_x) + 1.0).astype(dtype)
        expected_result = frobenius_norm(np_input, axis=axis, keepdims=keep_dim)
        result, = exe.run(feed={"X": np_input}, fetch_list=[out])
    self.assertEqual((np.abs(result - expected_result) < 1e-6).all(), True)
    if keep_dim and check_dim:
        self.assertEqual(
            (np.abs(np.array(result.shape) - np.array(expected_result.shape)) <
             1e-6).all(), True)


def run_pnorm(self, p, axis, shape_x, dtype, keep_dim, check_dim=False):
    with fluid.program_guard(fluid.Program()):
        data = fluid.data(name="X", shape=shape_x, dtype=dtype)
        out = paddle.norm(x=data, p=p, axis=axis, keepdim=keep_dim)
        place = fluid.CPUPlace()
        exe = fluid.Executor(place)
        np_input = (np.random.rand(*shape_x) + 1.0).astype(dtype)
        expected_result = p_norm(
            np_input, porder=p, axis=axis, keepdims=keep_dim).astype(dtype)
        result, = exe.run(feed={"X": np_input}, fetch_list=[out])
    self.assertEqual((np.abs(result - expected_result) < 1e-6).all(), True)
    if keep_dim and check_dim:
        self.assertEqual(
            (np.abs(np.array(result.shape) - np.array(expected_result.shape)) <
             1e-6).all(), True)


def run_graph(self, p, axis, shape_x, dtype):
    paddle.disable_static()
    shape = [2, 3, 4]
    np_input = np.arange(24).astype('float32') - 12
    np_input = np_input.reshape(shape)
    x = paddle.to_tensor(np_input)
    #[[[-12. -11. -10.  -9.] [ -8.  -7.  -6.  -5.] [ -4.  -3.  -2.  -1.]]
    # [[  0.   1.   2.   3.] [  4.   5.   6.   7.] [  8.   9.  10.  11.]]]
    out_pnorm = paddle.norm(x, p=2, axis=-1)

    # compute frobenius norm along last two dimensions.
    out_fro = paddle.norm(x, p='fro')
    out_fro = paddle.norm(x, p='fro', axis=0)
    out_fro = paddle.norm(x, p='fro', axis=[0, 1])
    # compute 2-order  norm along [0,1] dimension.
    out_pnorm = paddle.norm(x, p=2, axis=[0, 1])
    out_pnorm = paddle.norm(x, p=2)
    #out_pnorm = [17.43559577 16.91153453 16.73320053 16.91153453]
    # compute inf-order  norm
    out_pnorm = paddle.norm(x, p=np.inf)
    #out_pnorm = [12.]
    out_pnorm = paddle.norm(x, p=np.inf, axis=0)
    #out_pnorm = [[0. 1. 2. 3.] [4. 5. 6. 5.] [4. 3. 2. 1.]]

    # compute -inf-order  norm
    out_pnorm = paddle.norm(x, p=-np.inf)
    #out_pnorm = [0.]
    out_pnorm = paddle.norm(x, p=-np.inf, axis=0)
    # out_fro = [17.43559577 16.91153453 16.73320053 16.91153453]
    paddle.enable_static()


class API_NormTest(unittest.TestCase):
    def test_basic(self):
        keep_dims = {False, True}
        for keep in keep_dims:
            run_fro(
                self,
                p='fro',
                axis=None,
                shape_x=[2, 3, 4],
                dtype="float32",
                keep_dim=keep)
            run_fro(
                self,
                p='fro',
                axis=[0, 1],
                shape_x=[2, 3, 4],
                dtype="float64",
                keep_dim=keep,
                check_dim=True)
            run_pnorm(
                self,
                p=2,
                axis=None,
                shape_x=[3, 4],
                dtype="float32",
                keep_dim=keep)
            run_pnorm(
                self,
                p=2,
                axis=1,
                shape_x=[3, 4],
                dtype="float64",
                keep_dim=keep,
                check_dim=True)
            run_pnorm(
                self,
                p=np.inf,
                axis=0,
                shape_x=[2, 3, 4],
                dtype="float32",
                keep_dim=keep,
                check_dim=True)
            run_pnorm(
                self,
                p=np.inf,
                axis=None,
                shape_x=[2, 3, 4],
                dtype="float32",
                keep_dim=keep)
            run_pnorm(
                self,
                p=-np.inf,
                axis=0,
                shape_x=[2, 3, 4],
                dtype="float64",
                keep_dim=keep,
                check_dim=True)
            run_pnorm(
                self,
                p=-np.inf,
                axis=None,
                shape_x=[2, 3, 4],
                dtype="float64",
                keep_dim=keep)
            run_pnorm(
                self,
                p=0,
                axis=1,
                shape_x=[3, 4],
                dtype="float64",
                keep_dim=keep,
                check_dim=True)

            run_pnorm(
                self,
                p=1,
                axis=1,
                shape_x=[3, 4],
                dtype="float64",
                keep_dim=keep,
                check_dim=True)
            run_pnorm(
                self,
                p=0,
                axis=None,
                shape_x=[3, 4],
                dtype="float64",
                keep_dim=keep,
                check_dim=True)
            run_pnorm(
                self,
                p=2,
                axis=[0, 1],
                shape_x=[2, 3, 4],
                dtype="float64",
                keep_dim=keep,
                check_dim=True)
            run_pnorm(
                self,
                p=2,
                axis=-1,
                shape_x=[2, 3, 4],
                dtype="float64",
                keep_dim=keep,
                check_dim=True)
            run_pnorm(
                self,
                p=1,
                axis=[0, 1],
                shape_x=[2, 3, 4],
                dtype="float64",
                keep_dim=keep,
                check_dim=True)
            run_pnorm(
                self,
                p=np.inf,
                axis=[0, 1],
                shape_x=[2, 3, 4],
                dtype="float64",
                keep_dim=keep,
                check_dim=True)
            run_pnorm(
                self,
                p=-np.inf,
                axis=[0, 1],
                shape_x=[2, 3, 4],
                dtype="float64",
                keep_dim=keep,
                check_dim=True)

    def test_dygraph(self):
        run_graph(self, p='fro', axis=None, shape_x=[2, 3, 4], dtype="float32")

    def test_name(self):
        with fluid.program_guard(fluid.Program()):
            x = fluid.data(name="x", shape=[10, 10], dtype="float32")
            y_1 = paddle.norm(x, p='fro', name='frobenius_name')
            y_2 = paddle.norm(x, p=2, name='pnorm_name')
            self.assertEqual(('frobenius_name' in y_1.name), True)
            self.assertEqual(('pnorm_name' in y_2.name), True)

    def test_errors(self):
        with fluid.program_guard(fluid.Program(), fluid.Program()):

            def err_dtype(p, shape_x, xdtype, out=None):
                data = fluid.data(shape=shape_x, dtype=xdtype)
                paddle.norm(data, p=p, out=out)

            self.assertRaises(TypeError, err_dtype, "fro", [2, 2], "int64")
            self.assertRaises(ValueError, paddle.norm, "inf", [2], "int64")
            out = fluid.data(name="out", shape=[1], dtype="int64")
            self.assertRaises(TypeError, err_dtype, "fro", [2, 2], "float64",
                              out)
            self.assertRaises(TypeError, err_dtype, 2, [10], "int64")
            self.assertRaises(TypeError, err_dtype, 2, [10], "float64", out)

            data = fluid.data(name="data_2d", shape=[2, 2], dtype="float64")
            self.assertRaises(ValueError, paddle.norm, data, p="unsupport norm")
            self.assertRaises(ValueError, paddle.norm, data, p=[1])
            self.assertRaises(ValueError, paddle.norm, data, p=[1], axis=-1)
            self.assertRaises(ValueError, paddle.norm, 0, [1, 0], "float64")
            data = fluid.data(name="data_3d", shape=[2, 2, 2], dtype="float64")
            self.assertRaises(
                ValueError, paddle.norm, data, p='unspport', axis=[-3, -2, -1])


if __name__ == '__main__':
    unittest.main()
