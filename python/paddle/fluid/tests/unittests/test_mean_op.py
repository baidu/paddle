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
from op_test import OpTest
import paddle
import paddle.fluid.core as core
import paddle.fluid as fluid
from paddle.fluid import Program, program_guard

np.random.seed(10)


class TestMeanOp(OpTest):
    def setUp(self):
        self.op_type = "mean"
        self.dtype = np.float64
        self.init_dtype_type()
        self.inputs = {'X': np.random.random((10, 10)).astype(self.dtype)}
        self.outputs = {'Out': np.mean(self.inputs["X"])}

    def init_dtype_type(self):
        pass

    def test_check_output(self):
        self.check_output()

    def test_checkout_grad(self):
        self.check_grad(['X'], 'Out')


class TestMeanOpError(unittest.TestCase):
    def test_errors(self):
        with program_guard(Program(), Program()):
            # The input type of mean_op must be Variable.
            input1 = 12
            self.assertRaises(TypeError, fluid.layers.mean, input1)
            # The input dtype of mean_op must be float16, float32, float64.
            input2 = fluid.layers.data(
                name='input2', shape=[12, 10], dtype="int32")
            self.assertRaises(TypeError, fluid.layers.mean, input2)
            input3 = fluid.layers.data(
                name='input3', shape=[4], dtype="float16")
            fluid.layers.softmax(input3)


@unittest.skipIf(not core.is_compiled_with_cuda(),
                 "core is not compiled with CUDA")
class TestFP16MeanOp(TestMeanOp):
    def init_dtype_type(self):
        self.dtype = np.float16

    def test_check_output(self):
        place = core.CUDAPlace(0)
        if core.is_float16_supported(place):
            self.check_output_with_place(place, atol=2e-3)

    def test_checkout_grad(self):
        place = core.CUDAPlace(0)
        if core.is_float16_supported(place):
            self.check_grad_with_place(
                place, ['X'], 'Out', max_relative_error=0.8)


def ref_reduce_mean(x, axis=None, keepdim=False, reduce_all=False):
    if isinstance(axis, list):
        axis = tuple(axis)
    if reduce_all:
        axis = None
    return np.mean(x, axis=axis, keepdims=keepdim)


class TestReduceMeanOp(OpTest):
    def setUp(self):
        self.op_type = 'reduce_mean'
        self.dtype = 'float64'
        self.shape = [2, 3, 4, 5]
        self.axis = [0]
        self.keepdim = False
        self.reduce_all = False
        self.set_attrs()

        np.random.seed(10)
        x_np = np.random.uniform(-1, 1, self.shape).astype(self.dtype)
        out_np = ref_reduce_mean(x_np, self.axis, self.keepdim, self.reduce_all)
        self.inputs = {'X': x_np}
        self.outputs = {'Out': out_np}
        self.attrs = {
            'dim': self.axis,
            'keep_dim': self.keepdim,
            'reduce_all': self.reduce_all
        }

    def set_attrs(self):
        pass

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], ['Out'])


class TestReduceMeanOpDefaultAttrs(TestReduceMeanOp):
    def setUp(self):
        self.op_type = 'reduce_mean'
        self.dtype = 'float64'
        self.shape = [2, 3, 4, 5]

        x_np = np.random.uniform(-1, 1, self.shape).astype(self.dtype)
        out_np = np.mean(x_np, axis=0)
        self.inputs = {'X': x_np}
        self.outputs = {'Out': out_np}


class TestReduceMeanOpFloat32(TestReduceMeanOp):
    def set_attrs(self):
        self.dtype = 'float32'


class TestReduceMeanOpShape1D(TestReduceMeanOp):
    def set_attrs(self):
        self.shape = [100]


class TestReduceMeanOpShape6D(TestReduceMeanOp):
    def set_attrs(self):
        self.shape = [2, 3, 4, 5, 6, 7]


class TestReduceMeanOpAxisAll(TestReduceMeanOp):
    def set_attrs(self):
        self.axis = [0, 1, 2, 3]


class TestReduceMeanOpAxisTuple(TestReduceMeanOp):
    def set_attrs(self):
        self.axis = (0, 1, 2)


class TestReduceMeanOpAxisNegative(TestReduceMeanOp):
    def set_attrs(self):
        self.axis = [-2, -1]


class TestReduceMeanOpKeepdimTrue1(TestReduceMeanOp):
    def set_attrs(self):
        self.keepdim = True


class TestReduceMeanOpKeepdimTrue2(TestReduceMeanOp):
    def set_attrs(self):
        self.axis = [0, 1, 2, 3]
        self.keepdim = True


class TestReduceMeanOpReduceAllTrue(TestReduceMeanOp):
    def set_attrs(self):
        self.reduce_all = True


class TestMeanAPI(unittest.TestCase):
    # test paddle.tensor.stat.mean

    def setUp(self):
        self.x_shape = [2, 3, 4, 5]
        self.x = np.random.uniform(-1, 1, self.x_shape).astype(np.float32)
        self.place = paddle.CUDAPlace(0) if core.is_compiled_with_cuda() \
            else paddle.CPUPlace()

    def test_api_static(self):
        paddle.enable_static()
        with paddle.static.program_guard(paddle.static.Program()):
            x = paddle.fluid.data('X', self.x_shape)
            out1 = paddle.mean(x)
            out2 = paddle.tensor.mean(x)
            out3 = paddle.tensor.stat.mean(x)
            axis = np.arange(len(self.x_shape)).tolist()
            out4 = paddle.mean(x, axis)
            out5 = paddle.mean(x, tuple(axis))

            exe = paddle.static.Executor(self.place)
            res = exe.run(feed={'X': self.x},
                          fetch_list=[out1, out2, out3, out4, out5])
        out_ref = np.mean(self.x)
        for out in res:
            self.assertEqual(np.allclose(out, out_ref, rtol=1e-04), True)

    def test_api_dygraph(self):
        paddle.disable_static(self.place)

        def test_case(x, axis=None, keepdim=False):
            x_tensor = paddle.to_tensor(x)
            out = paddle.mean(x_tensor, axis, keepdim)
            if isinstance(axis, list):
                axis = tuple(axis)
                if len(axis) == 0:
                    axis = None
            out_ref = np.mean(x, axis, keepdims=keepdim)
            self.assertEqual(
                np.allclose(
                    out.numpy(), out_ref, rtol=1e-04), True)

        test_case(self.x)
        test_case(self.x, [])
        test_case(self.x, -1)
        test_case(self.x, keepdim=True)
        test_case(self.x, 2, keepdim=True)
        test_case(self.x, [0, 2])
        test_case(self.x, (0, 2))
        test_case(self.x, [0, 1, 2, 3])
        paddle.enable_static()

    def test_fluid_api(self):
        with fluid.program_guard(fluid.Program(), fluid.Program()):
            x = fluid.data("x", shape=[10, 10], dtype="float32")
            out = fluid.layers.reduce_mean(input=x, dim=1)
            place = fluid.CPUPlace()
            exe = fluid.Executor(place)
            x_np = np.random.rand(10, 10).astype(np.float32)
            res = exe.run(feed={"x": x_np}, fetch_list=[out])
        self.assertEqual(np.allclose(res[0], np.mean(x_np, axis=1)), True)

        with fluid.dygraph.guard():
            x_np = np.random.rand(10, 10).astype(np.float32)
            x = fluid.dygraph.to_variable(x_np)
            out = fluid.layers.reduce_mean(input=x, dim=1)
        self.assertEqual(np.allclose(out.numpy(), np.mean(x_np, axis=1)), True)

    def test_errors(self):
        paddle.disable_static()
        x = np.random.uniform(-1, 1, [10, 12]).astype('float32')
        x = paddle.to_tensor(x)
        self.assertRaises(Exception, paddle.mean, x, -3)
        self.assertRaises(Exception, paddle.mean, x, 2)
        paddle.enable_static()
        with paddle.static.program_guard(paddle.static.Program()):
            x = paddle.fluid.data('X', [10, 12], 'int32')
            self.assertRaises(TypeError, paddle.mean, x)


if __name__ == "__main__":
    unittest.main()
