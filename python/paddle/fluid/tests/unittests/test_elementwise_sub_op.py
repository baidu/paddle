#  Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
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
from op_test import OpTest, skip_check_grad_ci


class TestElementwiseOp(OpTest):
    def setUp(self):
        self.op_type = "elementwise_sub"
        self.inputs = {
            'X': np.random.uniform(0.1, 1, [2, 3, 4, 5]).astype("float64"),
            'Y': np.random.uniform(0.1, 1, [2, 3, 4, 5]).astype("float64")
        }
        self.outputs = {'Out': self.inputs['X'] - self.inputs['Y']}

    def test_check_output(self):
        self.check_output()

    def test_check_grad_normal(self):
        self.check_grad(['X', 'Y'], 'Out')

    def test_check_grad_ingore_x(self):
        self.check_grad(
            ['Y'], 'Out', max_relative_error=0.005, no_grad_set=set("X"))

    def test_check_grad_ingore_y(self):
        self.check_grad(
            ['X'], 'Out', max_relative_error=0.005, no_grad_set=set('Y'))


@skip_check_grad_ci(
    reason="[skip shape check] Use y_shape(1) to test broadcast.")
class TestElementwiseSubOp_scalar(TestElementwiseOp):
    def setUp(self):
        self.op_type = "elementwise_sub"
        self.inputs = {
            'X': np.random.rand(10, 3, 4).astype(np.float64),
            'Y': np.random.rand(1).astype(np.float64)
        }
        self.outputs = {'Out': self.inputs['X'] - self.inputs['Y']}


class TestElementwiseSubOp_Vector(TestElementwiseOp):
    def setUp(self):
        self.op_type = "elementwise_sub"
        self.inputs = {
            'X': np.random.random((100, )).astype("float64"),
            'Y': np.random.random((100, )).astype("float64")
        }
        self.outputs = {'Out': self.inputs['X'] - self.inputs['Y']}


class TestElementwiseSubOp_broadcast_0(TestElementwiseOp):
    def setUp(self):
        self.op_type = "elementwise_sub"
        self.inputs = {
            'X': np.random.rand(100, 3, 2).astype(np.float64),
            'Y': np.random.rand(100).astype(np.float64)
        }

        self.attrs = {'axis': 0}
        self.outputs = {
            'Out': self.inputs['X'] - self.inputs['Y'].reshape(100, 1, 1)
        }


class TestElementwiseSubOp_broadcast_1(TestElementwiseOp):
    def setUp(self):
        self.op_type = "elementwise_sub"
        self.inputs = {
            'X': np.random.rand(2, 100, 3).astype(np.float64),
            'Y': np.random.rand(100).astype(np.float64)
        }

        self.attrs = {'axis': 1}
        self.outputs = {
            'Out': self.inputs['X'] - self.inputs['Y'].reshape(1, 100, 1)
        }


class TestElementwiseSubOp_broadcast_2(TestElementwiseOp):
    def setUp(self):
        self.op_type = "elementwise_sub"
        self.inputs = {
            'X': np.random.rand(2, 3, 100).astype(np.float64),
            'Y': np.random.rand(100).astype(np.float64)
        }

        self.outputs = {
            'Out': self.inputs['X'] - self.inputs['Y'].reshape(1, 1, 100)
        }


class TestElementwiseSubOp_broadcast_3(TestElementwiseOp):
    def setUp(self):
        self.op_type = "elementwise_sub"
        self.inputs = {
            'X': np.random.rand(2, 10, 12, 3).astype(np.float64),
            'Y': np.random.rand(10, 12).astype(np.float64)
        }

        self.attrs = {'axis': 1}
        self.outputs = {
            'Out': self.inputs['X'] - self.inputs['Y'].reshape(1, 10, 12, 1)
        }


class TestElementwiseSubOp_broadcast_4(TestElementwiseOp):
    def setUp(self):
        self.op_type = "elementwise_sub"
        self.inputs = {
            'X': np.random.rand(2, 5, 3, 12).astype(np.float64),
            'Y': np.random.rand(2, 5, 1, 12).astype(np.float64)
        }
        self.outputs = {'Out': self.inputs['X'] - self.inputs['Y']}


class TestElementwiseSubOp_commonuse_1(TestElementwiseOp):
    def setUp(self):
        self.op_type = "elementwise_sub"
        self.inputs = {
            'X': np.random.rand(2, 3, 100).astype(np.float64),
            'Y': np.random.rand(1, 1, 100).astype(np.float64)
        }
        self.outputs = {'Out': self.inputs['X'] - self.inputs['Y']}


class TestElementwiseSubOp_commonuse_2(TestElementwiseOp):
    def setUp(self):
        self.op_type = "elementwise_sub"
        self.inputs = {
            'X': np.random.rand(10, 3, 1, 4).astype(np.float64),
            'Y': np.random.rand(10, 1, 12, 1).astype(np.float64)
        }
        self.outputs = {'Out': self.inputs['X'] - self.inputs['Y']}


class TestElementwiseSubOp_xsize_lessthan_ysize(TestElementwiseOp):
    def setUp(self):
        self.op_type = "elementwise_sub"
        self.inputs = {
            'X': np.random.rand(10, 12).astype(np.float64),
            'Y': np.random.rand(2, 3, 10, 12).astype(np.float64)
        }

        self.attrs = {'axis': 2}

        self.outputs = {
            'Out': self.inputs['X'].reshape(1, 1, 10, 12) - self.inputs['Y']
        }


class TestComplexElementwiseSubOp(OpTest):
    def setUp(self):
        self.op_type = "elementwise_sub"
        self.dtype = np.float64
        self.shape = (2, 3, 4, 5)
        self.init_input_output()
        self.init_grad_input_output()

        self.inputs = {
            'X': OpTest.np_dtype_to_fluid_dtype(self.x),
            'Y': OpTest.np_dtype_to_fluid_dtype(self.y)
        }
        self.attrs = {'axis': -1, 'use_mkldnn': False}
        self.outputs = {'Out': self.out}

    def init_base_dtype(self):
        self.dtype = np.float64

    def init_input_output(self):
        self.x = np.random.random(self.shape).astype(
            self.dtype) + 1J * np.random.random(self.shape).astype(self.dtype)
        self.y = np.random.random(self.shape).astype(
            self.dtype) + 1J * np.random.random(self.shape).astype(self.dtype)
        self.out = self.x - self.y

    def init_grad_input_output(self):
        self.grad_out = np.ones(self.shape, self.dtype) + 1J * np.ones(
            self.shape, self.dtype)
        self.grad_x = self.grad_out
        self.grad_y = -self.grad_out

    def test_check_output(self):
        self.check_output()

    def test_check_grad_normal(self):
        self.check_grad(
            ['X', 'Y'],
            'Out',
            user_defined_grads=[self.grad_x, self.grad_y],
            user_defined_grad_outputs=[self.grad_out])

    def test_check_grad_ingore_x(self):
        self.check_grad(
            ['Y'],
            'Out',
            no_grad_set=set("X"),
            user_defined_grads=[self.grad_y],
            user_defined_grad_outputs=[self.grad_out])

    def test_check_grad_ingore_y(self):
        self.check_grad(
            ['X'],
            'Out',
            no_grad_set=set('Y'),
            user_defined_grads=[self.grad_x],
            user_defined_grad_outputs=[self.grad_out])


class TestRealComplexElementwiseSubOp(TestComplexElementwiseSubOp):
    def init_input_output(self):
        self.x = np.random.random(self.shape).astype(self.dtype)
        self.y = np.random.random(self.shape).astype(
            self.dtype) + 1J * np.random.random(self.shape).astype(self.dtype)
        self.out = self.x - self.y

    def init_grad_input_output(self):
        self.grad_out = np.ones(self.shape, self.dtype) + 1J * np.ones(
            self.shape, self.dtype)
        self.grad_x = np.real(self.grad_out)
        self.grad_y = -self.grad_out


class TestSubtractApi(unittest.TestCase):
    def setUp(self):
        self._executed_api()

    def _executed_api(self):
        self.subtract = paddle.subtract

    def test_name(self):
        with fluid.program_guard(fluid.Program()):
            x = fluid.data(name="x", shape=[2, 3], dtype="float32")
            y = fluid.data(name='y', shape=[2, 3], dtype='float32')

            y_1 = self.subtract(x, y, name='subtract_res')
            self.assertEqual(('subtract_res' in y_1.name), True)

    def test_declarative(self):
        with fluid.program_guard(fluid.Program()):

            def gen_data():
                return {
                    "x": np.array([2, 3, 4]).astype('float32'),
                    "y": np.array([1, 5, 2]).astype('float32')
                }

            x = fluid.data(name="x", shape=[3], dtype='float32')
            y = fluid.data(name="y", shape=[3], dtype='float32')
            z = self.subtract(x, y)

            place = fluid.CPUPlace()
            exe = fluid.Executor(place)
            z_value = exe.run(feed=gen_data(), fetch_list=[z.name])
            z_expected = np.array([1., -2., 2.])
            self.assertEqual((z_value == z_expected).all(), True)

    def test_dygraph(self):
        with fluid.dygraph.guard():
            np_x = np.array([2, 3, 4]).astype('float64')
            np_y = np.array([1, 5, 2]).astype('float64')
            x = fluid.dygraph.to_variable(np_x)
            y = fluid.dygraph.to_variable(np_y)
            z = self.subtract(x, y)
            np_z = z.numpy()
            z_expected = np.array([1., -2., 2.])
            self.assertEqual((np_z == z_expected).all(), True)


class TestSubtractInplaceApi(TestSubtractApi):
    def _executed_api(self):
        self.subtract = paddle.subtract_


class TestSubtractInplaceBroadcastSuccess(unittest.TestCase):
    def init_data(self):
        self.x_numpy = np.random.rand(2, 3, 4).astype('float')
        self.y_numpy = np.random.rand(3, 4).astype('float')

    def test_broadcast_success(self):
        paddle.disable_static()
        self.init_data()

        x = paddle.to_tensor(self.x_numpy)
        y = paddle.to_tensor(self.y_numpy)

        inplace_result = paddle.subtract_(x, y)
        numpy_result = self.x_numpy - self.y_numpy
        self.assertEqual((inplace_result.numpy() == numpy_result).all(), True)
        paddle.enable_static()


class TestSubtractInplaceBroadcastSuccess2(TestSubtractInplaceBroadcastSuccess):
    def init_data(self):
        self.x_numpy = np.random.rand(1, 2, 3, 1).astype('float')
        self.y_numpy = np.random.rand(3, 1).astype('float')


class TestSubtractInplaceBroadcastSuccess3(TestSubtractInplaceBroadcastSuccess):
    def init_data(self):
        self.x_numpy = np.random.rand(2, 3, 1, 5).astype('float')
        self.y_numpy = np.random.rand(1, 3, 1, 5).astype('float')


class TestSubtractInplaceBroadcastError(unittest.TestCase):
    def init_data(self):
        self.x_numpy = np.random.rand(3, 4).astype('float')
        self.y_numpy = np.random.rand(2, 3, 4).astype('float')

    def test_broadcast_errors(self):
        paddle.disable_static()
        self.init_data()

        x = paddle.to_tensor(self.x_numpy)
        y = paddle.to_tensor(self.y_numpy)

        def broadcast_shape_error():
            paddle.subtract_(x, y)

        self.assertRaises(ValueError, broadcast_shape_error)
        paddle.enable_static()


class TestSubtractInplaceBroadcastError2(TestSubtractInplaceBroadcastError):
    def init_data(self):
        self.x_numpy = np.random.rand(2, 1, 4).astype('float')
        self.y_numpy = np.random.rand(2, 3, 4).astype('float')


class TestSubtractInplaceBroadcastError3(TestSubtractInplaceBroadcastError):
    def init_data(self):
        self.x_numpy = np.random.rand(5, 2, 1, 4).astype('float')
        self.y_numpy = np.random.rand(2, 3, 4).astype('float')


if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()
