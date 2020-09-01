# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
import paddle.fluid.core as core
from paddle.fluid import Program, program_guard


def create_kernel_case(op_type, numpy_op_type):
    class ArgMinMaxKernelBaseCase(OpTest):
        def initTestCase(self):
            self.op_type = op_type
            self.numpy_op_type = numpy_op_type
            self.axis = 0

        def setUp(self):
            np.random.seed(123)
            self.initTestCase()
            self.dims = (4, 5, 6)
            self.dtype = "float64"
            self.x = (1000 * np.random.random(self.dims).astype(self.dtype))
            self.inputs = {'X': self.x}
            self.attrs = {"axis": self.axis}
            self.numpy_op = eval("np.%s" % (numpy_op_type))
            self.outputs = {'Out': self.numpy_op(self.x, axis=self.axis)}

        def test_check_output(self):
            paddle.enable_static()
            self.check_output()

    class ArgMinMaxKernelCase0(ArgMinMaxKernelBaseCase):
        def initTestCase(self):
            self.op_type = op_type
            self.numpy_op_type = numpy_op_type
            self.axis = 1

    class ArgMinMaxKernelCase1(ArgMinMaxKernelBaseCase):
        def initTestCase(self):
            self.op_type = op_type
            self.numpy_op_type = numpy_op_type
            self.axis = 2

    class ArgMinMaxKernelCase2(ArgMinMaxKernelBaseCase):
        def initTestCase(self):
            self.op_type = op_type
            self.numpy_op_type = numpy_op_type
            self.axis = -1

    class ArgMinMaxKernelCase3(ArgMinMaxKernelBaseCase):
        def initTestCase(self):
            self.op_type = op_type
            self.numpy_op_type = numpy_op_type
            self.axis = -2

    class ArgMinMaxKernelCase4(ArgMinMaxKernelBaseCase):
        def setUp(self):
            self.initTestCase()
            self.dims = (4, 5, 6)
            self.dtype = "float64"
            self.x = (1000 * np.random.random(self.dims).astype(self.dtype))
            self.inputs = {'X': self.x}
            self.attrs = {"axis": self.axis, "keepdims": True}
            self.numpy_op = eval("np.%s" % (numpy_op_type))
            self.outputs = {
                'Out': self.numpy_op(
                    self.x, axis=self.axis).reshape((1, 5, 6))
            }

    class ArgMinMaxKernelCase5(ArgMinMaxKernelBaseCase):
        def setUp(self):
            self.initTestCase()
            self.dims = (4)
            self.dtype = "float64"
            self.x = (1000 * np.random.random(self.dims).astype(self.dtype))
            self.inputs = {'X': self.x}
            self.attrs = {"axis": self.axis, "flatten": True}
            self.numpy_op = eval("np.%s" % (numpy_op_type))
            self.outputs = {
                'Out': self.numpy_op(
                    self.x.flatten(), axis=self.axis)
            }

    class ArgMinMaxKernelCase6(ArgMinMaxKernelBaseCase):
        def setUp(self):
            self.initTestCase()
            self.dims = (4)
            self.dtype = "float64"
            self.x = (1000 * np.random.random(self.dims).astype(self.dtype))
            self.inputs = {'X': self.x}
            self.attrs = {"axis": self.axis, "flatten": True, "keepdims": True}
            self.numpy_op = eval("np.%s" % (numpy_op_type))
            self.outputs = {
                'Out':
                np.array(self.numpy_op(
                    self.x.flatten(), axis=self.axis))
            }

    cls_name = "ArgMinMaxKernelBaseCase_%s" % (op_type)
    ArgMinMaxKernelBaseCase.__name__ = cls_name
    globals()[cls_name] = ArgMinMaxKernelBaseCase

    cls_name = "ArgMinMaxKernelCase0_%s" % (op_type)
    ArgMinMaxKernelCase0.__name__ = cls_name
    globals()[cls_name] = ArgMinMaxKernelCase0

    cls_name = "ArgMinMaxKernelCase1_%s" % (op_type)
    ArgMinMaxKernelCase1.__name__ = cls_name
    globals()[cls_name] = ArgMinMaxKernelCase1

    cls_name = "ArgMinMaxKernelCase2_%s" % (op_type)
    ArgMinMaxKernelCase2.__name__ = cls_name
    globals()[cls_name] = ArgMinMaxKernelCase2

    cls_name = "ArgMinMaxKernelCase3_%s" % (op_type)
    ArgMinMaxKernelCase3.__name__ = cls_name
    globals()[cls_name] = ArgMinMaxKernelCase3

    cls_name = "ArgMinMaxKernelCase4_%s" % (op_type)
    ArgMinMaxKernelCase4.__name__ = cls_name
    globals()[cls_name] = ArgMinMaxKernelCase4

    cls_name = "ArgMinMaxKernelCase5_%s" % (op_type)
    ArgMinMaxKernelCase5.__name__ = cls_name
    globals()[cls_name] = ArgMinMaxKernelCase5

    cls_name = "ArgMinMaxKernelCase6_%s" % (op_type)
    ArgMinMaxKernelCase6.__name__ = cls_name
    globals()[cls_name] = ArgMinMaxKernelCase6


for op_type, numpy_op_type in zip(['arg_max', 'arg_min'], ['argmax', 'argmin']):
    create_kernel_case(op_type, numpy_op_type)


def create_test_case(op_type):
    class ArgMaxMinTestCase(unittest.TestCase):
        def setUp(self):
            np.random.seed(123)
            self.input_data = np.random.rand(10, 10).astype("float32")
            self.places = []
            self.places.append(fluid.CPUPlace())
            if core.is_compiled_with_cuda():
                self.places.append(paddle.CUDAPlace(0))
            self.op = eval("paddle.%s" % (op_type))
            self.numpy_op = eval("np.%s" % (op_type))

        def run_static(self, place):
            paddle.enable_static()
            with paddle.static.program_guard(paddle.static.Program()):
                data_var = paddle.static.data(
                    name="data", shape=[10, 10], dtype="float32")
                op = eval("paddle.%s" % (op_type))
                result = op(data_var)
                exe = paddle.static.Executor(place)
                result_data = exe.run(feed={"data": self.input_data},
                                      fetch_list=[result])
                expected_data = self.numpy_op(self.input_data)
                self.assertTrue((result_data == np.array(expected_data)).all(),
                                True)

            with paddle.static.program_guard(paddle.static.Program()):
                data_var = paddle.static.data(
                    name="data", shape=[10, 10], dtype="float32")
                op = eval("paddle.%s" % (op_type))
                result = op(data_var, axis=1)
                exe = paddle.static.Executor(place)
                result_data = exe.run(feed={"data": self.input_data},
                                      fetch_list=[result])
                expected_data = self.numpy_op(self.input_data, axis=1)
                self.assertTrue((result_data == expected_data).all(), True)

            with paddle.static.program_guard(paddle.static.Program()):
                data_var = paddle.static.data(
                    name="data", shape=[10, 10], dtype="float32")
                op = eval("paddle.%s" % (op_type))
                result = op(data_var, axis=-1)
                exe = paddle.static.Executor(place)
                result_data = exe.run(feed={"data": self.input_data},
                                      fetch_list=[result])
                expected_data = self.numpy_op(self.input_data, axis=-1)
                self.assertTrue((result_data == expected_data).all(), True)

            with paddle.static.program_guard(paddle.static.Program()):
                data_var = paddle.static.data(
                    name="data", shape=[10, 10], dtype="float32")

                op = eval("paddle.%s" % (op_type))
                result = op(data_var, axis=-1, keepdim=True)
                exe = paddle.static.Executor(place)
                result_data = exe.run(feed={"data": self.input_data},
                                      fetch_list=[result])
                expected_data = self.numpy_op(
                    self.input_data, axis=-1).reshape((10, 1))
                self.assertTrue((result_data == expected_data).all(), True)

            with paddle.static.program_guard(paddle.static.Program()):
                op = eval("paddle.%s" % (op_type))
                data_var = paddle.static.data(
                    name="data", shape=[10, 10], dtype="float32")
                result = op(data_var, axis=-1, name="test_arg_api")
                self.assertTrue("test_arg_api" in result.name)

        def run_dygraph(self, place):
            paddle.disable_static(place)
            op = eval("paddle.%s" % (op_type))
            data_tensor = paddle.to_tensor(self.input_data)

            #case 1 
            result_data = op(data_tensor)
            excepted_data = self.numpy_op(self.input_data)
            self.assertTrue((result_data.numpy() == excepted_data).all(), True)

            #case 2 
            result_data = op(data_tensor, axis=1)
            excepted_data = self.numpy_op(self.input_data, axis=1)
            self.assertTrue((result_data.numpy() == excepted_data).all(), True)

            #case 3 
            result_data = op(data_tensor, axis=-1)
            excepted_data = self.numpy_op(self.input_data, axis=-1)
            self.assertTrue((result_data.numpy() == excepted_data).all(), True)

            #case 4 
            result_data = op(data_tensor, axis=-1, keepdim=True)
            excepted_data = self.numpy_op(self.input_data, axis=-1)
            excepted_data = excepted_data.reshape((10, 1))
            self.assertTrue((result_data.numpy() == excepted_data).all(), True)

            #case 5 
            result_data = op(data_tensor, axis=-1, keepdim=True, dtype="int32")
            self.assertTrue(result_data.numpy().dtype == np.int32)

            # case for dim 4, 5, 6, for test case coverage
            input_data = np.random.rand(5, 5, 5, 5)
            excepted_data = self.numpy_op(input_data, axis=0)
            result_data = op(paddle.to_tensor(input_data), axis=0)
            self.assertTrue((result_data.numpy() == excepted_data).all(), True)

            input_data = np.random.rand(4, 4, 4, 4, 4)
            excepted_data = self.numpy_op(input_data, axis=0)
            result_data = op(paddle.to_tensor(input_data), axis=0)
            self.assertTrue((result_data.numpy() == excepted_data).all(), True)

            input_data = np.random.rand(3, 3, 3, 3, 3, 3)
            excepted_data = self.numpy_op(input_data, axis=0)
            result_data = op(paddle.to_tensor(input_data), axis=0)
            self.assertTrue((result_data.numpy() == excepted_data).all(), True)

        def test_case(self):
            for place in self.places:
                self.run_static(place)
                self.run_dygraph(place)

    cls_name = "ArgMaxMinTestCase_{}".format(op_type)
    ArgMaxMinTestCase.__name__ = cls_name
    globals()[cls_name] = ArgMaxMinTestCase


for op_type in ['argmin', 'argmax']:
    create_test_case(op_type)


class TestArgMinMaxOpError(unittest.TestCase):
    def test_errors(self):
        paddle.enable_static()
        with program_guard(Program(), Program()):

            def test_argmax_x_type():
                x1 = [1, 2, 3]
                output = paddle.argmax(x=x1)

            self.assertRaises(TypeError, test_argmax_x_type)

            def test_argmin_x_type():
                x2 = [1, 2, 3]
                output = paddle.argmin(x=x2)

            self.assertRaises(TypeError, test_argmin_x_type)

            def test_argmax_attr_type():
                data = paddle.static.data(
                    name="test_argmax", shape=[10], dtype="float32")
                output = paddle.argmax(x=data, dtype="float32")

            self.assertRaises(TypeError, test_argmax_attr_type)

            def test_argmin_attr_type():
                data = paddle.static.data(
                    name="test_argmax", shape=[10], dtype="float32")
                output = paddle.argmin(x=data, dtype="float32")

            self.assertRaises(TypeError, test_argmin_attr_type)

            def test_argmax_axis_type():
                data = paddle.static.data(
                    name="test_argmax", shape=[10], dtype="float32")
                output = paddle.argmax(x=data, axis=1.2)

            self.assertRaises(TypeError, test_argmax_axis_type)

            def test_argmin_axis_type():
                data = paddle.static.data(
                    name="test_argmin", shape=[10], dtype="float32")
                output = paddle.argmin(x=data, axis=1.2)

            self.assertRaises(TypeError, test_argmin_axis_type)


if __name__ == '__main__':
    unittest.main()
