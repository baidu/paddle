#  Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import numpy as np
import unittest
import sys
sys.path.append("..")
from op_test import OpTest, _set_use_system_allocator
import paddle
import paddle.fluid as fluid

paddle.enable_static()


@unittest.skipIf(not paddle.is_compiled_with_npu(),
                 "core is not compiled with NPU")
class TestElementwiseAddOp(OpTest):
    def setUp(self):
        self.set_npu()
        self.op_type = "elementwise_add"
        self.place = paddle.NPUPlace(0)
        self.init_dtype()
        self.init_input_output()
        self.init_kernel_type()
        self.init_axis()

        self.inputs = {
            'X': OpTest.np_dtype_to_fluid_dtype(self.x),
            'Y': OpTest.np_dtype_to_fluid_dtype(self.y)
        }
        self.attrs = {'axis': self.axis, 'use_mkldnn': self.use_mkldnn}
        self.outputs = {'Out': self.out}

    def set_npu(self):
        self.__class__.use_npu = True

    def init_kernel_type(self):
        self.use_mkldnn = False

    def init_input_output(self):
        self.x = np.random.uniform(0.1, 1, [13, 17]).astype(self.dtype)
        self.y = np.random.uniform(0.1, 1, [13, 17]).astype(self.dtype)
        self.out = np.add(self.x, self.y)

    def init_dtype(self):
        self.dtype = np.float32

    def init_axis(self):
        self.axis = -1

    def test_check_output(self):
        self.check_output_with_place(self.place, check_dygraph=False)

    def test_check_grad_normal(self):
        self.check_grad_with_place(
            self.place, ['X', 'Y'],
            'Out',
            max_relative_error=0.006,
            check_dygraph=False)

    def test_check_grad_ingore_x(self):
        self.check_grad_with_place(
            self.place, ['Y'],
            'Out',
            no_grad_set=set("X"),
            max_relative_error=0.006,
            check_dygraph=False)

    def test_check_grad_ingore_y(self):
        self.check_grad_with_place(
            self.place, ['X'],
            'Out',
            no_grad_set=set("Y"),
            max_relative_error=0.006,
            check_dygraph=False)


@unittest.skipIf(not paddle.is_compiled_with_npu(),
                 "core is not compiled with NPU")
class TestAddAPI(unittest.TestCase):
    def test_name(self):
        with paddle.static.program_guard(paddle.static.Program()):
            x = paddle.static.data(name="x", shape=[2, 3], dtype="float32")
            y = paddle.static.data(name='y', shape=[2, 3], dtype='float32')

            y_1 = paddle.add(x, y, name='add_res')
            self.assertEqual(('add_res' in y_1.name), True)

    def test_static(self):
        with paddle.static.program_guard(paddle.static.Program()):

            x_np = np.array([2, 3, 4]).astype('float32')
            y_np = np.array([1, 5, 2]).astype('float32')

            x = paddle.static.data(name="x", shape=[3], dtype='float32')
            y = paddle.static.data(name="y", shape=[3], dtype='float32')

            x_reshape = paddle.reshape(x, [3, 1])
            y_reshape = paddle.reshape(y, [3, 1])
            z = paddle.add(x_reshape, y_reshape)
            z = paddle.reshape(z, shape=[3])

            place = paddle.NPUPlace(0)
            exe = paddle.static.Executor(place)
            x_value, y_value, z_value = exe.run(feed={"x": x_np,
                                                      "y": y_np},
                                                fetch_list=[x, y, z])

            z_expected = np.array([3., 8., 6.])
            self.assertEqual(
                (x_value == x_np).all(),
                True,
                msg="x_value = {}, but expected {}".format(x_value, x_np))
            self.assertEqual(
                (y_value == y_np).all(),
                True,
                msg="y_value = {}, but expected {}".format(y_value, y_np))
            self.assertEqual(
                (z_value == z_expected).all(),
                True,
                msg="z_value = {}, but expected {}".format(z_value, z_expected))


@unittest.skipIf(not paddle.is_compiled_with_npu(),
                 "core is not compiled with NPU")
class TestAddError(unittest.TestCase):
    def test_errors(self):
        with paddle.static.program_guard(paddle.static.Program()):
            # the input of elementwise_add must be Variable.
            x1 = fluid.create_lod_tensor(
                np.array([-1, 3, 5, 5]), [[1, 1, 1, 1]], fluid.NPUPlace(0))
            y1 = fluid.create_lod_tensor(
                np.array([-1, 3, 5, 5]), [[1, 1, 1, 1]], fluid.NPUPlace(0))
            self.assertRaises(TypeError, paddle.add, x1, y1)

            # the input dtype must be float16 or float32 or float64 or int32 or int64
            x2 = paddle.static.data(
                name='x2', shape=[3, 4, 5, 6], dtype="uint8")
            y2 = paddle.static.data(
                name='y2', shape=[3, 4, 5, 6], dtype="uint8")
            self.assertRaises(TypeError, paddle.add, x2, y2)


if __name__ == '__main__':
    unittest.main()
