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
from op_test import OpTest, skip_check_grad_ci
import paddle.fluid as fluid
from paddle.fluid import compiler, Program, program_guard, core
import paddle


class TestConcatOp(OpTest):
    def setUp(self):
        self.op_type = "concat"
        self.dtype = self.get_dtype()
        self.init_test_data()
        self.inputs = {'X': [('x0', self.x0), ('x1', self.x1), ('x2', self.x2)]}
        self.attrs = {'axis': self.axis}
        if self.axis < 0:
            self.actual_axis = self.axis + len(self.x0.shape)
            self.actual_axis = self.actual_axis if self.actual_axis > 0 else 0
        else:
            self.actual_axis = self.axis

        self.outputs = {
            'Out': np.concatenate(
                (self.x0, self.x1, self.x2), axis=self.actual_axis)
        }

    def get_dtype(self):
        return "float64"

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['x0'], 'Out')
        self.check_grad(['x1'], 'Out')
        self.check_grad(['x2'], 'Out')

    def init_test_data(self):
        self.x0 = np.random.random((5, 1, 4, 5)).astype(self.dtype)
        self.x1 = np.random.random((5, 2, 4, 5)).astype(self.dtype)
        self.x2 = np.random.random((5, 3, 4, 5)).astype(self.dtype)
        self.axis = 1


class TestConcatOp2(TestConcatOp):
    def init_test_data(self):
        self.x0 = np.random.random((2, 3, 4, 5)).astype(self.dtype)
        self.x1 = np.random.random((2, 3, 4, 5)).astype(self.dtype)
        self.x2 = np.random.random((2, 3, 4, 5)).astype(self.dtype)
        self.axis = 1


@skip_check_grad_ci(
    reason="The function 'check_grad' for large inputs is too slow.")
class TestConcatOp3(TestConcatOp):
    def init_test_data(self):
        self.x0 = np.random.random((1, 256, 170, 256)).astype(self.dtype)
        self.x1 = np.random.random((1, 128, 170, 256)).astype(self.dtype)
        self.x2 = np.random.random((1, 128, 170, 256)).astype(self.dtype)
        self.axis = 1

    def test_check_grad(self):
        pass


@skip_check_grad_ci(
    reason="This test will meet fetch error when there is a null grad. The detailed information is in PR#17015."
)
class TestConcatOp4(TestConcatOp):
    def init_test_data(self):
        self.x0 = np.random.random((2, 3, 4, 5)).astype(self.dtype)
        self.x1 = np.random.random((2, 3, 4, 5)).astype(self.dtype)
        self.x2 = np.random.random((0, 3, 4, 5)).astype(self.dtype)
        self.axis = 0

    def test_check_grad(self):
        pass


class TestConcatOp5(TestConcatOp):
    def init_test_data(self):
        self.x0 = np.random.random((5, 1, 4, 5)).astype(self.dtype)
        self.x1 = np.random.random((5, 2, 4, 5)).astype(self.dtype)
        self.x2 = np.random.random((5, 3, 4, 5)).astype(self.dtype)
        self.axis = -3


class TestConcatOp6(TestConcatOp):
    def setUp(self):
        self.op_type = "concat"
        self.dtype = self.get_dtype()
        self.init_test_data()
        self.lod = [[20, 80]]
        self.out_lod = [[20, 80, 20, 80, 20, 80]]
        self.inputs = {
            'X': [('x0', (self.x0, self.lod)), ('x1', (self.x1, self.lod)),
                  ('x2', (self.x2, self.lod))]
        }
        self.attrs = {'axis': self.axis}
        if self.axis < 0:
            self.actual_axis = self.axis + len(self.x0.shape)
            self.actual_axis = self.actual_axis if self.actual_axis > 0 else 0
        else:
            self.actual_axis = self.axis
        out = np.concatenate((self.x0, self.x1, self.x2), axis=self.actual_axis)
        self.outputs = {'Out': (out, self.out_lod)}

    def test_check_output(self):
        self.check_output(check_dygraph=False)

    def test_check_grad(self):
        self.check_grad(['x0'], 'Out', check_dygraph=False)
        self.check_grad(['x1'], 'Out', check_dygraph=False)
        self.check_grad(['x2'], 'Out', check_dygraph=False)

    def init_test_data(self):
        self.x0 = np.random.random([100]).astype(self.dtype)
        self.x1 = np.random.random([100]).astype(self.dtype)
        self.x2 = np.random.random([100]).astype(self.dtype)
        self.axis = 0


def create_test_AxisTensor(parent):
    class TestConcatAxisTensor(parent):
        def setUp(self):
            self.op_type = "concat"
            self.dtype = self.get_dtype()
            self.init_test_data()

            self.inputs = {
                'X': [('x0', self.x0), ('x1', self.x1), ('x2', self.x2)],
                'AxisTensor': np.array([self.axis]).astype("int32")
            }
            self.attrs = {}

            if self.axis < 0:
                self.actual_axis = self.axis + len(self.x0.shape)
                self.actual_axis = self.actual_axis if self.actual_axis > 0 else 0
            else:
                self.actual_axis = self.axis

            self.outputs = {
                'Out': np.concatenate(
                    (self.x0, self.x1, self.x2), axis=self.actual_axis)
            }

    cls_name = "{0}_{1}".format(parent.__name__, "AxisTensor")
    TestConcatAxisTensor.__name__ = cls_name
    globals()[cls_name] = TestConcatAxisTensor


create_test_AxisTensor(TestConcatOp)
create_test_AxisTensor(TestConcatOp2)
create_test_AxisTensor(TestConcatOp3)
create_test_AxisTensor(TestConcatOp4)
create_test_AxisTensor(TestConcatOp5)
create_test_AxisTensor(TestConcatOp6)

#----------------Concat Fp16----------------


def create_test_fp16(parent):
    class TestConcatFp16(parent):
        def get_dtype(self):
            return np.float16

    cls_name = "{0}_{1}".format(parent.__name__, "Fp16")
    TestConcatFp16.__name__ = cls_name
    globals()[cls_name] = TestConcatFp16


create_test_fp16(TestConcatOp)
create_test_fp16(TestConcatOp2)
create_test_fp16(TestConcatOp3)
create_test_fp16(TestConcatOp4)
create_test_fp16(TestConcatOp5)
create_test_fp16(TestConcatOp6)


class TestConcatOpError(unittest.TestCase):
    def test_errors(self):
        with program_guard(Program(), Program()):
            # The input type of concat_op should be list.
            x1 = fluid.layers.data(shape=[4], dtype='int32', name='x1')
            fluid.layers.concat(x1)
            # The item in input must be Variable.
            x2 = fluid.create_lod_tensor(
                np.array([[-1]]), [[1]], fluid.CPUPlace())
            x3 = fluid.create_lod_tensor(
                np.array([[-1]]), [[1]], fluid.CPUPlace())
            self.assertRaises(TypeError, fluid.layers.concat, [x2])
            # The input dtype of concat_op must be float16, float32, float64, int32, int64.
            x4 = fluid.layers.data(shape=[4], dtype='uint8', name='x4')
            x5 = fluid.layers.data(shape=[4], dtype='uint8', name='x5')
            self.assertRaises(TypeError, fluid.layers.concat, [x4, x5])
            x6 = fluid.layers.data(shape=[4], dtype='float16', name='x6')
            x7 = fluid.layers.data(shape=[4], dtype='float16', name='x7')
            x8 = fluid.layers.data(shape=[4], dtype='float32', name='x8')
            fluid.layers.concat([x6, x7])

            # The type of axis in concat_op should be int or Variable.
            def test_axis_type():
                fluid.layers.concat([x6, x7], 3.2)

            self.assertRaises(TypeError, test_axis_type)

            def test_input_same_dtype():
                fluid.layers.concat([x7, x8])

            self.assertRaises(TypeError, test_input_same_dtype)


class TestConcatAPI(unittest.TestCase):
    def test_fluid_api(self):
        paddle.enable_static()
        x_1 = fluid.data(shape=[None, 1, 4, 5], dtype='int32', name='x_1')
        fluid.layers.concat([x_1, x_1], 0)

        input_2 = np.random.random([2, 1, 4, 5]).astype("int32")
        input_3 = np.random.random([2, 2, 4, 5]).astype("int32")
        x_2 = fluid.data(shape=[2, 1, 4, 5], dtype='int32', name='x_2')
        x_3 = fluid.data(shape=[2, 2, 4, 5], dtype='int32', name='x_3')
        positive_1_int32 = fluid.layers.fill_constant([1], "int32", 1)
        positive_1_int64 = fluid.layers.fill_constant([1], "int64", 1)
        out_1 = fluid.layers.concat(input=[x_2, x_3], axis=1)
        out_2 = fluid.layers.concat(input=[x_2, x_3], axis=positive_1_int32)
        out_3 = fluid.layers.concat(input=[x_2, x_3], axis=positive_1_int64)

        exe = fluid.Executor(place=fluid.CPUPlace())
        [res_1, res_2, res_3] = exe.run(
            fluid.default_main_program(),
            feed={"x_1": input_2,
                  "x_2": input_2,
                  "x_3": input_3},
            fetch_list=[out_1, out_2, out_3])
        assert np.array_equal(res_1, np.concatenate((input_2, input_3), axis=1))
        assert np.array_equal(res_2, np.concatenate((input_2, input_3), axis=1))
        assert np.array_equal(res_3, np.concatenate((input_2, input_3), axis=1))

    def test_api(self):
        paddle.enable_static()
        x_1 = paddle.fluid.data(
            shape=[None, 1, 4, 5], dtype='int32', name='x_1')
        paddle.concat([x_1, x_1], 0)

        input_2 = np.random.random([2, 1, 4, 5]).astype("int32")
        input_3 = np.random.random([2, 2, 4, 5]).astype("int32")
        x_2 = fluid.data(shape=[2, 1, 4, 5], dtype='int32', name='x_2')
        x_3 = fluid.data(shape=[2, 2, 4, 5], dtype='int32', name='x_3')
        positive_1_int32 = paddle.fluid.layers.fill_constant([1], "int32", 1)
        positive_1_int64 = paddle.fluid.layers.fill_constant([1], "int64", 1)
        negative_int64 = paddle.fluid.layers.fill_constant([1], "int64", -3)
        out_1 = paddle.concat(x=[x_2, x_3], axis=1)
        out_2 = paddle.concat(x=[x_2, x_3], axis=positive_1_int32)
        out_3 = paddle.concat(x=[x_2, x_3], axis=positive_1_int64)
        out_4 = paddle.concat(x=[x_2, x_3], axis=negative_int64)

        exe = paddle.static.Executor(place=paddle.CPUPlace())
        [res_1, res_2, res_3, res_4] = exe.run(
            paddle.static.default_main_program(),
            feed={"x_1": input_2,
                  "x_2": input_2,
                  "x_3": input_3},
            fetch_list=[out_1, out_2, out_3, out_4])
        assert np.array_equal(res_1, np.concatenate((input_2, input_3), axis=1))
        assert np.array_equal(res_2, np.concatenate((input_2, input_3), axis=1))
        assert np.array_equal(res_3, np.concatenate((input_2, input_3), axis=1))
        assert np.array_equal(res_4, np.concatenate((input_2, input_3), axis=1))

    def test_imperative(self):
        in1 = np.array([[1, 2, 3], [4, 5, 6]])
        in2 = np.array([[11, 12, 13], [14, 15, 16]])
        in3 = np.array([[21, 22], [23, 24]])
        paddle.disable_static()
        x1 = paddle.to_tensor(in1)
        x2 = paddle.to_tensor(in2)
        x3 = paddle.to_tensor(in3)
        out1 = fluid.layers.concat(input=[x1, x2, x3], axis=-1)
        out2 = paddle.concat(x=[x1, x2], axis=0)
        np_out1 = np.concatenate([in1, in2, in3], axis=-1)
        np_out2 = np.concatenate([in1, in2], axis=0)
        paddle.enable_static()
        self.assertEqual((out1.numpy() == np_out1).all(), True)
        self.assertEqual((out2.numpy() == np_out2).all(), True)

    def test_errors(self):
        with program_guard(Program(), Program()):
            # The item in input must be Variable.
            x2 = fluid.create_lod_tensor(
                np.array([[-1]]), [[1]], fluid.CPUPlace())
            x3 = fluid.create_lod_tensor(
                np.array([[-1]]), [[1]], fluid.CPUPlace())
            self.assertRaises(TypeError, paddle.concat, [x2])
            # The input dtype of concat_op must be float16, float32, float64, int32, int64.
            x4 = paddle.fluid.data(shape=[4], dtype='uint8', name='x4')
            x5 = paddle.fluid.data(shape=[4], dtype='uint8', name='x5')
            self.assertRaises(TypeError, fluid.layers.concat, [x4, x5])

            # The type of axis in concat_op should be int or Variable.
            x6 = fluid.layers.data(shape=[4], dtype='float16', name='x6')
            x7 = fluid.layers.data(shape=[4], dtype='float16', name='x7')
            x8 = fluid.layers.data(shape=[4], dtype='float32', name='x8')

            def test_axis_type():
                paddle.concat([x6, x7], 3.2)

            self.assertRaises(TypeError, test_axis_type)

            def test_input_same_dtype():
                paddle.concat([x7, x8])

            self.assertRaises(TypeError, test_input_same_dtype)


class TestConcatAPIWithLoDTensorArray(unittest.TestCase):
    """
    Test concat api when the input(x) is a LoDTensorArray.
    """

    def setUp(self):
        self.axis = 1
        self.iter_num = 3
        self.input_shape = [2, 3]
        self.x = np.random.random(self.input_shape).astype("float32")
        self.place = fluid.CUDAPlace(0) \
            if fluid.is_compiled_with_cuda() else fluid.CPUPlace()

    def set_program(self, use_fluid_api):
        paddle.enable_static()
        if use_fluid_api:
            self.program = fluid.Program()
            with fluid.program_guard(self.program):
                input = fluid.layers.assign(self.x)
                tensor_array = fluid.layers.create_array(dtype='float32')
                zero = fluid.layers.fill_constant(
                    shape=[1], value=0, dtype="int64")

                for i in range(self.iter_num):
                    fluid.layers.array_write(input, zero + i, tensor_array)

                self.out_var = fluid.layers.concat(tensor_array, axis=self.axis)
        else:
            self.program = paddle.static.Program()
            with paddle.static.program_guard(self.program):
                input = paddle.assign(self.x)
                tensor_array = fluid.layers.create_array(
                    dtype='float32'
                )  # Api create_array is not supported in paddle 2.0 yet.
                zero = paddle.zeros(shape=[1], dtype="int64")

                for i in range(self.iter_num):
                    # Api array_write is not supported in paddle 2.0 yet.
                    fluid.layers.array_write(input, zero + i, tensor_array)

                self.out_var = paddle.concat(tensor_array, axis=self.axis)

    def test_fluid_api(self):
        self._run_static_mode(use_fluid_api=True)

    def test_paddle_api(self):
        self._run_static_mode(use_fluid_api=False)

    def _run_static_mode(self, use_fluid_api):
        self.set_program(use_fluid_api)
        self.assertTrue(self.out_var.shape[self.axis] == -1)
        exe = fluid.Executor(self.place)
        res = exe.run(self.program, fetch_list=self.out_var)
        self.assertTrue(
            np.array_equal(
                res[0],
                np.concatenate(
                    [self.x] * self.iter_num, axis=self.axis)))


if __name__ == '__main__':
    unittest.main()
