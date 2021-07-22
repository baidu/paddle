#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
from functools import reduce
import numpy as np

from paddle.fluid.tests.unittests.op_test import OpTest, OpTestTool, convert_float_to_uint16
import paddle.fluid.core as core
import paddle
import paddle.fluid as fluid
import paddle.fluid.framework as framework


def reference_matmul(X, Y, trans_x=False, trans_y=False):
    """Reference forward implementation using np.matmul."""
    # np.matmul does not support the transpose flags, so we manually
    # transpose X and Y appropriately.
    if trans_x:
        if X.ndim == 1:
            X = X.reshape((X.size, ))
        elif X.ndim == 2:
            X = X.T
        else:
            dim = [i for i in range(len(X.shape))]
            dim[-1], dim[len(X.shape) - 2] = dim[len(X.shape) - 2], dim[-1]
            X = np.transpose(X, tuple(dim))
    if trans_y:
        if Y.ndim == 1:
            Y = Y.reshape((Y.size, ))
        else:
            dim = [i for i in range(len(Y.shape))]
            dim[-1], dim[len(Y.shape) - 2] = dim[len(Y.shape) - 2], dim[-1]
            Y = np.transpose(Y, tuple(dim))

    Out = np.atleast_1d(np.matmul(X, Y))
    return Out


class TestMatMulV2VectorXVectorOneDNNOp(OpTest):
    def config(self):
        self.x_shape = (100, )
        self.y_shape = (100, )
        self.trans_x = False
        self.trans_y = False

    def set_inputs(self, x, y):
        self.inputs = {'X': x, 'Y': y}

    def set_dtype_attr(self):
        self.attrs['mkldnn_data_type'] = "float32"

    def setUp(self):
        self.config()
        self.op_type = "matmul_v2"
        x = np.random.random(self.x_shape).astype("float32")
        y = np.random.random(self.y_shape).astype("float32")
        # -0.1 ~ 0.1
        x = -0.1 + 0.2 * x
        y = -0.1 + 0.2 * y
        result = reference_matmul(x, y, self.trans_x,
                                  self.trans_y).astype("float32")

        self.set_inputs(x, y)
        self.attrs = {
            'trans_x': self.trans_x,
            'trans_y': self.trans_y,
            'use_mkldnn': True
        }
        self.set_dtype_attr()
        self.outputs = {'Out': result}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X', 'Y'], 'Out')


class TestMatMulV2VectorXMatrixTransposeYOneDNNOp(
        TestMatMulV2VectorXVectorOneDNNOp):
    def config(self):
        self.x_shape = (100, )
        self.y_shape = (1, 3, 2, 100)
        self.trans_x = False
        self.trans_y = True


class TestMatMulV2VectorXMatrixOneDNNOp(TestMatMulV2VectorXVectorOneDNNOp):
    def config(self):
        self.x_shape = (100, )
        self.y_shape = (1, 1, 100, 2)
        self.trans_x = False
        self.trans_y = False


class TestMatMulV2MatrixXVectorTransposeXOneDNNOp(
        TestMatMulV2VectorXVectorOneDNNOp):
    def config(self):
        self.x_shape = (1, 1, 100, 1)
        self.y_shape = (100, )
        self.trans_x = True
        self.trans_y = False


class TestMatMulV2MatrixXVectorOneDNNOp(TestMatMulV2VectorXVectorOneDNNOp):
    def config(self):
        self.x_shape = (1, 2, 1, 100)
        self.y_shape = (100, )
        self.trans_x = False
        self.trans_y = False


class TestMatMulV2MatrixXMatrixOneDNNOp(TestMatMulV2VectorXVectorOneDNNOp):
    def config(self):
        self.x_shape = (1, 1, 2, 100)
        self.y_shape = (1, 1, 100, 1)
        self.trans_x = False
        self.trans_y = False


class TestMatMulV2MatrixXMatrixTransposeYOneDNNOp(
        TestMatMulV2VectorXVectorOneDNNOp):
    def config(self):
        self.x_shape = (1, 1, 1, 100)
        self.y_shape = (2, 1, 2, 100)
        self.trans_x = False
        self.trans_y = True


class TestMatMulV2MatrixXMatrix2OneDNNOp(TestMatMulV2VectorXVectorOneDNNOp):
    def config(self):
        self.x_shape = (2, 1, 12, 9)
        self.y_shape = (1, 3, 9, 12)
        self.trans_x = False
        self.trans_y = False


class TestMatMulV2MatrixXMatrix3OneDNNOp(TestMatMulV2VectorXVectorOneDNNOp):
    def config(self):
        self.x_shape = (2, 1, 2, 100)
        self.y_shape = (1, 1, 100, 2)
        self.trans_x = False
        self.trans_y = False


class TestMatMulV2MatrixXMatrixTranposeXOneDNNOp2(
        TestMatMulV2VectorXVectorOneDNNOp):
    def config(self):
        self.x_shape = (2, 1, 4, 25)
        self.y_shape = (1, 1, 4, 25)
        self.trans_x = True
        self.trans_y = False


class TestMatMulV2MatrixXMatrixTranposeX2OneDNNOp3(
        TestMatMulV2VectorXVectorOneDNNOp):
    def config(self):
        self.x_shape = (2, 2, 7, 4)
        self.y_shape = (2, 2, 7, 5)
        self.trans_x = True
        self.trans_y = False


class TestMatMulV2MatrixXMatrixTransposeX3OneDNNOp(
        TestMatMulV2VectorXVectorOneDNNOp):
    def config(self):
        self.x_shape = (3, 1, 6, 7)
        self.y_shape = (1, 2, 6, 9)
        self.trans_x = True
        self.trans_y = False


class TestMatMulV2MatrixXMatrix4OneDNNOp(TestMatMulV2VectorXVectorOneDNNOp):
    def config(self):
        self.x_shape = (3, 1, 6, 6)
        self.y_shape = (1, 2, 6, 9)
        self.trans_x = False
        self.trans_y = False


class TestMatMulV2VectorXMatrix5DOneDNNOp(TestMatMulV2VectorXVectorOneDNNOp):
    def config(self):
        self.x_shape = (100)
        self.y_shape = (1, 2, 2, 100, 2)
        self.trans_x = False
        self.trans_y = False


class TestMatMulV2Matrix3DXVectorOneDNNOp(TestMatMulV2VectorXVectorOneDNNOp):
    def config(self):
        self.x_shape = (2, 1, 100)
        self.y_shape = (100)
        self.trans_x = False
        self.trans_y = False


class TestMatMulV2MatrixXMatrixTransposeXTransposeYOneDNNOp(
        TestMatMulV2VectorXVectorOneDNNOp):
    def config(self):
        self.x_shape = (3, 1, 10, 8)
        self.y_shape = (1, 2, 9, 10)
        self.trans_x = True
        self.trans_y = True


class TestMatMulV2MatrixXMatrixTransposeY2OneDNNOp(
        TestMatMulV2VectorXVectorOneDNNOp):
    def config(self):
        self.x_shape = (3, 1, 10, 10)
        self.y_shape = (1, 2, 9, 10)
        self.trans_x = False
        self.trans_y = True


class TestMatMulV2MatrixXMatrix5DTranposeYOneDNNOp(
        TestMatMulV2VectorXVectorOneDNNOp):
    def config(self):
        self.x_shape = (1, 3, 1, 10, 10)
        self.y_shape = (3, 1, 2, 9, 10)
        self.trans_x = False
        self.trans_y = True


#   BF16 TESTS
def create_bf16_test_class(parent):
    @OpTestTool.skip_if_not_cpu_bf16()
    class TestMatMulV2Bf16OneDNNOp(parent):
        def set_inputs(self, x, y):
            self.inputs = {
                'X': convert_float_to_uint16(x),
                'Y': convert_float_to_uint16(y)
            }
            self.x_fp32 = x
            self.y_fp32 = y

        def set_dtype_attr(self):
            self.attrs['mkldnn_data_type'] = "bfloat16"

        def test_check_output(self):
            self.check_output_with_place(core.CPUPlace())

        def test_check_grad(self):
            self.calculate_grads()
            self.check_grad_with_place(
                core.CPUPlace(), ["X", "Y"],
                "Out",
                user_defined_grads=[self.dx, self.dy],
                user_defined_grad_outputs=[convert_float_to_uint16(self.dout)])

        def matmul_grad(self, x, transpose_x, y, transpose_y):
            x = np.transpose(
                x, self.shape_transpose_axes[x.ndim]) if transpose_x else x
            y = np.transpose(
                y, self.shape_transpose_axes[y.ndim]) if transpose_y else y

            return np.matmul(x, y)

        def calculate_grads(self):
            self.shape_transpose_axes = {
                2: [1, 0],
                3: [0, 2, 1],
                4: [0, 1, 3, 2],
                5: [0, 1, 2, 4, 3]
            }

            # expand vector so it will be a valid matrix for multiplication
            if self.x_fp32.ndim == 1:
                self.x_fp32 = np.expand_dims(self.x_fp32, axis=0)
            if self.y_fp32.ndim == 1:
                self.y_fp32 = np.expand_dims(self.y_fp32, axis=1)

            x_transpose_axes = self.shape_transpose_axes[self.x_fp32.ndim]
            y_transpose_axes = self.shape_transpose_axes[self.y_fp32.ndim]

            x = np.transpose(self.x_fp32, x_transpose_axes) if self.attrs[
                'trans_x'] is True else self.x_fp32
            y = np.transpose(self.y_fp32, y_transpose_axes) if self.attrs[
                'trans_y'] is True else self.y_fp32

            dout = np.matmul(x, y)

            x_shape = x.shape
            y_shape = y.shape

            if x.ndim <= 2 or y.ndim <= 2:
                is_broadcast = False
            elif x.ndim != y.ndim:
                is_broadcast = True
            else:
                is_broadcast = x.shape[0:-2] != y.shape[0:-2]

            if self.attrs['trans_x'] is True and self.attrs['trans_y'] is True:
                self.dx = self.matmul_grad(self.y_fp32, True, dout, True)
                self.dy = self.matmul_grad(dout, True, self.x_fp32, True)
            elif self.attrs['trans_x'] is True and self.attrs[
                    'trans_y'] is False:
                self.dx = self.matmul_grad(self.y_fp32, False, dout, True)
                self.dy = self.matmul_grad(self.x_fp32, False, dout, False)
            elif self.attrs['trans_x'] is False and self.attrs[
                    'trans_y'] is True:
                self.dx = self.matmul_grad(dout, False, self.y_fp32, False)
                self.dy = self.matmul_grad(dout, True, self.x_fp32, False)
            else:
                self.dx = self.matmul_grad(dout, False, self.y_fp32, True)
                self.dy = self.matmul_grad(self.x_fp32, True, dout, False)

            if is_broadcast:
                x_reduce_axis = []
                y_reduce_axis = []
                for index, (
                        first, second
                ) in enumerate(zip(x_shape[0:-2], self.dx.shape[0:-2])):
                    if first != second:
                        x_reduce_axis.append(index)

                for index, (
                        first, second
                ) in enumerate(zip(y_shape[0:-2], self.dy.shape[0:-2])):
                    if first != second:
                        y_reduce_axis.append(index)

                if x_reduce_axis:
                    self.dx = self.dx.sum(axis=tuple(x_reduce_axis),
                                          keepdims=True)
                if y_reduce_axis:
                    self.dy = self.dy.sum(axis=tuple(y_reduce_axis),
                                          keepdims=True)

            # after multiplying with vector one dimension is deleted from tensor
            if len(x_shape) == 2 and x_shape[0] == 1:
                dout = dout.sum(axis=-2)
            if len(y_shape) == 2 and y_shape[1] == 1:
                dout = dout.sum(axis=-1)

            self.dout = dout

    cls_name = "{0}_{1}".format(parent.__name__, "BF16")
    TestMatMulV2Bf16OneDNNOp.__name__ = cls_name
    globals()[cls_name] = TestMatMulV2Bf16OneDNNOp


create_bf16_test_class(TestMatMulV2VectorXMatrixTransposeYOneDNNOp)
create_bf16_test_class(TestMatMulV2VectorXMatrixOneDNNOp)
create_bf16_test_class(TestMatMulV2MatrixXVectorTransposeXOneDNNOp)
create_bf16_test_class(TestMatMulV2MatrixXVectorOneDNNOp)
create_bf16_test_class(TestMatMulV2MatrixXMatrixOneDNNOp)
create_bf16_test_class(TestMatMulV2MatrixXMatrixTransposeYOneDNNOp)
create_bf16_test_class(TestMatMulV2MatrixXMatrix2OneDNNOp)
create_bf16_test_class(TestMatMulV2MatrixXMatrix3OneDNNOp)
create_bf16_test_class(TestMatMulV2MatrixXMatrixTranposeXOneDNNOp2)
create_bf16_test_class(TestMatMulV2MatrixXMatrixTranposeX2OneDNNOp3)
create_bf16_test_class(TestMatMulV2MatrixXMatrixTransposeX3OneDNNOp)
create_bf16_test_class(TestMatMulV2MatrixXMatrix4OneDNNOp)
create_bf16_test_class(TestMatMulV2VectorXMatrix5DOneDNNOp)
create_bf16_test_class(TestMatMulV2Matrix3DXVectorOneDNNOp)
create_bf16_test_class(TestMatMulV2MatrixXMatrixTransposeXTransposeYOneDNNOp)
create_bf16_test_class(TestMatMulV2MatrixXMatrixTransposeY2OneDNNOp)
create_bf16_test_class(TestMatMulV2MatrixXMatrix5DTranposeYOneDNNOp)

if __name__ == "__main__":
    paddle.enable_static()
    unittest.main()
