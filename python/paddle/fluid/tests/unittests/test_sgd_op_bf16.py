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
import numpy as np
import paddle.fluid as fluid
import paddle.fluid.core as core
from paddle.fluid.op import Operator
from paddle.fluid.tests.unittests.op_test import (
    OpTest, convert_float_to_uint16, convert_uint16_to_float)
import paddle


@unittest.skipIf(not core.supports_bfloat16(),
                 'place does not support BF16 evaluation')
class TestSGDOpBF16(OpTest):
    def setUp(self):
        self.op_type = 'sgd'
        self.dtype = np.uint16
        self.conf()
        w = np.random.random((self.h, self.w)).astype('float32')
        w_bf16 = convert_float_to_uint16(w)
        g = np.random.random((self.h, self.w)).astype('float32')
        g_bf16 = convert_float_to_uint16(g)
        lr = np.array([0.1]).astype('float32')
        lr_bf16 = convert_float_to_uint16(lr)

        self.inputs = {'Param': w_bf16, 'Grad': g_bf16, 'LearningRate': lr_bf16}
        self.outputs = {'ParamOut': w - lr * g}

    def conf(self):
        self.h = 102
        self.w = 105

    def test_check_output(self):
        self.check_output_with_place(core.CPUPlace(), check_dygraph=False)


@unittest.skipIf(not core.supports_bfloat16(),
                 'place does not support BF16 evaluation')
class TestSGDOpCase8XBF16(TestSGDOpBF16):
    def conf(self):
        self.h = 10
        self.w = 64


class TestSparseSGDOpBF16(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        np.random.seed(12345)

    def ref_optimize(self, params, grad_rows, grad_array, lr_value):
        reference = np.copy(params)
        for index, id in enumerate(grad_rows):
            reference[id] = params[id] - lr_value * grad_array[index]
        return reference

    def check_output(self, actual_bf16, reference, atol=0, rtol=0.15e-2):
        actual_fp32 = convert_uint16_to_float(actual_bf16)
        np.testing.assert_allclose(actual_fp32, reference, atol=atol, rtol=rtol)

    def create_sparse_grad_var(self, scope, place, height, rows, row_numel):
        grad_selected_rows = scope.var('Grad').get_selected_rows()
        grad_selected_rows.set_height(height)
        grad_selected_rows.set_rows(rows)
        grad_array = np.random.random((len(rows), row_numel)).astype('float32')
        np_array_bf16 = convert_float_to_uint16(grad_array)

        grad_tensor = grad_selected_rows.get_tensor()
        grad_tensor.set(np_array_bf16, place)

        return grad_tensor, grad_array

    def create_dense_param_var(self, scope, place, height, width):
        param_tensor = scope.var('Param').get_tensor()
        param_array = np.random.random((height, width)).astype('float32')
        param_array_bf16 = convert_float_to_uint16(param_array)
        param_tensor.set(param_array_bf16, place)

        return param_tensor, param_array

    def create_sparse_param_var(self, scope, place, height, rows, row_numel):
        param_selected_rows = scope.var('Param').get_selected_rows()
        param_selected_rows.set_height(height)
        param_selected_rows.set_rows(rows)
        param_selected_rows.sync_index()
        param_array = np.random.random((len(rows), row_numel)).astype('float32')
        np_array_bf16 = convert_float_to_uint16(param_array)

        param_tensor = param_selected_rows.get_tensor()
        param_tensor.set(np_array_bf16, place)

        return param_tensor, param_array

    def create_dense_lr_var(self, scope, place):
        lr_tensor = scope.var('LearningRate').get_tensor()
        lr_value = np.random.uniform()
        lr_array = np.full((1), lr_value, np.float32)
        lr_array_bf16 = convert_float_to_uint16(lr_array)
        lr_tensor.set(lr_array_bf16, place)

        return lr_tensor, lr_value


@unittest.skipIf(not core.supports_bfloat16(),
                 'place does not support BF16 evaluation')
class TestSparseGradSGDOpBF16(TestSparseSGDOpBF16):
    def setUp(self):
        self.setup_params()

    def setup_params(self):
        self.grad_height = 10
        self.grad_rows = [0, 4, 7]
        self.grad_row_numel = 12

    def test_sparse_grad_sgd(self):
        scope = core.Scope()
        place = core.CPUPlace()
        _, grad_array = self.create_sparse_grad_var(
            scope, place, self.grad_height, self.grad_rows, self.grad_row_numel)
        param_tensor, param_array = self.create_dense_param_var(
            scope, place, self.grad_height, self.grad_row_numel)
        _, lr_value = self.create_dense_lr_var(scope, place)

        sgd_op = Operator(
            'sgd',
            Param='Param',
            Grad='Grad',
            ParamOut='Param',
            LearningRate='LearningRate')
        sgd_op.run(scope, place)

        reference = self.ref_optimize(param_array, self.grad_rows, grad_array,
                                      lr_value)
        output = np.array(param_tensor)
        self.check_output(output, reference, atol=5e-3, rtol=1e-1)


@unittest.skipIf(not core.supports_bfloat16(),
                 'place does not support BF16 evaluation')
class TestSparseGradSGDOpBF16Case2(TestSparseGradSGDOpBF16):
    def setup_params(self):
        self.grad_height = 14
        self.grad_rows = [1, 4, 12, 7, 8]
        self.grad_row_numel = 16


class TestSparseGradSGDOpBF16Case3(TestSparseGradSGDOpBF16):
    def setup_params(self):
        self.grad_height = 10
        self.grad_rows = [0, 4, 7]
        self.grad_row_numel = 120


@unittest.skipIf(not core.supports_bfloat16(),
                 'place does not support BF16 evaluation')
class TestSparseGradParamSGDOpBF16(TestSparseSGDOpBF16):
    def setUp(self):
        self.setup_params()

    def setup_params(self):
        self.grad_height = 10
        self.grad_rows = [0, 4, 7]
        self.grad_row_numel = 12
        self.param_rows = [a for a in range(self.grad_height)]

    def test_sparse_param_grad_sgd(self):
        scope = core.Scope()
        place = core.CPUPlace()
        _, grad_array = self.create_sparse_grad_var(
            scope, place, self.grad_height, self.grad_rows, self.grad_row_numel)
        param_tensor, param_array = self.create_sparse_param_var(
            scope, place, self.grad_height, self.param_rows,
            self.grad_row_numel)
        _, lr_value = self.create_dense_lr_var(scope, place)

        sgd_op = Operator(
            'sgd',
            Param='Param',
            Grad='Grad',
            ParamOut='Param',
            LearningRate='LearningRate')
        sgd_op.run(scope, place)

        reference = self.ref_optimize(param_array, self.grad_rows, grad_array,
                                      lr_value)
        output = np.array(param_tensor)
        self.check_output(output, reference, atol=5e-3, rtol=1e-1)


class TestSparseGradParamSGDOpBF16Case2(TestSparseGradParamSGDOpBF16):
    def setup_params(self):
        self.grad_height = 14
        self.grad_rows = [1, 4, 12, 7, 8]
        self.grad_row_numel = 16
        self.param_rows = [a for a in range(self.grad_height)]


if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()
