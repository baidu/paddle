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
from op_test import OpTest
import paddle
import paddle.fluid as fluid

paddle.enable_static()
SEED = 2021


@unittest.skipIf(not paddle.is_compiled_with_npu(),
                 "core is not compiled with NPU")
class TestConcat(OpTest):
    def setUp(self):
        self.set_npu()
        self.op_type = "concat"
        self.place = paddle.NPUPlace(0)
        self.init_dtype()
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

    def set_npu(self):
        self.__class__.use_npu = True

    def init_dtype(self):
        self.dtype = np.float32

    def test_check_output(self):
        self.check_output_with_place(self.place, check_dygraph=False)

    def init_test_data(self):
        self.x0 = np.random.random((1, 4, 50)).astype(self.dtype)
        self.x1 = np.random.random((2, 4, 50)).astype(self.dtype)
        self.x2 = np.random.random((3, 4, 50)).astype(self.dtype)
        self.axis = 0

    def test_check_grad(self):
        self.check_grad_with_place(
            self.place, ['x0', 'x2'], 'Out', check_dygraph=False)
        self.check_grad_with_place(
            self.place, ['x1'], 'Out', check_dygraph=False)
        self.check_grad_with_place(
            self.place, ['x2'], 'Out', check_dygraph=False)


class TestConcatFP16(OpTest):
    def setUp(self):
        self.set_npu()
        self.op_type = "concat"
        self.place = paddle.NPUPlace(0)
        self.init_dtype()
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

    def set_npu(self):
        self.__class__.use_npu = True
        self.__class__.no_need_check_grad = True

    def init_dtype(self):
        self.dtype = np.float16

    def test_check_output(self):
        self.check_output_with_place(self.place, check_dygraph=False)

    def init_test_data(self):
        self.x0 = np.random.random((1, 4, 50)).astype(self.dtype)
        self.x1 = np.random.random((2, 4, 50)).astype(self.dtype)
        self.x2 = np.random.random((3, 4, 50)).astype(self.dtype)
        self.axis = 0


if __name__ == '__main__':
    unittest.main()
