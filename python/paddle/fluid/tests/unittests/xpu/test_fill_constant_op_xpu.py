#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import sys
sys.path.append("..")
import unittest
from op_test import OpTest

import paddle
import numpy as np


# Situation 1: Attr(shape) is a list(without tensor)
@unittest.skipIf(not paddle.is_compiled_with_xpu(),
                 "core is not compiled with XPU")
class TestFillConstantOp1(OpTest):
    def setUp(self):
        '''Test fill_constant op with specified value'''
        self.op_type = "fill_constant"

        self.inputs = {}
        self.attrs = {'shape': [123, 92], 'dtype': 5, 'value': 3.8}
        self.outputs = {'Out': np.full((123, 92), 3.8)}

    def test_check_output(self):
        place = paddle.XPUPlace(0)
        self.check_output_with_place(place)


@unittest.skipIf(not paddle.is_compiled_with_xpu(),
                 "core is not compiled with XPU")
class TestFillConstantOp2(OpTest):
    def setUp(self):
        '''Test fill_constant op with default value'''
        self.op_type = "fill_constant"

        self.inputs = {}
        self.attrs = {'shape': [123, 92], 'dtype': 5}
        self.outputs = {'Out': np.full((123, 92), 0.0)}

    def test_check_output(self):
        place = paddle.XPUPlace(0)
        self.check_output_with_place(place)


@unittest.skipIf(not paddle.is_compiled_with_xpu(),
                 "core is not compiled with XPU")
class TestFillConstantOp3(OpTest):
    def setUp(self):
        '''Test fill_constant op with specified int64 value'''
        self.op_type = "fill_constant"

        self.inputs = {}
        self.attrs = {'shape': [123, 92], 'dtype': 3, 'value': 10000000000}
        self.outputs = {'Out': np.full((123, 92), 10000000000)}

    def test_check_output(self):
        place = paddle.XPUPlace(0)
        self.check_output_with_place(place)


@unittest.skipIf(not paddle.is_compiled_with_xpu(),
                 "core is not compiled with XPU")
class TestFillConstantOp4(OpTest):
    def setUp(self):
        '''Test fill_constant op with specified int value'''
        self.op_type = "fill_constant"

        self.inputs = {}
        self.attrs = {'shape': [123, 92], 'dtype': 2, 'value': 3}
        self.outputs = {'Out': np.full((123, 92), 3)}

    def test_check_output(self):
        place = paddle.XPUPlace(0)
        self.check_output_with_place(place)


# Situation 2: Attr(shape) is a list(with tensor)
@unittest.skipIf(not paddle.is_compiled_with_xpu(),
                 "core is not compiled with XPU")
class TestFillConstantOp1_ShapeTensorList(OpTest):
    def setUp(self):
        '''Test fill_constant op with specified value'''
        self.op_type = "fill_constant"
        self.init_data()
        shape_tensor_list = []
        for index, ele in enumerate(self.shape):
            shape_tensor_list.append(("x" + str(index), np.ones(
                (1)).astype('int32') * ele))

        self.inputs = {"ShapeTensorList": shape_tensor_list}
        self.attrs = {
            'shape': self.infer_shape,
            'dtype': 5,
            'value': self.value
        }
        self.outputs = {'Out': np.full(self.shape, self.value)}

    def init_data(self):
        self.shape = [123, 92]
        self.infer_shape = [-1, 92]
        self.value = 3.8

    def test_check_output(self):
        place = paddle.XPUPlace(0)
        self.check_output_with_place(place)


@unittest.skipIf(not paddle.is_compiled_with_xpu(),
                 "core is not compiled with XPU")
class TestFillConstantOp2_ShapeTensorList(OpTest):
    def setUp(self):
        '''Test fill_constant op with default value'''
        self.op_type = "fill_constant"
        self.init_data()
        shape_tensor_list = []
        for index, ele in enumerate(self.shape):
            shape_tensor_list.append(("x" + str(index), np.ones(
                (1)).astype('int32') * ele))

        self.inputs = {"ShapeTensorList": shape_tensor_list}
        self.attrs = {'shape': self.infer_shape, 'dtype': 5}
        self.outputs = {'Out': np.full(self.shape, 0.0)}

    def init_data(self):
        self.shape = [123, 92]
        self.infer_shape = [-1, -1]

    def test_check_output(self):
        place = paddle.XPUPlace(0)
        self.check_output_with_place(place)


@unittest.skipIf(not paddle.is_compiled_with_xpu(),
                 "core is not compiled with XPU")
class TestFillConstantOp3_ShapeTensorList(TestFillConstantOp1_ShapeTensorList):
    def init_data(self):
        self.shape = [123, 92]
        self.infer_shape = [123, -1]
        self.value = 10000000000


@unittest.skipIf(not paddle.is_compiled_with_xpu(),
                 "core is not compiled with XPU")
class TestFillConstantOp4_ShapeTensorList(TestFillConstantOp1_ShapeTensorList):
    def init_data(self):
        self.shape = [123, 92]
        self.infer_shape = [123, -1]
        self.value = 3


# Situation 3: shape is a tensor
@unittest.skipIf(not paddle.is_compiled_with_xpu(),
                 "core is not compiled with XPU")
class TestFillConstantOp1_ShapeTensor(OpTest):
    def setUp(self):
        '''Test fill_constant op with specified value'''
        self.op_type = "fill_constant"
        self.init_data()

        self.inputs = {"ShapeTensor": np.array(self.shape).astype("int32")}
        self.attrs = {'value': self.value, 'dtype': 5}
        self.outputs = {'Out': np.full(self.shape, self.value)}

    def init_data(self):
        self.shape = [123, 92]
        self.value = 3.8

    def test_check_output(self):
        place = paddle.XPUPlace(0)
        self.check_output_with_place(place)


# Situation 4: value is a tensor
@unittest.skipIf(not paddle.is_compiled_with_xpu(),
                 "core is not compiled with XPU")
class TestFillConstantOp1_ValueTensor(OpTest):
    def setUp(self):
        '''Test fill_constant op with specified value'''
        self.op_type = "fill_constant"
        self.init_data()

        self.inputs = {
            "ShapeTensor": np.array(self.shape).astype("int32"),
            'ValueTensor': np.array([self.value]).astype("float32")
        }
        self.attrs = {'value': self.value + 1.0, 'dtype': 5}
        self.outputs = {'Out': np.full(self.shape, self.value)}

    def init_data(self):
        self.shape = [123, 92]
        self.value = 3.8
        self.dtype = np.float32

    def test_check_output(self):
        place = paddle.XPUPlace(0)
        self.check_output_with_place(place)


# Situation 5: value is a tensor
@unittest.skipIf(not paddle.is_compiled_with_xpu(),
                 "core is not compiled with XPU")
class TestFillConstantOp2_ValueTensor(OpTest):
    def setUp(self):
        '''Test fill_constant op with specified value'''
        self.op_type = "fill_constant"
        self.init_data()

        self.inputs = {
            "ShapeTensor": np.array(self.shape).astype("int32"),
            'ValueTensor': np.array([self.value]).astype("int32")
        }
        self.attrs = {'value': self.value, 'dtype': 2}
        self.outputs = {'Out': np.full(self.shape, self.value)}

    def init_data(self):
        self.shape = [123, 92]
        self.value = 3
        self.dtype = np.int32

    def test_check_output(self):
        place = paddle.XPUPlace(0)
        self.check_output_with_place(place)


if __name__ == "__main__":
    paddle.enable_static()
    unittest.main()
