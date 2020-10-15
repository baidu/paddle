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
from paddle.fluid.tests.unittests.op_test import OpTest, convert_float_to_uint16


class TestDeQuantizeOp(OpTest):
    def setUp(self):
        self.op_type = 'dequantize'
        self.scale = 127.0
        self.shift = 0.0
        self.input_size = [1, 1, 5, 5]  # Naive nChw16c
        self.data_type = 'int8'
        self.set_scale()
        self.set_shift()
        self.set_data_type()
        self.set_input_size()
        if self.data_type == 'uint16':
            self.prepare_input_output_bf16()
        else:
            self.prepare_input_int8()
            self.prepare_output_int8()

    def prepare_input_output_bf16(self):
        output = np.random.random(self.input_size).astype(np.float32)
        input = convert_float_to_uint16(output)
        self.inputs = {'Input': OpTest.np_dtype_to_fluid_dtype(input)}
        self.outputs = {'Output': output}

    def prepare_input_int8(self):
        if self.data_type == 'int8':
            # input data values are integers from interval [-128, 128)
            self.input = (np.random.randint(0, 256, self.input_size) - 128
                          ).astype(self.data_type)
        else:
            # input data values are integers from interval [0, 256)
            self.input = (np.random.randint(
                0, 256, self.input_size)).astype(self.data_type)

        self.inputs = {'Input': OpTest.np_dtype_to_fluid_dtype(self.input)}
        self.attrs = {'Scale': self.scale, 'Shift': self.shift}

    def prepare_output_int8(self):
        output = (self.input / self.scale -
                  (self.shift / self.scale)).astype('float')
        self.outputs = {'Output': output}

    def test_check_output(self):
        # TODO(wangzhongpu): support mkldnn op in dygraph mode
        self.check_output(check_dygraph=False)

    def check_raise_error(self, msg):
        try:
            self.check_output()
        except Exception as e:
            if msg in str(e):
                raise AttributeError
            else:
                print(e)

    def set_scale(self):
        pass

    def set_shift(self):
        pass

    def set_data_type(OpTest):
        pass

    def set_input_size(self):
        pass


class TestDeQuantizeOp1(TestDeQuantizeOp):
    def set_scale(self):
        self.scale = 1.5

    def set_data_type(self):
        self.data_type = 'int8'


class TestDeQuantizeOp2(TestDeQuantizeOp):
    def set_scale(self):
        self.scale = 0.8

    def set_data_type(self):
        self.data_type = 'uint8'


class TestDeQuantizeOpBf16(TestDeQuantizeOp):
    def set_scale(self):
        self.scale = 1.0

    def set_data_type(self):
        self.data_type = 'uint16'


class TestDeQuantizeOp_ZeroScale(TestDeQuantizeOp):
    def set_scale(self):
        self.scale = 0.0

    def prepare_output_int8(self):
        self.output = np.zeros(self.input_size)
        self.outputs = {'Output': self.output}

    def test_check_output(self):
        self.assertRaises(AttributeError, self.check_raise_error,
                          'Dequantization scale cannot be 0.0')


# 2-dim input
# P - positive input, with shift
class TestDeQuantizeOpShift_2_P(TestDeQuantizeOp):
    def set_data_type(self):
        self.data_type = 'uint8'

    def set_scale(self):
        self.scale = 255.0

    def set_shift(self):
        self.shift = 128.0

    def set_input_size(self):
        self.input_size = [2, 3]


# 2-dim input
# N - negative input, with shift
class TestDeQuantizeOpShift_2_N(TestDeQuantizeOpShift_2_P):
    def set_data_type(self):
        self.data_type = 'int8'

    def set_scale(self):
        self.scale = 127.0

    def set_shift(self):
        self.shift = 10.0

    def set_input_size(self):
        self.input_size = [2, 3]


# 3-dim input
class TestDeQuantizeOpShift_3_P(TestDeQuantizeOpShift_2_P):
    def set_input_size(self):
        self.input_size = [2, 3, 4]


class TestDeQuantizeOpShift_3_N(TestDeQuantizeOpShift_2_N):
    def set_input_size(self):
        self.input_size = [2, 3, 4]


# 4-dim input
class TestDeQuantizeOpShift_4_P(TestDeQuantizeOpShift_2_P):
    def set_input_size(self):
        self.input_size = [2, 3, 4, 5]


class TestDeQuantizeOpShift_4_N(TestDeQuantizeOpShift_2_N):
    def set_input_size(self):
        self.input_size = [2, 3, 4, 5]


class TestDeQuantizeOp_NegativeShift(TestDeQuantizeOp):
    def set_shift(self):
        self.shift = -10.0

    def prepare_output_int8(self):
        self.output = np.zeros(self.input_size)
        self.outputs = {'Output': self.output}

    def test_check_output(self):
        self.assertRaises(AttributeError, self.check_raise_error,
                          'Dequantization shift must be nonnegative.')


class TestDeQuantizeOp_TooBigShift(TestDeQuantizeOp_NegativeShift):
    def set_shift(self):
        self.shift = 300.0

    def test_check_output(self):
        self.assertRaises(
            AttributeError, self.check_raise_error,
            'Dequantization shift must be less than or equal to 255.')


if __name__ == '__main__':
    unittest.main()
