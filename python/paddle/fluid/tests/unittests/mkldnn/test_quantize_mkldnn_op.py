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
from paddle.fluid.tests.unittests.op_test import OpTest


class TestQuantizeOp(OpTest):
    def setUp(self):
        self.op_type = 'quantize'
        self.scale = 255.0
        self.shift = 0.0
        self.input_size = [1, 1, 5, 5]  # Naive nChw16c
        self.is_negative = False
        self.output_format = 'NCHW'
        self.set_scale()
        self.set_shift()
        self.set_is_negative()
        self.set_input_size()
        self.set_output_format()
        self.prepare_input()
        self.prepare_output()

    def prepare_input(self):
        if self.is_negative:
            # input data values are from interval [-1.0, 1.0)
            self.input = (2 * np.random.random_sample(self.input_size) - 1
                          ).astype('float32')
        else:
            # input data values are from interval [0.0, 1.0)
            self.input = (
                np.random.random_sample(self.input_size)).astype('float32')

        self.inputs = {'Input': OpTest.np_dtype_to_fluid_dtype(self.input)}
        self.attrs = {
            'Scale': self.scale,
            'Shift': self.shift,
            'is_negative_input': self.is_negative,
            'output_format': self.output_format
        }

    def prepare_output(self):
        input_data_type = 'int8' if self.is_negative else 'uint8'
        output = np.rint(self.input * self.scale + self.shift).astype(
            input_data_type)
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

    def set_is_negative(self):
        pass

    def set_input_size(self):
        pass

    def set_output_format(self):
        pass


class TestQuantizeOp1(TestQuantizeOp):
    def set_scale(self):
        self.scale = 127.0

    def set_is_negative(self):
        self.is_nagative = True


class TestQuantizeOp2(TestQuantizeOp):
    def set_scale(self):
        self.scale = 255.0

    def set_is_negative(self):
        self.is_nagative = False


class TestQuantizeOp_ZeroScale(TestQuantizeOp):
    def set_scale(self):
        self.scale = 0.0

    def prepare_output(self):
        self.output = np.zeros(self.input_size)
        self.outputs = {'Output': self.output}

    def test_check_output(self):
        self.assertRaises(AttributeError, self.check_raise_error,
                          'Quantization scale cannot be 0.0')


# 2-dim input
# P - positive input
class TestQuantizeOpShift_NCHW_2_P(TestQuantizeOp):
    def set_output_format(self):
        self.output_format = 'NCHW'

    def set_is_negative(self):
        self.is_nagative = False

    def set_scale(self):
        self.scale = 255.0

    def set_shift(self):
        self.shift = 0.0

    def set_input_size(self):
        self.input_size = [2, 3]


# 2-dim input
# N - negative input
class TestQuantizeOpShift_NCHW_2_N(TestQuantizeOpShift_NCHW_2_P):
    def set_is_negative(self):
        self.is_nagative = True

    def set_scale(self):
        self.scale = 127.0

    def set_shift(self):
        self.shift = 128.0


class TestQuantizeOpShift_NHWC_2_P(TestQuantizeOpShift_NCHW_2_P):
    def set_output_format(self):
        self.output_format = 'NHWC'


class TestQuantizeOpShift_NHWC_2_N(TestQuantizeOpShift_NCHW_2_N):
    def set_output_format(self):
        self.output_format = 'NHWC'


# 3-dim input
class TestQuantizeOpShift_NCHW_3_P(TestQuantizeOpShift_NCHW_2_P):
    def set_input_size(self):
        self.input_size = [2, 3, 4]


class TestQuantizeOpShift_NCHW_3_N(TestQuantizeOpShift_NCHW_2_N):
    def set_input_size(self):
        self.input_size = [2, 3, 4]


class TestQuantizeOpShift_NHWC_3_P(TestQuantizeOpShift_NCHW_3_P):
    def set_output_format(self):
        self.output_format = 'NHWC'


class TestQuantizeOpShift_NHWC_3_N(TestQuantizeOpShift_NCHW_3_N):
    def set_output_format(self):
        self.output_format = 'NHWC'


# 4-dim input
class TestQuantizeOpShift_NCHW_4_P(TestQuantizeOpShift_NCHW_2_P):
    def set_input_size(self):
        self.input_size = [2, 3, 4, 5]


class TestQuantizeOpShift_NCHW_4_N(TestQuantizeOpShift_NCHW_2_N):
    def set_input_size(self):
        self.input_size = [2, 3, 4, 5]


class TestQuantizeOpShift_NHWC_4_P(TestQuantizeOpShift_NCHW_4_P):
    def set_output_format(self):
        self.output_format = 'NHWC'


class TestQuantizeOpShift_NHWC_4_N(TestQuantizeOpShift_NCHW_4_N):
    def set_output_format(self):
        self.output_format = 'NHWC'


class TestQuantizeOp_NegativeShift(TestQuantizeOp):
    def set_is_negative(self):
        self.is_nagative = False

    def set_scale(self):
        self.scale = 100.0

    def set_shift(self):
        self.shift = -10.0

    def prepare_output(self):
        self.output = np.zeros(self.input_size)
        self.outputs = {'Output': self.output}

    def test_check_output(self):
        self.assertRaises(AttributeError, self.check_raise_error,
                          'Quantization shift must be nonnegative.')


class TestQuantizeOp_TooBigShift(TestQuantizeOp_NegativeShift):
    def set_shift(self):
        self.shift = 300.0

    def test_check_output(self):
        self.assertRaises(
            AttributeError, self.check_raise_error,
            'Quantization shift must be less than or equal to 255.')


if __name__ == '__main__':
    unittest.main()
