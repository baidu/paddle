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
import paddle.fluid as fluid
import paddle.fluid.core as core
from paddle.fluid.tests.unittests.op_test import OpTest
from mkldnn_op_test import format_reorder


class TestReQuantizeOp(OpTest):
    def setUp(self):
        self.op_type = 'requantize'
        self.scale_in = 127.0
        self.shift_in = 0.0
        self.scale_out = 100.0
        self.shift_out = 0.0
        self.input_size = [1, 1, 10, 10]
        self.input_data_type = 'int8'
        self.set_scales()
        self.set_shifts()
        self.set_input_data_type()
        self.prepare_input()
        self.prepare_output()

    def prepare_input(self):
        if self.input_data_type == 'int8':
            # input data values are integers from interval [-128, 128)
            self.input = (np.random.randint(0, 256, self.input_size) - 128
                          ).astype(self.input_data_type)
        else:
            # input data values are integers from interval [0, 256)
            self.input = (np.random.randint(
                0, 256, self.input_size)).astype(self.input_data_type)

        self.inputs = {'Input': OpTest.np_dtype_to_fluid_dtype(self.input)}
        self.attrs = {
            'Scale_in': self.scale_in,
            'Scale_out': self.scale_out,
            'Shift_in': self.shift_in,
            'Shift_out': self.shift_out
        }

    def prepare_output(self):
        scale_ratio = self.scale_out / self.scale_in
        with_shift = (self.shift_in != 0.0 or self.shift_out != 0.0)

        if with_shift or self.input_data_type == 'uint8':
            dst_type = 'uint8'
            type_min = 0
            type_max = 255
            new_shift = np.clip(
                np.rint(self.shift_out - scale_ratio * self.shift_in), type_min,
                type_max)
        else:
            dst_type = 'int8'
            type_min = -128
            type_max = 127
            new_shift = 0

        output_tmp = np.clip(
            np.rint(self.input.astype('float32') * scale_ratio + new_shift),
            type_min, type_max).astype(dst_type)

        self.output = format_reorder(output_tmp, self.input_size)
        self.outputs = {'Output': self.output}

    def test_check_output(self):
        # TODO(wangzhongpu): support mkldnn op in dygraph mode
        self.assertTrue(self.input_data_type == 'uint8' or self.shift_in == 0.0,
                        'Input data must be unsigned if it has nonzero shift.')
        self.check_output(check_dygraph=False)

    def check_raise_error(self, msg):
        try:
            self.check_output()
        except Exception as e:
            if msg in str(e):
                raise AttributeError
            else:
                print(e)

    def set_scales(self):
        pass

    def set_shifts(self):
        pass

    def set_input_data_type(OpTest):
        pass


# ---------------test requantize with s8 input, no shift--------------------


class TestReQuantizeOp_S8_SameScales(TestReQuantizeOp):
    def set_scales(self):
        self.scale_in = 127.0
        self.scale_out = 127.0


class TestReQuantizeOp_S8_DifferentScales_1(TestReQuantizeOp):
    def set_scales(self):
        self.scale_in = 127.0
        self.scale_out = 100.0


class TestReQuantizeOp_S8_DifferentScales_2(TestReQuantizeOp):
    def set_scales(self):
        self.scale_in = 100.0
        self.scale_out = 127.0


class TestReQuantizeOp_S8_ZeroInputScale(TestReQuantizeOp):
    def set_scales(self):
        self.scale_in = 0.0
        self.scale_out = 127.0

    def prepare_output(self):
        self.output = np.zeros(self.input_size)
        self.outputs = {'Output': self.output}

    def test_check_output(self):
        self.assertRaises(AttributeError, self.check_raise_error,
                          'Scale of input cannot be 0.0')


class TestReQuantizeOp_S8_ZeroOutputScale(TestReQuantizeOp):
    def set_scales(self):
        self.scale_in = 127.0
        self.scale_out = 0.0

    def prepare_output(self):
        self.output = np.zeros(self.input_size)
        self.outputs = {'Output': self.output}

    def test_check_output(self):
        self.assertRaises(AttributeError, self.check_raise_error,
                          'Scale of output cannot be 0.0')


# ---------------test requantize with u8 input, no shift--------------------


class TestReQuantizeOp_U8_SameScales(TestReQuantizeOp_S8_SameScales):
    def set_input_data_type(self):
        self.input_data_type = 'uint8'


class TestReQuantizeOp_U8_DifferentScales_1(
        TestReQuantizeOp_S8_DifferentScales_1):
    def set_input_data_type(self):
        self.input_data_type = 'uint8'


class TestReQuantizeOp_U8_DifferentScales_2(
        TestReQuantizeOp_S8_DifferentScales_2):
    def set_input_data_type(self):
        self.input_data_type = 'uint8'


# ---------------test requantize with s8 input, with shift------------------


class TestReQuantizeOp_S8_WithShift(TestReQuantizeOp):
    def set_scales(self):
        self.scale_in = 60.0
        self.scale_out = 127.0

    def set_shifts(self):
        self.shift_in = 128.0
        self.shift_out = 128.0

    def test_check_output(self):
        self.assertRaises(
            AttributeError, self.check_raise_error,
            'Requantize does not support nonzero shift for signed input.')


class TestReQuantizeOp_S8_WithOutputShift(TestReQuantizeOp):
    def set_scales(self):
        self.scale_in = 127.0
        self.scale_out = 60.0

    def set_shifts(self):
        self.shift_in = 0.0
        self.shift_out = 120.0


# ---------------test requantize with u8 input, with shift------------------


class TestReQuantizeOp_U8_SameScales_SameShift(TestReQuantizeOp_U8_SameScales):
    def set_shifts(self):
        self.shift_in = 128.0
        self.shift_out = 128.0


class TestReQuantizeOp_U8_SameScales_DifferentShift_1(
        TestReQuantizeOp_U8_SameScales):
    def set_shifts(self):
        self.shift_in = 60.0
        self.shift_out = 128.0


class TestReQuantizeOp_U8_SameScales_DifferentShift_2(
        TestReQuantizeOp_U8_SameScales):
    def set_shifts(self):
        self.shift_in = 128.0
        self.shift_out = 60.0


class TestReQuantizeOp_U8_DifferentScales_1_SameShift(
        TestReQuantizeOp_U8_DifferentScales_1):
    def set_shifts(self):
        self.shift_in = 128.0
        self.shift_out = 128.0


class TestReQuantizeOp_U8_DifferentScales_2_SameShift(
        TestReQuantizeOp_U8_DifferentScales_2):
    def set_shifts(self):
        self.shift_in = 128.0
        self.shift_out = 128.0


class TestReQuantizeOp_U8_DifferentScales_1_DifferentShift_1(
        TestReQuantizeOp_U8_DifferentScales_1):
    def set_shifts(self):
        self.shift_in = 128.0
        self.shift_out = 60.0


class TestReQuantizeOp_U8_DifferentScales_2_DifferentShift_1(
        TestReQuantizeOp_U8_DifferentScales_2):
    def set_shifts(self):
        self.shift_in = 128.0
        self.shift_out = 60.0


class TestReQuantizeOp_U8_DifferentScales_1_DifferentShift_2(
        TestReQuantizeOp_U8_DifferentScales_1):
    def set_shifts(self):
        self.shift_in = 60.0
        self.shift_out = 128.0


class TestReQuantizeOp_U8_DifferentScales_2_DifferentShift_2(
        TestReQuantizeOp_U8_DifferentScales_2):
    def set_shifts(self):
        self.shift_in = 60.0
        self.shift_out = 128.0


# ---------------test reused requantize op, no shift------------------------


class TestReQuantizeOpReused(TestReQuantizeOp):
    def setUp(self):
        #  self.input_size = [1, 1, 10, 10]
        self.input_size = [1, 1, 2, 2]
        self.input_data_type = 'int8'
        self.set_scales()
        self.set_shifts()
        self.set_input_data_type()
        self.prepare_input()
        self.prepare_output()

    def set_scales(self):
        self.scale_in = 100.0
        self.scale_out = 120.0

    def set_shifts(self):
        self.shift_in = 0.0
        self.shift_out = 0.0

    def set_input_data_type(self):
        pass

    def test_check_output(self):
        variables = {
            "input": self.input,
            "output": self.output,
        }
        program = fluid.Program()
        with fluid.program_guard(program):
            block = program.global_block()
            for name in variables:
                block.create_var(
                    name=name, dtype="int8", shape=variables[name].shape)
            block.append_op(
                type="requantize",
                inputs={'Input': block.var('input'), },
                outputs={"Output": block.var('output')},
                attrs={
                    'Scale_in': self.scale_in,
                    'Scale_out': self.scale_out,
                    'Shift_in': self.shift_in,
                    'Shift_out': self.shift_out
                })
            place = core.CPUPlace()
            exe = fluid.Executor(place)
            for i in range(2):
                out = exe.run(program,
                              feed={'input': variables['input']},
                              fetch_list=['output'])

            self.assertTrue(
                np.allclose(
                    variables['output'], out[0], atol=1e-4), 'output')


# ---------------test reused requantize op, no shift------------------------


class TestReQuantizeOpReused_WithShift(TestReQuantizeOpReused):
    def set_input_data_type(self):
        self.input_data_type = 'uint8'

    def set_shifts(self):
        self.shift_in = 128
        self.shift_out = 60


if __name__ == '__main__':
    unittest.main()
