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

import op_test
import unittest
import numpy as np
import struct

from op_test import convert_uint16_to_float, convert_float_to_uint16
import paddle
import paddle.fluid.core as core
import paddle.fluid as fluid
from paddle.fluid import compiler, Program, program_guard


class TestCastOp1(op_test.OpTest):
    def setUp(self):
        ipt = np.random.random(size=[10, 10])
        self.inputs = {'X': ipt.astype('float32')}
        self.outputs = {'Out': ipt.astype('float64')}
        self.attrs = {
            'in_dtype': int(core.VarDesc.VarType.FP32),
            'out_dtype': int(core.VarDesc.VarType.FP64)
        }
        self.op_type = 'cast'

    def test_check_output(self):
        self.check_output()

    def test_grad(self):
        self.check_grad(['X'], ['Out'])


class TestCastOp2(op_test.OpTest):
    def setUp(self):
        ipt = np.random.random(size=[10, 10])
        self.inputs = {'X': ipt.astype('float16')}
        self.outputs = {'Out': ipt.astype('float32')}
        self.attrs = {
            'in_dtype': int(core.VarDesc.VarType.FP16),
            'out_dtype': int(core.VarDesc.VarType.FP32)
        }
        self.op_type = 'cast'

    def test_check_output(self):
        self.check_output(atol=1e-3)


class TestCastOp3(op_test.OpTest):
    def setUp(self):
        ipt = np.random.random(size=[10, 10])
        self.inputs = {'X': ipt.astype('float32')}
        self.outputs = {'Out': ipt.astype('float16')}
        self.attrs = {
            'in_dtype': int(core.VarDesc.VarType.FP32),
            'out_dtype': int(core.VarDesc.VarType.FP16)
        }
        self.op_type = 'cast'

    def test_check_output(self):
        self.check_output(atol=1e-3)


# bf16->fp32
class TestCastOp4(op_test.OpTest):
    def setUp(self):
        ipt = np.array(np.random.randint(10, size=(1, 10))).astype('uint16')
        self.inputs = {'X': ipt}
        self.outputs = {'Out': convert_uint16_to_float(ipt)}
        self.attrs = {
            'in_dtype': int(core.VarDesc.VarType.BF16),
            'out_dtype': int(core.VarDesc.VarType.FP32)
        }
        self.op_type = 'cast'

    def test_check_output(self):
        places = [core.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(core.CUDAPlace(0))
        for place in places:
            self.check_output_with_place(place, atol=1e-6)


# fp32->bf16
class TestCastOp5(op_test.OpTest):
    def setUp(self):
        ipt = np.random.random(size=[2, 10]).astype('float32')
        self.inputs = {'X': ipt}
        self.outputs = {'Out': convert_float_to_uint16(ipt)}
        self.attrs = {
            'in_dtype': int(core.VarDesc.VarType.FP32),
            'out_dtype': int(core.VarDesc.VarType.BF16)
        }
        self.op_type = 'cast'

    def test_check_output(self):
        places = [core.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(core.CUDAPlace(0))
        # bf16 results are represented as float32 in OpTest
        # besides, like TestCastOp4, this test is_bfloat16_op, 
        # therefore set check_dygraph as False.
        for place in places:
            self.check_output_with_place(place, atol=1e-6, check_dygraph=False)


class TestCastOpError(unittest.TestCase):
    def test_errors(self):
        with program_guard(Program(), Program()):
            # The input type of cast_op must be Variable.
            x1 = fluid.create_lod_tensor(
                np.array([[-1]]), [[1]], fluid.CPUPlace())
            self.assertRaises(TypeError, fluid.layers.cast, x1, 'int32')
            # The input dtype of cast_op must be bool, float16, float32, float64, int32, int64, uint8.
            x2 = fluid.layers.data(name='x2', shape=[4], dtype='int16')
            self.assertRaises(TypeError, fluid.layers.cast, x2, 'int32')

            def test_dtype_type():
                x4 = fluid.layers.data(name='x4', shape=[4], dtype='int32')
                output = fluid.layers.cast(x=x4, dtype='int16')

            self.assertRaises(TypeError, test_dtype_type)


if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()
