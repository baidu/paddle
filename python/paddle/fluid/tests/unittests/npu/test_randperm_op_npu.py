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
import paddle.fluid.core as core
from paddle.static import program_guard, Program

paddle.enable_static()


def check_randperm_out(n, data_np):
    assert isinstance(data_np, np.ndarray), \
        "The input data_np should be np.ndarray."
    gt_sorted = np.arange(n)
    out_sorted = np.sort(data_np)
    return list(gt_sorted == out_sorted)


def error_msg(data_np):
    return "The sorted ground truth and sorted out should " + \
 "be equal, out = " + str(data_np)


def convert_dtype(dtype_str):
    dtype_str_list = ['int32', 'int64', 'float32', 'float64', 'float16']
    dtype_num_list = [
        core.VarDesc.VarType.INT32, core.VarDesc.VarType.INT64,
        core.VarDesc.VarType.FP32, core.VarDesc.VarType.FP64,
        core.VarDesc.VarType.FP16
    ]
    assert dtype_str in dtype_str_list, dtype_str + \
        " should in " + str(dtype_str_list)
    return dtype_num_list[dtype_str_list.index(dtype_str)]


@unittest.skipIf(not paddle.is_compiled_with_npu(),
                 "core is not compiled with NPU")
class TestRandpermOp(OpTest):
    """ Test randperm op."""

    def setUp(self):
        self.set_npu()
        self.op_type = 'randperm'
        self.place = paddle.NPUPlace(0)

        self.init_dtype()

        self.set_inputs()
        self.set_attrs()
        self.set_outputs()

    def init_attrs(self):
        pass

    def set_npu(self):
        self.__class__.use_npu = True

    def init_dtype(self):
        self.dtype = 'int64'

    def set_inputs(self):
        self.inputs = {}

    def set_attrs(self):
        n = 100
        self.attrs = {
            'n': n,
            'dtype': convert_dtype(self.dtype),
        }

    def set_outputs(self):
        n = self.attrs['n']
        self.outputs = {'Out': np.zeros((n)).astype(self.dtype)}

    def test_check_output(self):
        self._get_places = lambda: [self.place]
        self.check_output_customized(self.verify_output)

    def verify_output(self, outs):
        out_np = np.array(outs[0])
        self.assertTrue(
            check_randperm_out(self.attrs['n'], out_np), msg=error_msg(out_np))


class TestRandpermOpN(TestRandpermOp):
    def set_attrs(self):
        super(TestRandpermOpN, self).set_attrs()
        self.attrs['n'] = 10000


class TestRandpermOpInt32(TestRandpermOp):
    def init_dtype(self):
        self.dtype = 'int32'


class TestRandpermOpFloat32(TestRandpermOp):
    def init_dtype(self):
        self.dtype = 'float32'


class TestRandpermOpFloat64(TestRandpermOp):
    def init_dtype(self):
        self.dtype = 'float64'


class TestRandpermOpFloat16(TestRandpermOp):
    def init_dtype(self):
        self.dtype = 'float16'


@unittest.skipIf(not paddle.is_compiled_with_npu(),
                 "core is not compiled with NPU")
class TestRandpermAPI(unittest.TestCase):
    def test_out(self):
        n = 10
        place = paddle.NPUPlace(0)
        with program_guard(Program(), Program()):
            x1 = paddle.randperm(n)
            x2 = paddle.randperm(n, 'float32')

            exe = paddle.static.Executor(place)
            res = exe.run(fetch_list=[x1, x2])

            self.assertEqual(res[0].dtype, np.int64)
            self.assertEqual(res[1].dtype, np.float32)
            self.assertTrue(check_randperm_out(n, res[0]))
            self.assertTrue(check_randperm_out(n, res[1]))


@unittest.skipIf(not paddle.is_compiled_with_npu(),
                 "core is not compiled with NPU")
class TestRandpermImperative(unittest.TestCase):
    def test_out(self):
        place = paddle.NPUPlace(0)
        with paddle.fluid.dygraph.base.guard(place=place):
            self.place = paddle.NPUPlace(0)
            n = 10
            for dtype in ['int32', 'int64', 'float32', 'float64']:
                data_p = paddle.randperm(n, dtype)
                data_np = data_p.numpy()
                self.assertTrue(
                    check_randperm_out(n, data_np), msg=error_msg(data_np))


if __name__ == "__main__":
    unittest.main()
