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
from paddle.fluid import core

paddle.enable_static()
SEED = 2021


@unittest.skipIf(not paddle.is_compiled_with_npu(),
                 "core is not compiled with NPU")
class TestMean(OpTest):
    def setUp(self):
        self.set_npu()
        self.place = paddle.NPUPlace(0)
        self.op_type = "mean"
        self.init_dtype()

        x = np.random.random([1, 100]).astype(self.dtype)
        self.inputs = {'X': x}

        self.attrs = {}
        np_out = np.mean(x)
        self.outputs = {'Out': np_out}

    def set_npu(self):
        self.__class__.use_npu = True

    def init_dtype(self):
        self.dtype = np.float32

    def test_check_output(self):
        self.check_output_with_place(self.place, check_dygraph=False)

    def test_check_grad(self):
        self.check_grad_with_place(
            self.place, ['X'], 'Out', check_dygraph=False)


@unittest.skipIf(not paddle.is_compiled_with_npu(),
                 "core is not compiled with NPU")
class TestMeanFP16(OpTest):
    def setUp(self):
        self.set_npu()
        self.place = paddle.NPUPlace(0)
        self.op_type = "mean"
        self.init_dtype()

        x = np.random.random([3, 200]).astype(self.dtype)
        self.inputs = {'X': x}

        self.attrs = {}
        np_out = np.mean(x)
        self.outputs = {'Out': np_out}

    def set_npu(self):
        self.__class__.use_npu = True
        self.__class__.no_need_check_grad = True

    def init_dtype(self):
        self.dtype = np.float16

    def test_check_output(self):
        self.check_output_with_place(self.place, check_dygraph=False)


if __name__ == '__main__':
    unittest.main()
