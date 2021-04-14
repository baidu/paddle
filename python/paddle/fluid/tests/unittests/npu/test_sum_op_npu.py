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
import paddle.fluid.core as core

paddle.enable_static()
SEED = 2021


@unittest.skipIf(not paddle.is_compiled_with_npu(),
                 "core is not compiled with NPU")
class TestSum1(OpTest):
    def setUp(self):
        self.set_npu()
        self.init_dtype()
        self.op_type = "sum"
        self.place = paddle.NPUPlace(0)

        x0 = np.random.random((3, 40)).astype(self.dtype)
        x1 = np.random.random((3, 40)).astype(self.dtype)
        x2 = np.random.random((3, 40)).astype(self.dtype)
        self.inputs = {'X': [("x0", x0), ("x1", x1), ("x2", x2)]}
        y = x0 + x1 + x2
        self.outputs = {'Out': y}

        self.attrs = {'use_mkldnn': False}

    def init_dtype(self):
        self.dtype = np.float32

    def set_npu(self):
        self.__class__.use_npu = True

    def test_check_output(self):
        self.check_output_with_place(self.place, check_dygraph=False)


class TestSum2(OpTest):
    def setUp(self):
        self.set_npu()
        self.init_dtype()
        self.op_type = "sum"
        self.place = paddle.NPUPlace(0)

        x0 = np.random.random((3, 3)).astype(self.dtype)
        x1 = np.random.random((3, 3)).astype(self.dtype)
        x2 = np.random.random((3, 3)).astype(self.dtype)
        x3 = np.random.random((3, 3)).astype(self.dtype)
        self.inputs = {'X': [("x0", x0), ("x1", x1), ("x2", x2), ("x3", x3)]}
        y = x0 + x1 + x2 + x3
        self.outputs = {'Out': y}

        self.attrs = {'use_mkldnn': False}

    def init_dtype(self):
        self.dtype = np.float16

    def set_npu(self):
        self.__class__.use_npu = True

    def test_check_output(self):
        self.check_output_with_place(self.place, check_dygraph=False)


if __name__ == '__main__':
    unittest.main()
