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
class TestReshape2(OpTest):
    def setUp(self):
        self.set_npu()
        self.op_type = "reshape2"
        self.place = paddle.NPUPlace(0)

        self.init_data()
        self.inputs = {"X": np.random.random(self.ori_shape).astype("float32")}
        self.attrs = {"shape": self.new_shape}
        self.outputs = {
            "Out": self.inputs["X"].reshape(self.infered_shape),
            'XShape': np.random.random(self.ori_shape).astype("float32")
        }

    def set_npu(self):
        self.__class__.use_npu = True

    def init_data(self):
        self.ori_shape = (2, 100)
        self.new_shape = (20, 10)
        self.infered_shape = (20, 10)

    def test_check_output(self):
        self.check_output_with_place(
            self.place, check_dygraph=False, no_check_set=['XShape'])

    def test_check_grad_normal(self):
        self.check_grad_with_place(
            self.place, ['X'], 'Out', check_dygraph=False)


class TestReshape2_case2(TestReshape2):
    def init_data(self):
        self.ori_shape = (2, 100)
        self.new_shape = (-1, 10)
        self.infered_shape = (20, 10)


class TestReshape2_case3(TestReshape2):
    def init_data(self):
        self.ori_shape = (100, 5, 6)
        self.new_shape = (-1, 0, 3)
        self.infered_shape = (200, 5, 3)


if __name__ == '__main__':
    unittest.main()
