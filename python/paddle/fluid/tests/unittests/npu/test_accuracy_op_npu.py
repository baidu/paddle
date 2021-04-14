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
class TestAccuracy(OpTest):
    def setUp(self):
        self.op_type = "accuracy"
        self.set_npu()
        self.init_dtype()
        np.random.seed(SEED)
        pred = np.random.uniform(1, 2, [11, 1]).astype(self.dtype)
        label = pred.copy()
        accuracy = np.array([1]).astype(self.dtype)
        correct = np.array([11 * 1]).astype(self.dtype)
        total = np.array([11 * 1]).astype(self.dtype)

        self.inputs = {
            "Out": OpTest.np_dtype_to_fluid_dtype(pred),
            "Label": OpTest.np_dtype_to_fluid_dtype(label),
            "Indices": OpTest.np_dtype_to_fluid_dtype(pred)
        }
        self.outputs = {
            "Accuracy": accuracy,
            "Correct": correct,
            "Total": total
        }

    def set_npu(self):
        self.__class__.use_npu = True
        self.place = paddle.NPUPlace(0)

    def init_dtype(self):
        self.dtype = np.float32

    def test_check_output(self):
        self.check_output_with_place(self.place, check_dygraph=False)


class TestAccuracy2(TestAccuracy):
    def setUp(self):
        self.op_type = "accuracy"
        self.set_npu()
        self.init_dtype()
        np.random.seed(SEED)
        pred = np.random.uniform(1, 2, [11, 1]).astype(self.dtype)
        label = np.random.uniform(4, 5, [11, 1]).astype(self.dtype)
        accuracy = np.array([0]).astype(self.dtype)
        correct = np.array([11 * 0]).astype(self.dtype)
        total = np.array([11 * 1]).astype(self.dtype)

        self.inputs = {
            "Out": OpTest.np_dtype_to_fluid_dtype(pred),
            "Label": OpTest.np_dtype_to_fluid_dtype(label),
            "Indices": OpTest.np_dtype_to_fluid_dtype(pred)
        }
        self.outputs = {
            "Accuracy": accuracy,
            "Correct": correct,
            "Total": total
        }


class TestAccuracy3(TestAccuracy):
    def setUp(self):
        self.op_type = "accuracy"
        self.set_npu()
        self.init_dtype()
        np.random.seed(SEED)
        a = np.random.randint(1, 2, [5, 1])
        b = np.random.randint(0, 1, [5, 1])
        pred = np.row_stack((a, b)).astype(self.dtype)
        label = np.random.randint(1, 2, [10, 1]).astype(self.dtype)
        accuracy = np.array([0.5]).astype(self.dtype)
        correct = np.array([5]).astype(self.dtype)
        total = np.array([10 * 1]).astype(self.dtype)

        self.inputs = {
            "Out": OpTest.np_dtype_to_fluid_dtype(pred),
            "Label": OpTest.np_dtype_to_fluid_dtype(label),
            "Indices": OpTest.np_dtype_to_fluid_dtype(pred)
        }
        self.outputs = {
            "Accuracy": accuracy,
            "Correct": correct,
            "Total": total
        }


class TestAccuracyInt(TestAccuracy):
    def init_dtype(self):
        self.dtype = np.int


if __name__ == '__main__':
    unittest.main()
