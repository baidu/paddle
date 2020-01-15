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

import unittest
import numpy as np
from op_test import OpTest
import paddle.fluid.core as core
from paddle.fluid.op import Operator
import random


class TestSumOp(OpTest):
    def setUp(self):
        self.op_type = "partial_sum"
        self.init_kernel_type()
        self.init_para()
        if self.length is -1:
            end_index = self.column
        else:
            end_index = self.start_index + self.length
        self.var_names = ['x' + str(num) for num in range(self.var_num)]
        self.vars = [np.random.random((self.batch_size, self.column)).astype(self.dtype)\
                     for num in range(self.var_num) ]
        self.inputs = {'X': zip(self.var_names, self.vars)}
        self.attrs = {'start_index': self.start_index, 'length': self.length}
        y = self.vars[0][:, self.start_index:end_index]
        for i in range(1, self.var_num):
            y = y + self.vars[i][:, self.start_index:end_index]

        self.outputs = {'Out': y}

    def init_kernel_type(self):
        self.dtype = np.float64

    def init_para(self):
        self.batch_size = random.randint(10, 20)
        self.column = random.randint(101, 200)
        self.start_index = random.randint(0, self.column - 1)
        self.length = random.randint(0, self.column - self.start_index)
        self.var_num = random.randint(1, 3)

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        for var_name in self.var_names:
            self.check_grad([var_name], 'Out')


class TestSumOp2(TestSumOp):
    def init_para(self):
        self.batch_size = random.randint(1, 10)
        self.column = random.randint(101, 200)
        self.start_index = random.randint(0, self.column - 1)
        self.length = -1
        self.var_num = 3


class TestSumOp3(TestSumOp):
    def init_para(self):
        self.batch_size = random.randint(1, 10)
        self.column = random.randint(101, 200)
        self.start_index = self.column - 1
        self.length = 1
        self.var_num = 2


class TestSumOp4(TestSumOp):
    def init_para(self):
        self.batch_size = random.randint(1, 10)
        self.column = random.randint(101, 200)
        self.start_index = self.column - 1
        self.length = 1
        self.var_num = 1


if __name__ == "__main__":
    unittest.main()
