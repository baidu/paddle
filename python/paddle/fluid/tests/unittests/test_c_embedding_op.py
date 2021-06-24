#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
import paddle
import paddle.fluid as fluid
from paddle.framework import core


def get_c_embedding(start, end, table, ids):
    index = ids.flatten()
    input_mask = (index < start) | (index >= end)
    masked_input = index - start
    masked_input[input_mask] = 0
    output = table[masked_input]
    output[input_mask] = 0.0
    return output


class TestCEmbeddingOp(OpTest):
    def setUp(self):
        self.op_type = "c_embedding"
        table = np.random.random((17, 31)).astype("float64")
        ids = np.random.randint(
            low=0, high=17 * 2, size=(2, 4, 5)).astype("int32")
        self.start_index = 10
        self.end_index = self.start_index + 17

        self.inputs = {'W': table, 'Ids': ids}
        np_out = get_c_embedding(self.start_index, self.end_index, table, ids)
        self.outputs = {'Out': np_out.reshape((2, 4, 5, 31))}
        self.attrs = {'start_index': self.start_index}

    def test_check_output_gpu(self):
        if core.is_compiled_with_cuda():
            self.check_output_with_place(core.CUDAPlace(0))

    def test_check_grad_gpu(self):
        if core.is_compiled_with_cuda():
            self.check_grad_with_place(core.CUDAPlace(0), ['W'], 'Out')


if __name__ == "__main__":
    unittest.main()
