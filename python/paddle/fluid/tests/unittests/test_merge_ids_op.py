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
from op_test import OpTest


class TestMergeIdsOp(OpTest):
    def setUp(self):
        self.op_type = "merge_ids"
        ids1 = np.array([[0], [2], [5], [6]]).astype('int64')
        ids2 = np.array([[0], [2], [2], [3]]).astype('int64')

        rows1 = np.array([[0], [2]]).astype('int64')
        rows2 = np.array([[3], [5]]).astype('int64')
        rows3 = np.array([[6]]).astype('int64')

        x0 = np.array([[0.1, 0.2], [0.2, 0.3]]).astype('float32')
        x1 = np.array([[0.3, 0.4], [0.4, 0.5]]).astype('float32')
        x2 = np.array([[0.5, 0.6]]).astype('float32')

        out1 = np.array(
            [[0.1, 0.2], [0.2, 0.3], [0.4, 0.5], [0.5, 0.6]]).astype('float32')
        out2 = np.array(
            [[0.1, 0.2], [0.2, 0.3], [0.2, 0.3], [0.3, 0.4]]).astype('float32')

        self.inputs = {
            'Ids': [('ids1', ids1), ('ids2', ids2)],
            "Rows": [('rows1', rows1), ('rows2', rows2), ('rows3', rows3)],
            "X": [('x0', x0), ('x1', x1), ('x2', x2)]
        }
        self.outputs = {'Out': [('out1', out1), ('out2', out2)]}

    def test_check_output(self):
        self.check_output()


if __name__ == '__main__':
    unittest.main()
