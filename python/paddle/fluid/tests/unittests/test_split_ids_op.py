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
import six
from op_test import OpTest
import paddle.fluid.core as core
from paddle.fluid.op import Operator


class TestSplitIdsOp(OpTest):
    def setUp(self):
        self.op_type = "split_ids"
        ids1 = np.array([[0], [2], [2], [3], [5], [5], [6]]).astype('int64')
        ids2 = np.array([[6], [2], [3], [3], [5], [2], [6]]).astype('int64')
        ids3 = np.array([[2], [2], [2], [3], [5], [5], [6]]).astype('int64')

        out0 = np.array([[0], [3], [6]]).astype('int64')
        out1 = np.array([[]]).astype('int64')
        out2 = np.array([[2], [5]]).astype('int64')
        self.inputs = {'Ids': [('ids1', ids1), ('ids2', ids2), ('ids3', ids3)]}
        self.outputs = {'Out': [('out0', out0), ('out1', out1), ('out2', out2)]}

    def test_check_output(self):
        self.check_output()


class TestSplitSelectedRows(unittest.TestCase):
    def get_places(self):
        places = [core.CPUPlace()]
        return places

    def test_check_output(self):
        for place in self.get_places():
            self.check_with_place(place)

    def check_with_place(self, place):
        scope = core.Scope()
        rows = [0, 5, 7, 4, 9]
        height = 20
        row_numel = 2

        # initialize input variable X
        x = scope.var('X').get_selected_rows()
        x.set_rows(rows)
        x.set_height(height)
        np_array = np.ones((len(rows), row_numel)).astype("float32")
        for i in range(len(rows)):
            for j in range(row_numel):
                np_array[i, j] = rows[i] + j
        x_tensor = x.get_tensor()
        x_tensor.set(np_array, place)

        outs_name = ["out%d" % i for i in six.moves.xrange(3)]
        outs = [
            scope.var(var_name).get_selected_rows() for var_name in outs_name
        ]

        # expected output selected rows
        expected_out_rows = [[0, 9], [7, 4], [5]]

        op = Operator("split_ids", Ids="X", Out=outs_name)

        for _ in range(3):
            op.run(scope, place)

            for i in range(len(outs)):
                expected_rows = expected_out_rows[i]
                self.assertEqual(outs[i].rows(), expected_rows)
                for j in range(len(expected_rows)):
                    row = expected_rows[j]
                    self.assertAlmostEqual(
                        float(row), np.array(outs[i].get_tensor())[j, 0])
                    self.assertAlmostEqual(
                        float(row + 1), np.array(outs[i].get_tensor())[j, 1])


if __name__ == '__main__':
    unittest.main()
