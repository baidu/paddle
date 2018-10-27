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
import paddle.fluid.core as core
from paddle.fluid.op import Operator


class TestSumOp(OpTest):
    def setUp(self):
        self.op_type = "sum"
        self.use_mkldnn = False
        self.init_kernel_type()
        x0 = np.random.random((3, 4)).astype('float32')
        x1 = np.random.random((3, 4)).astype('float32')
        x2 = np.random.random((3, 4)).astype('float32')
        self.inputs = {"X": [("x0", x0), ("x1", x1), ("x2", x2)]}
        y = x0 + x1 + x2
        self.outputs = {'Out': y}
        self.attrs = {'use_mkldnn': self.use_mkldnn}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['x0'], 'Out')

    def init_kernel_type(self):
        pass


class TestSelectedRowsSumOp(OpTest):
    def check_with_place(self, place, inplace):
        scope = core.Scope()

        self.height = 10
        self.row_numel = 12
        self.rows = [0, 1, 2, 3, 4, 5, 6]

        self.check_input_and_optput(scope, place, inplace, True, True, True)
        self.check_input_and_optput(scope, place, inplace, False, True, True)
        self.check_input_and_optput(scope, place, inplace, False, False, True)
        self.check_input_and_optput(scope, place, inplace, False, False, False)

    def _get_array(self, row_num, row_numel):
        array = np.ones((row_num, row_numel)).astype("float32")
        for i in range(row_num):
            array[i] *= i
        return array

    def check_input_and_optput(self,
                               scope,
                               place,
                               inplace,
                               w1_has_data=False,
                               w2_has_data=False,
                               w3_has_data=False):

        self.create_selected_rows(scope, place, "W1", w1_has_data)
        self.create_selected_rows(scope, place, "W2", w2_has_data)
        self.create_selected_rows(scope, place, "W3", w3_has_data)

        # create Out Variable
        if inplace:
            out_var_name = "W1"
        else:
            out_var_name = "Out"
        out = scope.var(out_var_name).get_selected_rows()

        # create and run sum operator
        sum_op = Operator("sum", X=["W1", "W2", "W3"], Out=out_var_name)
        sum_op.run(scope, place)

        has_data_w_num = 0
        for has_data in [w1_has_data, w2_has_data, w3_has_data]:
            if has_data:
                has_data_w_num += 1

        if has_data_w_num > 0:
            self.assertEqual(len(out.rows()), 7)
            self.assertTrue(
                np.array_equal(
                    np.array(out.get_tensor()),
                    self._get_array(len(self.rows), self.row_numel) *
                    has_data_w_num))
        else:
            self.assertEqual(len(out.rows()), 0)
            self.assertTrue(
                np.array_equal(
                    np.array(out.get_tensor()),
                    self._get_array(0, self.row_numel) * has_data_w_num))

    def create_selected_rows(self, scope, place, var_name, has_data):
        # create and initialize W Variable
        if has_data:
            rows = self.rows
        else:
            rows = []

        var = scope.var(var_name)
        w_selected_rows = var.get_selected_rows()
        w_selected_rows.set_height(self.height)
        w_selected_rows.set_rows(rows)
        w_array = self._get_array(len(rows), self.row_numel)
        w_tensor = w_selected_rows.get_tensor()
        w_tensor.set(w_array, place)

        return var

    def test_w_is_selected_rows(self):
        places = [core.CPUPlace()]
        # currently only support CPU
        for place in places:
            for inplace in [True, False]:
                self.check_with_place(place, inplace)


if __name__ == "__main__":
    unittest.main()
