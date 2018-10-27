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
import paddle.fluid.core as core
import numpy as np
from paddle.fluid.op import Operator


class TestSpliteSelectedRows(unittest.TestCase):
    def get_places(self):
        places = [core.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(core.CUDAPlace(0))
        return places

    def test_check_output(self):
        for place in self.get_places():
            self.check_with_place(place)

    def test_check_grad(self):
        for place in self.get_places():
            self.check_grad_with_place(place)

    def check_with_place(self, place):
        scope = core.Scope()
        rows = [0, 5, 7, 4, 20]
        height = 20
        row_numel = 2

        # initialize input variable X
        x = scope.var('X').get_selected_rows()
        x.set_rows(rows)
        x.set_height(height)
        np_array = np.ones((len(rows), row_numel)).astype("float32")
        np_array[0, 0] = 2.0
        np_array[2, 1] = 4.0
        np_array[4, 1] = 8.0
        x_tensor = x.get_tensor()
        x_tensor.set(np_array, place)

        height_sections = [5, 5, 5, 5, 3]

        # initialize output variables [out0, out1]
        outs_name = ["out%d" % i for i in range(len(height_sections))]
        outs = [
            scope.var(var_name).get_selected_rows() for var_name in outs_name
        ]

        # expected output selected rows
        expected_out0_rows = [0, 4]
        expected_out1_rows = [0, 2]
        expected_out4_rows = [0]

        op = Operator(
            "split_selected_rows",
            X="X",
            Out=outs_name,
            height_sections=height_sections)

        op.run(scope, place)

        self.assertEqual(outs[0].rows(), expected_out0_rows)
        self.assertEqual(outs[1].rows(), expected_out1_rows)
        self.assertEqual(outs[4].rows(), expected_out4_rows)

        self.assertEqual(outs[0].height(), height_sections[0])
        self.assertEqual(outs[4].height(), height_sections[4])

        self.assertAlmostEqual(2.0, np.array(outs[0].get_tensor())[0, 0])
        self.assertAlmostEqual(4.0, np.array(outs[1].get_tensor())[1, 1])
        self.assertAlmostEqual(8.0, np.array(outs[4].get_tensor())[0, 1])

    def check_grad_with_place(self, place):
        scope = core.Scope()
        height = 10
        row_numel = 2

        # attr
        height_sections = [5, 5]

        # initialize input variable X
        out0_grad = scope.var("out0@GRAD").get_selected_rows()
        rows0 = [0, 5]
        out0_grad.set_rows(rows0)
        out0_grad.set_height(height)
        out0_grad_tensor = out0_grad.get_tensor()
        np_array = np.ones((len(rows0), row_numel)).astype("float32")
        out0_grad_tensor.set(np_array, place)

        out1_grad = scope.var("out1@GRAD").get_selected_rows()
        rows1 = [2, 0]
        out1_grad.set_rows(rows1)
        out1_grad.set_height(height)
        out1_grad_tensor = out1_grad.get_tensor()
        np_array = np.ones((len(rows1), row_numel)).astype("float32")
        out1_grad_tensor.set(np_array, place)

        x_grad = scope.var("X@GRAD").get_selected_rows()

        grad_op = Operator(
            "sum",
            X=["out0@GRAD", "out1@GRAD"],
            Out="X@GRAD",
            height_sections=height_sections)

        grad_op.run(scope, place)

        merged_rows = set(rows0 + rows1)
        self.assertEqual(set(x_grad.rows()), set(rows0 + rows1))
        self.assertEqual(x_grad.height(), height)

        print(np.array(x_grad.get_tensor()))
        self.assertAlmostEqual(2.0, np.array(x_grad.get_tensor())[0, 0])
        self.assertAlmostEqual(1.0, np.array(x_grad.get_tensor())[2, 1])


if __name__ == "__main__":
    unittest.main()
