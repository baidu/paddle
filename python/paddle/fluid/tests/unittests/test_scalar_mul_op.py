# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import unittest
import numpy as np
from op_test import OpTest


class TestScalarMulOp(OpTest):
    def setUp(self):
        self.op_type = "scalar_mul"
        self.inputs = {
            # "X": np.random.random((5, 5)).astype("float32")
            "X": np.random.random((10, 10)).astype(np.float64)
        }
        self.attrs = {"a": -1.5, "b": 0.1}
        self.outputs = {
            "Out": self.inputs["X"] * self.attrs["a"] + self.attrs["b"]
        }

    def test_check_output(self):
        self.check_output()

    def test_check_grad_normal(self):
        self.check_grad(["X"], "Out", max_relative_error=0.5)


if __name__ == "__main__":
    unittest.main()
