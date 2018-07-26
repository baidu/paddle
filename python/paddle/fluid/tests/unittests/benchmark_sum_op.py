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

import unittest
import numpy as np

import paddle.fluid as fluid
from .benchmark import BenchmarkSuite
from .op_test import OpTest

# This is a demo op test case for operator benchmarking and high resolution number stability alignment.


class TestSumOp(BenchmarkSuite):
    def setUp(self):
        self.op_type = "sum"
        self.customize_testcase()
        self.customize_fetch_list()

    def customize_fetch_list(self):
        """
        customize fetch list, configure the wanted variables.
        >>> self.fetch_list = ["Out"]
        """
        self.fetch_list = ["Out"]
        # pass

    def customize_testcase(self):
        # a test case
        x0 = np.random.random((300, 400)).astype('float32')
        x1 = np.random.random((300, 400)).astype('float32')
        x2 = np.random.random((300, 400)).astype('float32')

        # NOTE: if the output is empty, then it will autofilled by benchmarkSuite.
        # only the output dtype is used, the shape, lod and data is computed from input.
        self.inputs = {"X": [("x0", x0), ("x1", x1), ("x2", x2)]}
        self.outputs = {"Out": x0 + x1 + x2}

    def test_check_output(self):
        """
        compare the output with customized output. In this case,
        you should set the correct output by hands.
        >>> self.outputs = {"Out": x0 + x1 + x2}
        """
        self.check_output(atol=1e-8)

    def test_output_stability(self):
        # compare the cpu gpu output in high resolution.
        self.check_output_stability()

    def test_timeit_output(self):
        """
        perf the op, time cost will be averged in iters.
        output example
        >>> One pass of (sum_op) at CPUPlace cost 0.000461330413818
        >>> One pass of (sum_op) at CUDAPlace(0) cost 0.000556070804596
        """
        self.timeit_output(iters=100)

    def test_timeit_grad(self):
        """
        perf the op gradient, time cost will be averged in iters.
        output example
        >>> One pass of (sum_grad_op) at CPUPlace cost 0.00279935121536
        >>> One pass of (sum_grad_op) at CUDAPlace(0) cost 0.00500632047653
        """
        self.timeit_grad(iters=100)


if __name__ == "__main__":
    unittest.main()
