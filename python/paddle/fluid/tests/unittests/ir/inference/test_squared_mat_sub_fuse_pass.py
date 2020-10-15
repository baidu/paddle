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

from __future__ import print_function

import unittest
import numpy as np
from inference_pass_test import InferencePassTest
import paddle
import paddle.fluid as fluid
import paddle.fluid.core as core
from paddle.fluid.core import AnalysisConfig
from paddle.fluid.core import PassVersionChecker


class SquaredMatSubFusePassTest(InferencePassTest):
    def setUp(self):
        with fluid.program_guard(self.main_program, self.startup_program):
            data_a = fluid.data(name="data_a", shape=[128, 1], dtype="float32")
            data_b = fluid.data(name="data_b", shape=[256, 1], dtype="float32")

            fc_a = fluid.layers.fc(data_a, size=256)
            fc_b = fluid.layers.fc(data_b, size=64)

            data_a_square = paddle.square(fc_a)
            data_b_square = paddle.square(fc_b)

            matmul_ab = paddle.matmul(fc_a, fc_b)
            matmul_ab_square = paddle.square(matmul_ab)
            matmul_square_ab = paddle.matmul(data_a_square, data_b_square)

            scale = paddle.fluid.layers.fill_constant(shape=[1], value=0.5, dtype='float32')

            sub_val = paddle.fluid.layers.elementwise_sub(matmul_ab_square, matmul_square_ab)
            squared_mat_sub_out = fluid.layers.elementwise_mul(sub_val, scale)

        self.feeds = {
            "data_a": np.random.random((128, 1)).astype("float32"),
            "data_b": np.random.random((256, 1)).astype("float32")
        }
        self.fetch_list = [squared_mat_sub_out]

    def test_check_output(self):
        use_gpu = False
        self.check_output_with_option(use_gpu)

        self.assertTrue(
            PassVersionChecker.IsCompatible('squared_mat_sub_fuse_pass'))


if __name__ == "__main__":
    unittest.main()
