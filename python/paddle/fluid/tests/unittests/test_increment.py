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
import paddle
import paddle.fluid as fluid


class TestIncrement(unittest.TestCase):
    def test_api(self):
        with fluid.program_guard(fluid.Program(), fluid.Program()):
            input = fluid.layers.fill_constant(
                shape=[1], dtype='int64', value=5)
            expected_result = np.array([8], dtype='int64')

            output = paddle.tensor.math.increment(input, value=3)
            exe = fluid.Executor(fluid.CPUPlace())
            result = exe.run(fetch_list=[output])
            self.assertEqual((result == expected_result).all(), True)

        with fluid.dygraph.guard():
            input = paddle.ones(shape=[1], dtype='int64')
            expected_result = np.array([2], dtype='int64')
            output = paddle.tensor.math.increment(input, value=1)
            self.assertEqual((output.numpy() == expected_result).all(), True)


if __name__ == "__main__":
    unittest.main()
