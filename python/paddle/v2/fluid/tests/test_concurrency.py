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
import paddle.v2.fluid.layers as layers
import paddle.v2.fluid as fluid
import paddle.v2.fluid.core as core
from paddle.v2.fluid.executor import Executor


class TestRoutineOp(unittest.TestCase):
    def test_simple_routine(self):
        counter = layers.zeros(shape=[1], dtype='int64')
        counter = layers.increment(counter)

        routine_op = fluid.Routine()
        with routine_op.block():
            counter = layers.increment(counter)

        cpu = core.CPUPlace()
        exe = Executor(cpu)

        outs = exe.run(fetch_list=[counter])
        self.assertEqual(2, np.sum(outs[0]))


if __name__ == '__main__':
    unittest.main()
