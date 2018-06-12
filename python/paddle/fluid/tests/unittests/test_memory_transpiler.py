# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

import paddle.fluid.layers as layers
import paddle.fluid.optimizer as optimizer
from paddle.fluid.framework import Program, program_guard
from paddle.fluid.transpiler import MemoryTranspiler


class TestMemoryTranspiler(unittest.TestCase):
    def test_normal_memory_transpiler(self):
        program = Program()
        with program_guard(program, startup_program=Program()):
            x = layers.data(name='x', shape=[13], dtype='float32')
            y_predict = layers.fc(input=x, size=1, act=None)
            y = layers.data(name='y', shape=[1], dtype='float32')
            cost = layers.square_error_cost(input=y_predict, label=y)
            avg_cost = layers.mean(cost)
            opt = optimizer.SGD(learning_rate=0.001)
            opt = opt.minimize(avg_cost)
        print("before transpile")
        print(program)
        MemoryTranspiler(program).transpile()
        print("after transpile")
        print(program)

    def test_sub_block_memory_transpiler(self):
        program = Program()
        with program_guard(program, startup_program=Program()):
            data = layers.data(name='X', shape=[1], dtype='float32')
            data.stop_gradient = False
            cond = layers.ConditionalBlock(inputs=[data])
            out = layers.create_tensor(dtype='float32')
            with cond.block():
                hidden = layers.fc(input=data, size=10)
                layers.assign(hidden, out)
            loss = layers.mean(out)
            append_backward(loss=loss)
        print("before transpile")
        print(program)
        MemoryTranspiler(program).transpile()
        print("after transpile")
        print(program)


if __name__ == "__main__":
    unittest.main()
