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
import paddle
import paddle.fluid.layers as layers
from paddle.fluid.executor import Executor
import paddle.fluid.core as core
import paddle.fluid as fluid
from paddle.fluid.backward import append_backward
import numpy
from paddle.fluid import compiler, Program, program_guard

paddle.enable_static()


class TestWhileOp(unittest.TestCase):
    def simple_net(self):
        d0 = layers.data(
            "d0", shape=[10], append_batch_size=False, dtype='float32')
        d1 = layers.data(
            "d1", shape=[10], append_batch_size=False, dtype='float32')
        d2 = layers.data(
            "d2", shape=[10], append_batch_size=False, dtype='float32')
        i = layers.zeros(shape=[1], dtype='int64')
        i.stop_gradient = True
        init = layers.zeros(shape=[10], dtype='float32')
        mem_array = layers.array_write(x=init, i=i)
        data_array = layers.array_write(x=d0, i=i)
        i = layers.increment(i)
        layers.array_write(d1, i, array=data_array)
        i = layers.increment(i)
        layers.array_write(d2, i, array=data_array)
        i = layers.zeros(shape=[1], dtype='int64')
        i.stop_gradient = True
        array_len = layers.fill_constant(shape=[1], dtype='int64', value=1)
        array_len.stop_gradient = True
        cond = layers.less_than(x=i, y=array_len)
        j = layers.fill_constant(shape=[1], dtype='int64', value=1)
        j.stop_gradient = True
        array_len2 = layers.fill_constant(shape=[1], dtype='int64', value=3)
        array_len2.stop_gradient = True
        cond2 = layers.less_than(x=j, y=array_len2)
        while_op = layers.While(cond=cond)
        while_op2 = layers.While(cond=cond2)
        with while_op.block():
            d = layers.array_read(array=data_array, i=i)
            prev = layers.array_read(array=mem_array, i=i)
            result = layers.sums(input=[d, prev])

            i = layers.increment(x=i, in_place=True)
            layers.array_write(result, i=i, array=mem_array)
            layers.less_than(x=i, y=array_len, cond=cond)

            with while_op2.block():
                d2 = layers.array_read(array=data_array, i=j)
                prev2 = layers.array_read(array=mem_array, i=j)
                result2 = layers.sums(input=[d2, prev2])

                j = layers.increment(x=j, in_place=True)
                layers.array_write(result2, i=j, array=mem_array)
                layers.less_than(x=j, y=array_len2, cond=cond2)
        sum_result = layers.array_read(array=mem_array, i=j)
        loss = layers.mean(sum_result)
        return loss, sum_result

    def test_simple_net(self):
        main_program = fluid.Program()
        startup_program = fluid.Program()
        with fluid.program_guard(main_program, startup_program):
            loss, sum_result = self.simple_net()

            append_backward(loss)

            cpu = core.CPUPlace()
            exe = Executor(cpu)
            d = []

            for i in range(3):
                d.append(numpy.random.random(size=[10]).astype('float32'))

            outs = exe.run(feed={'d0': d[0],
                                 'd1': d[1],
                                 'd2': d[2]},
                           fetch_list=[sum_result])
            self.assertAlmostEqual(numpy.sum(d), numpy.sum(outs[0]), delta=0.01)

    def test_simple_net_forward(self):
        main_program = fluid.Program()
        startup_program = fluid.Program()
        with fluid.program_guard(main_program, startup_program):
            self.simple_net()
            binary = fluid.compiler.CompiledProgram(main_program)

            cpu = core.CPUPlace()
            exe = Executor(cpu)
            d = []

            for i in range(3):
                d.append(numpy.random.random(size=[10]).astype('float32'))

            for _ in range(2):
                exe.run(binary, feed={'d0': d[0], 'd1': d[1], 'd2': d[2]})

    def test_exceptions(self):
        i = layers.zeros(shape=[2], dtype='int64')
        array_len = layers.fill_constant(shape=[2], dtype='int64', value=1)
        cond = layers.less_than(x=i, y=array_len)
        with self.assertRaises(TypeError):
            layers.While(cond=cond)
        cond = layers.cast(cond, dtype='float64')
        with self.assertRaises(TypeError):
            layers.While(cond=cond)


class BadInputTest(unittest.TestCase):
    def test_error(self):
        with fluid.program_guard(fluid.Program()):

            def test_bad_x():
                x = [1, 2, 3]
                fluid.layers.increment(x)

            self.assertRaises(TypeError, test_bad_x)


if __name__ == '__main__':
    unittest.main()
