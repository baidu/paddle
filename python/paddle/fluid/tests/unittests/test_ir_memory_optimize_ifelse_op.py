# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

# nlp model stack of op operate on lod. It's a classical test case in optimize pass.

from __future__ import print_function

import numpy as np

import paddle
import paddle.fluid as fluid
import paddle.fluid.layers as layers

import unittest
import paddle.fluid.core as core

from paddle.fluid import compiler, Program, program_guard
from paddle.fluid.executor import Executor
from paddle.fluid.backward import append_backward
from paddle.fluid.optimizer import MomentumOptimizer
from ir_memory_optimize_net_base import TestIrMemOptBase


class TestIrMemoryOptimizeIfElseOp(unittest.TestCase):
    def check_network_convergence(self,
                                  use_cuda=True,
                                  use_mem_opt=False,
                                  iter_num=5):
        paddle.seed(100)
        paddle.framework.random._manual_program_seed(100)
        prog = Program()
        startup_prog = Program()
        with program_guard(prog, startup_prog):
            image = layers.data(name='x', shape=[784], dtype='float32')

            label = layers.data(name='y', shape=[1], dtype='int64')

            limit = layers.fill_constant(shape=[1], dtype='int64', value=5)
            cond = layers.less_than(x=label, y=limit)
            ie = layers.IfElse(cond)

            with ie.true_block():
                true_image = ie.input(image)
                hidden = layers.fc(input=true_image, size=100, act='tanh')
                prob = layers.fc(input=hidden, size=10, act='softmax')
                ie.output(prob)

            with ie.false_block():
                false_image = ie.input(image)
                hidden = layers.fc(input=false_image, size=200, act='tanh')
                prob = layers.fc(input=hidden, size=10, act='softmax')
                ie.output(prob)

            prob = ie()
            loss = layers.cross_entropy(input=prob[0], label=label)
            avg_loss = layers.mean(loss)

            optimizer = MomentumOptimizer(learning_rate=0.001, momentum=0.9)
            optimizer.minimize(avg_loss, startup_prog)
            train_reader = paddle.batch(
                paddle.dataset.mnist.train(), batch_size=200)

            place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
            exe = Executor(place)

            exec_strategy = fluid.ExecutionStrategy()
            exec_strategy._use_device = core.DeviceType.CUDA if use_cuda else core.DeviceType.CPU

            build_strategy = fluid.BuildStrategy()
            build_strategy.memory_optimize = use_mem_opt

            train_cp = compiler.CompiledProgram(fluid.default_main_program())
            train_cp = train_cp.with_data_parallel(
                loss_name=avg_loss.name,
                exec_strategy=exec_strategy,
                build_strategy=build_strategy)
            fetch_list = [avg_loss.name]

            exe.run(startup_prog)
            PASS_NUM = 100
            loop = 0
            ret = []
            for pass_id in range(PASS_NUM):
                for data in train_reader():
                    x_data = np.array([x[0] for x in data]).astype("float32")
                    y_data = np.array([x[1] for x in data]).astype("int64")
                    y_data = y_data.reshape((y_data.shape[0], 1))

                    outs = exe.run(train_cp,
                                   feed={'x': x_data,
                                         'y': y_data},
                                   fetch_list=[avg_loss])

                    loop += 1
                    ret.append(outs[0])
                    if iter_num == loop:
                        return ret
            return ret

    def test_ifelse(self):
        ret1 = self.check_network_convergence(False, True)
        print(ret1)
        ret2 = self.check_network_convergence(False, False)
        print(ret2)
        self.assertTrue(np.allclose(ret1, ret2))

        if fluid.core.is_compiled_with_cuda():
            ret1 = self.check_network_convergence(True, True)
            print(ret1)
            ret2 = self.check_network_convergence(True, False)
            print(ret2)
            self.assertTrue(np.allclose(ret1, ret2))


if __name__ == "__main__":
    unittest.main()
