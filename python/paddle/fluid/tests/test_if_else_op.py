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

import paddle
import paddle.fluid.layers as layers
from paddle.fluid.framework import Program, program_guard
from paddle.fluid.executor import Executor
from paddle.fluid.optimizer import MomentumOptimizer
import paddle.fluid.core as core
import paddle.fluid as fluid
from paddle.fluid.layers.control_flow import split_lod_tensor
from paddle.fluid.layers.control_flow import merge_lod_tensor
from paddle.fluid.layers.control_flow import ConditionalBlock

import unittest
import numpy as np


class TestMNISTIfElseOp(unittest.TestCase):
    def test_raw_api(self):
        prog = Program()
        startup_prog = Program()
        with program_guard(prog, startup_prog):
            image = layers.data(name='x', shape=[784], dtype='float32')

            label = layers.data(name='y', shape=[1], dtype='int64')

            limit = layers.fill_constant(shape=[1], dtype='int64', value=5)
            cond = layers.less_than(x=label, y=limit)
            true_image, false_image = split_lod_tensor(input=image, mask=cond)

            true_out = layers.create_tensor(dtype='float32')
            true_cond = ConditionalBlock([cond])

            with true_cond.block():
                hidden = layers.fc(input=true_image, size=100, act='tanh')
                prob = layers.fc(input=hidden, size=10, act='softmax')
                layers.assign(input=prob, output=true_out)

            false_out = layers.create_tensor(dtype='float32')
            false_cond = ConditionalBlock([cond])

            with false_cond.block():
                hidden = layers.fc(input=false_image, size=200, act='tanh')
                prob = layers.fc(input=hidden, size=10, act='softmax')
                layers.assign(input=prob, output=false_out)

            prob = merge_lod_tensor(
                in_true=true_out, in_false=false_out, mask=cond, x=image)
            loss = layers.cross_entropy(input=prob, label=label)
            avg_loss = layers.mean(loss)

            optimizer = MomentumOptimizer(learning_rate=0.001, momentum=0.9)
            optimizer.minimize(avg_loss, startup_prog)

        train_reader = paddle.batch(
            paddle.reader.shuffle(
                paddle.dataset.mnist.train(), buf_size=8192),
            batch_size=10)

        place = core.CPUPlace()
        exe = Executor(place)

        exe.run(startup_prog)
        PASS_NUM = 100
        for pass_id in range(PASS_NUM):
            for data in train_reader():
                x_data = np.array([x[0] for x in data]).astype("float32")
                y_data = np.array([x[1] for x in data]).astype("int64")
                y_data = np.expand_dims(y_data, axis=1)

                outs = exe.run(prog,
                               feed={'x': x_data,
                                     'y': y_data},
                               fetch_list=[avg_loss])
                print((outs[0]))
                if outs[0] < 1.0:
                    return
        self.assertFalse(True)

    def test_ifelse(self):
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
            paddle.reader.shuffle(
                paddle.dataset.mnist.train(), buf_size=8192),
            batch_size=200)

        place = core.CPUPlace()
        exe = Executor(place)

        exe.run(startup_prog)
        PASS_NUM = 100
        for pass_id in range(PASS_NUM):
            for data in train_reader():
                x_data = np.array([x[0] for x in data]).astype("float32")
                y_data = np.array([x[1] for x in data]).astype("int64")
                y_data = y_data.reshape((y_data.shape[0], 1))

                outs = exe.run(prog,
                               feed={'x': x_data,
                                     'y': y_data},
                               fetch_list=[avg_loss])
                print((outs[0]))
                if outs[0] < 1.0:
                    return
        self.assertFalse(True)


class TestIfElse(unittest.TestCase):
    def set_test_case(self):
        # condiction is: self.data < self.cond_value
        self.cond_value = 0.5
        self.data = np.random.rand(25, 1).astype(np.float32)

    def compare_ifelse_op_and_numpy(self, place):
        self.set_test_case()

        prog = Program()
        startup_prog = Program()
        with program_guard(prog, startup_prog):
            src = layers.data(name='data', shape=[1], dtype='float32')
            cond = layers.fill_constant(
                [1], dtype='float32', value=self.cond_value)
            ifcond = layers.less_than(x=src, y=cond)
            ie = layers.IfElse(ifcond)
            with ie.true_block():
                true_target = ie.input(src)
                ie.output(true_target)

            with ie.false_block():
                false_target = ie.input(src)
                ie.output(false_target)
            if_out = ie()
            out = layers.reduce_sum(if_out)

            exe = fluid.Executor(place)
            exe.run(fluid.default_startup_program())
            fetch_list = [out]
            o1, = exe.run(fluid.default_main_program(),
                          feed={'data': self.data},
                          fetch_list=[out])
            o2 = np.sum(self.data)
            self.assertTrue(
                np.allclose(
                    o1, o2, atol=1e-8),
                "IfElse result : " + str(o1) + "\n Numpy result :" + str(o2))

    def test_cpu(self):
        self.compare_ifelse_op_and_numpy(fluid.CPUPlace())

    def test_cuda(self):
        if not core.is_compiled_with_cuda():
            return
        self.compare_ifelse_op_and_numpy(fluid.CUDAPlace(0))


class TestIfElseTrueBranch(TestIfElse):
    def set_test_case(self):
        # condiction is: self.data < self.cond_value
        self.cond_value = 10.
        self.data = np.random.rand(25, 1).astype(np.float32)


class TestIfElseFalseBranch(TestIfElse):
    def set_test_case(self):
        # condiction is: self.data < self.cond_value
        self.cond_value = -10.
        self.data = np.random.rand(25, 1).astype(np.float32)


if __name__ == '__main__':
    unittest.main()
