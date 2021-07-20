# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
from paddle.fluid import core
import paddle.fluid as fluid
import paddle


class testAdamWOpStatic(unittest.TestCase):
    def test_adamw_op_static(self):
        paddle.enable_static()
        place = fluid.CPUPlace()
        shape = [2, 3, 8, 8]
        train_prog = fluid.Program()
        startup = fluid.Program()
        with fluid.program_guard(train_prog, startup):
            with fluid.unique_name.guard():
                data = fluid.data(name="data", shape=shape)
                conv = fluid.layers.conv2d(data, 8, 3)
                loss = paddle.mean(conv)

                beta1 = fluid.layers.create_global_var(
                    shape=[1], value=0.85, dtype='float32', persistable=True)
                beta2 = fluid.layers.create_global_var(
                    shape=[1], value=0.95, dtype='float32', persistable=True)

                opt = fluid.optimizer.AdamWOptimizer(
                    learning_rate=1e-5,
                    beta1=beta1,
                    beta2=beta2,
                    weight_decay=0.01,
                    epsilon=1e-8)
                opt.minimize(loss)

        exe = fluid.Executor(place)
        exe.run(startup)
        data_np = np.random.random(shape).astype('float32')
        rets = exe.run(train_prog, feed={"data": data_np}, fetch_list=[loss])
        assert rets[0] is not None
        paddle.disable_static()

    def test_adamw_op_invalid_input(self):
        paddle.enable_static()
        train_prog = fluid.Program()
        startup = fluid.Program()
        with fluid.program_guard(train_prog, startup):
            with fluid.unique_name.guard():
                with self.assertRaises(ValueError):
                    adamw = fluid.optimizer.AdamWOptimizer(
                        learning_rate=0.01, beta1=-1)
                with self.assertRaises(ValueError):
                    adamw = fluid.optimizer.AdamWOptimizer(
                        learning_rate=0.01, beta2=-1)
                with self.assertRaises(ValueError):
                    adamw = fluid.optimizer.AdamWOptimizer(
                        learning_rate=0.01, epsilon=-1)
                with self.assertRaises(ValueError):
                    adamw = fluid.optimizer.AdamWOptimizer(
                        learning_rate=0.01, weight_decay=-1)
                with self.assertRaises(TypeError):
                    adamw = fluid.optimizer.AdamWOptimizer(
                        learning_rate=0.01, weight_decay="0.01")

    def test_adamw_exception(self):
        paddle.enable_static()
        a = paddle.static.data(name="a", shape=[32, 32], dtype='float32')
        b = paddle.static.data(name="b", shape=[32, 32], dtype='float32')
        label = paddle.static.data(name="label", shape=[32, 1], dtype='int64')

        sum = paddle.add(a, b)
        z = paddle.pow(sum, 2.0)

        fc_1 = fluid.layers.fc(input=z, size=128)
        prediction = fluid.layers.fc(input=fc_1, size=2, act='softmax')

        cost = fluid.layers.cross_entropy(input=prediction, label=label)
        loss = fluid.layers.reduce_mean(cost)
        opt = fluid.optimizer.AdamWOptimizer(use_global_beta_pow=True)
        opt.minimize(loss)
        self.assertRaises(Exception, opt._get_global_accumulator, 'tmp')
        opt._add_global_accumulator('tmp', type=core.VarDesc.VarType.LOD_TENSOR)
        opt._get_global_accumulator('tmp')
        self.assertRaises(
            Exception,
            opt._add_global_accumulator,
            opt._beta1_pow_acc_str,
            type=core.VarDesc.VarType.LOD_TENSOR)
        paddle.disable_static()

    def test_adamw_op_coverage(self):
        paddle.enable_static()
        train_prog = fluid.Program()
        startup = fluid.Program()
        with fluid.program_guard(train_prog, startup):
            with fluid.unique_name.guard():
                adamw = fluid.optimizer.AdamWOptimizer(
                    learning_rate=0.01, weight_decay=0.01)
                assert (adamw.__str__() is not None)


if __name__ == "__main__":
    unittest.main()
