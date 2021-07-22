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

import paddle
import paddle.fluid as fluid
import unittest


class TestFuseBatchNormActPass(unittest.TestCase):
    def build_program(self, main_program, startup_program, use_cuda, seed=1):
        with fluid.program_guard(main_program, startup_program):
            x = fluid.layers.data(name='x', shape=[1, 28, 28], dtype='float32')
            y = fluid.layers.data(name="y", shape=[1], dtype='int64')
            hidden1 = fluid.layers.conv2d(
                input=x,
                filter_size=3,
                num_filters=16,
                stride=1,
                padding=1,
                act=None,
                bias_attr=False,
                data_format='NHWC')
            param_attr = fluid.ParamAttr(
                name='batch_norm_w',
                initializer=fluid.initializer.Constant(value=1.0))
            bias_attr = fluid.ParamAttr(
                name='batch_norm_b',
                initializer=fluid.initializer.Constant(value=0.0))
            hidden2 = fluid.layers.batch_norm(
                input=hidden1,
                param_attr=param_attr,
                bias_attr=bias_attr,
                act='relu',
                data_layout='NHWC')
            hidden3 = fluid.layers.fc(input=hidden2, size=32, act='relu')
            hidden4 = fluid.layers.batch_norm(
                input=hidden3, act='relu', data_layout='NHWC')
            prediction = fluid.layers.fc(input=hidden4, size=10, act='softmax')
            loss = fluid.layers.cross_entropy(input=prediction, label=y)
            loss = fluid.layers.mean(loss)
            sgd = fluid.optimizer.SGD(learning_rate=0.001)
            if use_cuda:
                sgd = fluid.contrib.mixed_precision.decorate(
                    sgd, use_dynamic_loss_scaling=True, init_loss_scaling=128.0)
            sgd.minimize(loss)
        return x, y, loss

    def check(self, place, use_cuda):
        paddle.seed(1)
        paddle.framework.random._manual_program_seed(1)
        main_program = fluid.Program()
        startup_program = fluid.Program()
        x, y, loss = self.build_program(main_program, startup_program, use_cuda)
        exe = fluid.Executor(place)
        iters = 8
        batch_size = 16
        feeder = fluid.DataFeeder(feed_list=[x, y], place=place)

        # close fused_bn_act_ops
        build_strategy = fluid.BuildStrategy()
        build_strategy.fuse_bn_act_ops = False
        binary = fluid.CompiledProgram(main_program).with_data_parallel(
            loss_name=loss.name, build_strategy=build_strategy)
        train_reader = paddle.batch(
            paddle.dataset.mnist.train(), batch_size=batch_size)
        loss_vals = []
        scope = fluid.Scope()
        with fluid.scope_guard(scope):
            exe.run(startup_program)
            for _ in range(iters):
                data = next(train_reader())
                loss_v = exe.run(binary,
                                 feed=feeder.feed(data),
                                 fetch_list=[loss])
                loss_vals.append(loss_v[0][0])

        # open fused_bn_act_ops
        build_strategy_fused = fluid.BuildStrategy()
        build_strategy_fused.fuse_bn_act_ops = True
        binary_fused = fluid.CompiledProgram(main_program).with_data_parallel(
            loss_name=loss.name, build_strategy=build_strategy_fused)
        train_reader_fused = paddle.batch(
            paddle.dataset.mnist.train(), batch_size=batch_size)
        loss_vals_fused = []
        scope_fused = fluid.Scope()
        with fluid.scope_guard(scope_fused):
            exe.run(startup_program)
            for _ in range(iters):
                data = next(train_reader_fused())
                loss_v = exe.run(binary_fused,
                                 feed=feeder.feed(data),
                                 fetch_list=[loss])
                loss_vals_fused.append(loss_v[0][0])

        # check loss
        for i in range(iters):
            self.assertAlmostEqual(loss_vals[i], loss_vals_fused[i], delta=1e-5)

    def test_fuse_bn_act_pass_cpu(self):
        place = fluid.CPUPlace()
        self.check(place, use_cuda=False)

    def test_fuse_bn_act_pass_cuda(self):
        if fluid.core.is_compiled_with_cuda():
            if fluid.core.is_compiled_with_rocm():
                return
            place = fluid.CUDAPlace(0)
            self.check(place, use_cuda=True)


if __name__ == '__main__':
    unittest.main()
