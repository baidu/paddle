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

import paddle
import paddle.fluid as fluid
import contextlib
import unittest
import numpy as np
from paddle.fluid.contrib.mixed_precision.fp16_utils import cast_model_to_fp16
from paddle.fluid.contrib.mixed_precision.fp16_utils import cast_parameters_to_fp16

paddle.enable_static()


def resnet_cifar10(input, depth=32):
    def conv_bn_layer(input,
                      ch_out,
                      filter_size,
                      stride,
                      padding,
                      act='relu',
                      bias_attr=False):
        tmp = fluid.layers.conv2d(
            input=input,
            filter_size=filter_size,
            num_filters=ch_out,
            stride=stride,
            padding=padding,
            act=None,
            bias_attr=bias_attr)
        return fluid.layers.batch_norm(input=tmp, act=act)

    def shortcut(input, ch_in, ch_out, stride):
        if ch_in != ch_out:
            return conv_bn_layer(input, ch_out, 1, stride, 0, None)
        else:
            return input

    def basicblock(input, ch_in, ch_out, stride):
        tmp = conv_bn_layer(input, ch_out, 3, stride, 1)
        tmp = conv_bn_layer(tmp, ch_out, 3, 1, 1, act=None, bias_attr=True)
        short = shortcut(input, ch_in, ch_out, stride)
        return fluid.layers.elementwise_add(x=tmp, y=short, act='relu')

    def layer_warp(block_func, input, ch_in, ch_out, count, stride):
        tmp = block_func(input, ch_in, ch_out, stride)
        for i in range(1, count):
            tmp = block_func(tmp, ch_out, ch_out, 1)
        return tmp

    assert (depth - 2) % 6 == 0
    n = (depth - 2) // 6
    conv1 = conv_bn_layer(
        input=input, ch_out=16, filter_size=3, stride=1, padding=1)
    res1 = layer_warp(basicblock, conv1, 16, 16, n, 1)
    res2 = layer_warp(basicblock, res1, 16, 32, n, 2)
    res3 = layer_warp(basicblock, res2, 32, 64, n, 2)
    pool = fluid.layers.pool2d(
        input=res3, pool_size=8, pool_type='avg', pool_stride=1)
    return pool


def compile(program, loss_name=None):
    build_strategy = paddle.static.BuildStrategy()
    exec_strategy = paddle.static.ExecutionStrategy()

    exec_strategy.num_threads = 1
    exec_strategy.num_iteration_per_drop_scope = 10000

    build_strategy.fuse_bn_act_ops = True
    build_strategy.fuse_elewise_add_act_ops = True
    build_strategy.fuse_bn_add_act_ops = True

    compiled_program = paddle.static.CompiledProgram(
        program).with_data_parallel(
            loss_name=loss_name,
            build_strategy=build_strategy,
            exec_strategy=exec_strategy)

    return compiled_program


def train(use_pure_fp16=True, use_nesterov=False):
    classdim = 10
    data_shape = [3, 32, 32]
    BATCH_SIZE = 128
    PASS_NUM = 1

    train_program = fluid.Program()
    startup_prog = fluid.Program()
    train_program.random_seed = 123
    startup_prog.random_seed = 456
    with fluid.program_guard(train_program, startup_prog):
        images = fluid.layers.data(
            name='pixel', shape=data_shape, dtype='float32')
        label = fluid.layers.data(name='label', shape=[1], dtype='int64')
        net = resnet_cifar10(images, 32)

        logits = fluid.layers.fc(input=net, size=classdim, act="softmax")
        if use_pure_fp16:
            cast_model_to_fp16(fluid.default_main_program())
            logits_fp32 = fluid.layers.cast(x=logits, dtype="float32")
        else:
            logits_fp32 = logits
        cost = fluid.layers.softmax_with_cross_entropy(
            logits_fp32, label, return_softmax=False)
        sum_cost = fluid.layers.reduce_sum(cost)

        # Test program
        test_program = train_program.clone(for_test=True)

        optimizer = fluid.contrib.optimizer.Momentum(
            learning_rate=0.001,
            momentum=0.9,
            use_nesterov=use_nesterov,
            regularization=fluid.regularizer.L2Decay(1e-4),
            multi_precision=use_pure_fp16,
            rescale_grad=1.0 / BATCH_SIZE)

        optimizer.minimize(sum_cost)

    # no shuffle for unit test
    train_reader = paddle.batch(
        paddle.dataset.cifar.train10(), batch_size=BATCH_SIZE)

    test_reader = paddle.batch(
        paddle.dataset.cifar.test10(), batch_size=BATCH_SIZE)

    place = fluid.CUDAPlace(0)
    exe = fluid.Executor(place)
    feeder = fluid.DataFeeder(place=place, feed_list=[images, label])

    def train_loop(main_program):
        exe.run(startup_prog)
        if use_pure_fp16:
            cast_parameters_to_fp16(place, train_program, fluid.global_scope())
        compiled_program = compile(train_program, sum_cost.name)
        loss = 0.0
        for pass_id in range(PASS_NUM):
            train_loss_list = []
            for batch_id, data in enumerate(train_reader()):
                loss, = exe.run(compiled_program,
                                feed=feeder.feed(data),
                                fetch_list=[sum_cost])
                print('PassID {0:1}, Train Batch ID {1:04}, train loss {2:2.4}'.
                      format(pass_id, batch_id + 1, float(loss)))
                train_loss_list.append(float(loss))

                if batch_id >= 4:  # For speeding up CI
                    test_loss_list = []
                    for tid, test_data in enumerate(test_reader()):
                        loss_t, = exe.run(program=test_program,
                                          feed=feeder.feed(test_data),
                                          fetch_list=[sum_cost])
                        test_loss_list.append(float(loss_t))
                        print(
                            'PassID {0:1}, Test Batch ID {1:04}, test loss {2:2.4}'.
                            format(pass_id, tid + 1, float(loss_t)))
                        if tid >= 4:
                            break  # For speeding up CI
                    return train_loss_list, test_loss_list

    return train_loop(train_program)


class TestImageMultiPrecision(unittest.TestCase):
    def test_resnet_pure_fp16(self):
        if not fluid.core.is_compiled_with_cuda():
            return

        def do_test(use_nesterov=False):
            suffix = "with Nesterov" if use_nesterov else "without Nesterov"
            with self.scope_prog_guard():
                print("-----------------FP16 Train {}-----------------".format(
                    suffix))
                train_loss_fp16, test_loss_fp16 = train(
                    use_pure_fp16=True, use_nesterov=use_nesterov)
            with self.scope_prog_guard():
                print("-----------------FP32 Train {}-----------------".format(
                    suffix))
                train_loss_fp32, test_loss_fp32 = train(
                    use_pure_fp16=False, use_nesterov=use_nesterov)

            self.assertTrue(
                np.allclose(
                    np.array(train_loss_fp16),
                    np.array(train_loss_fp32),
                    rtol=1e-02,
                    atol=1e-05,
                    equal_nan=True),
                msg='Failed to train in pure FP16.')
            self.assertTrue(
                np.allclose(
                    np.array(test_loss_fp16),
                    np.array(test_loss_fp32),
                    rtol=1e-02,
                    atol=1e-05,
                    equal_nan=True),
                msg='Failed to test in pure FP16.')

        do_test(use_nesterov=False)
        do_test(use_nesterov=True)

    @contextlib.contextmanager
    def scope_prog_guard(self):
        prog = fluid.Program()
        startup_prog = fluid.Program()
        scope = fluid.core.Scope()
        with fluid.scope_guard(scope):
            with fluid.program_guard(prog, startup_prog):
                yield


class TestAmpWithNonIterableDataLoader(unittest.TestCase):
    def decorate_with_data_loader(self):
        main_prog = paddle.static.Program()
        start_prog = paddle.static.Program()
        with paddle.static.program_guard(main_prog, start_prog):
            with paddle.fluid.unique_name.guard():
                image = fluid.layers.data(
                    name='image', shape=[3, 224, 224], dtype='float32')
                label = fluid.layers.data(
                    name='label', shape=[1], dtype='int64')
                py_reader = fluid.io.DataLoader.from_generator(
                    feed_list=[image, label],
                    capacity=4,
                    iterable=False,
                    use_double_buffer=False)
                zero_var = fluid.layers.fill_constant(
                    shape=[1], dtype='int64', value=0)
                one_var = fluid.layers.fill_constant(
                    shape=[1], dtype='int64', value=1)
                with fluid.layers.control_flow.Switch() as switch:
                    with switch.case(label != zero_var):
                        fluid.layers.assign(input=zero_var, output=label)
                    with switch.default():
                        fluid.layers.assign(input=one_var, output=label)

                net = resnet_cifar10(image)
                logits = fluid.layers.fc(input=net, size=10, act="softmax")

        block = main_prog.global_block()
        for op in block.ops:
            if op.type == "mul":
                op._set_attr('in_dtype', fluid.core.VarDesc.VarType.FP32)
                op._set_attr('out_dtype', fluid.core.VarDesc.VarType.FP32)
                op._set_attr('dtype', fluid.core.VarDesc.VarType.FP32)

        cast_model_to_fp16(main_prog)

    def test_non_iterable_dataloader(self):
        self.decorate_with_data_loader()


if __name__ == '__main__':
    unittest.main()
