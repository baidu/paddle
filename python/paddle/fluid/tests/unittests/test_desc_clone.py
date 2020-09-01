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

from __future__ import print_function

import numpy as np
import argparse
import time
import math

import paddle
import paddle.fluid as fluid
import paddle.fluid.profiler as profiler
from paddle.fluid import core
import unittest
from multiprocessing import Process
import os
import signal
import six
import collections

SEED = 1
DTYPE = "float32"
paddle.dataset.mnist.fetch()


# random seed must set before configuring the network.
# fluid.default_startup_program().random_seed = SEED
def cnn_model(data):
    conv_pool_1 = fluid.nets.simple_img_conv_pool(
        input=data,
        filter_size=5,
        num_filters=20,
        pool_size=2,
        pool_stride=2,
        act="relu")
    conv_pool_2 = fluid.nets.simple_img_conv_pool(
        input=conv_pool_1,
        filter_size=5,
        num_filters=50,
        pool_size=2,
        pool_stride=2,
        act="relu")

    # TODO(dzhwinter) : refine the initializer and random seed settting
    SIZE = 10
    input_shape = conv_pool_2.shape
    param_shape = [six.moves.reduce(lambda a, b: a * b, input_shape[1:], 1)
                   ] + [SIZE]
    scale = (2.0 / (param_shape[0]**2 * SIZE))**0.5

    predict = fluid.layers.fc(
        input=conv_pool_2,
        size=SIZE,
        act="softmax",
        param_attr=fluid.param_attr.ParamAttr(
            initializer=fluid.initializer.NormalInitializer(
                loc=0.0, scale=scale)))
    return predict


def get_model(batch_size):
    # Input data
    images = fluid.layers.data(name='pixel', shape=[1, 28, 28], dtype=DTYPE)
    label = fluid.layers.data(name='label', shape=[1], dtype='int64')

    # Train program
    predict = cnn_model(images)
    cost = fluid.layers.cross_entropy(input=predict, label=label)
    avg_cost = fluid.layers.mean(x=cost)

    # Evaluator
    batch_size_tensor = fluid.layers.create_tensor(dtype='int64')
    batch_acc = fluid.layers.accuracy(
        input=predict, label=label, total=batch_size_tensor)

    inference_program = fluid.default_main_program().clone()
    # Optimization
    opt = fluid.optimizer.AdamOptimizer(
        learning_rate=0.001, beta1=0.9, beta2=0.999)

    # Reader
    train_reader = paddle.batch(
        paddle.dataset.mnist.train(), batch_size=batch_size)
    test_reader = paddle.batch(
        paddle.dataset.mnist.test(), batch_size=batch_size)
    opt.minimize(avg_cost)
    return inference_program, avg_cost, train_reader, test_reader, batch_acc, predict


def get_transpiler(trainer_id, main_program, pserver_endpoints, trainers):
    t = fluid.DistributeTranspiler()
    t.transpile(
        trainer_id=trainer_id,
        program=main_program,
        pservers=pserver_endpoints,
        trainers=trainers)
    return t


from paddle.fluid.transpiler.details import op_to_code


def operator_equal(a, b):
    if op_to_code(a) != op_to_code(b):
        raise ValueError("In operator_equal not equal\n")

    for k, v in six.iteritems(a.__dict__):
        if isinstance(v, fluid.framework.Program) or \
                isinstance(v, fluid.framework.Block):
            continue

        elif isinstance(v, core.OpDesc):
            continue

        elif isinstance(v, collections.OrderedDict):
            v0 = sorted(list(six.iteritems(v)), key=lambda x: x[0])
            v1 = sorted(list(six.iteritems(b.__dict__[k])), key=lambda x: x[0])

            if v0 != v1:
                raise ValueError("In operator_equal not equal:{0}\n".format(k))

        elif (v != b.__dict__[k]):
            raise ValueError("In operator_equal not equal:{0}\n".format(k))

    return True


def block_equal(a, b):
    for k, v in six.iteritems(a.__dict__):
        if isinstance(v, core.ProgramDesc) or isinstance(
                v, fluid.framework.Program) or isinstance(v, core.BlockDesc):
            continue

        elif k == "ops":
            assert (len(a.ops) == len(b.ops))
            for i in range(0, len(a.ops)):
                if not operator_equal(a.ops[i], b.ops[i]):
                    raise ValueError("In block_equal not equal:{0}\n".format(k))

        elif isinstance(v, collections.OrderedDict):
            for key, value in six.iteritems(v):
                if str(value) != str(b.__dict__[k][key]):
                    raise ValueError("In block_equal not equal:{0}\n".format(k))

        elif (v != b.__dict__[k]):
            raise ValueError("In block_equal not equal:{0}\n".format(k))

    return True


def program_equal(a, b):
    for k, v in six.iteritems(a.__dict__):
        if isinstance(v, core.ProgramDesc):
            continue

        elif k == 'blocks':
            for i in range(0, len(a.blocks)):
                if not block_equal(a.blocks[i], b.blocks[i]):
                    raise ValueError("In operator_equal not equal:{0}\n".format(
                        k))
                    return False
            assert (len(a.blocks) == len(b.blocks))
        elif k == '_auto_checkpoint_name':
            continue
        elif (v != b.__dict__[k]):
            raise ValueError("In program_equal not equal:{0}\n".format(k))

    return True


class TestDistMnist(unittest.TestCase):
    def test_desc_clone(self):
        get_model(batch_size=20)

        pserver_endpoints = "127.0.0.1:9123"
        trainers = 1
        current_endpoint = "127.0.0.1:9123"
        t = get_transpiler(0,
                           fluid.default_main_program(), pserver_endpoints,
                           trainers)

        pserver_prog = t.get_pserver_program(current_endpoint)
        startup_prog = t.get_startup_program(current_endpoint, pserver_prog)
        main = pserver_prog.clone()
        startup = startup_prog.clone()
        self.assertTrue(program_equal(main, pserver_prog))
        self.assertTrue(program_equal(startup, startup_prog))


class TestCloneWithStopGradient(unittest.TestCase):
    def test_clone_with_stop_gradient(self):
        train_program = fluid.Program()
        startup_program = fluid.Program()
        with fluid.program_guard(train_program, startup_program):
            img = fluid.layers.data(name='image', shape=[784])
            hidden1 = fluid.layers.fc(input=img, size=200, act='relu')
            hidden1.stop_gradient = True
            hidden2 = fluid.layers.dropout(hidden1, dropout_prob=0.5)
            loss = fluid.layers.cross_entropy(
                input=fluid.layers.fc(hidden2, size=10, act='softmax'),
                label=fluid.layers.data(
                    name='label', shape=[1], dtype='int64'))
            avg_loss = fluid.layers.mean(loss)
            test_program = train_program.clone(for_test=False)

        self.assertEqual(
            test_program.block(0).var(hidden1.name).stop_gradient, True)
        self.assertEqual(
            test_program.block(0).var(hidden2.name).stop_gradient, False)


class TestCloneWithStopGradientInSubBlock(unittest.TestCase):
    def test_clone_with_stop_gradient(self):
        train_program = fluid.Program()
        startup_program = fluid.Program()
        with fluid.program_guard(train_program, startup_program):
            img = fluid.layers.data(name='image', shape=[784])
            true = fluid.layers.ones(shape=[1], dtype="float32")
            hidden1 = fluid.layers.fc(input=img, size=200, act='relu')
            hidden1.stop_gradient = True

            cond = fluid.layers.equal(true, true)

            def true_fn():
                hidden2 = fluid.layers.dropout(hidden1, dropout_prob=0.5)
                hidden2.stop_gradient = True
                return hidden2

            def false_fn():
                hidden2 = fluid.layers.dropout(hidden1, dropout_prob=0.6)
                return hidden2

            hidden2 = fluid.layers.cond(cond, true_fn, false_fn)

            loss = fluid.layers.cross_entropy(
                input=fluid.layers.fc(hidden2, size=10, act='softmax'),
                label=fluid.layers.data(
                    name='label', shape=[1], dtype='int64'))
            avg_loss = fluid.layers.mean(loss)
            test_program = train_program.clone(for_test=False)

        self.assertEqual(
            test_program.block(0).var(hidden1.name).stop_gradient, True)
        for var in test_program.block(1).vars.values():
            var2 = train_program.block(1).var(var.name)
            self.assertEqual(var.stop_gradient, var2.stop_gradient)
        for var in test_program.block(2).vars.values():
            var2 = train_program.block(2).var(var.name)
            self.assertEqual(var.stop_gradient, var2.stop_gradient)


class TestCloneWithRaise(unittest.TestCase):
    def test_clone_with_stop_gradient(self):
        train_program = fluid.Program()
        startup_program = fluid.Program()
        with fluid.program_guard(train_program, startup_program):
            img = fluid.layers.data(name='image', shape=[784])
            true = fluid.layers.ones(shape=[1], dtype="float32")
            hidden1 = fluid.layers.fc(input=img, size=200, act='relu')
            hidden1.stop_gradient = True

            cond = fluid.layers.equal(true, true)

            def true_fn():
                hidden2 = fluid.layers.dropout(hidden1, dropout_prob=0.5)
                hidden2.stop_gradient = True
                return hidden2

            def false_fn():
                hidden2 = fluid.layers.dropout(hidden1, dropout_prob=0.6)
                return hidden2

            hidden2 = fluid.layers.cond(cond, true_fn, false_fn)
            loss = fluid.layers.cross_entropy(
                input=fluid.layers.fc(hidden2, size=10, act='softmax'),
                label=fluid.layers.data(
                    name='label', shape=[1], dtype='int64'))
            avg_loss = fluid.layers.mean(loss)
            test_program = train_program.clone(for_test=False)

        self.assertRaises(ValueError, train_program._copy_data_info_from,
                          startup_program)
        self.assertRaises(TypeError, train_program._copy_data_info_from,
                          startup_program.block(0))


if __name__ == "__main__":
    unittest.main()
