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

from test_dist_base import TestDistRunnerBase, runtime_main
import unittest
import paddle
import os
import paddle.distributed.fleet as fleet
import paddle.distributed.fleet.base.role_maker as role_maker
import numpy as np
from functools import reduce
import paddle.fluid as fluid

paddle.enable_static()

DTYPE = "float32"
paddle.dataset.mnist.fetch()

# Fix seed for test
fluid.default_startup_program().random_seed = 1
fluid.default_main_program().random_seed = 1


def cnn_model(data):
    conv_pool_1 = fluid.nets.simple_img_conv_pool(
        input=data,
        filter_size=5,
        num_filters=20,
        pool_size=2,
        pool_stride=2,
        act="relu",
        param_attr=fluid.ParamAttr(initializer=fluid.initializer.Constant(
            value=0.01)))
    conv_pool_2 = fluid.nets.simple_img_conv_pool(
        input=conv_pool_1,
        filter_size=5,
        num_filters=50,
        pool_size=2,
        pool_stride=2,
        act="relu",
        param_attr=fluid.ParamAttr(initializer=fluid.initializer.Constant(
            value=0.01)))

    SIZE = 10
    input_shape = conv_pool_2.shape
    param_shape = [reduce(lambda a, b: a * b, input_shape[1:], 1)] + [SIZE]
    scale = (2.0 / (param_shape[0]**2 * SIZE))**0.5

    predict = fluid.layers.fc(
        input=conv_pool_2,
        size=SIZE,
        act="softmax",
        param_attr=fluid.param_attr.ParamAttr(
            initializer=fluid.initializer.Constant(value=0.01)))
    return predict


class TestFleetMetaOptimizerFuseAllReducePrecision(TestDistRunnerBase):
    def get_model(self, batch_size=2, single_device=False):
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

        test_program = fluid.default_main_program().clone(for_test=True)

        # Reader
        train_reader = paddle.batch(
            paddle.dataset.mnist.test(), batch_size=batch_size)
        test_reader = paddle.batch(
            paddle.dataset.mnist.test(), batch_size=batch_size)

        optimizer = paddle.fluid.optimizer.Adam(0.01)
        if single_device:
            optimizer.minimize(avg_cost)
        else:
            role = role_maker.PaddleCloudRoleMaker(is_collective=True)
            fleet.init(role)
            strategy = paddle.distributed.fleet.DistributedStrategy()
            strategy.without_graph_optimization = True
            strategy.fuse_all_reduce_ops = True
            strategy._calc_comm_same_stream = False
            strategy.fuse_grad_size_in_num = 8
            optimizer = fleet.distributed_optimizer(
                optimizer, strategy=strategy)
            optimizer.minimize(avg_cost)

        return test_program, avg_cost, train_reader, test_reader, batch_acc, predict


if __name__ == "__main__":
    runtime_main(TestFleetMetaOptimizerFuseAllReducePrecision)
