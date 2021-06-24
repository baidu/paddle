# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
from test_dist_base import TestDistRunnerBase, runtime_main
from dist_mnist import cnn_model
# from paddle.fluid.incubate.fleet.collective import fleet
import paddle.distributed.fleet as fleet
import paddle.distributed.fleet.base.role_maker as role_maker
import paddle.distributed.fleet.meta_optimizers.sharding as sharding

import os
import sys
import pickle

# Fix seed for test
fluid.default_startup_program().random_seed = 1
fluid.default_main_program().random_seed = 1


def runtime_main():
    import paddle.distributed.fleet as fleet

    # model definition
    train_prog = paddle.fluid.Program()
    startup_prog = paddle.fluid.Program()
    role = role_maker.PaddleCloudRoleMaker(is_collective=True)
    fleet.init(role)
    with fluid.program_guard(train_prog, startup_prog):
        with fluid.unique_name.guard():
            input_x = paddle.fluid.layers.data(
                name="x", shape=[32], dtype='float32')
            input_y = paddle.fluid.layers.data(
                name="y", shape=[1], dtype='int64')

            fc_1 = paddle.fluid.layers.fc(input=input_x, size=64, act='tanh')
            fc_2 = paddle.fluid.layers.fc(input=fc_1, size=256, act='tanh')
            prediction = paddle.fluid.layers.fc(input=[fc_2],
                                                size=2,
                                                act='softmax')
            cost = paddle.fluid.layers.cross_entropy(
                input=prediction, label=input_y)
            avg_cost = paddle.fluid.layers.mean(x=cost)

            strategy = paddle.distributed.fleet.DistributedStrategy()
            strategy.sharding = True
            strategy.sharding_configs = {
                "sharding_segment_strategy": "segment_broadcast_MB",
                "segment_broadcast_MB": 0.2,
                "sharding_degree": 2,
            }

            optimizer = paddle.fluid.optimizer.Momentum(
                learning_rate=0.01, momentum=0.9)
            optimizer = fleet.distributed_optimizer(
                optimizer, strategy=strategy)
            optimizer.minimize(avg_cost)

    # execution
    device_id = int(os.getenv("FLAGS_selected_gpus", "0"))
    place = fluid.CUDAPlace(device_id)
    exe = fluid.Executor(place)
    exe.run(startup_prog)
    dirname = "./ut_sharding_save_model"
    sharding.utils.save_persistables(
        exe, dirname, main_program=train_prog, filename=None)

    out_losses = []
    sys.stdout.buffer.write(pickle.dumps(out_losses))


if __name__ == "__main__":
    #NOTE(liangjianzhong): dist unittest should be imlpement using runtime_main in test_dist_base.py
    # but the runtime_main in test_dist_base.py use the fleet, DistributedStrategy from 
    # paddle.fluid.incubate.fleet.collective which is not support by sharding (paddle.distributed.fleet). 
    # this should be update in future.
    # runtime_main(TestDistMnist2x2)
    runtime_main()
