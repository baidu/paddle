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
import os
import unittest
import numpy as np
import tempfile
import shutil
from op_test import OpTest, randomize_probability
import paddle
import paddle.fluid as fluid
import paddle.fluid.layers as layers
import paddle.distributed.fleet.base.role_maker as role_maker
from paddle.distributed.fleet import fleet


class TestSparseLoadProgram(unittest.TestCase):
    """ 
    Test Sparse load operator.
    """

    def setUp(self):
        os.environ[
            "PADDLE_PSERVERS_IP_PORT_LIST"] = "127.0.0.1:4001,127.0.0.1:4002"
        os.environ["PADDLE_TRAINERS_NUM"] = str(2)
        os.environ["TRAINING_ROLE"] = "PSERVER"
        os.environ["PADDLE_PORT"] = "4001"
        os.environ["POD_IP"] = "127.0.0.1"
        role = role_maker.PaddleCloudRoleMaker()
        fleet.init(role)
        self.strategy = paddle.distributed.fleet.DistributedStrategy()
        self.strategy.a_sync = True

    def net(self):
        train_program = fluid.Program()
        startup_program = fluid.Program()
        scope = fluid.Scope()
        with fluid.scope_guard(scope):
            with fluid.program_guard(train_program, startup_program):
                with fluid.unique_name.guard():
                    inputs = fluid.data('input', shape=[None, 1], dtype="int64")
                    emb = fluid.layers.embedding(
                        inputs, is_sparse=True, size=[10000, 128])
                    fc1 = fluid.layers.fc(input=emb, size=128, act="relu")
                    fc2 = fluid.layers.fc(input=fc1, size=64, act="relu")
                    loss = fluid.layers.reduce_mean(fc2)
            return scope, train_program, startup_program, loss


class TestSparseLoadProgramSGD(TestSparseLoadProgram):
    def test_server_init(self):
        scope, train_program, startup_program, loss = self.net()
        with fluid.scope_guard(scope):
            with fluid.program_guard(train_program, startup_program):
                optimizer = fluid.optimizer.SGD(1e-3)
                optimizer = fleet.distributed_optimizer(optimizer,
                                                        self.strategy)
                optimizer.minimize(loss)
                fleet.init_server()


if __name__ == "__main__":
    paddle.enable_static()
    unittest.main()
