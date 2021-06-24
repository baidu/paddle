#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
import sys
import time
import threading
import subprocess
import unittest
import numpy

import paddle
import paddle.fluid as fluid

import paddle.distributed.fleet.base.role_maker as role_maker
import paddle.distributed.fleet as fleet

from paddle.distributed.utils import find_free_ports

paddle.enable_static()


class TestCommunicatorGeoEnd2End(unittest.TestCase):
    def net(self):
        x = fluid.layers.data(name='x', shape=[13], dtype='float32')
        x1 = fluid.layers.data(name='x1', shape=[1], dtype='int64', lod_level=1)

        emb = fluid.layers.embedding(
            input=x1,
            size=[10000, 10],
            param_attr=fluid.ParamAttr(
                name="embedding",
                initializer=fluid.initializer.Constant(value=0.01)),
            is_sparse=True)

        pool = fluid.layers.sequence_pool(input=emb, pool_type="sum")
        z = fluid.layers.concat(input=[x, pool], axis=1)
        y_predict = fluid.layers.fc(input=z, size=1, act=None)
        y = fluid.layers.data(name='y', shape=[1], dtype='float32')

        cost = fluid.layers.square_error_cost(input=y_predict, label=y)
        avg_cost = fluid.layers.mean(cost)
        return avg_cost, x, x1, y

    def fake_reader(self):
        def reader():
            for i in range(10000):
                x = numpy.random.random((1, 13)).astype('float32')
                z = numpy.random.randint(0, 9999, (1, 1)).astype('int64')
                y = numpy.random.randint(0, 2, (1, 1)).astype('int64')
                yield x, z, y

        return reader

    def run_pserver(self, role, strategy):
        fleet.init(role)
        avg_cost, x, z, y = self.net()
        optimizer = fluid.optimizer.SGD(0.01)
        optimizer = fleet.distributed_optimizer(optimizer, strategy)
        optimizer.minimize(avg_cost)

        fleet.init_server()
        fleet.run_server()

    def run_trainer(self, role, strategy):
        place = fluid.core.CPUPlace()
        exe = fluid.Executor(place)

        fleet.init(role)
        avg_cost, x, z, y = self.net()
        optimizer = fluid.optimizer.SGD(0.01)
        optimizer = fleet.distributed_optimizer(optimizer, strategy)
        optimizer.minimize(avg_cost)

        exe.run(fluid.default_startup_program())
        fleet.init_worker()

        train_reader = paddle.batch(self.fake_reader(), batch_size=24)
        feeder = fluid.DataFeeder(place=place, feed_list=[x, z, y])

        for batch_id, data in enumerate(train_reader()):
            exe.run(fluid.default_main_program(),
                    feed=feeder.feed(data),
                    fetch_list=[])

        fleet.stop_worker()

    def run_ut(self):
        training_role = os.getenv("TRAINING_ROLE", "TRAINER")

        os.environ["PADDLE_PSERVER_NUMS"] = "1"
        os.environ["PADDLE_TRAINERS_NUM"] = "1"
        os.environ["PADDLE_TRAINER_ID"] = "0"
        os.environ["PADDLE_TRAINERS_NUM"] = "1"
        os.environ["POD_IP"] = "127.0.0.1"

        role = role_maker.PaddleCloudRoleMaker()

        strategy = paddle.distributed.fleet.DistributedStrategy()
        strategy.a_sync = True
        strategy.a_sync_configs = {"k_steps": 100}
        strategy.a_sync_configs = {"launch_barrier": False}

        if training_role == "TRAINER":
            self.run_trainer(role, strategy)
        else:
            self.run_pserver(role, strategy)

    def test_communicator(self):
        run_server_cmd = """
from __future__ import print_function

import sys
import os

import time
import threading
import subprocess
import unittest
import numpy

import paddle
import paddle.fluid as fluid

from paddle.fluid.communicator import Communicator
import paddle.fluid.incubate.fleet.base.role_maker as role_maker
from paddle.fluid.incubate.fleet.parameter_server.mode import DistributedMode
import paddle.distributed.fleet as fleet

from test_communicator_geo import TestCommunicatorGeoEnd2End

paddle.enable_static()

class RunServer(TestCommunicatorGeoEnd2End):
    def runTest(self):
        pass

os.environ["TRAINING_ROLE"] = "PSERVER"

half_run_server = RunServer()
half_run_server.run_ut()
"""

        server_file = "run_server_for_communicator_geo.py"
        with open(server_file, "w") as wb:
            wb.write(run_server_cmd)

        port = find_free_ports(1).pop()

        os.environ["TRAINING_ROLE"] = "PSERVER"
        os.environ["PADDLE_PORT"] = str(port)
        os.environ["PADDLE_PSERVERS_IP_PORT_LIST"] = "127.0.0.1:{}".format(port)

        _python = sys.executable

        ps_cmd = "{} {}".format(_python, server_file)

        ps_proc = subprocess.Popen(
            ps_cmd.strip().split(" "),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE)

        time.sleep(5)

        os.environ["TRAINING_ROLE"] = "TRAINER"

        self.run_ut()
        ps_proc.kill()
        ps_proc.wait()
        outs, errs = ps_proc.communicate()

        if os.path.exists(server_file):
            os.remove(server_file)


if __name__ == '__main__':
    unittest.main()
