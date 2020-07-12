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

from dist_test_utils import *

silentremove("test_handle_signal_in_serv_op.flag")
silentremove("test_list_and_serv_run_empty_optimize_block.flag")

import paddle
import paddle.fluid as fluid
import signal
import subprocess
import time
import unittest
from multiprocessing import Process
from op_test import OpTest


def run_pserver(use_cuda, sync_mode, ip, port, trainers, trainer_id):
    remove_ps_flag(os.getpid())
    x = fluid.layers.data(shape=[1], dtype='float32')
    y_predict = fluid.layers.fc(input=x, size=1, act=None)
    y = fluid.layers.data(name='y', shape=[1], dtype='float32')

    # loss function
    cost = fluid.layers.square_error_cost(input=y_predict, label=y)
    avg_cost = fluid.layers.mean(cost)

    # optimizer
    sgd_optimizer = fluid.optimizer.SGD(learning_rate=0.001)
    sgd_optimizer.minimize(avg_cost)

    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
    exe = fluid.Executor(place)

    pserver_endpoints = ip + ":" + port
    current_endpoint = ip + ":" + port

    config = fluid.DistributeTranspilerConfig()
    config.sync_mode = sync_mode
    t = fluid.DistributeTranspiler(config=config)
    t.transpile(
        trainer_id,
        pservers=pserver_endpoints,
        trainers=trainers,
        sync_mode=sync_mode)
    pserver_prog = t.get_pserver_program(current_endpoint)
    pserver_startup = t.get_startup_program(current_endpoint, pserver_prog)
    exe.run(pserver_startup)
    exe.run(pserver_prog)


def run_pserver_with_empty_block(use_cuda, sync_mode, ip, port, trainers,
                                 trainer_id):
    remove_ps_flag(os.getpid())
    x = fluid.layers.data(name='x', shape=[1], dtype='float32')
    y_predict = fluid.layers.fc(input=x, size=1, act=None, bias_attr=False)
    y = fluid.layers.data(name='y', shape=[1], dtype='float32')

    # loss function
    cost = fluid.layers.square_error_cost(input=y_predict, label=y)
    avg_cost = fluid.layers.mean(cost)

    # optimizer
    sgd_optimizer = fluid.optimizer.SGD(learning_rate=0.001)
    sgd_optimizer.minimize(avg_cost)

    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
    exe = fluid.Executor(place)

    ps1 = ip + ":" + str(int(port) + 1)
    ps2 = ip + ":" + port
    pserver_endpoints = ps1 + "," + ps2

    config = fluid.DistributeTranspilerConfig()
    config.sync_mode = sync_mode
    config.slice_var_up = False

    t = fluid.DistributeTranspiler(config=config)
    t.transpile(
        trainer_id,
        pservers=pserver_endpoints,
        trainers=trainers,
        sync_mode=sync_mode)
    pserver_prog = t.get_pserver_program(ps2)

    # pserver2 have no parameter
    assert (len(pserver_prog.blocks) == 2)
    assert (len(pserver_prog.blocks[1].ops) == 0)

    pserver_startup = t.get_startup_program(ps2, pserver_prog)
    exe.run(pserver_startup)
    exe.run(pserver_prog)


def gen_complete_file_flag(flag_file):
    with open(flag_file, "w") as f:
        f.write("complete")


class TestListenAndServOp(unittest.TestCase):
    def setUp(self):
        self.ps_timeout = 200
        self.ip = "127.0.0.1"
        self.port = "0"
        self.trainers = 1
        self.trainer_id = 0

    def _start_pserver(self, use_cuda, sync_mode, pserver_func):
        p = Process(
            target=pserver_func,
            args=(use_cuda, sync_mode, self.ip, self.port, self.trainers,
                  self.trainer_id))
        p.daemon = True
        p.start()
        return p

    def _wait_ps_ready(self, pid):
        start_left_time = self.ps_timeout
        sleep_time = 0.5
        while True:
            assert start_left_time >= 0, "wait ps ready failed"
            time.sleep(sleep_time)
            try:
                # the listen_and_serv_op would touch a file which contains the listen port
                # on the /tmp directory until it was ready to process all the RPC call.
                os.stat("/tmp/paddle.%d.port" % pid)
                return
            except os.error:
                start_left_time -= sleep_time

    def test_rpc_interfaces(self):
        # TODO(Yancey1989): need to make sure the rpc interface correctly.
        pass

    def test_handle_signal_in_serv_op(self):
        # run pserver on CPU in sync mode
        p1 = self._start_pserver(False, True, run_pserver)
        print("test_handle_signal_in_serv_op before _wait_ps_ready")
        self._wait_ps_ready(p1.pid)

        # raise SIGTERM to pserver
        os.kill(p1.pid, signal.SIGINT)
        print("test_handle_signal_in_serv_op after kill pid:", p1.pid)
        p1.join()

        # run pserver on CPU in async mode
        p2 = self._start_pserver(False, False, run_pserver)
        print("test_handle_signal_in_serv_op after start p2 pid:", p2.pid)
        self._wait_ps_ready(p2.pid)

        # raise SIGTERM to pserver
        os.kill(p2.pid, signal.SIGTERM)
        print("test_handle_signal_in_serv_op before join p2 pid:", p2.pid)
        p2.join()

        gen_complete_file_flag("test_handle_signal_in_serv_op.flag")

    def test_list_and_serv_run_empty_optimize_block(self):
        # run pserver on CPU in sync mode
        p1 = self._start_pserver(False, True, run_pserver_with_empty_block)
        print(
            "test_list_and_serv_run_empty_optimize_block before _wait_ps_ready")
        self._wait_ps_ready(p1.pid)

        # raise SIGTERM to pserver
        os.kill(p1.pid, signal.SIGINT)
        print("test_list_and_serv_run_empty_optimize_block after kill pid:",
              p1.pid)
        p1.join()

        # run pserver on CPU in async mode
        p2 = self._start_pserver(False, False, run_pserver_with_empty_block)
        print("test_list_and_serv_run_empty_optimize_block after start p2 pid:",
              p2.pid)
        self._wait_ps_ready(p2.pid)

        # raise SIGTERM to pserver
        os.kill(p2.pid, signal.SIGTERM)
        print("test_list_and_serv_run_empty_optimize_block before join p2 pid:",
              p2.pid)
        p2.join()
        gen_complete_file_flag(
            "test_list_and_serv_run_empty_optimize_block.flag")


if __name__ == '__main__':
    unittest.main()
