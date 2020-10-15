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
"""
    high level unit test for distribute fleet.
"""

import os
import sys
import subprocess

import six
import shutil
import numpy as np
import argparse
from contextlib import closing
import socket
import time
import tempfile
import unittest

import paddle
import paddle.fluid as fluid
import paddle.distributed.fleet.base.role_maker as role_maker
import paddle.distributed.fleet as fleet

__all__ = ['FleetDistHeterRunnerBase', 'TestFleetHeterBase', 'runtime_main']

RUN_STEP = 5
LEARNING_RATE = 0.01
DIST_UT_PORT = 0


class FleetDistHeterRunnerBase(object):
    """
        run_pserver,run_trainer : after init role, using transpiler split program
        net : implment by child class, the network of model
        do training : exe run program
    """

    def build_role(self, args):
        environs = {}
        environs["PADDLE_PSERVERS_IP_PORT_LIST"] = args.endpoints
        environs["PADDLE_TRAINER_ENDPOINTS"] = args.trainer_endpoints
        environs[
            "PADDLE_HETER_TRAINER_IP_PORT_LIST"] = args.heter_trainer_endpoints
        environs["PADDLE_HETER_TRAINER_DEVICE"] = args.heter_trainer_device
        environs["TRAINING_ROLE"] = args.role.upper()
        environs["PADDLE_TRAINERS_NUM"] = args.trainers
        environs["PADDLE_TRAINER_ID"] = args.current_id
        if args.role.upper() == "PSERVER":
            environs["POD_IP"] = args.endpoints.split(",")[int(
                args.current_id)].split(":")[0]
            environs["PADDLE_PORT"] = args.endpoints.split(",")[int(
                args.current_id)].split(":")[1]
        elif args.role.upper() == "HETER_TRAINER":
            environs["POD_IP"] = args.heter_trainer_endpoints.split(",")[int(
                args.current_id)].split(":")[0]
            environs["PADDLE_PORT"] = args.heter_trainer_endpoints.split(",")[
                int(args.current_id)].split(":")[1]
            environs["FLAGS_selected_gpus"] = args.current_id

        for k, v in environs.items():
            os.environ[k] = str(v)

        self.role = role_maker.PaddleCloudRoleMaker()
        return self.role

    def build_strategy(self, args):
        self.strategy = paddle.distributed.fleet.DistributedStrategy()
        self.strategy.a_sync = True
        self.strategy.a_sync_configs = {"launch_barrier": True}
        return self.strategy

    def build_optimizer(self, avg_cost, strategy):
        optimizer = fluid.optimizer.SGD(LEARNING_RATE)
        optimizer = fleet.distributed_optimizer(optimizer, strategy=strategy)
        optimizer.minimize(avg_cost)

    def run_pserver(self, args):
        fleet.init_server()
        fleet.run_server()

    def run_dataset_trainer(self, args):
        out = self.do_dataset_training(fleet)

    def run_pyreader_trainer(self, args):
        out = self.do_pyreader_training(fleet)

    def net(self, args, batch_size=4, lr=0.01):
        raise NotImplementedError(
            "get_model should be implemented by child classes.")

    def do_dataset_training(self, fleet):
        raise NotImplementedError(
            "do_dataset_training should be implemented by child classes.")

    def do_pyreader_training(self, fleet):
        raise NotImplementedError(
            "do_pyreader_training should be implemented by child classes.")


class TestFleetHeterBase(unittest.TestCase):
    """
        start_pserver,start_trainer : add start cmd to test
        run_cluster : using multi process to test distribute program
    """

    def _setup_config(self):
        raise NotImplementedError("tests should have _setup_config implemented")

    def tearDown(self):
        t = time.time() - self.startTime
        print('%s: %.3f' % (self.__class__.__name__, t))

    def setUp(self):
        self.startTime = time.time()

        self._mode = "async"
        self._reader = "pyreader"
        self._trainers = 2
        self._pservers = 2
        self._port_set = set()

        self._heter_device = "gpu"

        global DIST_UT_PORT
        if DIST_UT_PORT == 0 and os.getenv("PADDLE_DIST_UT_PORT"):
            DIST_UT_PORT = int(os.getenv("PADDLE_DIST_UT_PORT"))

        if DIST_UT_PORT:
            print("set begin_port:", DIST_UT_PORT)
            self._ps_endpoints = "127.0.0.1:%s,127.0.0.1:%s" % (
                DIST_UT_PORT, DIST_UT_PORT + 1)
            self._tr_endpoints = "127.0.0.1:%s,127.0.0.1:%s" % (
                DIST_UT_PORT + 2, DIST_UT_PORT + 3)
            self._heter_endpoints = "127.0.0.1:%s,127.0.0.1:%s" % (
                DIST_UT_PORT + 4, DIST_UT_PORT + 5)
            DIST_UT_PORT += 6
        else:
            self._ps_endpoints = "127.0.0.1:%s,127.0.0.1:%s" % (
                self._find_free_port(), self._find_free_port())
            self._tr_endpoints = "127.0.0.1:%s,127.0.0.1:%s" % (
                self._find_free_port(), self._find_free_port())
            self._heter_endpoints = "127.0.0.1:%s,127.0.0.1:%s" % (
                self._find_free_port(), self._find_free_port())

        self._python_interp = sys.executable
        self._geo_sgd_need_push_nums = 5
        self._grad_clip_mode = 0
        self._setup_config()

    def _find_free_port(self):
        def __free_port():
            with closing(socket.socket(socket.AF_INET,
                                       socket.SOCK_STREAM)) as s:
                s.bind(('', 0))
                return s.getsockname()[1]

        while True:
            port = __free_port()
            if port not in self._port_set:
                self._port_set.add(port)
                return port

    def _start_pserver(self, cmd, required_envs):
        ps0_cmd, ps1_cmd = cmd.format(0), cmd.format(1)

        ps0_pipe = open(tempfile.gettempdir() + "/ps0_err.log", "wb+")
        ps1_pipe = open(tempfile.gettempdir() + "/ps1_err.log", "wb+")

        ps0_proc = subprocess.Popen(
            ps0_cmd.strip().split(" "),
            stdout=subprocess.PIPE,
            stderr=ps0_pipe,
            env=required_envs)
        ps1_proc = subprocess.Popen(
            ps1_cmd.strip().split(" "),
            stdout=subprocess.PIPE,
            stderr=ps1_pipe,
            env=required_envs)
        return ps0_proc, ps1_proc, ps0_pipe, ps1_pipe

    def _start_trainer(self, cmd, required_envs):
        tr0_cmd, tr1_cmd = cmd.format(0), cmd.format(1)

        tr0_pipe = open(tempfile.gettempdir() + "/tr0_err.log", "wb+")
        tr1_pipe = open(tempfile.gettempdir() + "/tr1_err.log", "wb+")

        tr0_out = open(tempfile.gettempdir() + "/tr0_out.log", "wb+")
        tr1_out = open(tempfile.gettempdir() + "/tr1_out.log", "wb+")

        tr0_proc = subprocess.Popen(
            tr0_cmd.strip().split(" "),
            stdout=tr0_out,
            stderr=tr0_pipe,
            env=required_envs)
        tr1_proc = subprocess.Popen(
            tr1_cmd.strip().split(" "),
            stdout=tr1_out,
            stderr=tr1_pipe,
            env=required_envs)

        return tr0_proc, tr1_proc, tr0_pipe, tr1_pipe

    def _start_heter_trainer(self, cmd, required_envs):
        heter0_cmd, heter1_cmd = cmd.format(0), cmd.format(1)

        heter0_pipe = open(tempfile.gettempdir() + "/heter0_err.log", "wb+")
        heter1_pipe = open(tempfile.gettempdir() + "/heter1_err.log", "wb+")
        heter0_out = open(tempfile.gettempdir() + "/heter0_out.log", "wb+")
        heter1_out = open(tempfile.gettempdir() + "/heter1_out.log", "wb+")

        heter0_proc = subprocess.Popen(
            heter0_cmd.strip().split(" "),
            stdout=heter0_out,
            stderr=heter0_pipe,
            env=required_envs)
        heter1_proc = subprocess.Popen(
            heter1_cmd.strip().split(" "),
            stdout=heter1_out,
            stderr=heter1_pipe,
            env=required_envs)

        return heter0_proc, heter1_proc, heter0_pipe, heter1_pipe

    def _run_cluster(self, model, envs):
        env = {
            'GRAD_CLIP': str(self._grad_clip_mode),
            'FLAGS_eager_delete_tensor_gb': str(-1)
        }
        python_path = self._python_interp
        gloo_path = tempfile.mkdtemp()

        if os.getenv('WITH_COVERAGE', 'OFF') == 'ON':
            envs['COVERAGE_FILE'] = os.getenv('COVERAGE_FILE', '')
            python_path += " -m coverage run --branch -p"
        env.update(envs)

        tr_cmd = "{0} {1} --role trainer --endpoints {2} --trainer_endpoints {3} --current_id {{}} --trainers {4} --mode {5} --geo_sgd_need_push_nums {6} --reader {7} --gloo_path {8} --heter_trainer_endpoints {9} --heter_trainer_device {10}".format(
            python_path, model, self._ps_endpoints, self._tr_endpoints,
            self._trainers, self._mode, self._geo_sgd_need_push_nums,
            self._reader, gloo_path, self._heter_endpoints, self._heter_device)

        ps_cmd = "{0} {1} --role pserver --endpoints {2} --trainer_endpoints {3} --current_id {{}} --trainers {4} --mode {5} --geo_sgd_need_push_nums {6} --reader {7} --gloo_path {8} --heter_trainer_endpoints {9} --heter_trainer_device {10}".format(
            python_path, model, self._ps_endpoints, self._tr_endpoints,
            self._trainers, self._mode, self._geo_sgd_need_push_nums,
            self._reader, gloo_path, self._heter_endpoints, self._heter_device)

        heter_cmd = "{0} {1} --role heter_trainer --endpoints {2} --trainer_endpoints {3} --current_id {{}} --trainers {4} --mode {5} --geo_sgd_need_push_nums {6} --reader {7} --gloo_path {8} --heter_trainer_endpoints {9} --heter_trainer_device {10}".format(
            python_path, model, self._ps_endpoints, self._tr_endpoints,
            self._trainers, self._mode, self._geo_sgd_need_push_nums,
            self._reader, gloo_path, self._heter_endpoints, self._heter_device)

        # Run dist train to compare with local results
        ps0, ps1, ps0_pipe, ps1_pipe = self._start_pserver(ps_cmd, env)
        tr0, tr1, tr0_pipe, tr1_pipe = self._start_trainer(tr_cmd, env)
        heter0, heter1, heter0_pipe, heter1_pipe = self._start_heter_trainer(
            heter_cmd, env)

        # Wait until trainer process terminate
        while True:
            stat0 = tr0.poll()
            time.sleep(0.1)
            if stat0 is not None:
                break

        while True:
            stat1 = tr1.poll()
            time.sleep(0.1)
            if stat1 is not None:
                break

        tr0_out, tr0_err = tr0.communicate()
        tr1_out, tr1_err = tr1.communicate()
        print("tr end communicate")

        tr0_ret = tr0.returncode
        tr1_ret = tr1.returncode

        # close trainer file
        tr0_pipe.close()
        tr1_pipe.close()
        ps0_pipe.close()
        ps1_pipe.close()
        heter0_pipe.close()
        heter1_pipe.close()

        ps0.terminate()
        ps1.terminate()
        heter0.terminate()
        heter1.terminate()
        self.assertEqual(tr0_ret, 0, "something wrong in tr0, please check")
        self.assertEqual(tr1_ret, 0, "something wrong in tr1, please check")
        shutil.rmtree(gloo_path)
        return 0, 0

    def check_with_place(self,
                         model_file,
                         delta=1e-3,
                         check_error_log=False,
                         need_envs={}):
        required_envs = {
            "PATH": os.getenv("PATH", ""),
            "PYTHONPATH": os.getenv("PYTHONPATH", ""),
            "LD_LIBRARY_PATH": os.getenv("LD_LIBRARY_PATH", ""),
            "FLAGS_rpc_deadline": "5000",  # 5sec to fail fast
            "http_proxy": ""
        }

        required_envs.update(need_envs)

        if check_error_log:
            required_envs["GLOG_v"] = "3"
            required_envs["GLOG_logtostderr"] = "1"

        tr0_losses, tr1_losses = self._run_cluster(model_file, required_envs)


def runtime_main(test_class):
    parser = argparse.ArgumentParser(description='Run Fleet test.')
    parser.add_argument(
        '--role',
        type=str,
        required=True,
        choices=['pserver', 'trainer', 'heter_trainer'])
    parser.add_argument('--endpoints', type=str, required=False, default="")
    parser.add_argument(
        '--trainer_endpoints', type=str, required=False, default="")
    parser.add_argument(
        '--heter_trainer_endpoints', type=str, required=False, default="")
    parser.add_argument(
        '--heter_trainer_device', type=str, required=False, default="gpu")
    parser.add_argument('--gloo_path', type=str, required=False, default="")
    parser.add_argument('--current_id', type=int, required=False, default=0)
    parser.add_argument('--trainers', type=int, required=False, default=1)
    parser.add_argument('--mode', type=str, required=False, default='async')
    parser.add_argument(
        '--geo_sgd_need_push_nums', type=int, required=False, default=2)
    parser.add_argument('--reader', type=str, required=False, default='dataset')
    args = parser.parse_args()

    model = test_class()
    role = model.build_role(args)
    fleet.init(role)
    strategy = model.build_strategy(args)
    avg_cost = model.net(args)
    model.build_optimizer(avg_cost, strategy)

    if args.role == "pserver" or args.role == "heter_trainer":
        model.run_pserver(args)
    else:
        if args.reader == "dataset":
            model.run_dataset_trainer(args)
        else:
            model.run_pyreader_trainer(args)
