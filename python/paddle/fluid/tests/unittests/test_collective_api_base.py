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
import numpy as np
import unittest
import time
import argparse
import os
import six
import sys
import subprocess
import traceback
import functools
import pickle
from contextlib import closing
from six import string_types
import paddle
import paddle.fluid as fluid
import paddle.fluid.unique_name as nameGen
from paddle.fluid import core


class TestCollectiveAPIRunnerBase(object):
    def get_model(self, train_prog, startup_prog, rank):
        raise NotImplementedError(
            "get model should be implemented by child class.")

    def wait_server_ready(self, endpoints):
        assert not isinstance(endpoints, string_types)
        while True:
            all_ok = True
            not_ready_endpoints = []
            for ep in endpoints:
                ip_port = ep.split(":")
                with closing(
                        socket.socket(socket.AF_INET,
                                      socket.SOCK_STREAM)) as sock:
                    sock.settimeout(2)
                    result = sock.connect_ex((ip_port[0], int(ip_port[1])))
                    if result != 0:
                        all_ok = False
                        not_ready_endpoints.append(ep)
            if not all_ok:
                sys.stderr.write("server not ready, wait 3 sec to retry...\n")
                sys.stderr.write("not ready endpoints:" + str(
                    not_ready_endpoints) + "\n")
                sys.stderr.flush()
                time.sleep(3)
            else:
                break

    def run_trainer(self, args):
        train_prog = fluid.Program()
        startup_prog = fluid.Program()
        endpoints = args["endpoints"].split(",")
        rank = args["trainerid"]
        current_endpoint = args["currentendpoint"]
        nranks = 2
        result = self.get_model(train_prog, startup_prog, rank)
        paddle.distributed.init_parallel_env()
        if args['backend'] == 'nccl':
            device_id = int(os.getenv("FLAGS_selected_gpus", "0"))
            place = fluid.CUDAPlace(
                device_id)  #if args.use_gpu else fluid.CPUPlace()
        else:
            place = fluid.CPUPlace()
        exe = fluid.Executor(place)
        exe.run(startup_prog)
        np.random.seed(os.getpid())
        indata = np.random.random((10, 1000))
        fetch_list = []
        for elem in result:
            fetch_list.append(elem.name)
        out = exe.run(train_prog,
                      feed={'tindata': indata},
                      fetch_list=fetch_list)
        if six.PY2:
            print(pickle.dumps(out))
        else:
            sys.stdout.buffer.write(pickle.dumps(out))


def runtime_main(test_class, col_type):
    args = {}
    model = test_class()
    args["deviceid"] = os.getenv("FLAGS_selected_gpus")
    args["trainerid"] = int(os.getenv("PADDLE_TRAINER_ID"))
    args["trainernum"] = int(os.getenv("PADDLE_TRAINERS_NUM"))
    args["endpoints"] = os.getenv('PADDLE_TRAINER_ENDPOINTS')
    args["currentendpoint"] = os.getenv("PADDLE_CURRENT_ENDPOINT")
    args["col_type"] = col_type
    args["backend"] = os.getenv("BACKEND")
    args["path_id"] = int(os.getenv("PATH_ID"))
    model.run_trainer(args)


import paddle.compat as cpt
import socket
from contextlib import closing


class TestDistBase(unittest.TestCase):
    def setUp(self):
        self._port_set = set()
        self._trainers = 2
        self._ps_endpoints = "127.0.0.1:%s,127.0.0.1:%s" % (
            self._find_free_port(), self._find_free_port())
        self._python_interp = sys.executable

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

    def _run_cluster(self, model_file, envs):
        worker_endpoints = self._ps_endpoints.split(",")
        w0_ep, w1_ep = worker_endpoints
        #print("w0_ep:",w0_ep," w1_ep:",w1_ep)
        env0 = {
            "FLAGS_selected_gpus": "0",
            "PADDLE_TRAINER_ID": "0",
            "PADDLE_TRAINERS_NUM": "2",
            "PADDLE_TRAINER_ENDPOINTS": self._ps_endpoints,
            "PADDLE_CURRENT_ENDPOINT": w0_ep
        }

        env1 = {
            "FLAGS_selected_gpus": "1",
            "PADDLE_TRAINER_ID": "1",
            "PADDLE_TRAINERS_NUM": "2",
            "PADDLE_TRAINER_ENDPOINTS": self._ps_endpoints,
            "PADDLE_CURRENT_ENDPOINT": w1_ep
        }
        #update environment
        env0.update(envs)
        env1.update(envs)
        tr_cmd = "%s %s"
        tr0_cmd = tr_cmd % (self._python_interp, model_file)
        tr1_cmd = tr_cmd % (self._python_interp, model_file)
        tr0_pipe = open("/tmp/tr0_err.log", "w")
        tr1_pipe = open("/tmp/tr1_err.log", "w")
        #print(tr0_cmd) 
        tr0_proc = subprocess.Popen(
            tr0_cmd.strip().split(),
            stdout=subprocess.PIPE,
            stderr=tr0_pipe,
            env=env0)

        tr1_proc = subprocess.Popen(
            tr0_cmd.strip().split(),
            stdout=subprocess.PIPE,
            stderr=tr1_pipe,
            env=env1)

        tr0_out, tr0_err = tr0_proc.communicate()
        tr1_out, tr1_err = tr1_proc.communicate()
        sys.stderr.write('trainer 0 stderr: %s\n' % tr0_err)
        sys.stderr.write('trainer 1 stderr: %s\n' % tr1_err)
        # close trainer file
        tr0_pipe.close()
        tr1_pipe.close()
        with open("/tmp/tr0_err.log", "r") as f:
            sys.stderr.write('trainer 0 stderr file: %s\n' % f.read())
        with open("/tmp/tr1_err.log", "r") as f:
            sys.stderr.write('trainer 1 stderr file: %s\n' % f.read())
        return pickle.loads(tr0_out), pickle.loads(
            tr1_out), tr0_proc.pid, tr1_proc.pid

    def check_with_place(self,
                         model_file,
                         col_type,
                         backend="nccl",
                         path_id="0",
                         check_error_log=False,
                         need_envs={}):
        required_envs = {
            "FLAGS_fraction_of_gpu_memory_to_use": "0.15",
            "FLAGS_eager_delete_tensor_gb": "0.0",
            "PATH": os.getenv("PATH"),
            "PYTHONPATH": os.getenv("PYTHONPATH", ""),
            "LD_LIBRARY_PATH": os.getenv("LD_LIBRARY_PATH", ""),
            "LD_PRELOAD": os.getenv("LD_PRELOAD", ""),
            "GLOG_v": "0",
            "NCCL_P2P_DISABLE": "1",
            "BACKEND": backend,
            "PATH_ID": path_id
        }
        required_envs.update(need_envs)
        if check_error_log:
            required_envs["GLOG_v"] = "3"
            required_envs["GLOG_logtostderr"] = "1"
            required_envs["GLOO_LOG_LEVEL"] = "TRACE"
        tr0_out, tr1_out, pid0, pid1 = self._run_cluster(model_file,
                                                         required_envs)
        np.random.seed(pid0)
        input1 = np.random.random((10, 1000))
        np.random.seed(pid1)
        input2 = np.random.random((10, 1000))
        if col_type == "allgather":
            need_result = np.vstack((input1, input2))
            tr_out0 = np.vstack((tr0_out[0], tr0_out[1]))
            tr_out1 = np.vstack((tr1_out[0], tr1_out[1]))
            self.assertTrue(np.allclose(tr_out0, need_result))
            self.assertTrue(np.allclose(tr_out1, need_result))
        elif col_type == "broadcast":
            need_result = input2
            self.assertTrue(np.allclose(tr0_out, need_result))
            self.assertTrue(np.allclose(tr1_out, need_result))
        elif col_type == "reduce":
            need_result = input1 + input2
            self.assertTrue(np.allclose(tr0_out, need_result))
        elif col_type == "scatter":
            need_result = input2
            need_result1 = need_result[0:need_result.shape[0] // 2]
            need_result2 = need_result[need_result.shape[0] // 2:]
            self.assertTrue(np.allclose(tr0_out, need_result1))
            self.assertTrue(np.allclose(tr1_out, need_result2))
        elif col_type == "allreduce":
            need_result = input1 + input2
            self.assertTrue(
                np.allclose(
                    tr0_out, need_result, rtol=1e-05, atol=1e-05))
            self.assertTrue(
                np.allclose(
                    tr1_out, need_result, rtol=1e-05, atol=1e-05))
        else:
            pass
