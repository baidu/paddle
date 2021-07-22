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
"""Defination of Role Makers."""
import os
import time
import numpy as np
import warnings
from multiprocessing import Process, Manager

import paddle
import paddle.fluid as fluid
from paddle.distributed.fleet.base.private_helper_function import wait_server_ready

__all__ = []


class Role:
    WORKER = 1
    SERVER = 2
    HETER_WORKER = 3
    ALL = 4


class Gloo(object):
    """
    Gloo is a universal class for barrier and collective communication
    """

    class RENDEZVOUS:
        HDFS = 1
        FILE = 2
        HTTP = 3

    def __init__(self):
        self._worker_comm = None
        self._server_comm = None
        self._nodes_comm = None

        self._comm_world = ["worker", "server", "all"]
        self._err_init = "gloo is not initialized, will not communicator with other nodes"
        self._err_type = "gloo initialized error, please check arguments"
        self._err_world = "argument error, comm_world must in {}".format(
            self._comm_world)

        self._is_initialized = False
        self._init_timeout_seconds = 3600
        self._run_timeout_seconds = 9999999

        self._rendezvous = None
        self._role = None
        self._iface = None

        self._role_id = -1
        self._worker_num = -1
        self._server_num = -1
        self._need_init_all = False

    def init(self,
             rendezvous,
             role,
             role_id,
             worker_num,
             server_num,
             need_init_all=False,
             kwargs=None):

        self._rendezvous = rendezvous
        self._role = role
        self._role_id = role_id
        self._worker_num = worker_num
        self._server_num = server_num
        self._need_init_all = need_init_all
        self._iface = ""
        self._prefix = kwargs.get("store.prefix", "")

        http_server = None
        if self._rendezvous == Gloo.RENDEZVOUS.HDFS:
            dfs_name = kwargs.get("dfs.name", "")
            dfs_ugi = kwargs.get("dfs.ugi", "")
            dfs_path = kwargs.get("dfs.path", "")

            if not dfs_name or not dfs_ugi or not dfs_path:
                raise ValueError(self._err_type)
            self._init_dfs(dfs_name, dfs_ugi, dfs_path, self._prefix)

        elif self._rendezvous == Gloo.RENDEZVOUS.FILE:
            fs_path = kwargs.get("dfs.path", "")

            if not fs_path:
                raise ValueError(self._err_type)
            self._init_fs(fs_path, self._prefix)

        elif self._rendezvous == Gloo.RENDEZVOUS.HTTP:
            ip = kwargs.get("http.host", "")
            port = kwargs.get("http.port", "")
            start_http_server = kwargs.get("start_http_server", False)
            http_server_d = kwargs.get("http_server_d")

            if not ip or not port:
                raise ValueError(self._err_type)
            http_server = self._init_http(ip, port, self._prefix,
                                          start_http_server, http_server_d)
        else:
            raise ValueError(self._err_type)

        self._is_initialized = True
        self._http_server = http_server

    def _init_fs(self, fs_path, prefix):
        def init(rank, nodes, role):
            gloo = fluid.core.Gloo()
            gloo.set_rank(rank)
            gloo.set_size(nodes)
            gloo.set_prefix(prefix)
            gloo.set_iface(self._iface)
            gloo.set_timeout_seconds(self._init_timeout_seconds,
                                     self._run_timeout_seconds)
            gloo.set_hdfs_store(os.path.join(fs_path, role), "", "")
            gloo.init()
            return gloo

        if self._role == Role.WORKER:
            rank, nodes = self._get_rank_nodes(Role.WORKER)
            gloo = init(rank, nodes, "WORKER")
            self._worker_comm = gloo
        else:
            rank, nodes = self._get_rank_nodes(Role.SERVER)
            gloo = init(rank, nodes, "SERVER")
            self._server_comm = gloo

        if self._need_init_all:
            rank, nodes = self._get_rank_nodes(Role.ALL)
            gloo = init(rank, nodes, "ALL")
            self._nodes_comm = gloo

    def _init_dfs(self, dfs_name, dfs_ugi, dfs_path, prefix):
        def init(rank, nodes, role):
            gloo = fluid.core.Gloo()
            gloo.set_rank(rank)
            gloo.set_size(nodes)
            gloo.set_prefix(prefix)
            gloo.set_iface(self._iface)
            gloo.set_timeout_seconds(self._init_timeout_seconds,
                                     self._run_timeout_seconds)
            gloo.set_hdfs_store(os.path.join(dfs_path, role), dfs_name, dfs_ugi)
            gloo.init()
            return gloo

        if self._role == Role.WORKER:
            rank, nodes = self._get_rank_nodes(Role.WORKER)
            gloo = init(rank, nodes, "WORKER")
            self._worker_comm = gloo
        else:
            rank, nodes = self._get_rank_nodes(Role.SERVER)
            gloo = init(rank, nodes, "SERVER")
            self._server_comm = gloo

        if self._need_init_all:
            rank, nodes = self._get_rank_nodes(Role.ALL)
            gloo = init(rank, nodes, "ALL")
            self._nodes_comm = gloo

    def _init_http(self, ip, port, prefix, start_http_server, http_server_d):
        def __start_kv_server(http_server_d, size_d):
            print("start http_server: {}, {}".format(port, size_d))
            from paddle.distributed.fleet.utils.http_server import KVServer
            http_server = KVServer(port, size_d)
            http_server.start()
            wait_seconds = 5
            while http_server_d.get("running",
                                    False) or not http_server.should_stop():
                time.sleep(wait_seconds)
            http_server.stop()

        def init_kv_server(http_server_d):
            worker_key = prefix + '_' + 'worker'
            size_d = {worker_key: self._worker_num, }
            print("worker_key:{}, size: {}".format(worker_key, size_d))

            http_server_d["running"] = True
            # child process for http server
            _http_server = Process(
                target=__start_kv_server, args=(http_server_d, size_d))
            _http_server.daemon = True
            # set running status to True
            # start child process
            _http_server.start()
            return _http_server

        def init(rank, nodes, role):
            gloo = fluid.core.Gloo()
            gloo.set_rank(rank)
            gloo.set_size(nodes)
            gloo.set_prefix(prefix)
            gloo.set_iface(self._iface)
            gloo.set_timeout_seconds(self._init_timeout_seconds,
                                     self._run_timeout_seconds)
            gloo.set_http_store(ip, port, 'worker')
            ep = ":".join([ip, str(port)])
            wait_server_ready([ep])
            gloo.init()
            return gloo

        port = int(port)

        if start_http_server:
            print("to start http_server")
            http_server = init_kv_server(http_server_d)

        if self._role == Role.WORKER:
            rank, nodes = self._get_rank_nodes(Role.WORKER)
            gloo = init(rank, nodes, "WORKER")
            self._worker_comm = gloo
        # TODO (sandyhouse): initialize gloo for server and all

        if start_http_server:
            http_server_d["running"] = False
            http_server.join()

    def _get_rank_nodes(self, role):
        nodes = 0
        rank = -1

        if role == Role.WORKER:
            nodes = self._worker_num
            rank = self._role_id
        elif role == Role.SERVER:
            nodes = self._server_num
            rank = self._role_id
        elif role == Role.ALL:
            nodes = self._worker_num + self._server_num

            if self._role == Role.WORKER:
                rank = self._role_id
            else:
                rank = self._worker_num + self._role_id
        else:
            ValueError(self._err_type)

        return rank, nodes

    def __get_default_iface(self):
        """
        get default physical interface
        """
        default1 = self.__get_default_iface_from_gateway()
        default2 = self.__get_default_iface_from_interfaces()
        return default2 if default1 == "lo" else default1

    def __get_default_iface_from_gateway(self):
        """
        get default physical interface
        """
        res = os.popen("route -A inet").read().strip().split("\n")

        gateway_idx = None
        iface_idx = None
        for item in res:
            item = item.split()
            if "Gateway" in item and "Iface" in item:
                gateway_idx = item.index("Gateway")
                iface_idx = item.index("Iface")
            elif gateway_idx != None and iface_idx != None:
                gateway = None
                if len(item) > gateway_idx:
                    gateway = item[gateway_idx]
                if gateway and gateway != '*' and gateway != "0.0.0.0" and len(
                        item) > iface_idx:
                    return item[iface_idx]
        return "lo"

    def __get_default_iface_from_interfaces(self):
        """
        get default physical interface
        """
        res = os.popen("ip -f inet addr | awk NR%3==1").read().strip().split(
            "\n")
        for item in res:
            if "BROADCAST" in item:
                return item.split(":")[1].strip()
        return "lo"

    def barrier(self, comm_world):
        """
        dummy barrier, do nothing
        """
        if not self._is_initialized:
            warnings.warn(self._err_init)
            return

        if comm_world not in self._comm_world:
            raise ValueError(self._err_world)

        if comm_world == "worker":
            self._worker_comm.barrier()
        elif comm_world == "server":
            self._server_comm.barrier()
        else:
            self._nodes_comm.barrier()

    def all_reduce(self, input, mode="sum", comm_world="worker"):
        if not self._is_initialized:
            warnings.warn(self._err_init)
            return input

        if comm_world not in self._comm_world:
            raise ValueError(self._err_world)

        input = np.array(input)
        input_shape = input.shape
        input_list = input.reshape(-1).tolist()

        self.barrier(comm_world)

        if comm_world == "worker":
            ans = self._worker_comm.all_reduce(input_list, mode)
        elif comm_world == "server":
            ans = self._server_comm.all_reduce(input_list, mode)
        else:
            ans = self._nodes_comm.all_reduce(input_list, mode)

        output = np.array(ans).reshape(input_shape)
        return output

    def all_gather(self, input, comm_world="worker"):
        """
        dummy all gather, do nothing
        Args:
            obj(any): obj to do all gather
        """
        if not self._is_initialized:
            warnings.warn(self._err_init)
            return input

        if comm_world not in self._comm_world:
            raise ValueError(self._err_world)

        if comm_world == "worker":
            output = self._worker_comm.all_gather(input)
        elif comm_world == "server":
            output = self._server_comm.all_gather(input)
        else:
            output = self._nodes_comm.all_gather(input)

        return output

    def all_gather_list(self, input, comm_world="worker"):
        """
        dummy all gather, do nothing
        Args:
            obj(any): obj to do all gather
        """
        if not self._is_initialized:
            warnings.warn(self._err_init)
            return input

        if comm_world not in self._comm_world:
            raise ValueError(self._err_world)

        if comm_world == "worker":
            output = self._worker_comm.all_gather_list(input)
        elif comm_world == "server":
            output = self._server_comm.all_gather_list(input)
        else:
            output = self._nodes_comm.all_gather_list(input)

        return output


class RoleMakerBase(object):
    """
    RoleMakerBase is a base class for assigning a role to current process
    in distributed training.
    A paddle developer can implement RoleMakerBase to design a role maker
    for worker or pserver assignment.
    """

    def __init__(self):
        self._worker_endpoints = []
        self._server_endpoints = []
        self._role_is_generated = False
        self._role = None
        self._current_id = -1

        # for heter parameter server mode
        self._heter_trainer_endpoints = []
        self._heter_trainer_device = "CPU"
        self._is_heter_parameter_server_mode = False

    def _is_worker(self):
        """
        return is_worker() of current process
        """
        raise NotImplementedError("Please implement this method in child class")

    def _is_server(self):
        """
        return is_server() of current process
        """
        raise NotImplementedError("Please implement this method in child class")

    def _is_first_worker(self):
        """
        Check whether the node is the first instance of worker.
        Returns:
            bool: True if this is the first node of worker,
                  False if not.
        """
        raise NotImplementedError("Please implement this method in child class")

    def _worker_num(self):
        """
        Get current total worker number.

        Returns:
            int: worker number
        """
        raise NotImplementedError("Please implement this method in child class")

    def _server_num(self):
        """
        Get current total server number.

        Returns:
            int: server number
        """
        raise NotImplementedError("Please implement this method in child class")

    def _worker_index(self):
        """
        Get current worker id.

        Returns:
            int: node id
        """
        raise NotImplementedError("Please implement this method in child class")

    def _server_index(self):
        """
        Get current server id.

        Returns:
            int: node id
        """
        raise NotImplementedError("Please implement this method in child class")

    def _role_id(self):
        """
        Get current id.

        Returns:
            int: node id
        """
        raise NotImplementedError("Please implement this method in child class")

    def _node_num(self):
        """
        Get the training node number
        Returns:
            int: node num
        """
        raise NotImplementedError("Please implement this method in child class")

    def _get_trainer_endpoints(self):
        """
        return trainer endpoints
        """
        return self._worker_endpoints

    def _get_pserver_endpoints(self):
        """
        return pserver endpoints
        """
        return self._server_endpoints

    def to_string(self):
        return "role: {}, current_id: {}, worker_endpoints: {}, server_endpoints: {}".format(
            self._role, self._current_id, self._worker_endpoints,
            self._server_endpoints)

    def _all_gather(self, input, comm_world="worker"):
        print("warning: RoleMakerBase does not have all gather worker.")
        return None

    def _all_gather_list(self, input, comm_world="worker"):
        print("warning: RoleMakerBase does not have all gather list worker.")
        return None

    def _all_reduce(self, input, mode="sum", comm_world="worker"):
        """
        Args:
            input(list/numpy.array): array of one dim
            output(list/numpy.array): array of one dim
            mode(str): "sum" or "min" or "max"
        """
        print("warning: RoleMakerBase does not have all reduce worker.")
        return None

    def _barrier(self, comm_world):
        """
        barrier between trainers if current role is TRAINER
        """
        print("warning: RoleMakerBase does not have barrier worker.")

    def _is_heter_worker(self):
        """
        Return is_heter_worker() of current process
        """
        warnings.warn("RoleMakerBase does not have function: _is_heter_worker.")
        return False

    def _heter_worker_num(self):
        """
        Get current total heter-worker number.

        Returns:
            int: heter_worker number
        """
        warnings.warn(
            "RoleMakerBase does not have function: _heter_worker_num.")
        return 0

    def _get_heter_worker_endpoints(self):
        """
        Returns:
            string: all heter_trainers'endpoints
        """
        assert self._heter_trainer_endpoints != [], "Heter Worker Endpoints Not initialized"
        return self._heter_trainer_endpoints

    def _get_heter_worker_endpoint(self):
        """
        Returns:
            int: corresponding heter_trainer's endpoint

        e.g: if we have 4 cpu-trainer(default), 2 gpu-trainer(heter)
             then No.0 and No.2 cpu-trainer will work with No.0 gpu-trainer
             and No.1 and No.3 cpu-trainer will work with No.1 gpu-trainer
        """
        assert self._heter_trainer_endpoints != [], "Heter Worker Endpoints Not initialized"
        return self._heter_trainer_endpoints[(self._current_id) %
                                             self._heter_worker_num()]


class PaddleCloudRoleMaker(RoleMakerBase):
    def __init__(self, is_collective=False, **kwargs):
        super(PaddleCloudRoleMaker, self).__init__()
        self._is_collective = is_collective

        self._non_distributed = False

        self._kwargs = kwargs
        self._role_is_generated = False

        self._server_endpoints = []
        self._worker_endpoints = []

        self._gloo = Gloo()  # gloo instance

    def _barrier(self, comm_world):
        self._gloo.barrier(comm_world)

    def _all_gather(self, input, comm_world="worker"):
        return self._gloo.all_gather(input, comm_world)

    def _all_gather_list(self, input, comm_world="worker"):
        length = len(input)
        length_arr = self._gloo.all_gather(length, comm_world)
        max_len = max(length_arr)
        gathered_list = self._gloo.all_gather_list(
            input + [0] * (max_len - length), comm_world)
        output = []
        for i in range(len(length_arr)):
            output = output + gathered_list[i * max_len:i * max_len +
                                            length_arr[i]]
        return output

    def _all_reduce(self, input, mode="sum", comm_world="worker"):
        return self._gloo.all_reduce(input, mode, comm_world)

    def _is_worker(self):
        """
        whether current process is worker
        """
        if not self._role_is_generated:
            self._generate_role()
        return self._role == Role.WORKER

    def _is_server(self):
        """
        whether current process is server
        """
        if not self._role_is_generated:
            self._generate_role()
        return self._role == Role.SERVER

    def _is_first_worker(self):
        """
        whether current process is worker of rank 0
        """
        if not self._role_is_generated:
            self._generate_role()
        return self._role == Role.WORKER and self._current_id == 0

    def _worker_index(self):
        """
        get index of current worker
        """
        if not self._role_is_generated:
            self._generate_role()
        return self._current_id

    def _server_index(self):
        """
        get index of current server
        """
        if not self._role_is_generated:
            self._generate_role()
        return self._current_id

    def _role_id(self):
        """
        get index of current node
        """
        if not self._role_is_generated:
            self._generate_role()
        return self._current_id

    def _worker_num(self):
        """
        retrun the current number of worker
        """
        if not self._role_is_generated:
            self._generate_role()
        return self._trainers_num

    def _server_num(self):
        """
        return the current number of server
        """
        if not self._role_is_generated:
            self._generate_role()
        return len(self._get_pserver_endpoints(
        )) if self._get_pserver_endpoints() is not None else 0

    def _node_num(self):
        """
        return the training node number
        """
        if not self._role_is_generated:
            self._generate_role()
        return self._nodes_num

    def _get_node_num(self):
        """
        return the training node number
        """
        if not self._role_is_generated:
            self._generate_role()
        return self._nodes_num

    def _get_local_rank(self):
        if not self._role_is_generated:
            self._generate_role()
        return self._local_rank

    def _get_local_device_ids(self):
        if not self._role_is_generated:
            self._generate_role()
        return self._local_device_ids

    def _get_world_device_ids(self):
        if not self._role_is_generated:
            self._generate_role()
        return self._world_device_ids

    def _get_trainer_endpoints(self):
        """
        get endpoint of all trainers
        """
        if not self._role_is_generated:
            self._generate_role()
        return self._worker_endpoints

    def _get_pserver_endpoints(self):
        """
        get endpoint of all pservers
        """
        if not self._role_is_generated:
            self._generate_role()
        return self._server_endpoints

    def _is_non_distributed(self):
        """
        Return True if indispensable environment for fleetrun is not found
        (use python-run to launch fleet-code directly)
        """
        if not self._role_is_generated:
            self._generate_role()
        return self._non_distributed

    def _heter_worker_num(self):
        """
        get heter worker nums
        """
        if not self._role_is_generated:
            self._generate_role()
        return self._heter_trainers_num

    def _is_heter_worker(self):
        """
        whether current process is heter worker
        """
        if not self._role_is_generated:
            self._generate_role()
        return self._role == Role.HETER_WORKER

    def _ps_env(self):
        # Environment variable PADDLE_PSERVERS_IP_PORT_LIST must be set
        # format: string(ip:port,ip:port), eg. 127.0.0.1:6001,127.0.0.1:6002
        self._server_endpoints = os.getenv("PADDLE_PSERVERS_IP_PORT_LIST", None)

        if self._server_endpoints is None:
            # back to non_distributed execution.
            self._server_endpoints = ""
            self._trainers_num = 1
            self._role = Role.WORKER
            self._current_id = 0
            self._nodes_num = 1
            self._heter_trainers_num = 0
            self._heter_trainer_endpoints = None
            self._non_distributed = True
            return

        self._server_endpoints = self._server_endpoints.split(",")

        self._worker_endpoints = os.getenv("PADDLE_TRAINER_ENDPOINTS", None)
        if self._worker_endpoints != None:
            self._worker_endpoints = self._worker_endpoints.split(",")
        else:
            self._worker_endpoints = []

        trainers_num = os.getenv("PADDLE_TRAINERS_NUM", None)
        if trainers_num == None:
            raise ValueError(
                "Can not find PADDLE_TRAINERS_NUM, please check your environment."
            )
        trainers_num = int(trainers_num)

        training_role = os.getenv("TRAINING_ROLE", None)
        if training_role == None:
            raise ValueError(
                "Can not find TRAINING_ROLE, please check your environment.")

        if training_role not in ["TRAINER", "PSERVER", "HETER_TRAINER"]:
            raise ValueError(
                "TRAINING_ROLE must be PSERVER or TRAINER or HETER_TRAINER, but get {}, please check your environment.".
                format(training_role))

        # For heter parameter server env setting
        heter_trainer_eplist = os.getenv("PADDLE_HETER_TRAINER_IP_PORT_LIST",
                                         "")
        if heter_trainer_eplist != "":
            try:
                heter_trainer_eplist = os.environ[
                    "PADDLE_HETER_TRAINER_IP_PORT_LIST"].split(",")
            except:
                raise ValueError(
                    "Can not Find PADDLE_HETER_TRAINER_IP_PORT_LIST in env or its format doesn't match the requirement: 'IP:PORT,IP:PORT' ."
                )

            self._is_heter_parameter_server_mode = True
            heter_trainers_num = len(heter_trainer_eplist)
        else:
            self._is_heter_parameter_server_mode = False
            heter_trainers_num = 0

        if training_role == "TRAINER":
            role = Role.WORKER
            current_id = os.getenv("PADDLE_TRAINER_ID", None)
            if current_id == None:
                raise ValueError(
                    "Can not find PADDLE_TRAINER_ID, please check your environment."
                )
            current_id = int(current_id)
            if len(self._worker_endpoints) > 0:
                self._cur_endpoint = self._worker_endpoints[current_id]
        elif training_role == "PSERVER":
            role = Role.SERVER
            port = os.getenv("PADDLE_PORT", None)
            if port == None:
                raise ValueError(
                    "Can not find PADDLE_PORT, please check your environment.")
            ip = os.getenv("POD_IP", None)
            if ip == None:
                raise ValueError(
                    "Can not find POD_IP, please check your environment.")
            self._cur_endpoint = ip + ":" + port
            current_id = self._server_endpoints.index(self._cur_endpoint)
        elif training_role == "HETER_TRAINER":
            role = Role.HETER_WORKER
            cur_port = os.getenv("PADDLE_PORT", None)
            if cur_port == None:
                raise ValueError(
                    "Can not find PADDLE_PORT, please check your environment.")
            cur_ip = os.getenv("POD_IP", None)
            if cur_ip == None:
                raise ValueError(
                    "Can not find POD_IP, please check your environment.")
            curr_endpoint = ":".join([cur_ip, cur_port])
            current_id = heter_trainer_eplist.index(curr_endpoint)

        self._trainers_num = trainers_num
        self._role = role
        self._current_id = current_id
        self._nodes_num = len(
            set([x.split(':')[0] for x in self._worker_endpoints]))
        self._heter_trainers_num = heter_trainers_num
        self._heter_trainer_endpoints = heter_trainer_eplist

    def _collective_env(self):
        self._current_id = int(os.getenv("PADDLE_TRAINER_ID", "0"))
        self._training_role = os.getenv("PADDLE_TRAINING_ROLE", "TRAINER")
        assert (self._training_role == "TRAINER")
        self._role = Role.WORKER
        self._worker_endpoints = os.getenv("PADDLE_TRAINER_ENDPOINTS")
        self._cur_endpoint = os.getenv("PADDLE_CURRENT_ENDPOINT")
        if self._worker_endpoints is None:
            # back to non_distributed execution.
            self._worker_endpoints = "127.0.0.1:6170"
            self._cur_endpoint = self._worker_endpoints
            self._non_distributed = True
        self._worker_endpoints = self._worker_endpoints.split(",")
        self._trainers_num = len(self._worker_endpoints)
        self._nodes_num = len(
            set([x.split(':')[0] for x in self._worker_endpoints]))
        self._local_rank = os.getenv("PADDLE_RANK_IN_NODE")
        self._local_device_ids = os.getenv("PADDLE_LOCAL_DEVICE_IDS")
        self._world_device_ids = os.getenv("PADDLE_WORLD_DEVICE_IDS")

    def _gloo_init(self):
        # PADDLE_WITH_GLOO 1: trainer barrier, 2: all barrier
        use_gloo = int(os.getenv("PADDLE_WITH_GLOO", "0"))
        if use_gloo not in [1, 2]:
            return

        # PADDLE_GLOO_RENDEZVOUS 1: HDFS 2: FILE 3: HTTP
        rendezvous_type = int(os.getenv("PADDLE_GLOO_RENDEZVOUS", "0"))
        prefix = os.getenv("SYS_JOB_ID", "")
        if rendezvous_type not in [
                Gloo.RENDEZVOUS.HDFS, Gloo.RENDEZVOUS.HTTP, Gloo.RENDEZVOUS.FILE
        ]:
            raise ValueError(self._gloo._err_type)

        need_init_all = True if use_gloo == 2 else False

        if rendezvous_type == Gloo.RENDEZVOUS.HDFS:
            dfs_name = os.getenv("PADDLE_GLOO_FS_NAME", "")
            dfs_ugi = os.getenv("PADDLE_GLOO_FS_UGI", "")
            dfs_path = os.getenv("PADDLE_GLOO_FS_PATH", "")
            kwargs = {
                "dfs.name": dfs_name,
                "dfs.ugi": dfs_ugi,
                "dfs.path": dfs_path,
                "store.prefix": prefix,
            }
        elif rendezvous_type == Gloo.RENDEZVOUS.HTTP:
            start_http_server = False
            manager = Manager()
            http_server_d = manager.dict()
            http_server_d["running"] = False
            if self._is_collective:
                ep_rank_0 = self._worker_endpoints[0]
                if self._is_first_worker():
                    start_http_server = True
            else:
                ep_rank_0 = os.getenv("PADDLE_GLOO_HTTP_ENDPOINT", "")
                if self._is_server() and self._server_index() == 0:
                    start_http_server = True
            ip, port = ep_rank_0.split(':')
            kwargs = {
                "http.host": ip,
                "http.port": port,
                "store.prefix": prefix,
                'start_http_server': start_http_server,
                'http_server_d': http_server_d,
            }
        else:
            dfs_path = os.getenv("PADDLE_GLOO_FS_PATH", "")
            kwargs = {
                "dfs.path": dfs_path,
                "store.prefix": prefix,
            }

        if rendezvous_type == Gloo.RENDEZVOUS.HDFS:
            type = "HDFS"
        elif rendezvous_type == Gloo.RENDEZVOUS.HTTP:
            type = "HTTP"
        else:
            type = "FILE"
        print("Gloo init with {}: need_init_all: {}, args: {}".format(
            type, need_init_all, kwargs))

        self._gloo.init(
            rendezvous=rendezvous_type,
            role=self._role,
            role_id=self._role_id(),
            worker_num=self._worker_num(),
            server_num=self._server_num(),
            need_init_all=need_init_all,
            kwargs=kwargs)

        if rendezvous_type == Gloo.RENDEZVOUS.HTTP:
            http_server_d['running'] = False

    def _generate_role(self):
        """
        generate role for role maker
        """
        if not self._role_is_generated:
            if not self._is_collective:
                self._ps_env()
            else:
                self._collective_env()
            self._role_is_generated = True
            if not paddle.fluid.framework.in_dygraph_mode():
                self._gloo_init()


class UserDefinedRoleMaker(PaddleCloudRoleMaker):
    def __init__(self, is_collective=False, init_gloo=False, **kwargs):
        super(UserDefinedRoleMaker, self).__init__(
            is_collective=is_collective, init_gloo=init_gloo, **kwargs)
        self._init_gloo = init_gloo

    def _user_defined_ps_env(self):
        self._server_endpoints = self._kwargs.get("server_endpoints")
        self._worker_endpoints = self._kwargs.get("worker_endpoints", [])
        self._trainers_num = self._kwargs.get("worker_num", 0)

        if self._trainers_num == 0:
            assert (len(self._worker_endpoints) > 0)
            self._trainers_num = len(self._worker_endpoints)

        self._role = self._kwargs.get("role")
        self._current_id = self._kwargs.get("current_id")

        if self._role == Role.WORKER and len(
                self._worker_endpoints) > self._current_id:
            self._cur_endpoint = self._worker_endpoints[self._current_id]
        elif self._role == Role.SERVER:
            self._cur_endpoint = self._server_endpoints[self._current_id]
        self._nodes_num = len(
            set([x.split(':')[0] for x in self._worker_endpoints]))

    def _user_defined_collective_env(self):
        self._worker_endpoints = self._kwargs.get("worker_endpoints")
        self._current_id = self._kwargs.get("current_id")
        self._trainers_num = len(self._worker_endpoints)
        self._training_role = Role.WORKER
        self._nodes_num = len(
            set([x.split(':')[0] for x in self._worker_endpoints]))

    def _generate_role(self):
        """
        generate role for role maker
        """
        if not self._role_is_generated:
            if not self._is_collective:
                self._user_defined_ps_env()
            else:
                self._user_defined_collective_env()
            self._role_is_generated = True
